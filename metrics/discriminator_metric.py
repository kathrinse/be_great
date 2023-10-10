# basic utils
import numpy as np

# tools to fit the ML classifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# utils to conduct cross-validation
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# utils to get access to original and generated data
from metrics import BaseMetric, DataProvider


class DiscriminatorMetric(BaseMetric):
    """ This function computes the discriminator metric.
        For the training set, we fit a ML model to distinguish between the
        synthetically generated and the original data set. On the test set, 
        we then evaluate the ability of the classifier to distinguish between
        synthetic data points and original data points.
        params: dp DataProvider
        metric: performance measure for the discriminator
        internal_testsplit: If True only the train set is loaded and internally split into train and test.
    """
    
    def __init__(self, dp: DataProvider, metric, n_runs = 10, internal_testsplit = False, params=None, encoder=OrdinalEncoder, encoder_params={"handle_unknown":"use_encoded_value", "unknown_value":-1}):
        super().__init__(dp)

        self.metric = metric
        self.params = params
        self.encoder = encoder
        self.encoder_params = encoder_params
        self.n_runs= n_runs
        self.X = dict()
        self.y = dict()

        self.X_gen = dict()
        self.y_gen = dict()
        self.disc_estimator = None # Best estimator
        self.disc_params = None # Estimtor params.
        self.internal_testsplit = internal_testsplit

    def __call__(self, dataset_name: str, gen_model_name: str):
        """ Compute the metric. 
            :param dataset_name: Name of the dataset, used to obtain dataset meta-info from the config file.
            :param gen_model_name: Name of the model to use from the DataProvider
        """
        discrete_cols = self.dp.get_config(dataset_name)["discrete_columns"]
        label_col = self.dp.get_config(dataset_name)["label_col"]

        # apply appropriate encodings to all data sets
        self._prepare_encoder_and_data(dataset_name, gen_model_name, discrete_cols, label_col)

        # fit the discriminator model
        self._fit_cross_validated_model()

        # compute the discriminator score
        disc_score_mean, disc_score_std = self._compute_disc_score()
              
        return {"discriminator_mean": disc_score_mean,"discriminator_std": disc_score_std}

    def _prepare_encoder_and_data(self, dataset_name, gen_model_name, discrete_cols, label_col):
                
        #  fit cat encoder to the original training data set
        df_train = self.dp.get_full_data("original", train=True)
        feature_order = list(df_train.columns)
        self.cat_encoder = self.encoder(**self.encoder_params)
        self.cat_encoder.fit(df_train[discrete_cols])

        ## Load synthetic train/test and original train/test datasets        
        # original data set
        df_train[discrete_cols] = self.cat_encoder.transform(df_train[discrete_cols]) 

        # gen data set
        df_train_gen = self.dp.get_full_data(gen_model_name, train=True)
        df_train_gen[discrete_cols] = self.cat_encoder.transform(df_train_gen[discrete_cols])
        df_train_gen = df_train_gen[feature_order]

        if self.internal_testsplit:
             df_train_gen, df_test_gen = train_test_split(df_train_gen, test_size=0.2, train_size=0.8, random_state=42, shuffle=False)
             df_train, df_test = train_test_split(df_train, test_size=0.2, train_size=0.8, random_state=42, shuffle=False)
        else:
            df_test = self.dp.get_full_data("original", train=False)
            df_test[discrete_cols] = self.cat_encoder.transform(df_test[discrete_cols]) 

            df_test_gen = self.dp.get_full_data(gen_model_name, train=False)
            df_test_gen[discrete_cols] = self.cat_encoder.transform(df_test_gen[discrete_cols])
            df_test_gen = df_test_gen[feature_order]
        
        ## create final discriminator data set
        # train data
        X_train_original = df_train.to_numpy()
        y_train_original = np.zeros(X_train_original.shape[0])
    
        X_train_gen = df_train_gen.to_numpy()
        y_train_gen = np.ones(X_train_gen.shape[0])

        len_use_train = min(len(X_train_gen), len(X_train_original))
        self.X_train = np.r_[X_train_original[:len_use_train], X_train_gen[:len_use_train]]
        self.y_train = np.r_[y_train_original[:len_use_train], y_train_gen[:len_use_train]]

        # test data
        X_test_original = df_test.to_numpy()
        y_test_original = np.zeros(X_test_original.shape[0])
    
        X_test_gen = df_test_gen.to_numpy()
        y_test_gen = np.ones(X_test_gen.shape[0])
        len_use_test = min(len(X_test_gen), len(X_test_original))
        self.X_test = np.r_[X_test_original[:len_use_test], X_test_gen[:len_use_test]]
        self.y_test = np.r_[y_test_original[:len_use_test], y_test_gen[:len_use_test]]

    def _fit_cross_validated_model(self, cv=5):
        # fit 5-fold cross-validated classifier

        # Number of trees in random forest
        n_estimators = [100] # [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ['sqrt']
        # Maximum number of levels in tree
        max_depth = [10] # [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [2, 3, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        rf = RandomForestClassifier()
    	
        rf_random = GridSearchCV(estimator=rf, 
                                       param_grid=random_grid, 
                                       cv=cv, 
                                       n_jobs=1)


        rf_random.fit(self.X_train, self.y_train)
        #print(rf_random.best_params_)
        self.disc_params = rf_random.best_params_
        self.disc_estimator = RandomForestClassifier

    def _compute_disc_score(self):
        disc_score_list = []
        for i in range(self.n_runs):
            # Initiate classifier with best params but different seeds
            self.disc_params.update({"random_state": i*42})
            mymodel = RandomForestClassifier(**self.disc_params) 
            mymodel.fit(self.X_train, self.y_train)
            pred = mymodel.predict(self.X_test)        
            disc_score_list.append(self.metric(self.y_test, pred))

        disc_list_np = np.array(disc_score_list)
        return disc_list_np.mean(), disc_list_np.std()


