from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import numpy as np
import pandas as pd
from metrics import BaseMetric, DataProvider

class MLEfficiency(BaseMetric):
    """ This function computes the machine learning efficiency.
        A discriminative model is trained on the synthetic dataset and test with the real dataset.
    """
    
    def __init__(self, dp: DataProvider, model, metric, params=None, encoder=OrdinalEncoder, encoder_params={"handle_unknown":"use_encoded_value", "unknown_value":-1}, normalize=False, fillna=True, use_proba=False, metric_params={}):
        """ Initialize the Machine Learning Efficiency (MLE) Metric.
            :param dp: The data provider.
            :param model: The machine learning estimator that should be used
            :param metric: Evaluation function that can be called on the true labels, and predicted labels, for instance sklearn.accuracy_score
            :params params: Parameter for the constructor of the ML model, stored in a dict
            :param encoder: Encoding of categorical variables
            :param use_proba: Whether to use predict or predict_proba to make the predictions of the ML model. This depends 
                    on the metric used (e.g. accuracy-> predict, roc_score -> predict_proba)
        """ 
        super().__init__(dp)
        self.model = model
        self.metric = metric
        self.params = params if params is not None else {}
        self.encoder = encoder
        self.encoder_params = encoder_params
        self.normalize = normalize
        self.fillna = fillna
        self.use_proba = use_proba
        self.metric_params = metric_params
        
        self.cat_encoder = dict()
        self.sc = dict()
        
        self.X_test = dict()
        self.y_test = dict()
        
    def _prepare_encoder_and_test_data(self, dataset_name, discrete_cols, continuous_cols, label_col, task):
        # Get training data to fit the encoders
        df_train = self.dp.get_full_data("original")
        
        # Fill missing values
        if self.fillna:
            df_train = df_train.fillna(0)
        
        #df_train[discrete_cols] = df_train[discrete_cols].fillna("unk")
        
        # Fit the encoder to convert the categorical features into numbers
        self.cat_encoder[dataset_name] = self.encoder(**self.encoder_params)
        self.cat_encoder[dataset_name].fit(df_train[discrete_cols])
        
        # Fit the encoder to normalize the continuous featurs
        if self.normalize:
            self.sc = StandardScaler()
            self.sc.fit(df_train[continuous_cols])
            
        # If regression task, fit target scaler
        if task == "regression":
            self.sc_label = StandardScaler()
            self.sc_label.fit(df_train[label_col].values.reshape(-1,1))

        # Get the test data set
        df_test = self.dp.get_full_data("original", train=False)
        
        # Fill missing values
        if self.fillna:
            df_test = df_test.fillna(0)
            
        #df_test[discrete_cols] = df_test[discrete_cols].fillna("unk")
        
        # Encode the categorical features in the test data
        encoded_data = pd.DataFrame(self.cat_encoder[dataset_name].transform(df_test[discrete_cols]))
        df_test = df_test.drop(discrete_cols, axis = 1)
        df_test = df_test.join(encoded_data)
        
        # Normalize the contiuous features in the test data
        if self.normalize:
            df_test[continuous_cols] = self.sc.transform(df_test[continuous_cols])
            
        # If regression task, normalize target
        if task == "regression":
            df_test[label_col] = self.sc_label.transform(df_test[label_col].values.reshape(-1,1))
        
        # Split in features and target data
        self.X_test[dataset_name] = df_test.drop(label_col, axis=1).to_numpy()
        self.y_test[dataset_name] = df_test[label_col].to_numpy()

    def __call__(self, dataset_name: str, model_name: str, random_seeds=[512, 13, 23, 28, 21]):
        """ Compute the metric. 
            :param dataset_name: Name of the dataset, used to obtain dataset meta-info from the config file.
            :param model_name: Name of the model to use from the DataProvider
        """
        discrete_cols = self.dp.get_config(dataset_name)["cat_cols"]
        continuous_cols = self.dp.get_config(dataset_name)["num_cols"]
        label_col = self.dp.get_config(dataset_name)["label_col"]
        task = self.dp.get_config(dataset_name)["task"]
        
        if dataset_name not in self.cat_encoder:
            self._prepare_encoder_and_test_data(dataset_name, discrete_cols, continuous_cols, label_col, task)
        
        # Get the (synthetic) to test
        df = self.dp.get_full_data(model_name)
        df = df.reset_index(drop=True)
        
        #df[discrete_cols] = df[discrete_cols].apply(lambda x: x.str.strip())
        #df[label_col] = df[label_col].apply(lambda x: x.strip())
        
        # Fill NaNs
        if self.fillna:
            df = df.fillna(0)
        
        # Encode the categorical features
        encoded_data = pd.DataFrame(self.cat_encoder[dataset_name].transform(df[discrete_cols]))
        df = df.drop(discrete_cols, axis = 1)
        df = df.join(encoded_data)
        
        # Normalize the contiuous features
        if self.normalize:
            df[continuous_cols]  = self.sc.transform(df[continuous_cols])
            
         # If regression task, normalize target
        if task == "regression":
            #df[label_col] = [t.strip() for t in df[label_col].values]
            #df[label_col] = [t if t else "0.0" for t in df[label_col].values]
            df[label_col] = self.sc_label.transform(df[label_col].values.reshape(-1,1))
                    
        # Split in features and target data
        X_train = df.drop(label_col, axis=1).to_numpy()
        y_train = df[label_col].to_numpy()
        
        scores = []
        
        for seed in random_seeds:
            m = self.model(**self.params, random_state=seed)            
            m.fit(X_train, y_train)
            
            if self.use_proba:
                y_pred = m.predict_proba(self.X_test[dataset_name])[:,1]
            else:
                y_pred = m.predict(self.X_test[dataset_name])
            #print(y_pred)
            
            score = self.metric(self.y_test[dataset_name], y_pred, **self.metric_params)
            scores.append(score)
        
        return {"mle_scores": scores, "mle_mean": np.mean(scores), "mle_std": np.std(scores)}
