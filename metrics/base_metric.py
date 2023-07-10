from metrics import DataProvider
""" Basic metric interface. """

class BaseMetric(object):
    """ A base class for a metric. """
    def __init__(self, dp: DataProvider):
        self.dp = dp

    def __call__(self, dataset_name: str, model_name: str, **call_params) -> dict:
        """ Call the metric and compute the results.
        :param dataset_name: The dataset. See DataProvider for possible datasets.
        :param model_name: The model. See DataProvider for possible models. 
        
        Results are returned in a dict (to allow for multiple values being returned)
        """
        raise NotImplementedError("Plese implement this interface in a subclass.")


# An example metric to show how different metrics could possibly be implemented:
class NonDuplicateRate(BaseMetric):
    """ This function computes the percentage of points in the generated dataset that are unique. """
    
    def __init__(self, dp: DataProvider, n_samples = 5):
        super().__init__(dp)
        self.n_samples = n_samples

    def __call__(self, dataset_name: str, model_name: str):
        df = self.dp.get_random_data_sample(model_name,n_samples=self.n_samples)
        old_len = len(df)
        df_new = df.drop_duplicates()
        new_len = len(df_new)
        return {"non_dup_rate": new_len/old_len}

