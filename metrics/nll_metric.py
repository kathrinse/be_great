from sdv.metrics.tabular import BNLogLikelihood, GMLogLikelihood
from metrics import BaseMetric, DataProvider

class NLLMetric(BaseMetric):
    """ This function computes the percentage of points in the generated dataset that are unique. """
    
    def __init__(self, dp: DataProvider, density_model = "bn", fit_on_real=True):
        """ Initialize the NLL Metric.
            Parameters: dp: The data provider.
            density_model: Either "bn" or "gmm". If bn, a bayesian network model is fit to the data to compute NLL (discrete features).
                            If gmm a Gaussian mixture model is used (continous features)
            fit_on_real: If True, the density is fit on the real data, and the NLL of the generated data is computed.
                         If False, the density is fit on the generated data, and NLL of the true data is reported. This corresponds 
                         to the metric proposed by Xu et al. (2019).
        """
        super().__init__(dp)
        self.density_model = density_model
        self.fit_on_real = fit_on_real

    def __call__(self, dataset_name: str, model_name: str):
        data_org = self.dp.get_full_data("original", train=False)
        data_fake = self.dp.get_full_data(model_name, train=False)

        if not self.fit_on_real: # swap
            _tmp = data_org
            data_org = data_fake
            data_fake = _tmp
        
        key = "nll_metric_fit_" + ("real" if self.fit_on_real else "fake")
        if self.density_model == "bn":
            likelihood_provider = BNLogLikelihood()
            res = likelihood_provider.compute(data_org, data_fake)
        elif self.density_model == "gmm":
            likelihood_provider = GMLogLikelihood()
            res = likelihood_provider.compute(data_org, data_fake, n_components=3)
        else:
            raise ValueError("unknown density model.")
    
        return {key: res}

        
