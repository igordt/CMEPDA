import numpy as np
from sklearn import preprocessing
from nflows.flows.base import Flow
import scipy
import torch


class Preprocessor:
    '''
    Description: 
    '''

    def __init__(self, settings):
        self.index_log = settings['index_log']
        self.range_quantile = settings['range_quantile']
        self.n_quantiles = settings['n_quantiles']
        self.standard_scaler = preprocessing.StandardScaler()
        self.quantile_scaler = preprocessing.QuantileTransformer(n_quantiles=self.n_quantiles,output_distribution='normal')

    def forward(self,data):
        y = np.copy(data)

        for i in self.index_log:
            y[:,i] = np.log(10 + y[:,i])
        
        self.standard_scaler = preprocessing.StandardScaler().fit(y)
        y = self.standard_scaler.transform(y)

        self.quantile_scaler = preprocessing.QuantileTransformer(n_quantiles=self.n_quantiles,output_distribution='normal').fit(y[:,min(self.range_quantile):max(self.range_quantile)+1])
        y[:,min(self.range_quantile):max(self.range_quantile)+1] = self.quantile_scaler.transform(y[:,min(self.range_quantile):max(self.range_quantile)+1])
        data_preprocessed = y
        return data_preprocessed
    
    def backward(self, data_reconstructed_preprocessed):
        data_reconstructed = np.copy(data_reconstructed_preprocessed)
        data_reconstructed[:,min(self.range_quantile):max(self.range_quantile)+1] = self.quantile_scaler.inverse_transform(data_reconstructed[:,min(self.range_quantile):max(self.range_quantile)+1])
        data_reconstructed = self.standard_scaler.inverse_transform(data_reconstructed)
        for i in self.index_log:
            data_reconstructed[:,i] = np.exp(data_reconstructed[:,i]) - 10
        return data_reconstructed
    


class Compressor:
    '''
    Description: 
    '''

    def __init__(self, flow, N, limit):
        self.flow = flow
        self.N = N
        self.limit = limit
        self.maxabsscaler = preprocessing.MinMaxScaler()
        

    def compress(self,data):
        gaus_tensor = self.flow.transform_to_noise(data)
        gaus = (gaus_tensor.cpu().detach().numpy())

        self.maxabsscaler = preprocessing.MaxAbsScaler().fit(gaus)
        gaus = self.maxabsscaler.transform(gaus)
        gaus = self.limit*gaus

        unif = scipy.special.erf(gaus)
        unif = unif * 2**self.N
        data_compressed = unif.astype(int)

        return data_compressed
    
    def decompress(self,data_compressed):
        unif = data_compressed/2**self.N
        gaus = scipy.special.erfinv(unif)

        gaus = self.maxabsscaler.inverse_transform(gaus)
        gaus = gaus/self.limit

        gaus_tensor = torch.tensor(gaus).to('cuda').float()
        data_tensor_decompressed, _ = self.flow._transform.inverse(gaus_tensor)
        data_decompressed = data_tensor_decompressed.cpu().detach().numpy()

        return data_decompressed