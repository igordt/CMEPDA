import numpy as np
from sklearn import preprocessing
from nflows.flows.base import Flow
import scipy
import torch


class Preprocessor:
    """
    A class that performs data preprocessing.

    Attributes:
        index_log : List of indices of columns to apply logarithmic transformation.
        range_quantile : Range of columns to apply quantile transformation.
        n_quantiles : Number of quantiles to use for quantile transformation.
        standard_scaler : StandardScaler object for standardization.
        quantile_scaler : QuantileTransformer object for quantile transformation.

    Methods:
        forward(data): Preprocesses the input data.
        backward(data_reconstructed_preprocessed): Reconstructs the preprocessed data.

    """

    def __init__(self, settings):
        """
        Initializes the Preprocessor object.

        Args:
            settings : Dictionary containing the preprocessing settings.

        """
        self.index_log = settings['index_log']
        self.range_quantile = settings['range_quantile']
        self.n_quantiles = settings['n_quantiles']
        self.standard_scaler = preprocessing.StandardScaler()
        self.quantile_scaler = preprocessing.QuantileTransformer(n_quantiles=self.n_quantiles,output_distribution='normal')

    def forward(self,data):
        """
        Preprocesses the input data.

        Args:
            data : Input data to be preprocessed.

        Returns:
            numpy.ndarray: Preprocessed data.

        """
        y = np.copy(data)

        for i in self.index_log:
            y[:,i] = np.log(10 + y[:,i])
        
        self.standard_scaler = preprocessing.StandardScaler().fit(y)
        y = self.standard_scaler.transform(y)

        if self.range_quantile:
            self.quantile_scaler = preprocessing.QuantileTransformer(n_quantiles=self.n_quantiles,output_distribution='normal').fit(y[:,min(self.range_quantile):max(self.range_quantile)+1])
            y[:,min(self.range_quantile):max(self.range_quantile)+1] = self.quantile_scaler.transform(y[:,min(self.range_quantile):max(self.range_quantile)+1])    

        data_preprocessed = y
        return data_preprocessed
    
    def backward(self, data_reconstructed_preprocessed):
        """
        Reconstructs the preprocessed data.

        Args:
            data_reconstructed_preprocessed : Preprocessed data to be reconstructed.

        Returns:
            numpy.ndarray: Reconstructed data.

        """
        data_reconstructed = np.copy(data_reconstructed_preprocessed)
        if self.range_quantile:
            data_reconstructed[:,min(self.range_quantile):max(self.range_quantile)+1] = self.quantile_scaler.inverse_transform(data_reconstructed[:,min(self.range_quantile):max(self.range_quantile)+1])

        data_reconstructed = self.standard_scaler.inverse_transform(data_reconstructed)
        for i in self.index_log:
            data_reconstructed[:,i] = np.exp(data_reconstructed[:,i]) - 10
        return data_reconstructed
    


class Compressor:
    """
    A class that performs data compression using a flow model.

    Attributes:
        flow : Flow model for data compression.
        N : Number of bits used for compression.
        limit : Limit value for scaling the compressed data.
        maxabsscaler : MinMaxScaler object for scaling.

    Methods:
        compress(data): Compresses the input data.
        decompress(data_compressed): Decompresses the compressed data.

    """

    def __init__(self, flow, N, limit):
        """
        Initializes the Compressor object.

        Args:
            flow : Flow model for data compression.
            N : Number of bits used for compression.
            limit : Limit value for scaling the compressed data.

        """
        self.flow = flow
        self.N = N
        self.limit = limit
        self.maxabsscaler = preprocessing.MinMaxScaler()
        
    def compress(self,data):
        """
        Compresses the input data.

        Args:
            data : Input data to be compressed.

        Returns:
            tuple: A tuple containing the compressed data, the gaussian distribution, and the uniform distribution.

        """
        if isinstance(data, np.ndarray):
            data = torch.tensor(data).to('cuda').float()
    
        gaus_tensor = self.flow.transform_to_noise(data)
        gaus = (gaus_tensor.cpu().detach().numpy())

        self.maxabsscaler = preprocessing.MaxAbsScaler().fit(gaus)
        gaus_prep = self.maxabsscaler.transform(gaus)
        gaus_prep = self.limit*gaus_prep

        unif = scipy.special.erf(gaus_prep)
        unif_prep = unif * 2**self.N
        data_compressed = unif_prep.astype(int)

        return data_compressed, gaus, unif
    
    def decompress(self,data_compressed):
        """
        Decompresses the compressed data.

        Args:
            data_compressed : Compressed data to be decompressed.

        Returns:
            tuple: A tuple containing the decompressed data and the gaussian distribution after erfinv.

        """
        unif = data_compressed/2**self.N
        gaus = scipy.special.erfinv(unif)

        gaus_post = self.maxabsscaler.inverse_transform(gaus)
        gaus_post = gaus_post/self.limit

        gaus_tensor = torch.tensor(gaus_post).to('cuda').float()
        data_tensor_decompressed, _ = self.flow._transform.inverse(gaus_tensor)
        data_decompressed = data_tensor_decompressed.cpu().detach().numpy()

        return data_decompressed, gaus