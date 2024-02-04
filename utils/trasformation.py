import numpy as np
from sklearn import preprocessing

def prep(original,transformed, dir, index_log, range_quantile, n_quantiles):
    """
    description:
    
    """

    y = np.copy(original)

    for i in index_log:
        y[:,i] = np.log(10 + y[:,i])
    
    scaler = preprocessing.StandardScaler().fit(y)
    y = scaler.transform(y)

    quantile = preprocessing.QuantileTransformer(n_quantiles=n_quantiles,output_distribution='normal').fit(y[:,min(range_quantile):max(range_quantile)+1])
    y[:,min(range_quantile):max(range_quantile)+1] = quantile.transform(y[:,min(range_quantile):max(range_quantile)+1])

    if dir == 'forward':
        return y

    if dir == 'backward':
        transformed[:,min(range_quantile):max(range_quantile)+1] = quantile.inverse_transform(transformed[:,min(range_quantile):max(range_quantile)+1])
        transformed = scaler.inverse_transform(transformed)
        for i in index_log:
            transformed[:,i] = np.exp(transformed[:,i]) - 10
        return transformed
