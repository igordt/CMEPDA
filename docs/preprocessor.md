# class **Preprocessor**
A class that performs data preprocessing. 
The code is contained in the `our_classes.py` file.

## Attributes:

* `index_log` : List of indices of columns to apply logarithmic transformation.
* `range_quantile` : Range of columns to apply quantile transformation.
* `n_quantiles` : Number of quantiles to use for quantile transformation.
* `standard_scaler` : StandardScaler object for standardization.
* `quantile_scaler` : QuantileTransformer object for quantile transformation.


## Methods:

### **__init__**
Initializes the Preprocessor object.

**Args**:
* `settings` : Dictionary containing the preprocessing settings.

```
def __init__(self, settings):
    
    self.index_log = settings['index_log']
    self.range_quantile = settings['range_quantile']
    self.n_quantiles = settings['n_quantiles']
    self.standard_scaler = preprocessing.StandardScaler()
    self.quantile_scaler = preprocessing.QuantileTransformer(n_quantiles=self.n_quantiles,output_distribution='normal')
```

### **forward**
Preprocesses the input data.

**Args**:
* `data` : Input data to be preprocessed.

**Returns**:
* `numpy.ndarray`: Preprocessed data.

```
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
```

### **backward**
Reconstructs the preprocessed data.

**Args**:
* `data_reconstructed_preprocessed` : Preprocessed data to be reconstructed.

**Returns**:
* `numpy.ndarray` : Reconstructed data.

```
def backward(self, data_reconstructed_preprocessed):

    data_reconstructed = np.copy(data_reconstructed_preprocessed)
    if self.range_quantile:
        data_reconstructed[:,min(self.range_quantile):max(self.range_quantile)+1] = self.quantile_scaler.inverse_transform(data_reconstructed[:,min(self.range_quantile):max(self.range_quantile)+1])

    data_reconstructed = self.standard_scaler.inverse_transform(data_reconstructed)
    for i in self.index_log:
        data_reconstructed[:,i] = np.exp(data_reconstructed[:,i]) - 10
    return data_reconstructed
```