# class **Compressor**
A class that performs data compression using a flow model.
The code is contained in the `our_classes.py` file.
## Attributes:

* `flow` : Flow model for data compression.
* `N` : Number of bits used for compression.
* `limit` : Limit value for scaling the compressed data.
* `maxabsscaler` : MinMaxScaler object for scaling.


## Methods:

### **\_\_init\_\_**
Initializes the Compressor object.

**Args**:
* `flow` : Flow model for data compression.
* `N` : Number of bits used for compression.
* `limit` : Limit value for scaling the compressed data.

```
def __init__(self, flow, N, limit):

    self.flow = flow
    self.N = N
    self.limit = limit
    self.maxabsscaler = preprocessing.MinMaxScaler()
```


### **compress**
Compresses the input data.

**Args**:
* `data` : Input data to be compressed.

**Returns**:
* `tuple`: A tuple containing the compressed data, the gaussian distribution, and the uniform distribution.

```
def compress(self,data):
        
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
```


### **decompress**
Decompresses the compressed data.

**Args**:
* `data_compressed` : Compressed data to be decompressed.

**Returns**:
* `tuple`: A tuple containing the decompressed data and the gaussian distribution after erfinv.

```
def decompress(self,data_compressed):

    unif = data_compressed/2**self.N
    gaus = scipy.special.erfinv(unif)

    gaus_post = self.maxabsscaler.inverse_transform(gaus)
    gaus_post = gaus_post/self.limit

    gaus_tensor = torch.tensor(gaus_post).to('cuda').float()
    data_tensor_decompressed, _ = self.flow._transform.inverse(gaus_tensor)
    data_decompressed = data_tensor_decompressed.cpu().detach().numpy()

    return data_decompressed, gaus
```