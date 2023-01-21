# INCREMENTAL CLUSTERING FOR FEDERATED LEARNING ON NON-IID DATA
Code implementing article accepted at IJCNN 2022 Conference: 

Espinoza Castellon et al., Federated learning with incremental clustering for heterogeneous data [2022](https://ieeexplore.ieee.org/abstract/document/9892653).

# Built with
This section lists any major frameworks/libraries used in the code.
* Python 3.8.3
* Pandas 1.0.5
* scipy 1.5.0 
* numpy 1.18.5
* torch 1.7.1
* torchvision 0.8.2
* networkx 2.4
* community 0.15

# Before usage
Note that you should have a folder 'Results' containing subfolders 'labelSwap', 'imageRot' and 'emnist'. Results according to simulation cases will be stored in the corresponding folders.
Additionally, the data was downloaded from [here](https://www.nist.gov/itl/products-and-services/emnist-dataset)[1] in matlab format. You should store the .mat files in a folder 'Data' parallel to folder 'Code'. The structure should be as follows :
```
IncrementalClusteringFL
                      |-- Code
                      |-- Data
```

[1] Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters.

# Usage
Call function main_ic_fl.py with python followed by the arguments of your choice. They are :
  * (str) Non-IID case you want to simulate. Can be 'labelSwap', 'imageRot' or 'emnist'.
  * (int) Number of users you want to create. Each will have the same number of samples.
  * (int) Number of users' local epochs.
  * (int) Number of users' local batch size.
  * (float) Users' local learning rate.
  * (float) Proportion of clients to be sampled at each round.
  * (int) Number of federated rounds before clustering.
  * (int) Number of federated rounds after clustering.
  * (optional bool) True if computation will be made using a GPU.

We can image the following case : 100 users under "labelSwap" non-IID data, whose local parameters are E=5, B=10 and learning rate = 0.01. At each round, 10% of the users are sampled and 200 rounds are made before clustering. After clustering, 5 rounds are performed. We use a GPU to compute this case. The corresponding command line would be :
```bash
python main_ic_fl.py labelSwap 100 5 10 0.01 0.1 200 5 --cuda
```

# Contact
Fabiola Espinoza Castellon - fabiola.espinozacastellon@cea.fr
