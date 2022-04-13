# Feature-Block Importance Quantification
Importance of multiblock artificial neural networks

## Introduction

## Methods

## Experiments
### Simulated datasets
An identical, sequential M-ANN architecture is used for all four datasets, S1-S4. The final structures for input block branches are equal, consisting of 1 input layer and 3 dense layers with 16, 8, and 4 nodes (all with swish activation). The blender network comprises 1 concatenation layer, 2 dense layers with swish activations and 2 nodes each, and 1 output layer containing 1 node (linear activation). We use the RMSprop optimizer to train the network, a batch size of 32, and a learning rate of 1 and 150 epochs. All parameters were determined in a preliminary grid search analysis.

### Breast Cancer Wisconsin dataset [1](#1)
The M-ANN model consists of three input blocks where the structure for each block is sequential with three dense layers and 10 (input), 5, and 2 nodes, respectively, and a "swish" activation function. The blender part concatenates the single blocks and the output via two dense layers with two nodes each and "swish" activation. The batch size, epochs, and learning rate are set to 32, 100, and 0.1.

### Servo dataset
The M-ANN structure consists of an input layer, mapped directly to the concatenation layer. The blender part of the network consists of 4 layers with 12, 8, 4, and 2 layers, respectively. Each layer has "swish" activation. The final layer is the output layer with linear activation. To train the network we use a batchsize of 32, 500 epochs, and a learning rate of 5.


## References
<a id="1">[1]</a> 
Street, W.N., Wolberg, W.H., Mangasarian, O.L.: Nuclear feature extraction for
breast tumor diagnosis. In: Acharya, R.S., Goldgof, D.B. (eds.) Biomedical Image
Processing and Biomedical Visualization. vol. 1905, pp. 861 – 870. SPIE (1993),
doi: 10.1117/12.148698
