# Feature-Block Importance Quantification
This repository contains information about the experiments of the paper *Ranking Feature-Block Importance in Artificial Multiblock Neural Networks* [[1]](#1). 
Folder description:

### Repository structure
#### data:
- containns all simulated and real world data.

#### evaluations:
- The **figures** folder contains plots of the network structures for Section 3.1 and 3.2.
- **notebooks** stores the jupyter notebooks for the BCW and Servo analysis. In addition, the Spearman's rank correlation coefficient calculations are shown in a separate notebook.

#### raw_results:
- contains calculated metrics.

#### src:
- Folder **M_ANN** contains the code for the M-ANN and help functions in python.
- **data_generation** contains an R-file for the generation of simulated data.
- **evaluation** stores files for evaluating the M-ANN (in Python) and plotting the results in ggplot, together with a Wilcoxon-test for block differences (R).


## Experiments
We set a random seed in all computations to make the results reproducible.
The simulation experiments in Section 3.1 were conducted on a *CentOS Linux 7.9.2009, Intel Xeon(R) CPU E5-2650 @ 2.60GHz, 3 GB RAM* cluster where multiple simulations could be started in parallel. For the real world experiments (Section 3.2) we used a local *Ubuntu 20.04.4 LTS, Intel(R) Core(TM) i5-8265U CPU @ 1.60GHz* % RAM information? The average runtimes on the local machine were 35s (S1a), 4s (BCW), and 10s (Servo). 

In the following we describe the network structure of the different experimental setups and datasets. Though, it is important to mention that the focus of the article is not to optimize the network structure but rather the interpreation of a well-performing network.

### Simulated datasets
An identical, sequential M-ANN architecture is used for all six datasets, S1a-S2b. The M-ANN delivered an R2-score >0.9 for all datasets on a separate test dataset and is therefore, ideal to show our concept. The  structures for input block branches are equal, consisting of 1 input layer and 3 dense layers with 16, 8, and 4 nodes (all with relu activation). The blender network comprises 1 concatenation layer, 2 dense layers with relu activations and 2 nodes each, and 1 output layer containing 1 node (linear activation). We use the RMSprop optimizer to train the network, a batch size of 64, and a learning rate of 1 and 100 epochs. All parameters were determined in a preliminary grid search analysis.

<img src="https://github.com/annajenul/Block_Importance_Quantification/blob/master/evaluations/figures/Simulation_M_ANN.png" width="1400"/> The plot shows the M-ANN structure that is used for all simulation experiments. </img>

### Breast Cancer Wisconsin dataset [[2]](#2)
The M-ANN model consists of three input blocks where the structure for each block is sequential with 3 dense layers and 10 (input), 5, and 2 nodes, respectively, and a "swish" activation function. The blender part concatenates the single blocks and the output via one dense layers with two nodes and "swish" activation. The final output layer has a sigmoid activation. The batch size, epochs, and learning rate are set to 32, 100, and 0.1. Those four parameters were determined with a 3-fold cross validation.

<img src="https://github.com/annajenul/Block_Importance_Quantification/blob/master/evaluations/figures/BCW_M_ANN.png" width="1400"/> The plot shows the M-ANN structure that is used for BCW. </img>

### Servo dataset [[3]](#3)
The M-ANN structure consists of an input layer, mapped directly to the concatenation layer through one layer with the same number of nodes than the input layer and linear activation. The blender part of the network consists of 4 layers with 12, 8, 4, and 2 layers, respectively. Each layer has "swish" activation. The final layer is the output layer with linear activation. To train the network we use a batchsize of 32, 500 epochs, and a learning rate of 5, all determined via a 3-fold cross validation.

<img src="https://github.com/annajenul/Block_Importance_Quantification/blob/master/evaluations/figures/Servo.png" width="1400"/> The plot shows the M-ANN structure that is used for Servo. </img>


## References
<a id="1">[1]</a> 
A. Jenul, S. Schrunner, et al. Ranking Feature-Block Importance in Artificial Multiblock Neural Networks. arXiv, 2022.

<a id="2">[2]</a> 
Street, W.N., Wolberg, W.H., Mangasarian, O.L.: Nuclear feature extraction for
breast tumor diagnosis. In: Acharya, R.S., Goldgof, D.B. (eds.) Biomedical Image
Processing and Biomedical Visualization. vol. 1905, pp. 861 – 870. SPIE (1993),
doi: 10.1117/12.148698.

<a id="3">[3]</a> 
Quinlan, J.R.: Combining instance-based and model-based learning. In: Interna-
tional Conference on Machine Learning. pp. 236–243 (1993).
