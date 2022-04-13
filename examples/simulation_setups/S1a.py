from operator import pos
import numpy as np
import pandas as pd
import tensorflow as tf
import time

import os
os.chdir("/home/anna/GitHub/Block_Importance_Quantification")
from src.multiblock_network import multiblock_network
import src.help_functions as help_functions
from src.help_functions import remove_outliers

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential, optimizers

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
tf.random.set_seed(42)

X_train = pd.read_csv("examples/data/simulation/X_train.csv").iloc[:,1:]
X_test = pd.read_csv("examples/data/simulation/X_test.csv").iloc[:,1:]
y_train = pd.read_csv("examples/data/simulation/y_train_S1a.csv").iloc[:,1:]
y_test = pd.read_csv("examples/data/simulation/y_test_S1a.csv").iloc[:,1:]
beta = pd.read_csv("examples/data/simulation/beta_S1a.csv").iloc[:,1:].values

data_blocks = []
data_blocks_test = []
beta_blocks = []
j=0
for i in range(8):
    data_blocks.append(X_train.iloc[:,j:(j+32)])
    data_blocks_test.append(X_test.iloc[:,j:(j+32)])
    beta_blocks.append(beta[j:(j+32)])
    j = j+32


num_nodes_concat = 4

def build_block_network(data, activation="elu", learning_rate=0.1, prob = "class", epochs=1000):
    network = multiblock_network()

    for b in data:
            structure1 = [Input(shape=(np.shape(b)[1],)),
                  Dense(16, activation=activation),
                  Dense(8, activation=activation),
                  Dense(num_nodes_concat, activation=activation)]
            network.define_block_net(structure1.copy())

    structure = [Dense(2, activation=activation),
                 Dense(2, activation=activation),
                 Dense(1, activation="linear")]
    #concatenate
    network.define_block_concatenation(structure=structure)

    opt = tf.keras.optimizers.RMSprop()


    if prob =="class":
        network.compile(loss="binary_crossentropy", optimizer=opt,
                       metrics=[help_functions.matthews_correlation, help_functions.f1_m])
    elif prob == "regression":
        network.compile(loss='mean_squared_error', optimizer=opt,
                        metrics=[help_functions.coeff_determination])

    return network


batch_size = 64
epochs = 100
learning_rate = 1
activation = 'relu'


data_blocks_sc = []
data_blocks_test_sc = []

for i in range(len(data_blocks)):
    sc = StandardScaler()
    data_blocks_sc.append(sc.fit_transform(data_blocks[i]))
    data_blocks_test_sc.append(sc.transform(data_blocks_test[i]))


def create_list(num):
        return [None for _ in range(num)]


n = 30

# metrics
rmseiqr = create_list(n)
rmse_q1 = np.quantile(y_test, 0.25)
rmse_q3 = np.quantile(y_test, 0.75)

r2 = create_list(n)

KI = create_list(n)
KO = create_list(n)

vargrad_max = create_list(n)
vargrad_mean = create_list(n)

time_track = create_list(n)

for i in range(n):
    np.random.seed(i)
    tf.random.set_seed(i)
    print(i)
    start = time.time()
    network = build_block_network(data=data_blocks_sc, activation=activation,
                                learning_rate=learning_rate, prob="regression", epochs=epochs)

    network.fit(data_blocks_sc, y_train, epochs=epochs, batch_size=batch_size,
                validation_data = (data_blocks_test_sc,  y_test), verbose=2, problem="regression")
    
    end = time.time()
    time_track[i] = end-start
    
    pred = network.predict(data_blocks_test_sc)
    rmseiqr[i] = (mean_squared_error(y_test, pred, squared=False)/(rmse_q3-rmse_q1))
    r2[i] = r2_score(y_test, pred)
    
    l = 10 # number of bins
    KI[i] = network.MI(data_blocks_sc, type="mean", eps=1e-100, bins=l, knock_out=False, on_input=False,
                        density=True, plot=False)
    
    KO[i] = np.log2(l) - network.MI(data_blocks_sc, type="mean", eps=1e-100, bins=l, knock_out=True, on_input=False,
                         density=True, plot=False)


    vargrad_max[i] = network.vargrad_input(data_blocks_sc, type="max", seed=i)
    vargrad_mean[i] = network.vargrad_input(data_blocks_sc, type="mean", seed=i)

pd.DataFrame(np.vstack(KO)).to_csv("results/simulation/results_S1a/MI_knock_out.csv", index=False)
pd.DataFrame(np.vstack(KI)).to_csv("results/simulation/results_S1a/MI_knock_in.csv", index=False)
pd.DataFrame(np.vstack(vargrad_max)).to_csv("results/simulation/results_S1a/vargrad_max.csv", index=False)
pd.DataFrame(np.vstack(vargrad_mean)).to_csv("results/simulation/results_S1a/vargrad_mean.csv", index=False)

pd.DataFrame(rmseiqr).to_csv("results/simulation/results_S1a/rmseiqr.csv", index=False)
pd.DataFrame(r2).to_csv("results/simulation/results_S1a/r2.csv", index=False)


print("time", np.mean(time_track))
