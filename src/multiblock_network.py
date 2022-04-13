
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate
import tensorflow.keras.models
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import InputLayer
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold as SK
from sklearn.model_selection import KFold as KF
from sklearn.metrics import confusion_matrix, mean_squared_error
import sys


class multiblock_network():
    """
    Text
    """
    def __init__(self):
        self.block_net = []
        self.block_sizes = []
        self.train_log = []

    def define_block_net(self, structure):
        """
        The first layer in structure is the input layer.
        This is so the model is consistent with conv-nets as well as
        dense networks. "structure" must be defined by the user beforehand for
        each block.
        """
        train_data_size = structure[0].shape[1] # get the number of features
        #network_in = Input(shape=(train_data_size,))
        network_in = structure[0]
        for i, layer in enumerate(structure[1:]):
            if i == 0:
                network = layer(network_in)
            else:
                network = layer(network)
        network = Model(inputs=network_in, outputs = network)

        self.block_net.append(network)
        self.block_sizes.append(train_data_size)
        self.block_num = len(self.block_net)

    def define_block_concatenation(self, structure):
        """
        Build the concatenation layer and concatenate the independent block networks.
        """
        network_in = Concatenate(name='concat')([i.output for i in self.block_net])
        for i, layer in enumerate(structure):
            if i == 0:
                network = layer(network_in)
            else:
                network = layer(network)
        network = Model(inputs=[i.input for i in self.block_net], outputs = network)
        self.concat_net = Model(inputs=[i.input for i in self.block_net],
                                outputs = network.get_layer('concat').output)
        self.comb_net = network
        self.blender_net = [InputLayer(input_shape=(sum([i.output.shape[1] for i in self.block_net]),))]
        self.blender_net += structure
        self.blender_net = Sequential(self.blender_net)

        # Check that this is the same as in Annas version

    def compile(self, **kwargs):
        self.concat_net.compile(**kwargs)
        self.comb_net.compile(**kwargs)

    def fit(self,
            train_data,
            train_labels,
            cvfold=None,
            problem="classification",
            **kwargs):
        """
        if cross_validation: save the weights to start the cross validation folds with the same
        start conditions for each fold. Better to set cvfold to None. Otherwise model must be
        reinitialized if one wants to train the model after cross validation.
        """
        if self.check_input_consistency(train_data, train_labels):
            
            if train_labels.ndim == 1:
                if isinstance(train_labels, pd.Series):
                    train_labels = train_labels.values
                train_labels = train_labels.reshape(-1,1)
            
            if cvfold is None:
                self.train_log = [self.comb_net.fit(train_data,
                                    train_labels,
                                    **kwargs)]
            else:
                Wsave = self.comb_net.get_weights()
                if problem == "regression":
                    skf = KF(n_splits=cvfold, shuffle=True, random_state=0)
                else:
                    skf = SK(n_splits=cvfold, shuffle=True, random_state=0)
                for train_index, test_index in skf.split(train_data[0], train_labels):
                    model_fit = self.comb_net.fit([row[train_index,:] for row in train_data],
                                    train_labels[train_index,:],
                                    validation_data=([row[test_index,:] for row in train_data], train_labels[test_index,:]),
                                    **kwargs)
                    self.train_log.append(model_fit)
                    self.comb_net.set_weights(Wsave)
            return self.train_log
        else:
            sys.exit(("Number of rows in train data does not match"
                      "the length of train labels"))

    def get_concat_layer(self):
        concat_layer = Model(inputs=self.comb_net.input,
                             outputs=self.comb_net.get_layer('concat').output)
        return concat_layer

    def predict(self, newdata, **kwargs):
        if len(self.train_log) != 1:
            print("Prediction not possible before training")
        elif self.check_input_consistency(newdata):
            #return [net.predict(newdata, **kwargs) for net in self.comb_net]
            return self.comb_net.predict(newdata, **kwargs)
        else:
            sys.exit("New data do not fit to block sizes")

    def get_block_activations(self, train_data, on_input=False):
        """
        Activation of nodes  in the concatenation layer for each data block.
        """
        if on_input:
            return np.hstack(train_data) # for blocks with different feature numbers there is a problem
        else:
            return self.concat_net.predict(train_data)


    def get_pseudo_output(self, train_data, type="zeros", knock_out=True, on_input=False):
        """
        Compute pseudo output for each block.
        1. compute the block activation matrix (input matrix for the blender network with one row for each block and
        one column for each node in the concatenation layer).
        - on_input: compute pseudo activations by knocking in / out blocks on input layer; otherwise on concatenation layer (default)
        2. pseudo output: prediction output when only one block is active and the weights from the other blocks are set to the mean or to zero.
        """
        block_activations = self.get_block_activations(train_data, on_input)
        block_ac_mean = np.mean(block_activations, axis=0) # mean over train samples; one value for each sample and the mean of that
        self._train_length = np.shape(block_activations)[0]
        if type == "mean":
            if knock_out:
                pseudo_activation_matrix = np.vstack([block_activations for i in range(self.block_num)])
            else:
                pseudo_activation_matrix = np.repeat(block_ac_mean.reshape(1,len(block_ac_mean)),
                repeats=self.block_num*self._train_length, axis=0)
        elif type == "zeros":
            if knock_out:
                pseudo_activation_matrix = np.vstack([block_activations for i in range(self.block_num)])
            else:
                pseudo_activation_matrix = np.zeros((self.block_num*self._train_length, np.shape(block_activations)[1]))
        else:
            sys.exit("Unknown type!")

        c = 0
        block_activation_list = []
        for i in range(self.block_num):
            num_block_nodes = self.block_net[i].layers[0].input.shape[1] if on_input else self.block_net[i].layers[-1].output.shape[1]
            for j in np.arange(c, c+num_block_nodes):
                if knock_out:
                    if type=="mean":
                        pseudo_activation_matrix[(i*self._train_length):((i+1)*self._train_length),j] = block_ac_mean[j]
                    else:
                        pseudo_activation_matrix[(i*self._train_length):((i+1)*self._train_length),j] = 0
                else:
                    pseudo_activation_matrix[(i*self._train_length):((i+1)*self._train_length),j] = block_activations[:,j]
            block_activation_list.append(pseudo_activation_matrix[:,c:(c+num_block_nodes)])
            
            c = c+num_block_nodes

        if on_input:
            pseudo_output_vec = self.comb_net.predict(block_activation_list)
        else:
            pseudo_output_vec = self.blender_net.predict(pseudo_activation_matrix)
            
        if pseudo_output_vec.ndim == 1:
                pseudo_output_vec = pseudo_output_vec.reshape(-1,1)
        
        pseudo_output_vec = pseudo_output_vec.reshape(-1,1, order="F")
        #pseudo_output_mat = np.reshape(pseudo_output_vec, newshape=(self._train_length, self.block_num), order ="F")
        pseudo_output_mat = np.reshape(pseudo_output_vec, newshape=(self._train_length, self.block_num, 
                                                           self.blender_net.layers[-1].output.shape[1]), order ="F")

        return(pseudo_output_mat)


    def evaluate_block_contribution(self, train_data, target=None, on_target = False, type="zeros", knock_out=True, on_input=False, bins=10):
        pseudo_output = self.get_pseudo_output(train_data, type=type, knock_out=knock_out, on_input=on_input)
        output = self.comb_net.predict(train_data)
        
        if isinstance(target, pd.DataFrame) or isinstance(target, pd.Series):
            target = target.values
        if isinstance(output, pd.DataFrame) or isinstance(output, pd.Series):
            output = output.values
        
        if output.ndim == 1:
            output = output.reshape(-1,1)
        if target.ndim == 1:
            target = target.reshape(-1,1)
                   
        
        def internal_block_contribution_evaluation(pseudo_output, output, target):
            
            if output.ndim!=2 or np.shape(output)[1]!=1:
                sys.exit("Wrong dimensions in output.")
            
            if np.shape(output)[0] != np.shape(pseudo_output)[0]:
                sys.exit("dimensions in output and pseudo output don't match.")
            
            #if np.shape(pseudo_output)[1] != self.block_num:
            #    sys.exit("wrong dimension of pseudo output blocks.")
            
            r = [np.min(np.hstack([pseudo_output, output])), np.max(np.hstack([pseudo_output, output]))]

            
            def create_hist(subset, bins=bins):
                """
                subset: if one wants, he can take only a subset of the training data but this is not the default.
                """

                hist_out, bins = np.histogram(output[subset,0], bins=bins, range=r, density=True)
                hist_out += 0.01
                hist_out /= np.sum(hist_out)

                histograms = {"out":hist_out}
                js = []
                kl = []
                cor = []
                train_eval =[]

                for i in range(self.block_num):
                    
                    hist_pout = np.histogram(pseudo_output[subset,i], bins=bins, range=r, density=True)[0]
                    hist_pout += 0.01
                    hist_pout /= np.sum(hist_pout)
                    histograms["block_" + str(i)] = hist_pout
                    kl.append(self.kl_divergence(hist_pout, hist_out))
                    
                    if knock_out == False:
                        js.append(1-self.js_divergence(hist_pout, hist_out))
                    else:
                        js.append(self.js_divergence(hist_pout, hist_out))
                    cor.append(pearsonr(output[subset,0], pseudo_output[subset,i])[0])
                    if target is not None:
                        if len(np.unique(target)>2):
                            train_eval.append(mean_squared_error(y_true=target, y_pred=pseudo_output[:,i], squared=False))
                        else:   
                            train_eval.append(confusion_matrix(y_true=target, y_pred=pseudo_output[:,i]))
                return {"bins": bins, "histograms": histograms, "kl": kl, "js":js, "cor":cor, "train_eval":train_eval}

            if on_target == False:
                return create_hist(subset= range(np.shape(output)[0]))
            else:
                return {"0": create_hist(subset = np.where(target==0)[0]),
                        "1": create_hist(subset = np.where(target==1)[0])}

        res = []
        
        for i in range(np.shape(output)[1]):
            res.append(internal_block_contribution_evaluation(pseudo_output[:,:,i], output[:,i].reshape(-1,1), target[:,i]))
            
        if len(res) == 1:
            res = res[0]
            
        return res

    def get_block_weights(self):
        num = None
        for i, layer in enumerate(self.comb_net.layers):
            if layer._name == "concat":
                num = (i+1)
            if i == num:
                return(layer.get_weights())


    def kl_divergence(self, p, q):
	    return sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p)))

    def js_divergence(self, p, q):
        m = (p+q)/2
        return (self.kl_divergence(p,m) + self.kl_divergence(q,m)) / 2

    def check_input_consistency(self, data, labels = None):
        row_check = False
        col_check = False
        lab_check = False
        if len(np.unique([np.shape(i)[0] for i in data])) == 1:
            row_check = True
        else:
            print("Warning: rows do not match between input blocks")
        if all(np.array([np.shape(i)[1] for i in data]) - np.array(self.block_sizes) == 0):
            col_check = True
        else:
            print("Warning: columns of input blocks do not match model specification")
        if labels is None:
            lab_check = True
        elif np.shape(data[0])[0] == len(labels):
            lab_check = True
        else:
            print("Warning: rows do not match between input blocks and labels")
        return col_check & row_check & lab_check


    def plot(self,metric=['accuracy']):
        if len(self.train_log) == 0:
            print("Plotting not possible before training or cross validation study")
        elif all(me not in self.train_log[0].model.metrics_names for me in metric):
            print("Metric not found!")
        else:
            for m in metric:
                validation = 'val_'+m in self.train_log[0].history.keys()
                for i, tl in enumerate(self.train_log):
                    if(len(self.train_log) > 1):
                        if(i == 0):
                            mean = np.array(tl.history[m].copy()).reshape(-1,1)
                            if validation:
                                val_mean = np.array(tl.history['val_'+m].copy()).reshape(-1,1)
                        else:
                            values = np.array(tl.history[m].copy())
                            mean = np.hstack([mean, values.reshape(-1,1)])
                            if validation:
                                val_values = np.array(tl.history['val_'+m].copy())
                                val_mean = np.hstack([val_mean,
                                                      np.array(val_values).reshape(-1,1)])
                    else:
                        mean = np.array(tl.history[m])
                        if validation:
                            val_mean = np.array(tl.history['val_'+m])


                self.mean = mean.copy()
                if validation:
                    self.val_mean = val_mean.copy()
                print(np.shape(mean))
                if np.array(mean).ndim > 1:
                    mean = np.mean(mean, 1)
                    if validation:
                        val_mean = np.mean(val_mean, 1)

                plt.plot(mean)
                if validation:
                    plt.plot(val_mean)
                    plt.legend(['train '+m, 'validation '+ m], loc='lower right')
                else:
                    plt.legend(['train '+ m], loc='lower right')

                plt.title('model ')
                plt.ylabel('metric')
                plt.xlabel('epoch')
                plt.show()



    def compute_MI(self, x,y, bins, eps, density, plot=True):
        '''
        x ... a single pseudo output
        y ... the real output
        bins ... number of bins
        eps ... small value
        '''
        bivariate_hist = np.histogram2d(y, x, bins=bins, density=density)[0] + eps
        bivariate_hist = bivariate_hist / np.sum(bivariate_hist)

        x_hist = np.histogram(x, density=density, bins=bins)[0].reshape(1, bins) + eps
        y_hist = np.histogram(y, density=density, bins=bins)[0].reshape(bins,1) + eps

        x_hist = x_hist / np.sum(x_hist)
        y_hist = y_hist / np.sum(y_hist)

        prod_hist = np.matmul(y_hist, x_hist)

        MI = np.sum(bivariate_hist * np.log(np.divide(bivariate_hist, prod_hist)))

        if plot:
            f, axarr = plt.subplots(2,2)
            axarr[0,0].imshow(bivariate_hist)
            axarr[0,0].set_title("Bivariate plot")
            axarr[0,1].imshow(prod_hist)
            axarr[0,1].set_title("product plot")
            axarr[1,0].plot(np.transpose(x_hist))
            axarr[1,0].set_title("pseudo output hist")
            axarr[1,1].plot(y_hist)
            axarr[1,1].set_title("output hist")

        return MI

    def MI(self, blocks_train, type, knock_out, on_input, eps, bins, density, plot):
        pseudo_output = self.get_pseudo_output(blocks_train, type=type, knock_out=knock_out,on_input=on_input)
        out_num = np.shape(pseudo_output)[2]
        output = self.comb_net.predict(blocks_train)
        pseudo_output = pseudo_output[:,:,0]
        
        MI_matrix = np.hstack([output,pseudo_output])[..., np.newaxis]
        
        #ps = []
        #for i in range(self.block_num):
        #    ps.append(pseudo_output[(i*self._train_length):((i+1)*self._train_length)])

        #MI_matrix = np.hstack([output,pseudo_output])
        # need to update for multiple outputs
        return np.apply_along_axis(self.compute_MI, 0, MI_matrix[:,1:,0], y=MI_matrix[:,0,0], eps=eps, bins=bins, density=density, plot=plot)

    def vargrad_input(self, blocks_train, type="max", ensemble_num=20, quantile=0.9, seed=1):
        np_random_gen = np.random.default_rng(seed)
        all_grads = [[] for _ in range(len(blocks_train))]
        out = []
        for j in range(ensemble_num):
            noised_data_blocks = [data_block + np_random_gen.normal(loc=0, scale=1, size=(np.shape(data_block)[0],
                                                                    np.shape(data_block)[1])) for data_block in blocks_train]


            input_tensors = [tf.Variable(tf.cast(datablock, tf.float64)) for datablock in noised_data_blocks]
            with tf.GradientTape() as tape:
                tape.watch(input_tensors)
                output = self.comb_net(input_tensors)
                positive = tf.reduce_sum(output, axis=1)
            positive_grads = tape.gradient(positive, input_tensors)
            positive_grads = [grad.numpy() for grad in positive_grads]

            for i in range(len(blocks_train)):
                all_grads[i].append(positive_grads[i])
            out.append(output)

        vargrad = [np.array(all_grad).std(axis=0) for all_grad in all_grads]

        if type=="max":
            return [grad.max() for grad in vargrad]
        elif type=="mean":
            return [grad.mean() for grad in vargrad]
        elif type=="sum":
            return [grad.sum() for grad in vargrad]
        elif type=="quantile":
            return [np.quantile(grad, quantile) for grad in vargrad]
            
            
    def vargrad_concat(self, blocks_train, block_num, type="max", ensemble_num=20, seed=1): # currently only works when each block has the same number of layers in concat
        tf_random_gen = tf.random.experimental.Generator.from_seed(seed)
        def get_concat_layer(model):
            for i, layer in enumerate(model.layers):
                if 'concat' in layer.name:
                    concat_layer = layer
                    break
            return concat_layer, i
        concat_layer, layer_idx = get_concat_layer(self.comb_net)

        # feed forward the input until concat layer
        input_layer = self.comb_net.inputs
        #output_layer = self.comb_net.output

        before_concat = Model(input_layer, concat_layer.output)
        concat_tensors = before_concat(blocks_train) # the output at concat layer
        # use for creating the random noise
        mean = tf.reduce_mean(concat_tensors, axis=0)
        std = tf.math.reduce_std(concat_tensors, axis=0)

        all_grads = [[] for _ in range(len(blocks_train))]
        for i in range(ensemble_num):
            input_tensors = concat_tensors + tf_random_gen.normal(concat_tensors.shape, mean, std)

            with tf.GradientTape() as tape:
                tape.watch(input_tensors)

                next_output = input_tensors
                for layer in self.comb_net.layers[layer_idx+1:]:
                    next_output = layer(next_output)
                output = next_output
                positive = tf.reduce_sum(output, axis=1)

            positive_grads = tape.gradient(positive, input_tensors)
            # print(positive_grads)

            positive_grads = np.array([grad.numpy() for grad in positive_grads])

            for j, data_block in enumerate(blocks_train):
                all_grads[j].append(positive_grads[...,j*block_num: j*block_num + block_num])
        # calculate covariance of 50 runs
        var_grad = np.square(np.array(all_grads).std(axis=1))
        if type=="sum":
            return var_grad.sum(axis=(1,2))
        elif type=="max":
            return var_grad.max(axis=(1,2))