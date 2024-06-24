"""
Contains a network class and various related utility methods
"""



import numpy as np

import config

import sys
sys.path.append( config.MARABOU_PATH )
from maraboupy import MarabouNetworkNNet




class Network:
    """
    A class representing a DNN as a sequence or affine and relu transforms.

    Notation: Given any network, layer indices are always all-inclusive. That
    is, layer 0 is the input layer, and the last layer is the output layer.
    
    Members are:
    
    weights     -   List of weight matrices. Result of applying weight w on x is
                    given as x@w.
    biases      -   List of bias vectors
    end_relu    -   Does the last layer have ReLU or no
    num_layers  -   Holds the total number of layers in the network, including
                    input and output
    layer_sizes -   A list with sizes of each layer, including input and output
    out_size    -   The size of the output layer
    in_size     -   The size of the input layer
    """
    def __init__(self, weights, biases, end_relu = False):

        # Check for consistency in number of layers
        if config.ASSERTS:
            assert len(weights) == len(biases)
        self.num_layers = len(weights) + 1

        self.in_size = weights[0].shape[0]
        self.out_size = weights[-1].shape[1]

        # Check dimensions of weights and biases
        if config.ASSERTS:
            assert weights[0].shape[1] == biases[0].shape[0]
            for i in range( 1, self.num_layers-1 ):
                assert weights[i-1].shape[1] == weights[i].shape[0]
                assert weights[i].shape[1] == biases[i].shape[0]

        # Set up layer sizes
        self.layer_sizes = [ self.in_size ]
        for i in range( 1, self.num_layers-1 ):
            self.layer_sizes.append( weights[i].shape[0] )
        self.layer_sizes.append( self.out_size )
            
        # Set up weights and biases
        self.weights = weights
        self.biases = biases

        self.end_relu = end_relu

    def eval( self, inp ):
        """
        Evaluates the network on the given input `inp`. Input can also be a
        stack of vectors.
        
        Returns:
        
        1.  The vector of return values
        2.  A list with the values at each layer. At each layer, if the layer
            has a ReLU, the value is the output of the ReLU, or if the layer is
            an input layer, the value is the input, and otherwise the value is
            the output of the linear layer.
        """
        vals = [inp]
        cval = inp

        # Evaluate inner layers
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            cval = np.where(cval @ w + b > 0, cval @ w + b, 0)
            vals.append(cval)

        # Evaluate last layer
        cval = cval @ self.weights[-1] + self.biases[-1]
        if self.end_relu:
            cval = np.where(cval > 0, cval, 0)
        vals.append(cval)

        return cval, vals


    def append_layer( self, weight, bias, relu = False,
                        fold_perms = True ):
        """
        Adds a new layer to the end of network. Collapses consecutive linear
        layers into a single layer.
        
        Arguments:
        
        weight      -   The weight matrix of the new layer
        bias        -   The bias vector for the new layer.
        relu        -   Whether the newly added layer contains a ReLU layer.
        fold_perms  -   If true, and if the new layer performs a positively
                        scaled permutation, it is folded into the last layer.
        """
        
        # Check sizes
        if config.ASSERTS:
            assert self.out_size == weight.shape[0]
            assert bias.shape[0] == weight.shape[1]

        # If there already was relu, create new layer
        if self.end_relu:
            self.weights.append( weight )
            self.biases.append( bias )
            self.num_layers += 1
            self.layer_sizes.append( bias.shape[0] )

            # Fold permutations
            # if fold_perms and 

        # Otherwise, collapse into last layer
        else:
            self.weights[-1] = self.weights[-1] @ weight
            self.biases[-1] = self.biases[-1] @ weight + bias
            self.layer_sizes[-1] = bias.shape[0]
            
        # Set new end_relu
        self.end_relu = relu
        
        # Set new output size
        self.out_size = bias.shape[0]

    def __str__( self ):
        """
        Simply return weights and biases, etc.
        """
        return '\n'.join(
            ["Network has {} layers with sizes {}".format( 
                self.num_layers, self.layer_sizes )] +
            ["Connected to output of layer {}: \nWeight: \n{} \nBias: \n{}".format(
                i, w, b) for i, (w,b) in enumerate( 
                    zip( self.weights, self.biases )) ] +
            ["The network {} with ReLU".format(
                "ends" if self.end_relu else "does not end" )]
        )

    def dump_npz( self, fname, extra_entries = {} ):
        """
        Saves network to given filename/path. Saves it as an .npz file with the
        following named entries:

        layer_sizes -   An array of layer sizes including input and output layer
        weight_i    -   Weight connected to output of layer i
        bias_i      -   Bias connected to output of layer i
        end_relu    -   A boolean value, if true, network ends with ReLU

        Entries in `extra_entries` are also added if given. Should be numbers or
        np arrays only.

        NOTE: a .npz extension is auto-added to filename if its not there
        """
        save_dict = {}
        for i, (w,b) in enumerate( zip( self.weights, self.biases )):
            save_dict['weight_{}'.format( i )] = np.float32(w)
            save_dict['bias_{}'.format( i )] = np.float32(b)
        save_dict['layer_sizes'] = np.array( self.layer_sizes ) 
        save_dict['end_relu'] = np.array([ self.end_relu ]) 
        
        for entry_name, entry_val in extra_entries.items():
            save_dict[ entry_name ] = entry_val

        if config.DEBUG:
            print("Layer sizes of network being dumped: ",
                save_dict['layer_sizes']
            )
            print("Total size: ", sum( save_dict['layer_sizes'] ))
            print("Dumping to: ", fname)

        np.savez( fname, **save_dict )


def load_npz( npz_data ):
    """
    Loads a Network from given npz data
    """
    data_dict = npz_data
    layer_sizes = data_dict[ 'layer_sizes' ]
    end_relu = data_dict[ 'end_relu' ] 
    wb_num = len(layer_sizes) - 1
    weights = [ data_dict[ 'weight_{}'.format( wb_idx ) ] 
            for wb_idx in range( wb_num ) ]
    biases = [ data_dict[ 'bias_{}'.format( wb_idx ) ] 
            for wb_idx in range( wb_num ) ]
    return Network( weights, biases, end_relu )


def load_nnet( nnet_fname ):
    """
    Loads a network from an nnet file. Assumes that the network does not end
    with a ReLU.
    """

    # read nnetfile into Marabou Network
    marabou_net = MarabouNetworkNNet.MarabouNetworkNNet( filename=nnet_fname )

    net = Network(
        weights = [ np.array( w ).T for w in marabou_net.weights ],
        biases =  [ np.array( b )   for b in marabou_net.biases ],
        end_relu = False
    )

    return net

def load_nnet_from_tf(nnet_fname):
    with open(nnet_fname, 'r') as file:
        lines = file.readlines()

    weights = []
    biases = []

    for i in range(0, len(lines), 3):
    
        weight_str = lines[i + 1].strip()
        bias_str = lines[i + 2].strip()

        # Parse weight matrix
        weight = np.array(eval(weight_str)).T

        # Parse bias vector
        bias = np.array(eval(bias_str))

        weights.append(weight)
        biases.append(bias)

    net = Network(weights, biases, end_relu=False)
    return net

    # return activations, weights, biases
        

if __name__ == "__main__":
    
    fname = "networks/ACASXU_run2a_1_1_batch_2000.nnet"
    
    net = load_nnet( fname )
    
    print( net.layer_sizes )
    print( net.eval( np.array([ 1., 1., 1., 1., 1. ]) )) 
