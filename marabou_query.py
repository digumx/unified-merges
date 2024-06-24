"""
Encodes a neural network verification problem as a Marabou query and solves it.
"""



import sys
import math

import numpy as np

import config

# Load marabou
sys.path.append( config.MARABOU_PATH )
from maraboupy import MarabouCore
from maraboupy import Marabou



def marabou_query( net, inp_bnds, out_ub ):
    """
    Given a network with a single output, input-side properties and output side
    properties, this uses Marabou to query weather the given property is valid.
    
    Arguments:

    net      -   The network. Must have exactly one output.
    inp_bnds -   The input side bounds. This is a list of tuples, each tuple
                 giving the upper and lower bounds of the corresponding input. If
                 a bound is given as None, that bound is left unbounded.
    out_ub   -   The upper bound on the output.

    Returns a counterexample as an input vector if it exists, none otherwise.
    """
    print(net.weights)
    print(net.biases)
    if config.ASSERTS:
        assert net.out_size == 1

    # Variables are given by inputs, then the pre_relu and post_relu for each
    # layer, ending with the outputs
    num_vars = net.in_size
    num_vars += sum( net.layer_sizes[ 1 : -1 ] * 2 )
    num_vars += net.out_size * 2 if net.end_relu else net.out_size
    
    # Create input query
    inputQuery = MarabouCore.InputQuery()
    inputQuery.setNumberOfVariables( num_vars )

    # Set up bounds on inputs
    for v_idx, (lb, ub) in enumerate(inp_bnds):
        if lb is not None:
            inputQuery.setLowerBound( v_idx, lb )
        if ub is not None:
            inputQuery.setUpperBound( v_idx, ub )

    # Encode network layer by layer
    pre_var_base = 0
    pst_var_base = net.in_size
    for w, b in zip( net.weights[:-1], net.biases[:-1] ):

        # Loop over equations and add
        for dv_idx, (col, bias) in enumerate( zip( w.T, b )):
            eqn = MarabouCore.Equation()
            eqn.addAddend( -1, pst_var_base + dv_idx )
            for sv_idx, term in enumerate( col ):
                eqn.addAddend( term, pre_var_base + sv_idx )
            eqn.setScalar( -bias )
            inputQuery.addEquation( eqn )

        # Shift variable bases
        pre_var_base = pst_var_base
        pst_var_base += w.shape[1]

        # Relu Constraints
        for v_idx in range( w.shape[1] ):
            MarabouCore.addReluConstraint( 
                inputQuery, pre_var_base + v_idx, pst_var_base + v_idx )

        # Shift variable bases
        pre_var_base = pst_var_base
        pst_var_base += w.shape[1]

    # Encode weights and bias of last layer
    w, b = net.weights[-1], net.biases[-1]
    for dv_idx, (col, bias) in enumerate( zip( w.T, b )):
        eqn = MarabouCore.Equation()
        eqn.addAddend( -1, pst_var_base + dv_idx )
        for sv_idx, term in enumerate( col ):
            eqn.addAddend( term, pre_var_base + sv_idx )
        eqn.setScalar( bias )
        inputQuery.addEquation( eqn )

    # Encode relu at end if needed
    if net.end_relu:

        # Shift variable bases
        pre_var_base = pst_var_base
        pst_var_base += w.shape[1]
        
        # Relu Constraints
        for v_idx in range( w.shape[1] ):
            MarabouCore.addReLUConstraint( 
                inputQuery, pre_var_base + v_idx, pst_var_base + v_idx )

    # Check that pst_var_base is correct
    if config.ASSERTS:
        assert pst_var_base == num_vars - 1

    # Encode postcondition
    inputQuery.setLowerBound( pst_var_base, out_ub )

    #if config.DEBUG:
    #    MarabouCore.saveQuery( inputQuery, '/tmp/query2' )

    # Run marabou
    options = Marabou.createOptions()
    exit_code, model, stats = MarabouCore.solve(inputQuery, options, "")

    # Return input vector for cex
    if config.DEBUG:
        print("Marabou exit code: ", exit_code)
    if exit_code == 'sat':

        # Check for a weird situation where model can have nans
        if config.ASSERTS:
            if any([ math.isnan( model[ v_idx ]) for v_idx in range( num_vars ) ]):
                if config.DEBUG:
                    print("Network producing nans: ", net)
                    print("Values: ", model)
                    print("Exit_code: ", exit_code)
                    print("stats: ", stats)
                assert False
        
        cex = np.array([ model[ v_idx ] for v_idx in range( net.in_size ) ])
        if config.DEBUG:
            print("Net: ", net )
            print("Cex: ", cex )
            print("Eval: ", net.eval( cex ))
            print("Model: ", model)
            print("out_ub: ", out_ub)
            print( net.eval( cex )[0][0] >= out_ub )
        if config.ASSERTS:
            assert net.eval( cex )[0][0] + config.FLOAT_TOL >= out_ub
        return cex
    elif exit_code == 'unsat':
        return None
    else:
        raise RuntimeError("Unknown Marabou exit code: {}".format( exit_code ))



if __name__ == "__main__":

    from split_and_merge import split_net, merge_net
    from network import Network
    from cegar import saturation_partition

    # Input network
    in_net = Network(
        weights = [
            np.array([  
                [1., 2., 3.], 
                [3., 1., 2.] 
            ]),
            np.array([  
                [ 1., -1., -1.], 
                [ 1.,  1., -1.],
                [ 1., -1.,  1.],
            ]),
            np.array([  
                [ 1.], 
                [-1.],
                [ 1.],
            ])
        ],
        biases = [
            np.array([ 0., 1., 0. ]),
            np.array([ 1., 2., 3. ]),
            np.array([ 0 ]),
        ],
        end_relu = False
    )

    # Split and merge
    split_net, inc_dec_vects= split_net( in_net)
    merge_dir = saturation_partition( inc_dec_vects)

    # Merge net
    merge_net_1 = merge_net( split_net, merge_dir[:1], inc_dec_vects )
    merge_net_2 = merge_net( merge_net_1, merge_dir[1:], inc_dec_vects )

    # Set up queries
    inp_bnds = [ (-1, 1), (-1, 1) ]
    out_ub = 40
    out_q1 = marabou_query( in_net, inp_bnds, out_ub )
    out_q2 = marabou_query( merge_net_1, inp_bnds, out_ub )
    out_q3 = marabou_query( merge_net_2, inp_bnds, out_ub )
    
    print("Output of query 1: ", out_q1)
    print("Output of query 2: ", out_q2)
    print("Output of query 3: ", out_q3)
