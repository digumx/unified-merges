"""
The main file
"""


import time
from itertools import count
import argparse
import os.path
from datetime import datetime
import traceback
import copy


import numpy as np
from split_and_merge import split_net, merge_redundant
from marabou_query import marabou_query
from cegar import saturation_partition
from property_encode import encode_property
from network import load_nnet, load_nnet_from_tf ,Network
from utils import start_timer, record_time, log_times
from clustering import (
    gen_vects_sim, gen_vects_char, get_refining_linkage,
    merge_net_cluster, modified_linkages, get_parents_from_child, 
    least_common_ancestor, weights_for_merges, biases_for_possible_refines, 
    compute_gradient_on_cex, get_val_on_cex, identify_culprit_neurons, 
    identify_new_abs_indices, find_net_size
)
from extract_cex import extract_cex_from_file 
from dump_custom_vnnlib import dump_vnnlib
import config
import subprocess
import dump_onnx_helper

def save_stats( stats, stats_fname, res, t0, size, refine_steps, time_for_solver_calls, times_for_each_refine_step ):
    """
    A helper function to save the statistics dict to given file

    Arguments:
    
    stats       -   The existing stats dict, gets updated.
    stats_fname -   The file to save to
    res         -   The result so far
    t0          -   The time when timer was started
    size        -   The size of the network so far
    """
    if stats_fname is None:
        return

    stats['result'] = res
    stats['time'] = time.perf_counter() - t0 
    stats['net_size'] = size
    stats['no_of_refine_steps'] = refine_steps
    stats['time_for_solver_calls'] = time_for_solver_calls
    stats['times_for_each_refine_step'] = times_for_each_refine_step

    if config.DEBUG:
        print("Stats: ", stats )
    
    with open( stats_fname, 'w' ) as f:
        f.write( str( stats ))


def abs_refine_loop( net, inp_bnds, out_ub, 
        abstraction = 'none',
        refinement = 'cegar', vectors = 'simulation',
        stats = {}, stats_fname = None, solver='marabou'):
    """
    Tries to verify the given property on the given network using an
    abstraction-refinement loop.

    Supports the following abstraction methods:
    
    'none'      -   No abstraction, the network is directly passed to Marabou
    'saturation'-   Merge to saturation.

    Supports the following refinement methods:
    
    'cegar'     -   Refining based on the distance of the values produced by the
                    abstract and concrete networks, as described in the paper.
    'hcluster'  -   Refining based on splitting hierarchial clusters

    For clustering based refinement, the following kinds of vectors may be used:
    
    'simulation'-   Vectors generated via simulation
    'decision'  -   Vectors from decision boundary

    This function can be passed a stats_dict, and a stats_file. It will
    periodically update the stats dict with the below statistics, and save the
    resulting dict into the given stats file. Any entries already present in the
    dict will also be saved. The statistics tracked are:
    
    'result'    -   'Sat' if a cex was found, 'Unsat' if cex proven to not
                    exist, 'Unknown' if the process is still running, and
                    'Error' if an error has occurred
    'time'      -   The total time taken
    'net_size'  -   The size of the network
    
    Arguments:

    net         -   The network. Must have exactly one output.
    inp_bnds    -   The input bounds. This is a list of tuples, each tuple
                    giving the upper and lower bounds of the corresponding
                    input. If a bound is given as None, that bound is left
                    unbounded.
    out_ub      -   The upper bound on the output of the network 
    abstraction -   The abstraction method to use.
    refinement  -   The refinement method to use
    vectors     -   The kind of vectors to use for clustering
    stats       -   A dict with existing entries to be saved to the stats file
    stats_fname -   Where to save the stats dict

    Returns:
    
    1.  Cex if it is found, else None.
    2.  The statistics dict.
    """
    conc_net = net
    
    # Start timer
    t0 = time.perf_counter()

    try:
        
        # If no abstraction, do nothing
        if abstraction == 'none':
            abs_net = net
            conc_net = net

        # Else, split the net, classify, do saturation merging.
        elif abstraction == 'saturation':
            
            conc_net, inc_dec_vects = split_net( net )
            merge_dir = saturation_partition( inc_dec_vects )
            merge_dir = sorted(merge_dir)
            
            indices_sat_pat = []
            
            for lyr_merge in merge_dir:
                lyr_no, merge_grps = lyr_merge
                if len(merge_grps)>1:
                    merge_grp1, merge_grp2 = merge_grps
                    indices_abs_lyr = np.array([None for i in range(len(merge_grp1)+
                                                                    len(merge_grp2))])
                    indices_abs_lyr[merge_grp1] = 2*len(merge_grp1)+len(merge_grp2)-2
                    indices_abs_lyr[merge_grp2] = 2*len(merge_grp1)+2*len(merge_grp2)-3
                    indices_sat_pat.append((lyr_no, indices_abs_lyr))
                else:
                    merge_grp1 = merge_grps[0]
                    indices_abs_lyr = np.array([None for i in range(len(merge_grp1))])
                    indices_abs_lyr[merge_grp1] = 2*len(merge_grp1)-2
                    indices_sat_pat.append((lyr_no, indices_abs_lyr))
            
            conc_pat = []
            for i in range(1, len(conc_net.layer_sizes)-1):
                indices_conc_lyr = [j for j in range(conc_net.layer_sizes[i])]
                conc_pat.append((i, np.array(indices_conc_lyr)))
            

        # Save stats
        save_stats( stats, stats_fname, 'Unknown', t0, 
                   find_net_size(conc_net, indices_sat_pat), [], [], [] )

        # If using hierarchial clustering based refinement, set it up
        if abstraction != 'none':

            # Generate vectors as requested
            if vectors == 'simulation':
                cl_vects = gen_vects_sim( conc_net, inp_bnds )
            elif vectors == 'decision':
                cl_vects = gen_vects_char( conc_net )
            elif vectors == 'cegar-hcluster':
                cl_vects = gen_vects_sim(conc_net, inp_bnds)
            else:
                raise NotImplementedError()

            # Get linkage, set up initial threshold
            linkages = sorted(modified_linkages(
                get_refining_linkage( conc_net, merge_dir, cl_vects )), 
                key= lambda x: x[0])
            parents = get_parents_from_child(linkages)
            __, children_leaves_nodes, __, possible_refines =least_common_ancestor(
                                                        conc_net, linkages, parents)
            weights_for_possible_refines = weights_for_merges(conc_net, linkages, 
                                                              inc_dec_vects, 
                                                              children_leaves_nodes)
            biases_possible_refines = biases_for_possible_refines( conc_net, linkages,
                                                                inc_dec_vects, 
                                                                children_leaves_nodes)
            
            
        # Save stats
        save_stats( stats, stats_fname, 'Unknown', t0, 
                   find_net_size(conc_net, indices_sat_pat), [], [], [])

        # CEGAR loop
        refined_net = conc_net        

        if abstraction != 'none':
            refine_indices = indices_sat_pat

        net_sizes_trail = []
        cex_sizes_trail = []
        times_for_solver_calls = []
        times_for_each_refine_step = []


        visited_culprit = [[0 for i in range(conc_net.layer_sizes[j])] 
                           for j in range(1,len(conc_net.layer_sizes)-1)]

        
        for num in count():

            test_net = merge_net_cluster(refined_net, weights_for_possible_refines, 
                                        biases_possible_refines, 
                                        refine_indices, children_leaves_nodes)
            test_net = merge_redundant( test_net )
            chopped_refined_net = Network(test_net.weights[:-1],test_net.biases[:-1])
            chopped_refined_net.dump_npz("chopped_refined_net.npz")
            

            print("\n\n------- Refine Loop {} -------\n\n".format( num ))
            print("Network layer sizes: {}".format( chopped_refined_net.layer_sizes ))
            
            # Save stats
            net_sizes_trail.append( [ n for n in chopped_refined_net.layer_sizes ])
            stats['net_sizes'] = net_sizes_trail
            save_stats( 
                stats, stats_fname, 'Unknown', t0, 
                sum( chopped_refined_net.layer_sizes
            ), cex_sizes_trail, times_for_solver_calls, times_for_each_refine_step)

            # Do a query, exit if return is positive.
            if solver == 'marabou':
                cex = marabou_query( chopped_refined_net, inp_bnds, out_ub )
            elif solver=='alpha-beta-crown':
                dump_vnnlib('custom_vnnlib.vnnlib',inp_bnds,
                            chopped_refined_net.layer_sizes)
                # exit(0) 
                command = "python alpha-beta-CROWN/complete_verifier/abcrown.py --model 'Customized(\"abs_net_loader\", \"model_from_file\", \"{}\")' --input_shape 1 {}  --save_adv_example --cex_path {} --vnnlib_path {} --config {} ".format( 'chopped_refined_net.npz', refined_net.layer_sizes[0], 'cex.txt', 'custom_vnnlib.vnnlib','acasxu.yaml')
                os.system("rm cex.txt")

                
                time_before_call = time.perf_counter()

                completed_process = subprocess.run(command, shell=True)

                time_after_call = time.perf_counter()

                times_for_solver_calls.append(time_after_call-time_before_call)

                # Check the return code to see if the command was successful (0 indicates success)
                if completed_process.returncode == 0:
                    # Command completed successfully, proceed with extracting cex
                    cex = extract_cex_from_file('cex.txt')
                    print("Extracted cex: ", type(cex), cex)
                    os.system("rm custom_vnnlib.vnnlib* ")
                    os.system("rm chopped_refined_net.npz")
                else:
                    print("The command did not complete successfully.")
                    raise RuntimeError()
                exit(0)
            elif solver == 'neuralsat':
                os.system("rm res.txt")

                dump_vnnlib('custom_vnnlib.vnnlib',inp_bnds, 
                        chopped_refined_net.layer_sizes)
                
                dump_onnx_helper.dump_onnx(chopped_refined_net, "chopped_dump_net.onnx", 1)

                command = "python neuralsat/neuralsat-pt201/main.py --net {} --spec {} --result_file {} --export_cex ".format( 'chopped_dump_net.onnx', 'custom_vnnlib.vnnlib', 'res.txt')

                time_before_call = time.perf_counter()

                completed_process = subprocess.run(command, shell=True)

                time_after_call = time.perf_counter()

                times_for_solver_calls.append(time_after_call-time_before_call)

                if completed_process.returncode == 0:
                    # Command completed successfully, proceed with extracting cex
                    cex = extract_cex_from_file('res.txt')
                    print("Extracted cex: ", type(cex), cex)
                    os.system("rm custom_vnnlib.vnnlib* ")
                    os.system("rm chopped_refined_net.npz")
                else:
                    print("The command did not complete successfully.")
                    raise RuntimeError()
                
                os.system("rm chopped_dump_net.onnx")
                os.system("rm res.txt")




            print("Type of cex: ", type(cex))

            if config.ASSERTS:
                if cex is not None and len(cex)>0:     # Sanity check
                    assert (
                            get_val_on_cex(refined_net, weights_for_possible_refines, 
                                    biases_possible_refines,refine_indices, cex)[0][0] + 
                                    config.FLOAT_TOL >= 
                                    get_val_on_cex(conc_net, 
                                    weights_for_possible_refines, 
                                    biases_possible_refines,conc_pat, cex)[0][0]
                            )
                    
            if cex is None or len(cex)==0:
                print("No cex, property proved")
                break
            
            elif conc_net.eval( cex )[0][0] >= out_ub:
                print(conc_net.eval( cex )[0][0])
                print("Found cex, property disproved")
                break                  
                
            else:
                # This should never happen if abstraction method is 'none'
                assert not abstraction == 'none'
                grads = compute_gradient_on_cex(conc_net, cex)
                num_steps = 0
                if config.DEBUG:
                    if not get_val_on_cex(refined_net, weights_for_possible_refines,
                                        biases_possible_refines, 
                                        refine_indices ,cex)[0][0] >= out_ub:
                        print("Mismatch")
                        print(cex)
                        print(get_val_on_cex(refined_net, weights_for_possible_refines,
                                        biases_possible_refines, conc_pat ,cex)[0][0] )
                        print(get_val_on_cex(refined_net, weights_for_possible_refines,
                                        biases_possible_refines, refine_indices ,cex)[0][0] )
                        print(out_ub)
                        assert False



            
            # Refine network until cex is no longer spurious
            refine_time_start = time.perf_counter()

            while get_val_on_cex(refined_net, weights_for_possible_refines, 
                                biases_possible_refines,refine_indices, 
                                cex)[0][0] >= out_ub:
                
                num_steps+=1
                # Save stats
                save_stats( 
                    stats, stats_fname, 'Unknown', t0, 
                    find_net_size(refined_net, refine_indices
                    ), cex_sizes_trail, times_for_solver_calls, times_for_each_refine_step)


                if refinement == 'cegar-hcluster':
                    
                    start_timer("identify_culprit_neurons")
                    culprit_neuron = identify_culprit_neurons(conc_net, refined_net, 
                                    weights_for_possible_refines, 
                                    biases_possible_refines, conc_pat, 
                                    refine_indices, cex, grads, visited_culprit)  
                    refine_indices = identify_new_abs_indices(conc_net, refine_indices, 
                                    culprit_neuron, possible_refines)       
                    record_time("identify_culprit_neurons")
                    log_times()

            refine_time_end = time.perf_counter()

            times_for_each_refine_step.append(refine_time_end-refine_time_start) 

            cex_sizes_trail.append(num_steps)

            

        # Save stats
        save_stats( 
            stats, 
            stats_fname, 
            'Safe' if cex is None or len(cex)==0 else 'Unsafe', 
            t0, 
            sum( chopped_refined_net.layer_sizes ),
            cex_sizes_trail, times_for_solver_calls,
            times_for_each_refine_step
        )

        # Return cex
        return cex, stats

    except Exception as e:
        print("Exception: ", e)
        traceback.print_exc()
        save_stats( stats, stats_fname, 'Error', t0, -1,cex_sizes_trail, times_for_solver_calls,
            times_for_each_refine_step )
        raise e
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network',
        dest='net_fname',
        required=True,
        type=str,
        help="The network file to check"
    )
    parser.add_argument('-p', '--property',
        dest='prop_fname',
        required=True,
        type=str,
        help="The file with the property to check"
    )
    parser.add_argument('-a', '--abstraction', 
        dest='abs_method',
        default='none',
        choices=['saturation', 'none'],
        type=str,
        help="The abstraction method to use, none for direct Marabou"
    )
    parser.add_argument('-r', '--refinement',
        dest='ref_method',
        default='cegar',
        choices=['cegar-hcluster'],
        type=str,
        help="The refinement method to use"
    )
    parser.add_argument('-v', '--vectors',
        dest='vect_method',
        default='simulation',
        choices=['simulation', 'decision'],
        type=str,
        help="The method to use to generate vectors for clustering"
    )
    parser.add_argument('--stats-file',
        dest='stats_fname',
        default=None,
        type=str,
        help="The file to which to store statistics"
    )
    parser.add_argument('--solver-type','-s',
        dest='solver_type',
        default='marabou',
        choices=['marabou', 'alpha-beta-crown', 'neuralsat' ],
        type=str,
        help="The solver can marabou or alpha-beta-crown "
    )
    args = parser.parse_args()

    # Load network
    net_fname = args.net_fname
    if net_fname.endswith( 'nnet' ):
        net = load_nnet( net_fname )
    elif net_fname.endswith('.tf'):
        net = load_nnet_from_tf(net_fname)
    else:
        raise RuntimeError("Unknown network filetype")

    # Load property
    prop_fname = args.prop_fname
    if prop_fname.endswith( 'prop' ):
        with open( prop_fname, 'r' ) as f:
            prop = eval(f.read())
    else:
        raise RuntimeError("Unknown property filetype")
        
    # Encode property
    encoded_net, inp_bnds, out_ub, property = encode_property( net, prop ) 
      

    stats = {}
    for arg_name, arg_val in vars( args ).items():
        stats[ arg_name ] = arg_val
    if args.stats_fname is None:
        stats_fname = os.path.join( 
            config.STATS_PATH, 
            str( datetime.today() ))
    else:
        stats_fname = args.stats_fname

    # Call loop
    cex, stats = abs_refine_loop( encoded_net, inp_bnds, out_ub, 
        abstraction = args.abs_method, 
        refinement = args.ref_method,
        vectors = args.vect_method,
        stats = stats,
        stats_fname = stats_fname,
        solver = args.solver_type
    )

    print("Output: {}".format( stats ))
    if cex is not None and len(cex)>0:
        print("Counterexample found: ", cex)
        print("Network output on cex: ", net.eval( np.array( cex ))[0] )
