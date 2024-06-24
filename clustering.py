"""
Methods for splitting and merging based on clustering
"""
# TODO Use the min of pairwise distance in hcluster to generate linkage



from collections import defaultdict
from tqdm import tqdm

import numpy as np
import torch
import torch.optim
from scipy.cluster.hierarchy import linkage, fcluster

import config
import copy
import utils
from network import Network

def get_hashes(arr):

	n, __ = arr.shape

	hash_weights = np.random.rand(n)

	def get_hash(a): return np.dot(a, hash_weights)

	return np.apply_along_axis(get_hash, 0, arr)



def gen_vects_char( net ):
    """
    Generates vectors using the characteristic method. The distance between the
    generated vectors characterises the distance between the decision boundaries
    for neurons. 
    
    The decision boundary can be thought of as a plane perpendicular to some
    vector v, with perpendicular distance d from the origin. For 2 decision
    boundaries to match, they should be similarly oriented, so normalized v
    should match, and they should be at similar perpendicular distances. Thus,
    the characteristic vector is made by taking normalized v and conctenating
    the perpendicular distance. Note that v is simply the weights.

    Arguments:
    
    net -   The network whose characteristics to generate

    Returns:
    
    A list of a stack of vectors, one for each layer. Each stack contains one
    vector for each neuron. This list has some values for the input (first) and
    output (last) layer, but these should be ignored.
    """

    vects = [ None ]

    for w, b in zip( net.weights, net.biases ):
        v_stack = w.T
        vects.append(
            np.concatenate( ( v_stack, np.expand_dims( -b, axis=1 ) ), axis=1 ) /
            np.linalg.norm( v_stack, axis=1, keepdims=True )
        )

    return vects

def find_net_size(net, indices):
    """
    Helper function to find the size of the network

    Arguements:
    
    net - The concrete network
    
    indices - list of tuple where the second element corresponds to mapping 
              from neurons to the merge group number 
    
    Returns:
    
    Returns the size of network 
    """
    net_size = sum(
        [np.unique(ind[1]).size for ind in indices]
        )+net.layer_sizes[0]+ net.layer_sizes[-1]
    return net_size



def get_potential_cex(net, inp_bnds, num_samples):

    """
    Helper function to generate a random bunch of samples within the bounds

    Arguements:
    
    net - The concrete network

    inp_bnds - The list of tuples where the first element corresponds to 
               lower bound and second element corresponds to upper bound 
    
    num_samples - no of random samples that you want to generate

    Returns:
    
    Returns a list of potential counter-examples

    """


    lower_bounds = [inp_bnd[0] for inp_bnd in inp_bnds]
    upper_bounds = [inp_bnd[1] for inp_bnd in inp_bnds]
    samples = np.random.uniform(lower_bounds,upper_bounds,
                                (num_samples,net.layer_sizes[0]))
    return samples


def pgd_attack(net, abs_indices, num_steps, 
        bounds, weights, biases, num_samples = 100):
    """
    Arguments:
    
    num_samples     -   Number of parallel potential cexes
    net             -   Concrete network, used for last layer weights.
    """
    
    samples =  get_potential_cex(net, bounds, num_samples )
    loss_fn = torch.nn.CrossEntropyLoss()
    x_adv = torch.from_numpy(samples).clone().detach().type( torch.float32 )
    x_adv.requires_grad = True

    optim = torch.optim.SGD([ x_adv ], lr = config.PGD_LR)

    lbs = torch.tensor([ b[0] for b in bounds ])
    ubs = torch.tensor([ b[1] for b in bounds ])

    for i in range(num_steps):
    
        prediction_on_abs, __ = get_val_on_cex_autograd(net, weights, biases,
                abs_indices, x_adv)

        loss = -torch.sum( prediction_on_abs )
        bst_idx = torch.argmax( prediction_on_abs ).item()

        optim.zero_grad()
        loss.backward()

        optim.step()

        # Projection 
        with torch.no_grad():
            x_adv[:,:] = torch.clamp( x_adv, lbs, ubs )


    ret_val = x_adv[ bst_idx ].detach().numpy()
    return ret_val


def get_culprit_neuron(conc_net, abs_net, weights_for_possible_refines,
        biases_possible_refines , indices_conc, indices_abs, counter_examples,
        visited_culprit, grads):
    # TODO Potentially redundant with identify culprit neurons, fix
    
    __, conc_vals = get_val_on_cex(conc_net, weights_for_possible_refines, 
                                biases_possible_refines, indices_conc, 
                                counter_examples)
    __, abs_vals = get_val_on_cex(abs_net, weights_for_possible_refines, 
                                  biases_possible_refines, indices_abs, 
                                  counter_examples)
    
    culprit = None
    
    for i in range(1, len(conc_vals)-1):
        c_val, a_val, grad_lyr    = conc_vals[i], abs_vals[i], grads[i]
        
        val_diff_abs_conc_gradients = np.abs(np.multiply( 
            np.array(c_val)-np.array(a_val),grad_lyr))
        val_mean = np.mean(val_diff_abs_conc_gradients, axis=0)
        
        indices_not_visited         = np.where(
            np.array(visited_culprit[i-1])==0)[0].tolist()
       
        if len(indices_not_visited) == 0:
            continue 

        pos_max_elem = indices_not_visited[val_mean[indices_not_visited].argmax()]

        max_val_elem                = val_mean[pos_max_elem]

        if culprit is None and visited_culprit[i-1][pos_max_elem]==0:
            culprit = (i, pos_max_elem, max_val_elem)
        elif culprit is not None and (culprit[2] < max_val_elem) and visited_culprit[i-1][pos_max_elem]==0:
            culprit = (i, pos_max_elem, max_val_elem)
    
    if culprit is not None:
        visited_culprit[culprit[0]-1][culprit[1]] = 1
        return culprit[0:-1]
    else: 
        None


    


def simulate_get_cex(net,indices, weights, biases, out_ub, samples, is_conc):

    """
    Helper function to find actual counter-examples from a bunch of samples

    Arguements:
    
    net - The concrete network

    indices - list of tuple where the second element corresponds to mapping 
              from neurons to the merge group number 

    weights - A list of list which gives the refinement weigths for different 
              type of merging possible

    biases - A list of list which gives the biases  for different type of 
             merging possible

    out_ub - upper bound on the output neuron

    samples - samples that you check if they indeed are spurious 

    Returns:

    A list of spurious counter-examples 
    """

    vals = [samples]
    cval = samples 
    # Evaluate inner layers   
    lyr_no = 0
    for w, b in zip(weights, biases):
        cval = np.where(
            cval @w[1][:, np.array(indices[lyr_no][1]).tolist()] +
            b[1][np.array(indices[lyr_no][1]).tolist()]  > 0, 
            cval @w[1][:, np.array(indices[lyr_no][1]).tolist()] + 
            b[1][np.array(indices[lyr_no][1]).tolist()] , 0
            )
        vals.append(cval)
        lyr_no = lyr_no+1

    # Evaluate last layer
    cval = cval @ net.weights[-1]+ net.biases[-1]
    if net.end_relu:
        cval = np.where(cval > 0, cval, 0)
    vals.append(cval) 
    last_lyr_vals = vals[-1].T
    last_lyr_vals = last_lyr_vals[0]

    if is_conc:
        ind_cex =  np.where(last_lyr_vals < out_ub)[0].tolist()
        if len(ind_cex) !=0 :
            return vals[0][np.where(last_lyr_vals < out_ub)[0].tolist()]
        else:
            return None
    else:
        ind_cex =  np.where(last_lyr_vals >= out_ub)[0].tolist()
        if len(ind_cex) !=0 :
            return vals[0][np.where(last_lyr_vals >= out_ub)[0].tolist()]
        else:
            return None



def gen_vects_sim( net, inp_bnds, data = None, n_samps = config.NUM_SAMPS ):
    """
    Generates n_samps number of samples for clustering via simulation

    Argument:
    
    net         -   The network
    inp_bnds    -   The input bounds. This is a list of tuples, each tuple
                    giving the upper and lower bounds of the corresponding
                    input. If a bound is given as None, that bound is left
                    unbounded. Not required if data is given.
    data        -   Data to do simulation on.
    n_samps     -   The number of samples to generate

    Returns:
    
    A list of a stack of vectors, one for each layer. Each stack contains one
    vector for each neuron. This list also has vectors for input and output
    layers, but they should be ignored.
    """

    # Generate random samples for input
    if inp_bnds is not None and data is None:
        rand_samps = np.random.rand( n_samps, net.in_size )
        for i, (lb, ub) in enumerate( inp_bnds ):
            lb = -config.MAX_VALUE if lb is None else lb
            ub =  config.MAX_VALUE if ub is None else ub
            rand_samps[ :, i ] = rand_samps[ :, i ] * (ub - lb) + lb 
        samps = rand_samps

    # Get samps
    elif inp_bnds is None and data is not None:
        samps = data

    else:
        raise RuntimeError("Bad options combo")

    # Run the network 
    _, vals = net.eval( samps )
    
    # Return vectors, exclude first and last layer.
    return [ v.T for v in vals ]


def net_to_dump_before_merge(net, weights_for_possible_refines, biases_possible_refines, indices, property, children_nodes):
    
    #print(children_nodes)
    net = merge_net_cluster(net, weights_for_possible_refines, biases_possible_refines, indices, children_nodes )
    #print(weights_for_possible_refines[0])
    weights = net.weights[:-1]
    biases = net.biases[:-1]
    weights_to_replace = weights[-1]
    biases_to_replace = biases[-1]
    weights = weights[:-1]
    biases = biases[:-1]
    weights_to_replace = weights_to_replace @ np.linalg.pinv(property)
    biases_to_replace = biases_to_replace @ np.linalg.pinv(property)
    weights.append(weights_to_replace)
    biases.append(biases_to_replace)
    return  Network(weights, biases, False)    


def merge_net_cluster(  abs_net, weights_for_possible_refines, 
                    biases_possible_refines, indices, children_nodes):
    """
    Merges given network using given partitions. The merging is done in the
    order of the partitions given.

    Arguments:
    
    net         -   Network to merge
    merge_dir   -   The merge directive to merge using.

    Returns:
    
    1.  The merged network
    """


    # Copy of weights and biases
    layer_sizes = [abs_net.layer_sizes[0]]
    weights = []
    biases = []

    for i, w in enumerate(weights_for_possible_refines):
        if len(w) > 1:
            lyr_no_1 , weights_for_refines_1 = w
            weights.append(
                weights_for_refines_1[: , np.array(indices[lyr_no_1-1][1]).tolist()] 
            )
            layer_sizes.append(len(set(indices[lyr_no_1-1][1])))

    layer_sizes.append(abs_net.layer_sizes[-1])
    weights.append(abs_net.weights[-1])
    
    for i, b in enumerate(biases_possible_refines):
        if len(b) > 1:
            lyr_no , baises_for_refines_1 = b

            biases.append( 
                baises_for_refines_1[np.array(indices[lyr_no-1][1]).tolist()] 
            )
    biases.append(abs_net.biases[-1])
    

    for idx in indices:
        lyr_no, indices_lyr =  idx
        o_w = weights[lyr_no]
        i_w, i_b = weights[lyr_no - 1], biases[lyr_no - 1]
        indices_lyr = set(indices_lyr)
        partition = []
        for idx in indices_lyr:
            partition.append(children_nodes[str(lyr_no)][idx])

        n_i_w = np.empty((i_w.shape[0], len(partition)), dtype=config.FLOAT_TYPE_NP)
        n_o_w = np.empty((len(partition), o_w.shape[1]), dtype=np.float32)
        n_i_b = np.empty((len(partition),), dtype=config.FLOAT_TYPE_NP)
        
        for i, part in enumerate(partition):
            n_o_w[i, :] = np.sum(o_w[part, :], axis=0)

            i_w_part = i_w[:, part]
            
            n_i_w[:, i] = i_w[:, part][:, 0]

            n_i_b[i] = i_b[part][0]
        
        weights[lyr_no] = n_o_w
        weights[lyr_no - 1] = n_i_w
        biases[lyr_no - 1] = n_i_b

    return Network( weights, biases, end_relu = abs_net.end_relu )

def get_parents_from_child(linkages):
    """
    linkages    -  A linkage list which represents the way in which the 
                   neurons were merged.

    Returns:

    A list in which we get the parent node of a child node 
    """

    parents = []
    for linkage in linkages:
        #get the lyr_no and merge_dir for a particular layer
        lyr_no, hierarchy = linkage[0], linkage[1]
        parent_for_lyr = [None for i in range(len(hierarchy))]
        for neur, children in enumerate(hierarchy):
            child_1, child_2 = children
            if child_1 is not None and child_2 is not None:
                parent_for_lyr[child_1] = neur
                parent_for_lyr[child_2] = neur
        parents.append((lyr_no, parent_for_lyr))   

    return parents

def biases_for_possible_refines(net, linkages, inc_dec_vects, children_leaves_nodes):

    """
    net         -   The network

    linkages    -  A linkage list which represents the way in which the neurons 
                   were merged.

    Returns:

    A list of list which gives the biases  for different type of merging possible
    """

    biases_possible_refines = []
    
    for linkage in linkages:
        
        lyr_no, hierarchy = linkage
        
        bias_lyr = np.empty((len(hierarchy),),dtype = np.float32) 
        
        for i, children in enumerate(hierarchy):
            
            child1, child2 = children
            
            if child1 is None and child2 is None:
                bias_lyr[i] = net.biases[lyr_no-1][i]
                bias_lyr[i] = net.biases[lyr_no-1][i]
            else:
                children_leaves_nodes_lyr = children_leaves_nodes[str(lyr_no)] 
                if inc_dec_vects[lyr_no][children_leaves_nodes_lyr[i][0]] > 0:
                    bias_lyr[i] = np.maximum(bias_lyr[child1], bias_lyr[child2])
                else:
                    bias_lyr[i] = np.minimum(bias_lyr[child1], bias_lyr[child2])
        
        biases_possible_refines.append((lyr_no, bias_lyr))
    
    return biases_possible_refines

def weights_for_merges(net, linkages, inc_dec_vects, children_leaves_nodes):

    """
    net         -   The network

    linkages    -  A linkage list which represents the way in which the neurons 
                   were merged.

    Returns:

    A list of list which gives the refinement weigths for different type of 
    merging possible
    """
    weights_for_possible_refines = []

    for linkage in linkages:

        lyr_no, hierarchy = linkage

        weigths_lyr = np.empty((    
            net.layer_sizes[lyr_no-1], len(hierarchy)) ,dtype = np.float32)
        
        for i, children in enumerate(hierarchy):
            
            child1, child2 = children
            
            if child1 is None and child2 is None:
                weigths_lyr[:, i] = net.weights[lyr_no-1][:, i]
                weigths_lyr[:, i] = net.weights[lyr_no-1][:, i]
            else:
                children_leaves_nodes_lyr = children_leaves_nodes[str(lyr_no)] 
                if inc_dec_vects[lyr_no][children_leaves_nodes_lyr[i][0]] > 0:
                    weigths_lyr[:, i ] = np.maximum(
                        weigths_lyr[:, int(child1)], weigths_lyr[:, int(child2)])
                else:
                    weigths_lyr[:, i ] = np.minimum(
                        weigths_lyr[:, int(child1)], weigths_lyr[:, int(child2)])

        weights_for_possible_refines.append((lyr_no, weigths_lyr))

    return weights_for_possible_refines



def least_common_ancestor(net, linkages, parents):
    """
    net         -   The network

    linkages    -   A linkage list which represents the way in which the neurons 
                    were merged.
 
    parents     -   A list of tuple where the first index of tuple is the 
                    lyr_no and second ind corresponds to the parent 
                    (i.e the merge group) for that neuron
    
    Returns :

    lca - The least common ancestor between every pair of leaf nodes
    
    children_lyrs - It is a list of tuple where the first element corresponds 
                    to the lyr_no and the second index corresponds the all the 
                    chidlren nodes in the merge group 
    
    children_nodes-  It is a list of tuple where the first element corresponds 
                     to the lyr_no and the second index corresponds the children 
                     leaf nodes in the merge group 
    
    refines - It it a list of tuples where where the first element 
              corresponds to the lyr_no and the second element is a list of 
              list which stores the possible refinments for the cuts for the 
              leaf nodes
    """

    lca = []
    children_nodes = []
    children_lyrs = {}
    refines = []
    #iterating over linkage list of all of layers
    for linkage in linkages:
        #All the linkages for different merge groups in linkage[1], 
        #linkage[0] is lyr_no
        lyr_no, hierarchy = linkage
        child_ptr_1 = {}
        child_ptr_2 = {}
        possible_refines = [[0 for i in range(net.layer_sizes[lyr_no])] 
                            for j in range(net.layer_sizes[lyr_no])]        
        lca_for_mrg_grp = {}
        for i in range(net.layer_sizes[lyr_no]):
            child_ptr_1[i] = [i]   
            child_ptr_2[i] = [i]     
        for i, link in enumerate(hierarchy):
            child1, child2 = link
            if child1 is not None and child2 is not None:
                child_ptr_1[i] = []
                child_ptr_2[i] = [child1, child2]
                for children in child_ptr_1[child1]:
                    child_ptr_1[i].append(children)
            
                for children in child_ptr_1[child2]:
                    child_ptr_1[i].append(children)

                for c1 in child_ptr_1[child1]:
                    for c2 in child_ptr_1[child2]:
                        temp = sorted([c1, c2])
                        temp = (temp[0], temp[1])
                        lca_for_mrg_grp[str(temp)] = i
             
        lca.append((lyr_no, lca_for_mrg_grp))
        children_lyrs [str(lyr_no)] = child_ptr_1
        children_nodes.append((lyr_no, child_ptr_2))

        for i, refine in enumerate(possible_refines):
            for j in range(len(refine)):
                key = sorted([i,j])
                key = (key[0], key[1])
                if (str(key) in lca_for_mrg_grp.keys()):
                    common_ances   =  lca_for_mrg_grp[str(key)]
                    left_child  = hierarchy[common_ances][0]
                    right_child = hierarchy[common_ances][1]
                    if i in child_ptr_1[left_child]:
                        possible_refines[i][i] = i
                    elif i in child_ptr_1[right_child]:
                        possible_refines[i][i] = i
                    
                    if j in child_ptr_1[left_child]:
                        possible_refines[i][j] = left_child
                    elif j in child_ptr_1[right_child]:
                        possible_refines[i][j] = right_child
                elif i!=j:
                    previous = None
                    current = j
                    while(current is not None):
                        previous = current
                        current = parents[lyr_no-1][1][previous]
                    possible_refines[i][i] = i 
                    possible_refines[i][j] = previous
        
        refines.append((lyr_no, possible_refines))

    return lca, children_lyrs, children_nodes, refines

def get_val_on_cex(abs_net, weights, biases, indices, inp):

    """
    Helper function to get value on a counter-example 

    """

    vals = [inp]
    cval = inp


    # Evaluate inner layers   
    lyr_no = 0
    for w, b in zip(weights, biases):
        cval = cval @ w[1][:, np.array(indices[lyr_no][1]).tolist()] + b[1][np.array(indices[lyr_no][1]).tolist()]
        cval = np.where( cval > 0, cval , 0)
        vals.append(cval)
        lyr_no = lyr_no+1

    # Evaluate last layer
    cval = cval @ abs_net.weights[-1]+ abs_net.biases[-1]

    if abs_net.end_relu:
        cval = np.where(cval > 0, cval, 0)

    vals.append(cval) 


    return cval, vals



def get_val_on_cex_autograd(net, weights, biases, indices, inp):

    """
    Helper function to get value on a counter-example 

    Arguments
    net     -   Concrete network, used for last layer weights.
    """

    vals = [inp]
    cval = inp

    # TODO fix mixing floats and doubles

    # Evaluate inner layers   
    lyr_no = 0
    for w, b in zip(weights, biases):
        w = torch.from_numpy( w[1] )
        b = torch.from_numpy( b[1] )
        cval = cval @ w[:, np.array(indices[lyr_no][1]).tolist()] + b[np.array(indices[lyr_no][1]).tolist()]
        cval = torch.nn.functional.relu( cval )
        vals.append(cval)
        lyr_no = lyr_no+1

    # Evaluate last layer
    cval = ( 
            cval @ torch.from_numpy( net.weights[-1]).type( torch.float32 ) + 
            torch.from_numpy(        net.biases[-1] ).type( torch.float32 )
    )

    if net.end_relu:
        cval = torch.nn.functional( cval )

    vals.append(cval) 

    return cval, vals


def compute_gradient_on_cex( net, inp, out_coeffs = None ):
    """
    Evaluates the network on the given stacked input `inp` and returns the
    gradient of output wrt value at each layer value. 
    
    Returns: A list over the layers of `net` of stack of gradients, one for each
    input in `inp`.
    """
    relu = torch.nn.ReLU()

    # Loop over input points
    if len(inp.shape) <= 1: inp = np.expand_dims( inp, 0 )
    grad_stacks = [ [] for _ in net.layer_sizes ]
    it = tqdm( inp ) if inp.shape[0] >= 100 else inp
    for inp_i in it:

        cval = torch.tensor(inp_i, requires_grad = True,dtype=torch.float32)
        vals = [cval]
        #print("Cex shape: ", cval.shape)

        # Evaluate inner layers
        count = 0
        for w, b in zip(net.weights[:-1], net.biases[:-1]):
            count = count+1
            cval = relu(
                cval @ torch.from_numpy(w).float() + 
                torch.from_numpy(b).float())
            vals.append(cval)

        # Evaluate last layer
        cval = cval @ torch.from_numpy(
                net.weights[-1]).float() + torch.from_numpy(net.biases[-1]).float()
        if net.end_relu:
            cval = relu(cval)
        vals.append(cval)

        # Get value to take grad of
        if out_coeffs is None:
            if config.ASSERTS:
                assert vals[-1].shape[0] == 1
            fin_val = vals[-1]
        else:
            fin_val = torch.sum( vals[-1] * torch.from_numpy(out_coeffs).float(), 
                    dim = -1 )
        
        fin_val.backward(inputs = vals)
        for grad, v in zip( grad_stacks, vals ):
            grad.append( v.grad.numpy().copy() )
            v.grad = None
    
    return [ (np.stack(grad) if len(grad) > 1 else grad[0]) 
            for grad in grad_stacks ] 


def identify_culprit_neurons(conc_net, abs_net, weights_for_possible_refines,
        biases_possible_refines , indices_conc, indices_abs, cex, grads,
        visited_culprit):

    """
    Returns the culprit neuron which should be picked for refinement
    """
    
    __, conc_vals = get_val_on_cex(conc_net, weights_for_possible_refines, 
                                   biases_possible_refines, indices_conc, cex)
    __, abs_vals = get_val_on_cex(abs_net, weights_for_possible_refines, 
                                  biases_possible_refines, indices_abs, cex)
    culprit = None
    
    for i in range(1, len(conc_vals)-1):
        c_val, a_val, grad_lyr      = conc_vals[i], abs_vals[i], grads[i]
        
        val_diff_abs_conc_gradients = np.abs(np.multiply( 
                                    np.array(c_val)-np.array(a_val),grad_lyr))
        
        indices_not_visited         = np.where(
                                    np.array(visited_culprit[i-1])==0)[0].tolist()
       
        if len(indices_not_visited) == 0:
            continue 

        pos_max_elem = indices_not_visited[
                       val_diff_abs_conc_gradients[indices_not_visited].argmax()]

        max_val_elem                = val_diff_abs_conc_gradients[pos_max_elem]

        if culprit is None and visited_culprit[i-1][pos_max_elem]==0:
            culprit = (i, pos_max_elem, max_val_elem)
        elif culprit is not None and (
            culprit[2] < max_val_elem) and visited_culprit[i-1][pos_max_elem]==0:
            culprit = (i, pos_max_elem, max_val_elem)

    visited_culprit[culprit[0]-1][culprit[1]] = 1
    
    return culprit[0:-1]

def identify_new_abs_indices(conc_net, indices_abs, culprit_neuron, possible_refines):

    """
    Returns new refinement indices after one refinement
    """

    lyr_no = culprit_neuron[0]
    
    neur_no = culprit_neuron[1]

    modified_indices = indices_abs[:lyr_no-1]

    if lyr_no < len(indices_abs):
        modified_indices+= indices_abs[lyr_no:]

    lyr_cut_refine = possible_refines[lyr_no-1][1][neur_no]

    new_indices_in_lyr = np.minimum(lyr_cut_refine, indices_abs[lyr_no-1][1])

    new_indices_in_lyr = (lyr_no, np.array(new_indices_in_lyr))

    modified_indices.append(new_indices_in_lyr)

    return sorted(modified_indices) 

def get_refining_linkage( net, sat_parts, cl_vects ):
    """
    Gets the linkage matrices for hierarchial clustering.

    Arguments:
    
    net         -   The network
    sat_parts   -   The merge directive the linkages should refined. Any
                    clustering produced from the linkages will refine this merge
                    directive. This should usually be the merge-to-saturation
                    directive.
    cl_vects    -   A list of stack of vectors, one stack for each layer. Each
                    vector in the stack corresponds to one neuron.

    Returns:
    
    A linkage list following the layer order from `sat_parts`, and refining the
    neuron groups present in `sat_parts`.
    """
    
    if config.ASSERTS:
        # Each layer index makes sense
        for lyr_idx, _ in sat_parts:
            assert 1 <= lyr_idx and lyr_idx < net.num_layers

        # sat_parts should have merge instructions for each layer
        assert ( 
            set( [ i for i, _ in sat_parts ] ) == 
            set( range( 1, net.num_layers - 1 ))
        ) 
        
        # Each neuron is in exactly one partition
        for lyr_idx, partition in sat_parts:
            for n_idx in range( net.layer_sizes[ lyr_idx ] ):
                assert sum([ 1 for part in partition if n_idx in part ]) == 1

    # Loop over layers, generate linkages
    linkages = []
    for lyr_idx, partition in sat_parts:

        # Loop over neuron classes, generate linkages for each
        llist = []
        for neur_list in partition:

            # Class has more than one neuron
            if len(neur_list) > 1:
                vects = cl_vects[ lyr_idx ][ neur_list, : ]

                llist.append((
                    neur_list,
                    linkage( vects, method = config.CLUSTERING_METHOD )
                ))

            # Class is singleton
            else:
                llist.append(( [neur_list[0]], None ))
                
        linkages.append(( lyr_idx, llist ))
        
    return linkages



def merge_dir_from_linkage( linkage, dist_th ):
    """
    Returns a merge directive generated from the given linkage, using `dist_th`
    as the threshold distance.
    """
    
    # Loop over layer directives
    merge_dir = []
    for lyr_idx, lkg_list in linkage:

        # Build partition
        partition = []
        for neur_list, lkg in lkg_list:
            
            # If not singleton, use fcluster
            if lkg is not None:
                yhat = fcluster( lkg, dist_th, criterion='distance' )
                clust_dict = defaultdict( list )
                for elem_idx, clust_idx in enumerate( yhat ):
                    clust_dict[ clust_idx ].append( neur_list[ elem_idx ] )

                partition.extend( clust_dict.values() )

            # Else, singleton partition
            else:
                partition.append( neur_list )

        merge_dir.append(( lyr_idx, partition ))
    
    return merge_dir


def modified_linkages(linkages):

    """
    Returns a list of tuples where the first element of the tuple is 
    lyr_no and the second element is a list which contain a mapping from a 
    node to its left and right children
    """

    modified_linkages = []

    for linkage in linkages:
        lyr_no, partition =  linkage
        merge_grp1, hierarchy_struct_grp1 = partition[0][0], partition[0][1]
        if len(partition) == 2:
            merge_grp2, hierarchy_struct_grp2 = partition[1][0], partition[1][1]

        neurs_len  = len(merge_grp1)
        if len(partition)==2:
            neurs_len += len(merge_grp2)
        modified_linkages_lyr = []
        for i in range(neurs_len):
            modified_linkages_lyr.append((None,None))
        
        mapping_for_grp1 = {}

        
        merge_grp_exhaust_grp1 = neurs_len

        if hierarchy_struct_grp1 is not None:

            for i, level in enumerate(hierarchy_struct_grp1):
                neur1, neur2 =  level[0], level[1]
                children_nodes = []
                mapping_for_grp1[len(merge_grp1)+i] = neurs_len+i
                merge_grp_exhaust_grp1 = neurs_len+i
                if neur1 < len(merge_grp1):
                    children_nodes.append(merge_grp1[int(neur1)])
                else:
                    children_nodes.append(mapping_for_grp1[int(neur1)])
                                    
                if neur2 < len(merge_grp1):
                    children_nodes.append(merge_grp1[int(neur2)])
                else:
                    children_nodes.append(mapping_for_grp1[int(neur2)])
                modified_linkages_lyr.append((children_nodes[0], children_nodes[1]))
        
        if neurs_len < merge_grp_exhaust_grp1:
            merge_grp_exhaust_grp1 = merge_grp_exhaust_grp1+1
        
        mapping_for_grp2 = {}

        if len(partition)==2:

            if hierarchy_struct_grp2 is not None:

                for i, level in enumerate(hierarchy_struct_grp2):
                    neur1, neur2 =  level[0], level[1]
                    children_nodes = []
                    mapping_for_grp2[len(merge_grp2)+i] = merge_grp_exhaust_grp1+i
                    if neur1 < len(merge_grp2):
                        children_nodes.append(merge_grp2[int(neur1)])
                    else:
                        children_nodes.append(mapping_for_grp2[int(neur1)])
                                        
                    if neur2 < len(merge_grp2):
                        children_nodes.append(merge_grp2[int(neur2)])
                    else:
                        children_nodes.append(mapping_for_grp2[int(neur2)])
                    
                    modified_linkages_lyr.append((children_nodes[0], 
                                                  children_nodes[1]))

        modified_linkages.append((lyr_no, modified_linkages_lyr))

    return modified_linkages
   

if __name__ == "__main__":

    from cegar import saturation_partition
    from split_and_merge import split_net
    from network import Network

    import sys

    test_no = int(sys.argv[1])

    if test_no == 1:
        net = Network(
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
        splitted_net, inc_dec_vects = split_net( net )
        conc_net = copy.deepcopy(splitted_net)
        indices_abs = [(1, [6,7,6,7,6]),(2, [3,1,3])]
        indices_conc = [(1, [0,1,2,3,4]), (2,[0,1,2])]
        grads = compute_gradient_on_cex(conc_net, [1,2])
        sat_parts = saturation_partition( inc_dec_vects, pos_neg_vects )
        print("sat_parts", sat_parts)
        cl_vects = gen_vects_char( splitted_net )
        print( [ (v.shape if v is not None else None) for v in cl_vects ] )
        linkages = get_refining_linkage( splitted_net, sat_parts, cl_vects )
        linkages =  modified_linkages(linkages)
        linkages = sorted(linkages, key = lambda x: x[0])
        # print("linkages", linkages)
        parents = get_parents_from_child(linkages)
        lca, children_leaves_nodes, children_nodes, possible_refines =least_common_ancestor(splitted_net, linkages, parents)
        weights_for_possible_refines =  weights_for_merges(splitted_net, linkages, inc_dec_vects, children_leaves_nodes)
        biases_possible_refines = biases_for_possible_refines(splitted_net, linkages, inc_dec_vects, children_leaves_nodes )
        actual_cex =  simulate_get_cex(conc_net, indices_abs, weights_for_possible_refines, biases_possible_refines, [[1,2],[1,2]],4)
        visited_culprit = [[0 for i in range(conc_net.layer_sizes[j])] for j in range(1,len(conc_net.layer_sizes)-1)]

        culprit_neuron = identify_culprit_neurons(conc_net, splitted_net, weights_for_possible_refines, biases_possible_refines, indices_conc, indices_abs, [1,2], grads, visited_culprit)
        print("=======================")
        print("culprit_neuron", culprit_neuron)
        
        # # print("possible", possible_refines)
        indices_abs = identify_new_abs_indices(conc_net, indices_abs, culprit_neuron, possible_refines)
        print("indices_abs",indices_abs)
        abs_net = splitted_net 
        print("========================")
        culprit_neuron = identify_culprit_neurons(conc_net, splitted_net, weights_for_possible_refines, biases_possible_refines, indices_conc, indices_abs, [1,2], grads, visited_culprit)
        print("culprit_neuron", culprit_neuron)
        indices_abs = identify_new_abs_indices(conc_net, indices_abs, culprit_neuron, possible_refines)
        print("indices_abs",indices_abs)
        print("=========================")
        culprit_neuron = identify_culprit_neurons(conc_net, splitted_net, weights_for_possible_refines, biases_possible_refines, indices_conc, indices_abs, [1,2], grads, visited_culprit)
        print("culprit_neuron", culprit_neuron)
        indices_abs = identify_new_abs_indices(conc_net, indices_abs, culprit_neuron, possible_refines)
        print("indices_abs",indices_abs)
        print("=========================")
        culprit_neuron = identify_culprit_neurons(conc_net, splitted_net, weights_for_possible_refines, biases_possible_refines, indices_conc, indices_abs, [1,2], grads, visited_culprit)
        print("culprit_neuron", culprit_neuron)
        indices_abs = identify_new_abs_indices(conc_net, indices_abs, culprit_neuron, possible_refines)
        print("indices_abs",indices_abs)
        # # print("conc_net", conc_net.weights)
        print("========================")
        # # print("abs_net", abs_net.weights)
        # # print("========================")
        merged_net = merge_net_cluster( abs_net , weights_for_possible_refines, biases_possible_refines, indices_abs, children_leaves_nodes)
        print("mege_net", merged_net.weights)
        
    elif test_no == 3:
        net = Network(
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
                    [-1.], 
                    [-1.],
                    [-1.],
                ])
            ],
            biases = [
                np.array([ 0., 1., 0. ]),
                np.array([ 1., 2., 3. ]),
                np.array([ 0 ]),
            ],
            end_relu = False
        ) 
        splitted_net, inc_dec_vects= split_net( net)
        print("Split net: ", splitted_net)
        conc_net = copy.deepcopy(splitted_net)
        indices_abs = [(1, [6,7,6,7,6,7]),(2, [4,4,4])]
        indices_conc = [(1, [0,1,2,3,4,5]), (2,[0,1,2])]
        sat_parts = saturation_partition( inc_dec_vects, pos_neg_vects )
        print("sat_parts", sat_parts)
        cl_vects = gen_vects_char( splitted_net )
        print( [ (v.shape if v is not None else None) for v in cl_vects ] )
        linkages = get_refining_linkage( splitted_net, sat_parts, cl_vects )
        linkages =  modified_linkages(linkages)
        linkages = sorted(linkages, key = lambda x: x[0])
        # print("linkages", linkages)
        parents = get_parents_from_child(linkages)
        lca, children_leaves_nodes, children_nodes, possible_refines =least_common_ancestor(splitted_net, linkages, parents)
        weights_for_possible_refines =  weights_for_merges(splitted_net, linkages, inc_dec_vects, children_leaves_nodes)
        biases_possible_refines = biases_for_possible_refines(splitted_net, linkages, inc_dec_vects, children_leaves_nodes )

        bounds = [ (0., 1.), (0., 1.) ]

        print("Starting pgd attack")

        pgd_cex = pgd_attack(net, indices_abs, 100, 
            bounds, weights_for_possible_refines, biases_possible_refines, )
        print("Cex: ", pgd_cex)
        print("Val: ", get_val_on_cex(
            net, weights_for_possible_refines, biases_possible_refines,
            indices_abs, pgd_cex))

        visited_culprit = [[0 for i in range(conc_net.layer_sizes[j])] for j in range(1,len(conc_net.layer_sizes)-1)]
        grads = compute_gradient_on_cex(conc_net, pgd_cex)
        culprit_neuron = identify_culprit_neurons(conc_net, splitted_net,
                weights_for_possible_refines, biases_possible_refines,
                indices_conc, indices_abs, pgd_cex, grads, visited_culprit)
        indices_abs = identify_new_abs_indices(conc_net, indices_abs,
                culprit_neuron, possible_refines)
        print("=======================")
        print("culprit_neuron", culprit_neuron)

        print("Starting pgd attack")

        pgd_cex = pgd_attack(net, indices_abs, 100, 
            bounds, weights_for_possible_refines, biases_possible_refines, )
        print("Cex: ", pgd_cex)
        print("Val: ", get_val_on_cex(
            net, weights_for_possible_refines, biases_possible_refines,
            indices_abs, pgd_cex))

        visited_culprit = [[0 for i in range(conc_net.layer_sizes[j])] for j in range(1,len(conc_net.layer_sizes)-1)]
        grads = compute_gradient_on_cex(conc_net, pgd_cex)
        culprit_neuron = identify_culprit_neurons(conc_net, splitted_net,
                weights_for_possible_refines, biases_possible_refines,
                indices_conc, indices_abs, pgd_cex, grads, visited_culprit)
        indices_abs = identify_new_abs_indices(conc_net, indices_abs,
                culprit_neuron, possible_refines)
        print("=======================")
        print("culprit_neuron", culprit_neuron)
        
