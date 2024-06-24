"""
Implements the original merge upto saturation and counterexample guided
refinement
"""


import numpy as np

import config

from clustering import (gen_vects_sim, get_refining_linkage)

import torch


class LinkageTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def get_leaf_nodes_linkage_tree(root, leaf_node):

    if root.left is None and root.right is None:
        leaf_node[root.value] = [root.value]
        return [root.value]

    left_leaves = get_leaf_nodes_linkage_tree(root.left,leaf_node)
    right_leaves = get_leaf_nodes_linkage_tree(root.right,leaf_node)
    leaf_node[root.value] = left_leaves + right_leaves
    return left_leaves + right_leaves

def compute_gradient( net, inp ):
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
    cval = torch.tensor(inp, requires_grad = True,dtype=torch.float32)
    #print("Shape",cval.shape)
    vals = [cval]

    relu = torch.nn.ReLU()

    #print('Cval: ', cval)
    #print('Cval req grad: ', cval.requires_grad)

    # Evaluate inner layers
    count = 0
    for w, b in zip(net.weights[:-1], net.biases[:-1]):
        count = count+1
        cval = relu(cval @ torch.from_numpy(w).float() + torch.from_numpy(b).float())
        vals.append(cval)

    # Evaluate last layer
    cval = cval @ torch.from_numpy(net.weights[-1]).float() + torch.from_numpy(net.biases[-1]).float()
    if net.end_relu:
        cval = relu(cval)
    vals.append(cval)

    vals[-1][0].backward(inputs = vals)

    grads = [ v.grad.numpy() for v in vals ]
    
    #print('grads: ', grads )

    return grads




def saturation_partition( inc_dec_vects, merge_first_lyr = False):
    """
    Given inc-dec and pos-neg classifications, generate partitions for merging
    upto saturation. 
    
    Arguments:

    inc_dec_vects   -   A list of vectors for each layer storing the inc-dec
                        classification.  Each vector has one element for each
                        neuron. Each element is either +1 for inc, or -1 for
                        dec. There is no vector for the input layer, instead
                        there is None.
    merge_first_lyr -   If true, the first layer is merged, otherwise it is not.

    Returns: A merge directive representing merging to saturation.
    """
    merge_dir = []
    
    # Loop over layers
    for lyr_idx in range( len(inc_dec_vects) - 2, 0, -1 ):

        # Collects masks for each partition
        masks = []

        # Masks from inc-dec
        inc_dec = inc_dec_vects[ lyr_idx ]
        masks.append( inc_dec > 0 )
        masks.append( inc_dec < 0 )
        
        # Add partition list
        merge_dir.append( ( lyr_idx,
            [ np.nonzero( m )[0] for m in masks if np.any( m ) ] ) )
    
    return merge_dir

def identify_culprit_neurons( conc_net, abs_net, merge_dir, cex, layer_no ,cl_vects, inp_bnds, grads):
    """
    Uses a given cex to find a concrete neuron to split out of the abstract
    neuron and returns the resulting partitions

    Arguments:
    
    conc_net    -   The concrete network
    abs_net     -   The abstract network
    merge_dir   -   The merge directive to refine
    cex         -   The counterexample to use for splitting
    layer_no    -
    cl_vects    -
    linkages    -

    Returns the merge_dir after splitting
    """
    flag_changed = False

    if config.ASSERTS:
        assert conc_net.num_layers == abs_net.num_layers

    # Simulate abstract and concrete

    conc_vals_last_lyr, conc_vals = conc_net.eval( cex )
    a, abs_vals = abs_net.eval( cex )

    #print("Right now value of abs network", a)


    #print("Layer No", layer_no)

    partition = next((t for t in merge_dir if t[0] == layer_no), None)
    unchanged_merge_dir = [t for t in merge_dir if t[0] != layer_no]
    #print("Unchanged merge dir", unchanged_merge_dir)

    # Collect distance between abstract and concrete neurons to identify which neurons is actually deviating

    
    dists = []
    #print(layer_no)
    #print("Partition", partition)
    if partition[1] is not None: 
        for part_no, part in enumerate(partition[1]):
            dist_within_partition = []   
            for i, abs_idx in enumerate(part):
                dist_within_partition.append((str(part),part, i, np.abs((conc_vals[ layer_no ][ abs_idx ]-abs_vals[layer_no][part_no])*grads[layer_no][part_no])))
                #print("Length of conc_vals at layer",abs_idx, part_no, conc_vals[ layer_no ][ abs_idx ],abs_vals[layer_no][part_no])
            dist_within_partition = sorted(dist_within_partition, key=lambda x: x[3], reverse= True)
            dists.append(dist_within_partition[0])
    

    dists = sorted(dists, key=lambda x: x[3], reverse= True)
    #print("Dists",dists)

    inp_bnds_new = []
    for i,c in enumerate(cex):
        l = None
        u = None
        if c-config.CEX_REFINE>=inp_bnds[i][0]:
            l = c-config.CEX_REFINE
        else:
            l = inp_bnds[i][0]
        
        if c+config.CEX_REFINE<=inp_bnds[i][1]:
            u = c+config.CEX_REFINE
        else:
            u = inp_bnds[i][1]

        inp_bnds_new.append((l, u))

    new_clus = gen_vects_sim(conc_net, inp_bnds_new )

    clus_vect = [[] for cl_vect in cl_vects]


    for i, cl_vect in enumerate(cl_vects):
        for j, cex_inp in enumerate(new_clus[i]):
            cex_inp_array = np.array(cex_inp)
            clus_vect[i].append( np.concatenate((cl_vect[j], cex_inp_array)))


    converted_list = []
    [converted_list.append(np.vstack(sub_list)) for sub_list in clus_vect]


    linkages = get_refining_linkage( conc_net, merge_dir, converted_list )
    _, linkage_for_layer = next((t for t in linkages if t[0] == layer_no), None)

    tree_for_partition = []
    
    linkage_dict_part = {}
    leaf_node_linkage_dict = []


    for neur_list, lkg_list in linkage_for_layer:
        if lkg_list is not None:
            linkage_dict = {} 
            leaf_node = {}
            for i,neur in enumerate(neur_list):
                linkage_dict[i] = LinkageTree(i)  
            root = None
            for iter,l in enumerate(lkg_list):
                root = LinkageTree(len(neur_list)+iter)
                linkage_dict[len(neur_list)+iter] = root
                root.left = linkage_dict[int(l[0])]           
                root.right = linkage_dict[int(l[1])]
            get_leaf_nodes_linkage_tree(root, leaf_node)
            leaf_node_linkage_dict.append(leaf_node)
            linkage_dict_part[str(neur_list)] =  leaf_node
        else:
            linkage_dict_part[str(neur_list)] =  None


    new_merge_dir = []

    #print("DISTANCES", dists)

    for q, b in enumerate(dists):
        part, parti, indx, distance = b
        if distance == 0:
            new_merge_dir.append(parti)  

        elif q  == 0:
            if linkage_dict_part[part] is not None:
                temp = sorted(linkage_dict_part[part].keys())
                t = linkage_dict_part[part][temp[-1]]
                if len(t)>2:
                    for subtree in temp:
                        if indx in linkage_dict_part[part][subtree] and len(linkage_dict_part[part][subtree])>1 and len(linkage_dict_part[part][subtree])<len(parti):
                            d = linkage_dict_part[part][subtree]
                            y = []
                            for i in d:
                                y.append(parti[i])
                            x = list(set(parti)-set(y))
                            new_merge_dir.append(list(set(y)))                    
                            new_merge_dir.append(x)
                            flag_changed = True
                            
                            break
                        elif indx in linkage_dict_part[part][subtree] and len(linkage_dict_part[part][subtree])>1 and len(linkage_dict_part[part][subtree]) == len(parti):
                            d = linkage_dict_part[part][subtree]
                            #print("DDDD", d)
                            y = []
                            for i in d:
                                if i!= indx:
                                    y.append(parti[i])
                            new_merge_dir.append(y)
                            #print("YYYY", y)
                            x = list(set(parti)-set(y))
                            y = []
                            #print("XXX",x )
                            new_merge_dir.append(x)
                            flag_changed = True
                            break


                elif len(t)==2:
                    new_merge_dir.append([parti[t[0]]])
                    new_merge_dir.append([parti[t[1]]])  
                    flag_changed = True
                else :
                    new_merge_dir.append([parti[t[0]]])
            else:
                    new_merge_dir.append(parti) 
        else:
            new_merge_dir.append(parti)           

    #print(new_merge_dir)
    merge_dir = []
    for arr in new_merge_dir:
        merge_dir.append(np.array(arr))

    
    #print([(layer_no,merge_dir)]+unchanged_merge_dir)
    return flag_changed,[(layer_no,merge_dir)]+unchanged_merge_dir

def get_new_merge_dir(conc_net, culprit_neuron, merge_dir):

    split_set = set()
    split_set.add( culprit_neuron )
   
    new_merge_dir = []
    for lyr_idx, partition in merge_dir:
        if lyr_idx!= culprit_neuron[0]:
            new_partition = partition
        else:            
            new_partition = []
            for part in partition:
                rem_part = [ i for i in part if (lyr_idx,i) not in split_set ]
                if len(rem_part) > 0:
                    new_partition.append( rem_part )
                new_partition.extend([ 
                    [i] for i in part if (lyr_idx,i) in split_set ])
        new_merge_dir.append(( lyr_idx, new_partition ))
        
    return new_merge_dir


def get_cegar_culprit_neuron(conc_net, abs_net, counter_examples, visited_culprit, grads):
    # TODO Potentially redundant with identify culprit neurons, fix
    
    __, conc_vals = conc_net.eval(counter_examples)
    __, abs_vals  = conc_net.eval(counter_examples)
    
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




def cex_guided_refine( conc_net, abs_net, merge_dir, cex, 
        n_refine = config.NUM_REFINE ):
    """
    Uses a given cex to find a concrete neuron to split out of the abstract
    neuron and returns the resulting partitions

    Arguments:
    
    conc_net    -   The concrete network
    abs_net     -   The abstract network
    merge_dir   -   The merge directive to refine
    cex         -   The counterexample to use for splitting
    n_refine    -   The number of neurons to split out.

    Returns the merge_dir after splitting
    """

    if config.ASSERTS:
        assert conc_net.num_layers == abs_net.num_layers

    # Simulate abstract and concrete
    _, conc_vals = conc_net.eval( cex )
    a, abs_vals = abs_net.eval( cex )

    # Collect distance between abstract and concrete neurons
    neurs_list = []
    dists = []
    for lyr_idx, partition in merge_dir:
        for abs_idx, part in enumerate(partition):
            if len(part) > 1:
                neurs_list.extend([ (lyr_idx, ci) for ci in part ])
                dists.append( np.abs( 
                    conc_vals[ lyr_idx ][ part ] - abs_vals[ lyr_idx ][ abs_idx ]
                ))

    # Sort by distance
    split_idxs = np.argsort( np.concatenate( dists ))
    if split_idxs.shape[0] > n_refine:
        split_idxs = split_idxs[ -n_refine : ]

    # Collect indices into set
    split_set = set()
    for i in split_idxs:
        split_set.add( neurs_list[i] )

    # Create new partition
    new_merge_dir = []
    for lyr_idx, partition in merge_dir:
        new_partition = []
        for part in partition:
            rem_part = [ i for i in part if (lyr_idx,i) not in split_set ]
            if len(rem_part) > 0:
                new_partition.append( rem_part )
            new_partition.extend([ 
                [i] for i in part if (lyr_idx,i) in split_set ])
        new_merge_dir.append(( lyr_idx, new_partition ))
        
    return new_merge_dir



if __name__ == "__main__":
    
    from itertools import count
    
    from split_and_merge import split_net, merge_net
    from network import Network
    from marabou_query import marabou_query

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
    print("Input net")
    print(in_net)

    # Get split network
    split_net, inc_dec_vects = split_net( in_net )
    print("Split net") 
    print(split_net)
    print("Inc-dec classification")
    print(inc_dec_vects)
    print("Pos-neg classification")
    print(pos_neg_vects)

    # Get merge_dir
    merge_dir = saturation_partition( inc_dec_vects, pos_neg_vects )
    print("Merge lists")
    print(merge_dir)

    # Merge net
    merged_net = merge_net( split_net, merge_dir, inc_dec_vects )
    print("Merged net")
    print(merged_net) 

    # Do a query with merged net
    inp_bnds = [ (-1, 1), (-1, 1) ]
    out_ub = 14
    
    cl_vects = gen_vects_sim( split_net, inp_bnds )
    linkages = get_refining_linkage( split_net, merge_dir, cl_vects )

    # CEGAR loop
    refined_net = merged_net
    refined_merge_dir = merge_dir
    layer_no = 2
    for num in count():

        print("\n\n------- Refine Loop {} -------\n\n".format( num ))
        
        cex = marabou_query( refined_net, inp_bnds, out_ub )
        if cex is None:
            print("No cex, breaking")
            break
        if split_net.eval(cex)[0]>=out_ub:
            print("Counter example found breaking")
            break
        if split_net.eval(cex)[0] < out_ub:
            print("Current output therefore refining", split_net.eval(cex)[0] )

        print("Refining")

        if split_net.layer_sizes[layer_no] <= refined_net.layer_sizes[layer_no]:
            layer_no-=1

        if config.ASSERTS:
            assert split_net.num_layers == refined_net.num_layers

        if layer_no >= 1:
          
            _,refined_merge_dir = identify_culprit_neurons( 
                split_net, 
                refined_net,
                refined_merge_dir,
                cex, layer_no, cl_vects, inp_bnds)
        else:
            break
        print("Lyr  after refinement: ", refined_merge_dir)
        refined_net = merge_net( split_net, refined_merge_dir, inc_dec_vects )
        print("Merged net: ", refined_net)
