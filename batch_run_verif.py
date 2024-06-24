
from multiprocessing import Pool
import os
import itertools
from argparse import ArgumentParser
import argparse


import config_acasxu

import sys
import os
import os.path



def run_task(nt, prp, net_path, prop_path, outs_path, stats_path, conf, timeout,
        tidx, pyfile):
    """
    conf is a dict with:
    abs_mth -   Abstraction method to use
    ref_mth -   Refine method to use
    cl_vecs -   The clustering vector generation method to use
    """
    outname = "{}/Task_{}_net_{}_prp_{}_abs_{}_ref_{}_2c_{}_v_{}.out".format(
        outs_path, tidx, nt, prp, 
        conf['abs_mth'], 
        conf['ref_mth'],
        conf['cl_vecs'],
        conf['--solver-type']
    )
    statsname = "{}/Task_{}_net_{}_prp_{}_abs_{}_ref_{}_2c_{}_v_{}.stat".format(
        stats_path, tidx, nt, prp, 
        conf['abs_mth'], 
        conf['ref_mth'],
        conf['cl_vecs'],
        conf['--solver-type']
    )

    
    comb_net_path = os.path.join( net_path, nt )
    comb_prop_path = os.path.join( prop_path, prp )
    command = "timeout -k 0 {} python3 {} -n {} -p {} -a {} -r {} -v {} --solver-type {} --stats-file {} 2>&1 | tee {}".format(
        timeout, pyfile,
        comb_net_path, 
        comb_prop_path,
        conf['abs_mth'], 
        conf['ref_mth'],
        conf['cl_vecs'],
        conf['--solver-type'],

        statsname,
        
        outname
    )
    print("In Thread {0} : {1}".format( tidx, command ))
    os.system(command)

def run_per_cpu( fargs ): 
    tasks, net_path, prop_path, outs_path, stats_path, timeout, tidx, pyf = fargs
    print(tasks)
    for nt, prp, conf in tasks:
        run_task(nt, prp, net_path, prop_path, outs_path, stats_path, conf,
                timeout, tidx, pyf)



 
if __name__ == '__main__':

    # Set number of cpus, and timeout here
    timeout = 200
    n_cpu =  1

    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--network',
        dest='net_path',
        required=True,
        type=str,
        help="The network path"
    )
    parser.add_argument('-p', '--property',
        dest='prop_path',
        required=True,
        type=str,
        help="The the property path"
    )
    parser.add_argument('-o', '--outs_path',
        dest='outs_path',
        required=True,
        type=str,
        help="The the outs path"
    )
    parser.add_argument('-s', '--stats_path',
        dest='stats_path',
        required=True,
        type=str,
        help="The the stats path"
    )
    parser.add_argument('-m', '--mode',
        dest='mode',
        required=True,
        type=str,
        choices=['ours','baseline'],
        help="Whether to run our method or the baseline method"
    )

    args = parser.parse_args()
    
    file_to_run = None
    if args.mode == 'ours':
        file_to_run = 'main_tree_cegar.py'
    elif args.mode == 'baseline':
        file_to_run = 'main_cegar.py'

    net_path =  args.net_path
    prop_path = args.prop_path
    outs_path = args.outs_path
    stats_path = args.stats_path
    # Collect names for all networks
    netnames = [ 
        n for n in os.listdir( net_path ) 
    ]

    # Collect names for all properties
    properties = [ 
        p for p in os.listdir( prop_path ) if p.endswith('.prop') 
    ]

    print("Network names", netnames)
    print("Properties",properties)

    choice = 3

    # Collect configurations
    if choice==1:
        confs = [
                {
                    'abs_mth' : 'saturation',  
                    'ref_mth' : 'cegar-hcluster', 
                    'cl_vecs' : 'simulation',
                    '--solver-type': 'neuralsat'
                }, 

        ]
    elif choice==2:
        confs = [
                {
                    'abs_mth' : 'saturation',  
                    'ref_mth' : 'cegar', 
                    'cl_vecs' : 'simulation',
                    '--solver-type': 'neuralsat'
                }, 

        ]
    else:
        confs = [
                {
                    'abs_mth' : 'none',  
                    'ref_mth' : 'cegar', 
                    'cl_vecs' : 'simulation',
                    '--solver-type': 'neuralsat'
                }, 

        ]

    tasks = itertools.product(netnames, properties, confs)

    # Distribute
    task_per_cpu = [ [] for _ in range(n_cpu) ]
    cpu_idx = 0
    for task in tasks:
        task_per_cpu[ cpu_idx ].append( task )
        cpu_idx += 1
        if cpu_idx >= n_cpu:
            cpu_idx = 0

    #print(task_per_cpu)
    with Pool(processes=len(task_per_cpu)) as p:
        p.map(
            run_per_cpu, 
            ( (tasks, net_path, prop_path, outs_path, stats_path, timeout, i,
                    file_to_run)
                for i,tasks in enumerate(task_per_cpu) )
        )
    

