"""
Some configuration settings
"""

import numpy as np

FLOAT_TYPE_NP = np.float128  # The datatype for float
FLOAT_TOL = 1e-5            # A small enough floating point value

# Some paths
MARABOU_PATH =  "Marabou/"
PROPERTY_PATH = "properties/acasxu_prop"
NETWORK_PATH = "networks/acasxu"
STATS_PATH = "acasxu_logs/"

# Simulation config
NUM_SAMPS = 10000           # Number of samples for clustering
MAX_VALUE = 1000            # The maximum (or minimum) value an unbounded input
                            # can take. Only used for sampling

# Clustering config
CLUSTERING_METHOD = 'complete'
INIT_THRESHOLD = 100        # Initial threshold
DELTA_THRESHOLD = 5         # How much to reduce threshold by in each iteration

# Cegar config
NUM_REFINE = 50             # Number of neurons to refine in one step

ASSERTS = True              # Enables asserts checking consistency
DEBUG = True

CEX_REFINE = 0.001
