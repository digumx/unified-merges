"""
Some configuration settings
"""

import numpy as np

FLOAT_TYPE_NP = np.float128  # The datatype for float
FLOAT_TOL = 1e-5            # A small enough floating point value

# Some paths
MARABOU_PATH =  "Marabou/"
STATS_PATH = "mnist_logs/"
PLOTS_PATH = "plots/"

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

CEX_REFINE = 0.001

# Learning rate for pgd
PGD_LR = 0.001

# Number of samples used in calculating score when being guided by multiple
# cexes. Applies only when guided by randomly sampled points (for prop)
NO_SAMPLES_MULTI_CEX_GUIDED_SCORE = 10 #10000

# Number of subsamples to calculate score on at each stage. This applies for
# both random and dataset samples. -1 means take whole cex.
NO_SUBSAMPLES_MULTI_CEX_GUIDED_SCORE = 1000

# Configure pgd attack
NO_SAMPLES_PGD_ATTACK = 50
NO_STEPS_PGD = 1000
