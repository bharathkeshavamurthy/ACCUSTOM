"""
This script encapsulates the operations involved in studying the functionalities and performance capabilities of our
Hierarchical Reinforcement Learning (HRL) framework to solve for the optimal orchestration of rotary-wing UAVs that
are involved in the MIMO-enabled harvesting of prioritized data traffic from MIMO-capable Ground Nodes (GNs).

REFERENCE PAPER:
Ma, Y., Hao, X., Hao, J., Lu, J., Liu, X., Xialiang, T., Yuan, M., Li, Z., Tang, J. and Meng, Z., 2021.
A hierarchical reinforcement learning based optimization framework for large-scale dynamic pickup and delivery problems.
Advances in Neural Information Processing Systems, 34, pp.23609-23620.

Author: Bharath Keshavamurthy <bkeshav1@asu.edu>
Organization: School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.
Copyright (c) 2023. All Rights Reserved.
"""

import os
import warnings

# Ignore DeprecationWarnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Tensorflow logging setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit ' \
                             '/home/bkeshav1/workspace/repos/ACCUSTOM/src/HRLEvaluations.py'  # EXXACT GPUs

import plotly
import numpy as np
import tensorflow as tf

"""
SCRIPT SETUP
"""

# NumPy seed
np.random.seed(1337)

# Plotly API access credentials
plotly.tools.set_credentials_file(username='<insert_username_here>', api_key='<insert_api_key_here>')

"""
CONFIGURATIONS
"""

"""
UTILITIES
"""

"""
CORE OPERATIONS
"""
