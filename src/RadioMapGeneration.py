"""
This script encapsulates the operations involved in generating the radio map for the deployment site using an iterative
Maximum Likelihood Estimation (MLE) procedure exploiting the ray-tracing measurements obtained on Wireless InSite.

REFERENCE PAPER:
    J. Chen, U. Yatnalli and D. Gesbert,
    "Learning radio maps for UAV-aided wireless networks: A segmented regression approach,"
    2017 IEEE International Conference on Communications (ICC), Paris, France, 2017, pp. 1-6.

Author: Bharath Keshavamurthy <bkeshav1@asu.edu>
Organization: School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.
Copyright (c) 2023. All Rights Reserved.
"""

import warnings

# Ignore DeprecationWarnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import plotly
import numpy as np
import pandas as pd

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

# Converters
deg2rad, rad2deg = lambda _x: _x * (np.pi / 180), lambda _x: _x * (180 / np.pi)
db_watts, dbm_watts = lambda _x: 10 ** (_x / 10), lambda _x: 10 ** ((_x - 30) / 10)
watts_db, watts_dbm = lambda _x: 10 * np.log10(_x), lambda _x: 10 * np.log10(_x) + 30

pi, tx_p, n_u, n_g, n_a_u, n_a_g = np.pi, dbm_watts(23), 6, 36, 16, 4
x_max, y_max, z_max, x_d, y_d, z_d, h_u, h_g = 3000, 3000, 150, 10, 10, 10, None, None
ptn_cnt, msm_cnt, msm_file = 24, 2400, 'E:/Workspaces/ACCUSTOM/data/wireless_insite_slc_measurements.xlsx'

"""
UTILITIES
"""

# 3D (Euclidean) distance between two grid-voxels in this deployment
distance_3d = lambda _p_voxel, _q_voxel: np.linalg.norm(np.subtract([_p_voxel['x'], _p_voxel['y'], _p_voxel['z']],
                                                                    [_q_voxel['x'], _q_voxel['y'], _q_voxel['z']]))

# Angle (in radians) between two grid-voxels in this deployment
angle = lambda _p_voxel, _q_voxel: np.arcsin(abs(_p_voxel['z'] - _q_voxel['z']) / distance_3d(_p_voxel, _q_voxel))

"""
CORE OPERATIONS: Initializations, Iterative Search, Reconstruction, and Visualizations
"""

# Simulation begins...
print('[INFO] RadioMapGeneration core_operations: Setting up processes for Wireless InSite based radio map generation!')

''' Grid Tessellation '''

n_x, n_y, n_z = int(x_max / x_d), int(y_max / y_d), int(z_max / z_d)

voxels = [{'id': _i + _j + _k, 'x': ((2 * _i) + 1) * (x_d / 2), 'y': ((2 * _j) + 1) * (y_d / 2),
           'z': ((2 * _k) + 1) * (z_d / 2)} for _k in range(n_z) for _j in range(n_y) for _i in range(n_x)]


def coord_voxel(_coord: str) -> object:
    """
    Map rectangular coordinates to grid voxels
    """
    __coord = np.array(_coord.replace('[', '').replace(']', '').replace(' ', '').split(','), dtype=np.float64)
    _x, _y, _z = __coord[0], __coord[1], __coord[2]

    for _voxel in voxels:
        _x_min, _x_max = _voxel['x'] - (x_d / 2), _voxel['x'] + (x_d / 2)
        _y_min, _y_max = _voxel['y'] - (y_d / 2), _voxel['y'] + (y_d / 2)
        _z_min, _z_max = _voxel['z'] - (z_d / 2), _voxel['z'] + (z_d / 2)

        if _x_min <= _x < _x_max and _y_min <= _y < _y_max and _z_min <= _z < _z_max:
            return _voxel

    print('[INFO] RadioMapGeneration coord_voxel: Unknown coordinates encountered! Unable to map to a grid voxel!')
    return None


''' Initializations '''

h_g = z_d / 2 if h_g is None else h_g  # Heights of the GNs

# Assertions for model validations...
assert h_g != 0 and 0 < h_g < z_max, 'Unsupported or Impractical GN height values!'
assert h_g % (z_d / 2) == 0, 'GN height values do not adhere to the current grid tessellation!'
assert x_max % x_d == 0 and y_max % y_d == 0 and z_max % z_d == 0, 'Potential error in given grid tessellation!'
assert msm_file is not None and msm_file != '' and os.path.isfile(msm_file), 'The given measurements file is invalid!'

# A dataframe for the Wireless InSite measurements...
msm = pd.read_excel(msm_file, header=None, usecols=[0, 1, 2],
                    names=['uav_coord', 'gn_coord', 'ch_gain_db'],
                    engine='openpyxl', dtype={'uav_coord': str, 'gn_coord': str, 'ch_gain_db': np.float64})

''' K-Means Clustering for the Measurements '''

# Forgy initialization
clusters = [{'id': _i, 'centroid': msm[_j]['ch_gain_db'],
             'obs_s': [msm[_j]]} for _i, _j in enumerate(np.random.choice(msm_cnt, size=ptn_cnt))]

# Cluster convergence check routine
cluster_converge = lambda _msm_cnt, _obs_s: sum([_obs['prev_cluster'] ==
                                                 _obs['curr_cluster'] for _obs in _obs_s]) == _msm_cnt

obs_s = msm

# Until cluster assignments change...
while not cluster_converge(msm_cnt, obs_s):
    [_cluster['obs_s'].clear() for _cluster in clusters]

    # E-step (Assign)
    for obs in obs_s:
        obs['prev_cluster'] = obs['curr_cluster']
        obs_var = obs['ch_gain_db']

        obs['curr_cluster'] = min([_ for _ in range(ptn_cnt)],
                                  key=lambda _id: (obs_var - clusters[_id]['centroid']) ** 2)

        cluster_id = obs['curr_cluster']
        clusters[cluster_id]['obs_s'].append(obs)

    # M-step (Re-calculate)
    for cluster in clusters:
        if len(cluster['obs_s']) != 0:
            cluster['centroid'] = np.mean(np.array([_obs['ch_gain_db'] for _obs in cluster['obs_s']], dtype=np.float64))

''' Iterative Search '''

''' Reconstruction '''

''' Visualizations '''

# Simulation ends...
print('[INFO] RadioMapGeneration core_operations: Terminated all processes in our radio map generation procedure!')
