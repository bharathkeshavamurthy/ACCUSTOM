"""
This script encapsulates the operations involved in evaluating the performance of our proposed cross-layer optimization
solution (ZF + mTSP + LCSO) for harvesting prioritized traffic from MIMO-capable Ground Nodes (GNs) via MIMO UAVs.

REFERENCES:
    1. Y. Zeng, J. Xu and R. Zhang, "Energy Minimization for Wireless Communication With Rotary-Wing UAV,"
       IEEE Transactions on Wireless Communications, vol. 18, no. 4, pp. 2329-2345, April 2019.
    2. H. Yan, Y. Chen and S. H. Yang, "New Energy Consumption Model for Rotary-Wing UAV Propulsion,"
       IEEE Wireless Communications Letters, vol. 10, no. 9, pp. 2009-2012, Sept. 2021.
    3. T. Bektas, “The Multiple Traveling Salesman Problem: An Overview of Formulations and Solution Procedures,”
       Omega, vol. 34, no. 3, pp. 209–219, 2006.
    4. B. Borowska, “Learning Competitive Swarm Optimization,” Entropy, vol. 24, no. 2, 2022.
    5. https://developers.google.com/optimization/routing/tsp
    6. https://developers.google.com/optimization/routing/vrp
    7. https://developers.google.com/optimization/routing/vrptw

DEPLOYMENT MECHANISM:
    a. Naive K-Means Clustering (Forgy Initialization) to cluster the GNs based on their deployment proximity;
    b. Within each cluster, one assigned UAV (MU-MIMO) positioned optimally via coarse- & fine-grained grid search;
    c. Coarse-Search: A Bounding-Box (BB) method to obtain the set of candidate voxels for the subsequent fine-search;
    d. Fine-Search: A Zero-Forcing (ZF) MU-MIMO beam-forming construction determines channel capacities and latencies;
    e. A Learning Competitive Swarm Optimization (LCSO) algorithm to design the 3D trajectories between these positions;
    f. A Multiple Traveling Salesman Problem (mTSP) setup to graphically obtain the GN association/scheduling mechanism.

REPORTED METRICS:
    a. Total Cumulative Fleet-wide Reward (vs configurable Number of UAVs);
    b. Total Cumulative Fleet-wide Reward (vs configurable Number of Users/GNs);
    c. Total Cumulative Fleet-wide Reward and (vs) Average Per-UAV Energy Consumption.

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
                             '/home/bkeshav1/workspace/repos/ACCUSTOM/src/CrossLayerMTSPEvaluations.py'  # EXXACT GPUs

# import plotly
import numpy as np
import tensorflow as tf
from scipy import constants
# import plotly.graph_objs as go

# noinspection PyUnresolvedReferences
from tensorflow.compat.v1 import assign

from ortools.constraint_solver import pywrapcp
from scipy.interpolate import UnivariateSpline
from concurrent.futures import ThreadPoolExecutor
from ortools.constraint_solver import routing_enums_pb2

"""
SCRIPT SETUP
"""

# NumPy seed
np.random.seed(1337)

# Plotly API access credentials
# plotly.tools.set_credentials_file(username='<insert_your_username_here>', api_key='<insert_your_api_key_here>')

"""
CONFIGURATIONS
"""

# Converters
db_watts, dbm_watts = lambda _x: 10 ** (_x / 10), lambda _x: 10 ** ((_x - 30) / 10)
watts_db, watts_dbm = lambda _x: 10 * np.log10(_x), lambda _x: 10 * np.log10(_x) + 30

# Simulation setup (MKS/SI units)
# TO-DO Config | Core analysis variables: Number of UAVs, Number of GNs, and P_avg
g, n_u, n_g, n_c, n_a_u, n_a_g, m_u, v_min, v_p_min, v_max, v_num = 9.8, 6, 36, 8, 16, 4, 3, 0, 22, 55, 25
temp, k_1, k_2, z_1, z_2, alpha, alpha_, kappa, bw = 300, 1, np.log(100) / 90, 9.61, 0.16, 2, 2.8, 0.2, 5e6
t_min, t_max, x_max, y_max, z_max, x_d, y_d, z_d, h_g, n_ss = 300, 3000, 5000, 5000, 200, 5, 5, 5, None, 60
tx_p, beta_0, w_var = dbm_watts(23), db_watts(20), constants.Boltzmann * temp * bw  # 23dBm, 20dB, Thermal noise
n_w, m_sg, m_sg_ip, n_sw, omega, pwr_avg = 1024, 126, 2, 540, 1, np.arange(start=1e3, stop=2.25e3, step=0.25e3)[0]
ss_cnt, ld_omega, v_0, u_tip, p_0, p_1, p_2 = 9, 1, 7.2, 200, 580.65, 790.6715, 0.0073  # p_2 = (d_0*rho*eps*zeta) / 2
a_min, a_max, m_sg_post, eval_cnt_max, n_sw_div = -25, 10, m_sg_ip * (m_sg + 2), 100, {_s: n_ss for _s in range(ss_cnt)}

# Quality-of-Service table for GN traffic upload requests
traffic = {'file': {'n': 12, 'priority': 24, 'latency': 19 * 60, 'size': 536e6, 'discount_factor': 0.8},
           'image': {'n': 12, 'priority': 72, 'latency': 14.5 * 60, 'size': 512e6, 'discount_factor': 0.33},
           'video': {'n': 6, 'priority': 84, 'latency': 11.6 * 60, 'size': 1387e6, 'discount_factor': 0.24},
           'telemetry': {'n': 6, 'priority': 100, 'latency': 9.1 * 60, 'size': 256e6, 'discount_factor': 0.1}}

# Min & Max costs for the transit_callback arguments
f_max = max(traffic.values(), key=lambda _f: _f['priority'])
c_max, c_min = 0, -1 * n_g * f_max['priority'] * (f_max['discount_factor'] ** (t_min - f_max['latency']))

"""
UTILITIES
"""

# 3D (Euclidean) distance between two grid-voxels in this deployment
distance_3d = lambda _p_voxel, _q_voxel: np.linalg.norm(np.subtract([_p_voxel['x'], _p_voxel['y'], _p_voxel['z']],
                                                                    [_q_voxel['x'], _q_voxel['y'], _q_voxel['z']]))

# Angle (in radians) between two grid-voxels in this deployment
angle = lambda _p_voxel, _q_voxel: np.arcsin(abs(_p_voxel['z'] - _q_voxel['z']) / distance_3d(_p_voxel, _q_voxel))


def energy_1(_v, _t=1):
    """
    A constant 2D horizontal velocity model for UAV mobility energy consumption

    Y. Zeng, J. Xu and R. Zhang, "Energy Minimization for Wireless Communication With Rotary-Wing UAV,"
    IEEE Transactions on Wireless Communications, vol. 18, no. 4, pp. 2329-2345, April 2019.
    """
    return ((p_0 * (1 + ((3 * (_v ** 2)) / (u_tip ** 2)))) + (p_2 * (_v ** 3)) +
            (p_1 * ld_omega * ((((ld_omega ** 2) + ((_v ** 4) / (4 * (v_0 ** 4)))
                                 ) ** 0.5) - ((_v ** 2) / (2 * (v_0 ** 2)))) ** 0.5)) * _t


def energy_2(_vs, _as):
    """
    An arbitrary 2D horizontal velocity model for UAV mobility energy consumption

    H. Yan, Y. Chen and S. H. Yang, "New Energy Consumption Model for Rotary-Wing UAV Propulsion,"
    IEEE Wireless Communications Letters, vol. 10, no. 9, pp. 2009-2012, Sept. 2021.
    """
    _ke_d = 0.5 * m_u * ((_vs[-1] - _vs[0]) ** 2)
    _a_cf = lambda _v, _a: ((_a ** 2) - (((_a * _v) ** 2) / (_v ** 2))) ** 0.5
    _c_1, _c_2, _c_3, _c_4, _c_5 = p_0, 3 / (u_tip ** 2), p_1 * ld_omega, 2 * (v_0 ** 2), p_2

    _term_1 = sum([_c_5 * (__v ** 3) for __v in _vs])
    _term_2 = sum([_c_1 * (1 + (_c_2 * (__v ** 2))) for __v in _vs])

    _term_3 = sum([_c_3 * ((1 + ((_a_cf(_vs[_i], _as[_i]) ** 2) / (g ** 2))) ** 0.5) *
                   ((((1 + ((_a_cf(_vs[_i], _as[_i]) ** 2) / (g ** 2)) +
                       ((_vs[_i] ** 4) / (_c_4 ** 2))) ** 0.5) -
                     ((_vs[_i] ** 2) / _c_4)) ** 0.5)
                   for _i in range(len(_vs))])

    return _term_1 + _term_2 + _term_3 + _ke_d


def energy_3(_v, _t=1):
    """
    An arbitrary vertical transition model for UAV mobility energy consumption (segmented 2D + 'vertical transitions')

    <A simplified vertical transitions model with a fixed lift velocity for computational ease in trajectory design...>

    H. Yan, Y. Chen and S. H. Yang, "New Energy Consumption Model for Rotary-Wing UAV Propulsion,"
    IEEE Wireless Communications Letters, vol. 10, no. 9, pp. 2009-2012, Sept. 2021.
    """
    return ((p_0 * (1 + ((3 * (_v ** 2)) / (u_tip ** 2)))) +
            (p_1 * ((((1 + ((_v ** 4) / (4 * (v_0 ** 4)))) ** 0.5) - ((_v ** 2) / (2 * (v_0 ** 2)))) ** 0.5))) * _t


def cl_mtsp_channel(_c_uav, _voxel=None):
    """
    Cross-Layer mTSP MU-MIMO Cluster-UAV channel generation considering both large- and small-scale fading statistics
    """
    _h_matrix = [[] for _ in range(n_a_u)]

    if _voxel is not None:
        _c_uav['serv_voxel'] = _voxel

    for _a_u in range(n_a_u):
        for _gn in _c_uav['gns']:  # GNs served by '_c_uav'
            _a_gu = angle(_gn['voxel'], _c_uav['serv_voxel'])
            _d_gu = distance_3d(_gn['voxel'], _c_uav['serv_voxel'])
            _k_factor, _p_los = k_1 * np.exp(k_2 * _a_gu), 1 / (z_1 * np.exp(-z_2 * (_a_gu - z_1)))
            _beta = (_p_los * (beta_0 * (_d_gu ** -alpha))) + ((1 - _p_los) * (kappa * beta_0 * (_d_gu ** -alpha_)))

            _g_sigma = np.sqrt(1 / (2 * (_k_factor + 1)))
            _g_mu = np.sqrt(_k_factor / (2 * (_k_factor + 1)))

            # TO-DO: This difference in '_beta' and '_g' gen might warrant another look...
            [_h_matrix[_a_u].append(np.sqrt(_beta) * complex(np.random.normal(_g_mu, _g_sigma),
                                                             np.random.normal(_g_mu, _g_sigma))) for _ in range(n_a_g)]

    return _h_matrix


def comm_link(_gn, _c_uav):
    """
    Render the GN-UAV link in the MU-MIMO paradigm (with ZF receive beam-forming and receiver thermal noise)
    """
    _h_matrix = cl_mtsp_channel(_c_uav)
    _payload_size = _gn['traffic']['size']
    _w_vector = np.random.multivariate_normal(np.zeros(2), 0.5 * np.eye(2), size=n_a_u).view(np.complex128)

    # noinspection PyUnresolvedReferences
    _w_hat_vector = np.linalg.pinv(_h_matrix.conj().T @ h_matrix) @ h_matrix.conj().T @ _w_vector

    return _payload_size / (bw * np.log2(1 + (tx_p / ((np.linalg.norm(_w_hat_vector) ** 2) / _w_hat_vector.shape[0]))))


def lcso_initialize(_x_i, _x_f):
    """
    LCSO: Deterministic generation of the initial set of trajectories '_x_i' --> '_x_f'
    """
    _x_mid = tf.divide(tf.add(_x_i, _x_f), 2)
    _x_mid_i = tf.divide(tf.add(_x_i, _x_mid), 2)
    _x_mid_f = tf.divide(tf.add(_x_mid, _x_f), 2)
    _traj = tf.concat([_x_mid_i, _x_mid, _x_mid_f], axis=0)

    _i_s = [_ for _ in range(_traj.shape[0] + 2)]
    _x_s = np.linspace(0, len(_i_s) - 1, m_sg_post)

    _clip_value_min = [x_d, y_d, z_d]
    _clip_value_max = [x_max - x_d, y_max - y_d, z_max - z_d]

    # TO-DO: Add randomizations to this deterministic initialization...
    return tf.tile(tf.expand_dims(tf.clip_by_value(tf.constant(list(zip(
        UnivariateSpline(_i_s, tf.concat([_x_i[:, 0], _traj[:, 0], _x_f[:, 0]], axis=0), s=0)(_x_s),
        UnivariateSpline(_i_s, tf.concat([_x_i[:, 1], _traj[:, 1], _x_f[:, 1]], axis=0), s=0)(_x_s),
        UnivariateSpline(_i_s, tf.concat([_x_i[:, 2], _traj[:, 2], _x_f[:, 2]], axis=0), s=0)(_x_s)))),
        clip_value_min=_clip_value_min, clip_value_max=_clip_value_max), axis=0), multiples=[n_sw, 1, 1])


def lcso_eval_obj_(_traj_vels, _traj_accs, _traj_vert_vel, _traj_vert_time):
    """
    LCSO energy model: Arbitrary horizontal 2D motion + Constant vertical transition
    """
    return (energy_3(_traj_vert_vel.numpy(), _traj_vert_time.numpy()) +
            energy_2([_ for _ in _traj_vels.numpy()], [_ for _ in _traj_accs.numpy()]))


def lcso_eval_obj(_traj_wps, _traj_vels, _eval_assign, _eval_obj):
    """
    LCSO objective: Lagrangian design similar to the HCSO cost function in MAESTRO-X

    B. Keshavamurthy, M. A. Bliss and N. Michelusi,
    "MAESTRO-X: Distributed Orchestration of Rotary-Wing UAV-Relay Swarms,"
    in IEEE Transactions on Cognitive Communications and Networking, vol. 9, no. 3, pp. 794-810, June 2023.
    """
    _traj_vels = tf.clip_by_value(_traj_vels, clip_value_min=v_min, clip_value_max=v_max)

    _traj_times = tf.divide(tf.norm(tf.roll(_traj_wps, shift=-1, axis=0)[:-1, :2] - _traj_wps[:-1, :2], axis=1),
                            tf.where(tf.equal(_traj_vels[:-1], 0), tf.ones_like(_traj_vels[:-1]), _traj_vels[:-1]))

    _traj_accs = tf.clip_by_value(tf.divide(tf.roll(_traj_vels, shift=-1)[:-1] - _traj_vels[:-1],
                                            tf.where(tf.equal(_traj_times[:-1], 0), tf.ones_like(_traj_times[:-1]),
                                                     _traj_times[:-1])), clip_value_min=a_min, clip_value_max=a_max)

    # Enforcing acceleration constraints...
    for _i in range(_traj_accs.numpy().shape[0]):
        assign(_traj_vels[_i], (_traj_vels[_i - 1] if _i > 0 else 0) +
               (_traj_accs[_i] * _traj_times[_i]), validate_shape=True, use_locking=True)

    _traj_times = tf.divide(tf.norm(tf.roll(_traj_wps, shift=-1, axis=0)[:-1, :2] - _traj_wps[:-1, :2], axis=1),
                            tf.where(tf.equal(_traj_vels[:-1], 0), tf.ones_like(_traj_vels[:-1]), _traj_vels[:-1]))

    _traj_time = tf.reduce_sum(_traj_times).numpy()

    # TO-DO: Using mean(_traj_vels) & _traj_time for vert might need another look...
    # NOTE: Although the modeling allows arbitrary vertical transitions between highly-interpolated horizontal
    #       segments, for computational ease, we use a single transition at the end with mean(_traj_vels) & _traj_time.
    _traj_nrg = lcso_eval_obj_(_traj_vels, _traj_accs, tf.mean(_traj_vels), _traj_time)

    if _eval_assign and _eval_obj is not None:
        assign(_eval_obj, ((1.0 - (nu * pwr_avg)) * _traj_time) +
               (nu * _traj_nrg), validate_shape=True, use_locking=True)

    return _traj_nrg, _traj_time


def lcso_tournament(_traj_wps_ss, _wp_vels_ss, _traj_vels_ss, _vel_vels_ss, _s_ss, _evals, _w_save, _w_idxs):
    """
    LCSO: Organizing a tournament among the members of a sub-swarm for competitive optimization (regular- & post-season)
    """
    _t_ss = tf.random.shuffle([_ for _ in range(_traj_wps_ss.shape[0])])

    for _i_ss in range(0, _traj_wps_ss.shape[0], 3):

        with ThreadPoolExecutor(max_workers=n_w) as _exec:
            [_exec.submit(lcso_eval_obj, _traj_wps_ss[_t_ss[_i_ss + _j_ss]],
                          _traj_vels_ss[_t_ss[_i_ss + _j_ss]], _w_save, _evals[
                              (_s_ss * n_ss) + _t_ss[_i_ss + _j_ss]] if (_s_ss > 0) and (_evals is not None) and
                                                                        _w_save else None) for _j_ss in range(3)]

        _podium = sorted([_t_ss[_i_ss], _t_ss[_i_ss + 1], _t_ss[_i_ss + 2]], key=lambda __t: _evals[__t])

        if (_s_ss > 0) and _w_save and (_w_idxs is not None):
            assign(_w_idxs[int(_i_ss / 3)], (_s_ss * n_ss) + _podium[0], validate_shape=True, use_locking=True)

        _r_vals = tf.random.uniform(shape=[3, ])

        _wp_vels_runner = tf.add(tf.multiply(_r_vals[0], _wp_vels_ss[_podium[1]]),
                                 tf.multiply(_r_vals[1], tf.subtract(_traj_wps_ss[_podium[0]],
                                                                     _traj_wps_ss[_podium[1]])))

        _traj_wps_runner = tf.add(_traj_wps_ss[_podium[1]], _wp_vels_runner)

        assign(_traj_wps_ss[_podium[1]], _traj_wps_runner, validate_shape=True, use_locking=True)
        assign(_wp_vels_ss[_podium[1]], _wp_vels_runner, validate_shape=True, use_locking=True)

        _vel_vels_runner = tf.add(tf.multiply(_r_vals[0], _vel_vels_ss[_podium[1]]),
                                  tf.multiply(_r_vals[1], tf.subtract(_traj_vels_ss[_podium[0]],
                                                                      _traj_vels_ss[_podium[1]])))

        _traj_vels_runner = tf.add(_traj_vels_ss[_podium[1]], _vel_vels_runner)

        assign(_traj_vels_ss[_podium[1]], _traj_vels_runner, validate_shape=True, use_locking=True)
        assign(_vel_vels_ss[_podium[1]], _vel_vels_runner, validate_shape=True, use_locking=True)

        _wp_vels_loser = tf.add_n([tf.multiply(_r_vals[0], _wp_vels_ss[_podium[2]]),
                                   tf.multiply(_r_vals[1], tf.subtract(_traj_wps_ss[_podium[0]],
                                                                       _traj_wps_ss[_podium[2]])),
                                   tf.multiply(_r_vals[2], tf.subtract(_traj_wps_ss[_podium[1]],
                                                                       _traj_wps_ss[_podium[2]]))])

        _traj_wps_loser = tf.add(_traj_wps_ss[_podium[2]], _wp_vels_loser)

        assign(_traj_wps_ss[_podium[2]], _traj_wps_loser, validate_shape=True, use_locking=True)
        assign(_wp_vels_ss[_podium[2]], _wp_vels_loser, validate_shape=True, use_locking=True)

        _vel_vels_loser = tf.add_n([tf.multiply(_r_vals[0], _vel_vels_ss[_podium[2]]),
                                    tf.multiply(_r_vals[1], tf.subtract(_traj_vels_ss[_podium[0]],
                                                                        _traj_vels_ss[_podium[2]])),
                                    tf.multiply(_r_vals[2], tf.subtract(_traj_vels_ss[_podium[1]],
                                                                        _traj_vels_ss[_podium[2]]))])

        _traj_vels_loser = tf.add(_traj_vels_ss[_podium[2]], _vel_vels_loser)

        assign(_traj_vels_ss[_podium[2]], _traj_vels_loser, validate_shape=True, use_locking=True)
        assign(_vel_vels_ss[_podium[2]], _vel_vels_loser, validate_shape=True, use_locking=True)


def lcso_execute(_traj_wps, _wp_vels, _traj_vels, _vel_vels):
    """
    LCSO: Core execution routine within LCSO to trigger tournaments among sub-swarms in the swarm
    """
    _start_idx, _end_idx = 0, n_sw_div[0]

    _win_idxs = tf.Variable(tf.zeros(shape=[ss_cnt, int(n_ss / 3)]))  # Winner indices
    _evals = tf.Variable(np.inf * tf.ones(shape=[n_sw,]))  # Force an inf upper bound for min evals later...

    for _eval_cnt in range(eval_cnt_max):
        with ThreadPoolExecutor(max_workers=n_w) as _exec:

            for _s_ss, _n_ss in n_sw_div.items():
                _exec.submit(lcso_tournament, _traj_wps[_start_idx:_end_idx],
                             _wp_vels[_start_idx:_end_idx], _traj_vels[_start_idx:_end_idx],
                             _vel_vels[_start_idx:_end_idx], _s_ss, _evals, True, _win_idxs[_s_ss])  # Regular season...

                _start_idx += _end_idx
                _end_idx += _n_ss

        # Randomly choosing winners within each sub-swarm...
        _r_idxs = (np.arange(start=0, stop=n_sw, step=n_ss) +
                   np.array([np.random.choice(_win_idxs[__s_ss].numpy()) for __s_ss in n_sw_div.keys()]))

        # Post-season...
        lcso_tournament(tf.gather(_traj_wps, _r_idxs, axis=0), tf.gather(_wp_vels, _r_idxs, axis=0), tf.gather(
            _traj_vels, _r_idxs, axis=0), tf.gather(_vel_vels, _r_idxs, axis=0), -1, None, False, None)

    with ThreadPoolExecutor(max_workers=n_w) as _exec:
        [_exec.submit(lcso_eval_obj, _traj_wps[__i_sw], _traj_vels[__i_sw],
                      True, _evals[__i_sw]) for __i_sw in range(n_sw)]  # Final computation of lcso_eval_objs...

    _i_sw = tf.argmin(_evals)  # Best index
    return _traj_wps[_i_sw], _traj_vels[_i_sw]  # Best 3D UAV way-points & UAV velocities


def lcso_design(_x_i, _x_f):
    """
    LCSO: Design 3D energy-conscious UAV trajectories '_x_i' --> '_x_f'

    B. Borowska, “Learning Competitive Swarm Optimization,” Entropy, vol. 24, no. 2, 2022.
    """
    __traj_wps = lcso_initialize(_x_i, _x_f)  # Way-point particles
    __wp_vels = tf.Variable(tf.ones(shape=[n_sw, m_sg_post, 3]))  # Way-point particle velocities

    _vel_range = np.linspace(v_min, v_max, v_num)

    __traj_vels = tf.Variable(np.random.choice(_vel_range, size=[n_sw, m_sg_post]))  # Velocity particles
    __vel_vels = tf.Variable(tf.ones(shape=[n_sw, m_sg_post]))  # Velocity particle velocities

    _traj_wps, _traj_vels = lcso_execute(__traj_wps, __wp_vels, __traj_vels, __vel_vels)

    return lcso_eval_obj(_traj_wps, _traj_vels, False, None)


def mtsp_cost_model(_time_costs, _c_uavs, _curr_time, _from_idx, _to_idx):
    """
    mTSP cost model: Given the energy & time cost matrices and the current time, compute the cost for serving '_to_idx'
    """
    _cost = 0

    if _to_idx != 0:
        _gns = _c_uavs[_to_idx - 1]['gns']

        _cost = 0
        for _gn in _gns:
            _gn_id, _gn_traffic = _gn['id'], _gn['traffic_params']  # ID & Traffic params for '_gn'
            _f_pr, _f_lt, _f_df = _gn_traffic['priority'], _gn_traffic['latency'], _gn_traffic['discount_factor']

            _trans_time = _time_costs[_from_idx][_to_idx] - c_uavs[_to_idx - 1]['serv_time']
            _serv_time = _curr_time + _trans_time + _c_uavs[_to_idx - 1]['serv_times'][_gn_id]

            _cost -= _f_pr * (_f_df ** (_serv_time - _f_lt))

    return _cost


def mtsp_data_model(_nrg_costs, _time_costs, _cost_callback):
    """
    mTSP data model: Number of vehicles, Number of cities & depot, Energy & Time cost matrices, and Cost callback
    """
    return {'depot_idx': 0, 'num_uavs': n_u, 'num_clusters': n_c + 1, 'min_nrg': e_min,
            'max_nrg': e_max, 'nrg_costs': _nrg_costs, 'min_time': t_min, 'max_time': t_max,
            'time_costs': _time_costs, 'min_cost': c_min, 'max_cost': c_max, 'cost_callback': _cost_callback}


def mtsp_solve(_nrg_costs, _time_costs, _c_uavs, _cost_callback):
    """
    mTSP solver: Google OR-Tools API

    T. Bektas, “The Multiple Traveling Salesman Problem: An Overview of Formulations and Solution Procedures,”
    Omega, vol. 34, no. 3, pp. 209–219, 2006.

    https://developers.google.com/optimization/routing/tsp
    https://developers.google.com/optimization/routing/vrp
    https://developers.google.com/optimization/routing/vrptw
    """
    _data_model = mtsp_data_model(_nrg_costs, _time_costs, _cost_callback)

    _route_manager = pywrapcp.RoutingIndexManager(len(_data_model['nrg_costs']),
                                                  _data_model['num_uavs'], _data_model['depot_idx'])

    _route_model = pywrapcp.RoutingModel(_route_manager)

    def nrg_callback(_from_idx, _to_idx):
        """
        mTSP energy model '_from_idx' --> '_to_idx': Transition energy (LCSO) + Service energy (hover) at '_to_idx' city
        """
        return _data_model['nrg_costs'][_route_manager.IndexToNode(_from_idx)][_route_manager.IndexToNode(_to_idx)]

    _route_model.AddDimension(nrg_callback, _data_model['min_nrg'], _data_model['max_nrg'], False, 'nrg')
    _nrg = _route_model.GetDimensionOrDie('nrg')

    def time_callback(_from_idx, _to_idx):
        """
        mTSP temporal model '_from_idx' --> '_to_idx': Travel time (LCSO) + Service time (hover) at '_to_idx' city
        """
        return _data_model['time_costs'][_route_manager.IndexToNode(_from_idx)][_route_manager.IndexToNode(_to_idx)]

    _route_model.AddDimension(time_callback, _data_model['min_time'], _data_model['max_time'], False, 'time')
    _time = _route_model.GetDimensionOrDie('time')

    def transit_callback(_from_idx, _to_idx):
        """
        mTSP cost (transit optimization variable) model: Arc cost '_from_idx' --> '_to_idx'
        """
        # time_dim = route_model.GetDimensionOrDie('time')
        return _data_model['cost_callback'](_time_costs, _c_uavs, _time.CumulVar(_from_idx), _from_idx, _to_idx)

    _transit_callback_idx = _route_model.RegisterTransitCallback(transit_callback)
    _route_model.SetArcCostEvaluatorOfAllVehicles(_transit_callback_idx)

    _route_model.AddDimension(_transit_callback_idx, _data_model['min_cost'], _data_model['max_cost'], False, 'cost')
    _cost = _route_model.GetDimensionOrDie('cost')

    for _uav_idx in range(_data_model['num_uavs']):
        _route_model.AddVariableMinimizedByFinalizer(_cost.CumulVar(_route_model.Start(_uav_idx)))
        _route_model.AddVariableMinimizedByFinalizer(_cost.CumulVar(_route_model.End(_uav_idx)))

    _search_params = pywrapcp.DefaultRoutingSearchParameters()
    _search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    _solution = _route_model.SolveWithParameters(_search_params)

    _cumul_nrgs, _cumul_times, _cumul_rewards = {}, {}, {}

    if _solution:
        # nrg_dim = route_model.GetDimensionOrDie('nrg')
        # time_dim = route_model.GetDimensionOrDie('time')
        # cost_dim = route_model.GetDimensionOrDie('cost')

        for _uav_idx in range(_data_model['num_uavs']):
            _idx = _route_model.Start(_uav_idx)

            while not _route_model.IsEnd(_idx):
                _idx = _solution.Value(_route_model.NextVar(_idx))

            _cumul_nrgs[_uav_idx] = _nrg.CumulVar(_idx)
            _cumul_times[_uav_idx] = _time.CumulVar(_idx)
            _cumul_rewards[_uav_idx] += abs(_solution.Min(_cost.CumulVar(_idx)))

    else:
        print('[ERROR] CrossLayerMTSPEvaluations mtsp_solve: Obtaining a feasible solution for the provided '
              'mTSP formulation unsuccessful! Please try again with different num_uavs or num_clusters or cost_model!')

    return _cumul_nrgs, _cumul_times, _cumul_rewards


"""
CORE OPERATIONS
"""

# Simulation begins...
print('[INFO] CrossLayerMTSPEvaluations core_operations: Setting up the simulation for our cross-layer mTSP solution!')

# LCSO Energy boundary conditions
e_min, e_max = energy_1(v_p_min, t_min), energy_1(v_max, t_max)

# LCSO Lagrangian multiplier
nu = 0.99 / pwr_avg  # Lagrangian definition similar to HCSO in MAESTRO-X

# Heights of the GNs
h_g = z_d / 2 if h_g is None else h_g  # Heights of the UAVs optimized via LCSO...

# Assertions for model validations...
assert h_g != 0 and 0 < h_g < z_max, 'Unsupported or Impractical GN height values!'
assert h_g % (z_d / 2) == 0, 'GN height values do not adhere to the current grid tessellation!'
assert x_max % x_d == 0 and y_max % y_d == 0 and z_max % z_d == 0, 'Potential error in given grid tessellation!'
assert int(energy_1(v_min)) == 1371 and int(energy_1(v_p_min)) == 937, 'Potential error in energy_1 computation!'
assert sum([_f['n'] for _f in traffic.values()]) == n_g, 'Traffic QoS does not match the script simulation setup!'
assert n_u < n_c, 'The number of UAVs should be smaller than the number of GN clusters for better perf. comparisons!'
assert ss_cnt % 3 == 0 and n_sw % ss_cnt == 0 and sum(n_sw_div.values()) == n_sw, 'Incorrect swarm configs for LCSO!'
assert sum([_sc == n_ss and _sc % 3 == 0 for _sc in n_sw_div.values()]) == ss_cnt, 'Incorrect sub-swarm divs for LCSO!'

''' Grid Tessellation '''

n_x, n_y, n_z = int(x_max / x_d), int(y_max / y_d), int(z_max / z_d)

voxels = [{'id': _i + _j + _k, 'x': ((2 * _i) + 1) * (x_d / 2), 'y': ((2 * _j) + 1) * (y_d / 2),
           'z': ((2 * _k) + 1) * (z_d / 2)} for _k in range(n_z) for _j in range(n_y) for _i in range(n_x)]

gn_voxels_ = [{'id': _i + _j, 'x': ((2 * _i) + 1) * (x_d / 2),
               'y': ((2 * _j) + 1) * (y_d / 2), 'z': h_g} for _j in range(n_y) for _i in range(n_x)]

''' GNs Randomized Deployment '''

gn_voxels = [gn_voxels_[_i] for _i in np.random.choice(n_x * n_y, size=n_g)]

prev_n, gns_ = 0, []
for f_type, f_params in traffic.items():
    gns_.append([{'id': _i, 'voxel': gn_voxels[_i], 'traffic_type': f_type, 'traffic_params': f_params,
                  'prev_cluster': -2, 'curr_cluster': -1} for _i in range(prev_n, prev_n + f_params['n'])])
    prev_n += f_params['n']

gns = np.array(gns_).flatten()

''' K-Means Clustering for the GNs '''

# Forgy initialization (ID-0 is the depot node)
clusters = [{'id': _i + 1, 'centroid': [gns[_j]['voxel']['x'], gns[_j]['voxel']['y'], gns[_j]['voxel']['z']],
             'obs_s': [gns[_j]]} for _i, _j in enumerate(np.random.choice(n_g, size=n_c))]

# Cluster convergence check routine
cluster_converge = lambda _n_c, _obs_s: sum([_obs['prev_cluster'] == _obs['curr_cluster'] for _obs in _obs_s]) == _n_c

obs_s = gns

# Until cluster assignments change...
while not cluster_converge(n_c, obs_s):
    [_cluster['obs_s'].clear() for _cluster in clusters]

    # E-step (Assign)
    for obs in obs_s:
        obs['prev_cluster'] = obs['curr_cluster']
        obs_coord = [obs['voxel']['x'], obs['voxel']['y'], obs['voxel']['z']]

        obs['curr_cluster'] = min([_ for _ in range(n_c)],
                                  key=lambda _id: np.linalg.norm(np.subtract(obs_coord,
                                                                             clusters[_id]['centroid'])) ** 2)

        cluster_id = obs['curr_cluster']
        clusters[cluster_id]['obs_s'].append(obs)

    # M-step (Re-calculate)
    for cluster in clusters:
        if len(cluster['obs_s']) != 0:
            cluster['centroid'] = [_ for _ in np.mean(np.array([
                [_obs['voxel']['x'], _obs['voxel']['y'], _obs['voxel']['z']] for _obs in cluster['obs_s']]), axis=0)]

''' Cross-Layer mTSP Deployment '''

# UAV instances (cluster-specific)
# NOTE: These are not the actual vehicles used in the OR-Tools mTSP API...
c_uavs = [{'id': _cluster['id'], 'cumul_reward': 0, 'serv_voxel': voxels[0],
           'serv_voxels': [], 'bb_voxels': [], 'serv_nrg': 0, 'trans_nrg': 0,
           'start_time': 0, 'end_time': 0, 'serv_time': 0, 'trans_time': 0, 'serv_times': {},
           'gns': _cluster['obs_s'], 'start_voxel': voxels[0], 'end_voxel': voxels[0]} for _cluster in clusters]

for c_uav in c_uavs:
    search_data = []  # Collection for fine-search data
    voxels_z = [_ for _ in np.arange(start=z_d, stop=z_max, step=z_d)]

    voxels_x_ = [_gn['voxel']['x'] for _gn in c_uav['gns']]
    voxels_y_ = [_gn['voxel']['y'] for _gn in c_uav['gns']]

    voxels_x = [_ for _ in np.arange(start=min(voxels_x_), stop=max(voxels_x_) + x_d, step=x_d)]
    voxels_y = [_ for _ in np.arange(start=min(voxels_y_), stop=max(voxels_y_) + y_d, step=y_d)]

    bb_voxels = [{'id': int((_x / x_d) - 0.5) + int((_y / y_d) - 0.5) + int((_z / z_d) - 0.5),
                  'x': _x, 'y': _y, 'z': _z} for _z in voxels_z for _y in voxels_y for _x in voxels_x]

    c_uav['bb_voxels'] = bb_voxels  # Bounding-Box voxels (coarse-search candidate voxels)

    for bb_voxel in c_uav['bb_voxels']:
        rewards, serv_times = [], {}
        h_matrix = cl_mtsp_channel(c_uav, bb_voxel)

        for gn in c_uav['gns']:  # GNs served by 'c_uav'
            gn_id, gn_traffic = gn['id'], gn['traffic_params']  # ID & Traffic params for 'gn'
            f_pr, f_lt, f_df = gn_traffic['priority'], gn_traffic['latency'], gn_traffic['discount_factor']

            serv_time = comm_link(gn, c_uav)
            serv_times[gn_id].append(serv_time)  # For mTSP
            rewards.append(f_pr * (f_df ** (serv_time - f_lt)))

        search_data.append((bb_voxel, serv_times, max(serv_times.values()), sum(rewards)))

    best_item = max(search_data, key=lambda _sd: _sd[3])

    c_uav['serv_time'] = best_item[2]  # Service time at 'serv_voxel'
    c_uav['serv_voxel'] = best_item[0]  # Service voxel by fine-search
    c_uav['serv_times'] = best_item[1]  # Service times for each GN at 'serv_voxel'
    c_uav['cumul_reward'] = best_item[3]  # Temporary reward (without transition time)
    c_uav['serv_nrg'] = c_uav['serv_time'] * energy_1(0, c_uav['serv_time'])  # Hover at position

# Upon clustering & positioning, add the depot node (ID-0)
c_uavs.append({'id': 0, 'cumul_reward': 0, 'serv_voxel': voxels[0],
               'serv_voxels': [], 'bb_voxels': [], 'serv_nrg': 0, 'trans_nrg': 0,
               'start_time': 0, 'end_time': 0, 'serv_time': 0, 'trans_time': 0, 'serv_times': {},
               'gns': [], 'start_voxel': voxels[0], 'end_voxel': voxels[0]})  # For OR-Tools mTSP and DTO conformity...

nrg_costs = [[0 for _ in range(n_c) + 1] for _ in range(n_c) + 1]
time_costs = [[0 for _ in range(n_c) + 1] for _ in range(n_c) + 1]

for c_uav in c_uavs:
    for c_uav_ in c_uavs:
        nrg_cost, time_cost = 0, 0
        c_id, c_id_ = c_uav['id'], c_uav_['id']

        if c_id != c_id_:
            nrg_cost, time_cost = lcso_design(c_uav['serv_voxel'], c_uav_['serv_voxel'])  # LCSO trajectory design

        nrg_costs[c_id][c_id_] = int(nrg_cost) + c_uav_['serv_nrg']
        time_costs[c_id][c_id_] = int(time_cost) + c_uav_['serv_time']


'''
COLLISION AVOIDANCE:

In ACCUSTOM, enforcing collision avoidance in an offline centralized setting is nearly impossible due to the 
scheduling/association that is to-be-determined by mTSP. So, we assume that the UAVs are equipped with LIDARs and 
other sensing mechanisms (along with UAV-UAV control communication) to avoid collisions with each other (and obstacles).

So, if a UAV nears a collision during its LCSO optimal trajectory, it moves to the nearest 'collision-free' voxel.
'''

# OR-Tools mTSP solution
cumul_nrgs, cumul_times, cumul_rewards = mtsp_solve(nrg_costs, time_costs, c_uavs, mtsp_cost_model)

# Report metrics
print('[INFO] CrossLayerMTSPEvaluations core_operations: Avg. UAV Power Consumption = {} W | Fleet Reward = {}!'.format(
    np.mean(np.array([cumul_nrgs[_uav] / cumul_times[_uav] for _uav in range(n_u)])), sum(cumul_rewards.values())))

# Simulation ends...
print('[INFO] CrossLayerMTSPEvaluations core_operations: Finished the simulation for our cross-layer mTSP solution!')
