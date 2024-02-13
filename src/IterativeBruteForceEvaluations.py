"""
The Iterative Brute Force (IBF) algorithm to optimize UAV positioning in a MU-MIMO prioritized data harvesting
application. The IBF algorithm determines the optimal receive antenna positions in a distributed MIMO setup to serve
each Ground Node (GN) in a UAV's designated cluster; then, a 3D (Euclidean) distance minimization heuristic determines
the cluster-wide optimal serving position of the UAV. This mechanism is adapted from the implementation in Hanna et al.

REFERENCE PAPER:
    S. Hanna, H. Yan and D. Cabric,
    "Distributed UAV Placement Optimization for Cooperative Line-of-sight MIMO Communications,"
    ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).

DEPLOYMENT MECHANISM:
    a. One UAV per GN cluster at the optimal position determined by the IBF algorithm;
    b. Horizontal transitions (takeoff-pad to service to landing-pad) at constant (max) horizontal velocity;
    c. Vertical transitions (ground-level to fixed-height to ground-level) at constant (max) vertical velocity;

REPORTED METRICS:
    a. Total Cumulative Fleet-wide Reward (vs configurable Number of UAVs);
    b. Total Cumulative Fleet-wide Reward (vs configurable Number of Users/GNs);
    c. Total Cumulative Fleet-wide Reward and (vs) Average Per-UAV Energy Consumption.

Author: Bharath Keshavamurthy <bkeshav1@asu.edu>
Organization: School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.
Copyright (c) 2023. All Rights Reserved.
"""

import warnings

# Ignore DeprecationWarnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import plotly
import numpy as np
from scipy import constants
import plotly.graph_objs as go

"""
SCRIPT SETUP
"""

# NumPy seed
np.random.seed(1337)

# Plotly API access credentials
plotly.tools.set_credentials_file(username='total.academe', api_key='XQAdsDUeESbbgI0Pyw3E')

"""
CONFIGURATIONS
"""

# Converters
deg2rad, rad2deg = lambda _x: _x * (np.pi / 180), lambda _x: _x * (180 / np.pi)
db_watts, dbm_watts = lambda _x: 10 ** (_x / 10), lambda _x: 10 ** ((_x - 30) / 10)
watts_db, watts_dbm = lambda _x: 10 * np.log10(_x), lambda _x: 10 * np.log10(_x) + 30

# Simulation setup (MKS/SI units)
# TO-DO Configuration | Core analysis variables: Number of UAVs and Number of GNs
ex_vec, ey_vec, ez_vec = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
pi, ld, it_max, a_bf, b_bf, df_bf = np.pi, constants.speed_of_light / 1e9, 1000, 0.99, 0.3, 0.99
t_max, x_max, y_max, z_max, x_d, y_d, z_d, h_g, h_u = 3000, 3000, 3000, 150, 10, 10, 10, None, None
temp, k_1, k_2, z_1, z_2, alpha, alpha_, kappa, bw = 300, 1, np.log(100) / 90, 9.61, 0.16, 2, 2.8, 0.2, 5e6
g, n_u, n_g, n_c, n_a_u, n_a_g, wgt_uav, v_max, v_h_max, v_v_max = constants.g, 6, 36, 6, 16, 4, 80, 50, 50, 50
r_tw, delta, rho, rtr_rad, inc_corr, fp_area, n_bld, bld_len, rpm = 1, 0.012, 1.225, 0.4, 0.1, 0.0302, 8, 0.0157, 5730
v_min, v_stp, v_p_min, tx_p, beta_0, w_var = 0, 0.1, 20.1, dbm_watts(23), db_watts(20), constants.Boltzmann * temp * bw
# r_tw, delta, rho, rtr_rad, inc_corr, fp_area, n_bld, bld_len, rpm = 1, 0.012, 1.225, 0.4, 0.1, 0.0151, 4, 0.0157, 2865

# Quality-of-Service table for GN traffic upload requests
traffic = {'file': {'n': 12, 'priority': 24, 'latency': 19 * 60, 'size': 536e6, 'discount_factor': 0.8},
           'image': {'n': 12, 'priority': 72, 'latency': 14.5 * 60, 'size': 512e6, 'discount_factor': 0.33},
           'video': {'n': 6, 'priority': 84, 'latency': 11.6 * 60, 'size': 1387e6, 'discount_factor': 0.24},
           'telemetry': {'n': 6, 'priority': 100, 'latency': 9.1 * 60, 'size': 256e6, 'discount_factor': 0.1}}

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
    # Primary constants for a rotary-wing UAV
    _dsc_area, _f_sl, _ang_vel = pi * (rtr_rad ** 2), (n_bld * bld_len) / (pi * rtr_rad), rpm * ((2 * pi) / 60)

    # Secondary constants for a rotary-wing UAV
    _u_tip, _v_0 = _ang_vel * rtr_rad, (wgt_uav / (2 * rho * _dsc_area)) ** 0.5
    _r_fdr = fp_area / (_f_sl * _dsc_area)

    # Tertiary constants for evaluating energy consumption...
    _p_0, _kappa = (delta / 8) * rho * _f_sl * _dsc_area * (_ang_vel ** 3) * (rtr_rad ** 3), r_tw
    _p_1 = (1 + inc_corr) * ((wgt_uav ** 1.5) / ((2 * rho * _dsc_area) ** 0.5))
    _p_2 = 0.5 * _r_fdr * rho * _f_sl * _dsc_area

    return ((_p_0 * (1 + ((3 * (_v ** 2)) / (_u_tip ** 2)))) +
            (_p_1 * _kappa * ((((_kappa ** 2) + ((_v ** 4) / (4 * (_v_0 ** 4)))) ** 0.5) -
                              ((_v ** 2) / (2 * (_v_0 ** 2)))) ** 0.5) + (_p_2 * (_v ** 3))) * _t


def energy_2(_vs, _as):
    """
    An arbitrary horizontal velocity model for UAV mobility energy consumption (separated [horz. + vert.] components)

    H. Yan, Y. Chen and S. H. Yang, "New Energy Consumption Model for Rotary-Wing UAV Propulsion,"
    IEEE Wireless Communications Letters, vol. 10, no. 9, pp. 2009-2012, Sept. 2021.
    """
    # Primary constants for a rotary-wing UAV
    _dsc_area, _f_sl = pi * (rtr_rad ** 2), (n_bld * bld_len) / (pi * rtr_rad)
    _mass_uav, _ang_vel = wgt_uav / g, rpm * ((2 * pi) / 60)

    # Secondary constants for a rotary-wing UAV
    _u_tip, _v_0 = _ang_vel * rtr_rad, (wgt_uav / (2 * rho * _dsc_area)) ** 0.5
    _r_fdr = fp_area / (_f_sl * _dsc_area)

    # Tertiary constants for evaluating energy consumption...
    _p_0 = (delta / 8) * rho * _f_sl * _dsc_area * (_ang_vel ** 3) * (rtr_rad ** 3)
    _p_1 = (1 + inc_corr) * ((wgt_uav ** 1.5) / ((2 * rho * _dsc_area) ** 0.5))
    _p_2 = 0.5 * _r_fdr * rho * _f_sl * _dsc_area

    ''' Core constants in the energy equation '''

    _ke_d = 0.5 * _mass_uav * ((_vs[-1] - _vs[0]) ** 2)

    _c_0, _c_1, __c_2, _c_3, _c_4 = _p_0, 3 / (_u_tip ** 2), _p_1, 2 * (_v_0 ** 2), _p_2

    _kappa = lambda _v, _a: (1 + ((((rho * _r_fdr * _f_sl * _dsc_area * (_v ** 2)) +
                                    (2 * _mass_uav * _a)) ** 2) / (4 * (wgt_uav ** 2)))) ** 0.5

    ''' Split individual terms from the energy equation '''

    _term_0 = sum([_c_4 * (__v ** 3) for __v in _vs])

    _term_1 = sum([_c_0 * (1 + (_c_1 * (__v ** 2))) for __v in _vs])

    _term_2 = sum([__c_2 * _kappa(_vs[__i], _as[__i]) * (((((_kappa(_vs[__i], _as[__i]) ** 2) + (
            (_vs[__i] ** 4) / (_c_3 ** 2))) ** 0.5) - ((_vs[__i] ** 2) / _c_3)) ** 0.5) for __i in range(len(_vs))])

    return _term_0 + _term_1 + _term_2 + _ke_d


def energy_3(_vs, _as):
    """
    An arbitrary vertical velocity model for UAV mobility energy consumption (separated [horz. + vert.] components)

    H. Yan, Y. Chen and S. H. Yang, "New Energy Consumption Model for Rotary-Wing UAV Propulsion,"
    IEEE Wireless Communications Letters, vol. 10, no. 9, pp. 2009-2012, Sept. 2021.
    """
    # Primary constants for a rotary-wing UAV
    _dsc_area, _f_sl = pi * (rtr_rad ** 2), (n_bld * bld_len) / (pi * rtr_rad)
    _mass_uav, _ang_vel = wgt_uav / g, rpm * ((2 * pi) / 60)

    # Secondary constants for a rotary-wing UAV
    _u_tip, _v_0 = _ang_vel * rtr_rad, (wgt_uav / (2 * rho * _dsc_area)) ** 0.5
    _r_fdr = fp_area / (_f_sl * _dsc_area)

    # Tertiary constants for evaluating energy consumption...
    _p_0 = (delta / 8) * rho * _f_sl * _dsc_area * (_ang_vel ** 3) * (rtr_rad ** 3)
    _p_1 = (1 + inc_corr) * ((wgt_uav ** 1.5) / ((2 * rho * _dsc_area) ** 0.5))
    _p_2 = 0.5 * _r_fdr * rho * _f_sl * _dsc_area

    ''' Core constants in the energy equation '''

    _c_0, _c_1, __c_2, _c_3 = _p_0, 3 / (_u_tip ** 2), _p_1, 2 * (_v_0 ** 2)

    _kappa = lambda _v, _a: (1 + ((((rho * _r_fdr * _f_sl * _dsc_area * (_v ** 2)) +
                                    (2 * _mass_uav * _a)) ** 2) / (4 * (wgt_uav ** 2)))) ** 0.5

    ''' Split individual terms from the energy equation '''

    _term_0 = sum([_c_0 * (1 + (_c_1 * (__v ** 2))) for __v in _vs])

    _term_1 = sum([__c_2 * _kappa(_vs[__i], _as[__i]) * (((((_kappa(_vs[__i], _as[__i]) ** 2) + (
            (_vs[__i] ** 4) / (_c_3 ** 2))) ** 0.5) - ((_vs[__i] ** 2) / _c_3)) ** 0.5) for __i in range(len(_vs))])

    return _term_0 + _term_1


def ibf_channel_1(_p_m, _uav):
    """
    IBF distributed SU-MIMO GN-UAV channel generation considering only large-scale fading statistics (no prob. LoS-NLoS)

    S. Hanna, H. Yan and D. Cabric,
    "Distributed UAV Placement Optimization for Cooperative Line-of-sight MIMO Communications,"
    ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).
    """
    _h_matrix = [[] for _ in range(len(_uav['serv_voxels']))]

    for _a_u, _q_n in enumerate(_uav['serv_voxels']):
        for _a_g in range(n_a_g):
            _p_m_err = np.add(np.array([_p_m['x'], _p_m['y'], _p_m['z']]),
                              np.random.normal(0, max([x_d, y_d, z_d]), size=3))

            # TO-DO: This pos error for diversification might need another look...
            _p_m = {'x': _p_m_err[0], 'y': _p_m_err[1], 'z': _p_m_err[2], 'id': int(
                (_p_m_err[0] / x_d) - 0.5) + int((_p_m_err[1] / y_d) - 0.5) + int((_p_m_err[2] / z_d) - 0.5)}

            _gamma_mn = ld / (4 * pi * distance_3d(_p_m, _q_n))
            _theta_mn = ((2 * pi) / ld) * distance_3d(_p_m, _q_n)
            _h_matrix[_a_u].append(_gamma_mn * complex(np.cos(_theta_mn), -np.sin(_theta_mn)))

    return _h_matrix


def ibf_obj(_a_u, _p_m, __e_vec, _b_bf, _uav):
    """
    IBF evaluation objective to determine the minimizing e-vector for a UAV (SU-MIMO GN-UAV) in each iteration

    S. Hanna, H. Yan and D. Cabric,
    "Distributed UAV Placement Optimization for Cooperative Line-of-sight MIMO Communications,"
    ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).
    """
    _q_m = _uav['serv_voxels'][_a_u]
    _q_m_vec = np.array([_q_m['x'], _q_m['y'], _q_m['z']])

    _bf_mv_vec = np.add(_q_m_vec, _b_bf * __e_vec)
    _uav['serv_voxels'][_a_u] = {'x': _bf_mv_vec[0], 'y': _bf_mv_vec[1], 'z': _bf_mv_vec[2], 'id': int(
        (_bf_mv_vec[0] / x_d) - 0.5) + int((_bf_mv_vec[1] / y_d) - 0.5) + int((_bf_mv_vec[2] / z_d) - 0.5)}

    _objective = 0
    _h_matrix = ibf_channel_1(_p_m, _uav)

    for _a_g_l in range(n_a_g):
        for _a_g_k in range(n_a_g):

            if _a_g_l == _a_g_k:
                continue

            _h_l, _h_k = _h_matrix[:, _a_g_l], _h_matrix[:, _a_g_k]
            _objective += np.abs(sum([_h_l[__r].conj() * _h_k[__r] for __r in range(n_a_u)])) ** 2

    return _objective


def ibf_channel_2(_uav, _los):
    """
    IBF MU-MIMO Cluster-UAV channel generation considering both large- and small-scale fading statistics
    """
    _h_matrix = [[] for _ in range(n_a_u)]

    for _a_u in range(n_a_u):
        for _gn in _uav['gns']:  # GNs served by '_uav'
            _a_gu = angle(_gn['voxel'], _uav['serv_voxel'])
            _k_factor = k_1 * np.exp(k_2 * _a_gu) if _los else 0
            _d_gu = distance_3d(_gn['voxel'], _uav['serv_voxel'])

            _beta = beta_0 * (_d_gu ** -alpha) if _los else kappa * beta_0 * (_d_gu ** -alpha_)

            _g_sigma = np.sqrt(1 / (2 * (_k_factor + 1)))
            _g_mu = np.sqrt(_k_factor / (2 * (_k_factor + 1)))

            [_h_matrix[_a_u].append(np.sqrt(_beta) * complex(np.random.normal(_g_mu, _g_sigma),
                                                             np.random.normal(_g_mu, _g_sigma))) for _ in range(n_a_g)]

    return _h_matrix


def comm_link(_gn, _uav):
    """
    Render the GN-UAV link in the MU-MIMO paradigm (with ZF receive beam-forming and receiver thermal noise)
    """
    _a_gu = np.clip(rad2deg(angle(_gn['voxel'], _uav['serv_voxel'])), 0, 89.9)
    _p_los = 1 / (z_1 * np.exp(-z_2 * (_a_gu - z_1)))

    _w_vector = np.random.multivariate_normal(np.zeros(2), 0.5 * w_var * np.eye(2), size=n_a_u).view(np.complex128)
    _h_los_matrix, _h_nlos_matrix, _payload_size = _uav['los_channel'], _uav['nlos_channel'], _gn['traffic']['size']

    # noinspection PyUnresolvedReferences
    # ZF beam-forming modification to the LoS noise vector when symbols are perfectly recovered...
    _w_hat_los_vector = np.linalg.pinv(_h_los_matrix.conj().T @ _h_los_matrix) @ _h_los_matrix.conj().T @ _w_vector

    # noinspection PyUnresolvedReferences
    # ZF beam-forming modification to the NLoS noise vector when symbols are perfectly recovered...
    _w_hat_nlos_vector = np.linalg.pinv(_h_nlos_matrix.conj().T @ _h_nlos_matrix) @ _h_nlos_matrix.conj().T @ _w_vector

    _tgpt_los = bw * np.log2(1 + (tx_p / ((np.linalg.norm(_w_hat_los_vector) ** 2) / _w_hat_los_vector.shape[0])))
    _tgpt_nlos = bw * np.log2(1 + (tx_p / ((np.linalg.norm(_w_hat_nlos_vector) ** 2) / _w_hat_nlos_vector.shape[0])))

    return _payload_size / ((_p_los * _tgpt_los) + ((1 - _p_los) * _tgpt_nlos))


"""
CORE OPERATIONS
"""

# Simulation begins...
print('[INFO] IBFEvaluations core_operations: Setting up the simulation for the state-of-the-art IBF deployment!')

# Heights of the UAVs and the GNs
h_g = z_d / 2 if h_g is None else h_g
h_u = z_max - (z_d / 2) if h_u is None else h_u

# Assertions for model validations...
assert x_max % x_d == 0 and y_max % y_d == 0 and z_max % z_d == 0, 'Potential error in given grid tessellation!'
assert h_u != 0 and h_g != 0 and 0 < h_u < z_max and 0 < h_g < z_max, 'Unsupported or Impractical height values!'
assert sum([_f['n'] for _f in traffic.values()]) == n_g, 'Traffic QoS does not match the script simulation setup!'
assert int(energy_1(v_min)) == 1985 and int(energy_1(v_p_min)) == 1734, 'Potential error in energy_1 computation!'
assert n_c == n_u, 'The number of UAVs should be equal to the number of GN clusters for this static UAV deployment!'
assert h_u % (z_d / 2) == 0 and h_g % (z_d / 2) == 0, 'Height values do not adhere to the current grid tessellation!'
assert int(energy_2([v_min], [0])) == 1985 and int(energy_2([v_p_min], [0])) == 1736, 'Error in energy_2 computation!'
assert int(energy_3([v_min], [0])) == 1985 and int(energy_3([v_p_min], [0])) == 1586, 'Error in energy_3 computation!'

# Deployment model parameters
print('[INFO] IBFEvaluations core_operations: Deployment model parameters in this simulation are as follows - '
      f'Max simulation time = [{t_max}] s, Max site length = [{x_max}] m, Max site breadth = [{y_max}] m, '
      f'Max site height = [{z_max}] m, Voxel length = [{x_d}] m, Voxel breadth = [{y_d}] m, '
      f'Voxel height = [{z_d}] m, Std. UAV height = [{h_u}] m, GN height = [{h_g}] m, '
      f'Number of UAV antennas = [{n_a_u}], Number of GN antennas = [{n_a_g}], '
      f'Number of UAVs = [{n_u}], Number of GNs = [{n_g}], and '
      f'Number of clusters = [{n_c}].')

# Channel model parameters
print('[INFO] IBFEvaluations core_operations: Rotary-wing UAV mobility model simulation parameters are as follows - '
      f'Rx chain temperature = [{temp}] K, k1 = [{k_1}], k2 = [{k_2}], z1 = [{z_1}], z2 = [{z_2}], '
      f'NLoS attenuation factor = [{kappa}], GN-UAV channel bandwidth = [{bw}] Hz, '
      f'LoS pathloss exponent = [{alpha}], NLoS pathloss exponent = [{alpha_}].')

# Mobility model parameters
print('[INFO] IBFEvaluations core_operations: Rotary-wing UAV mobility model simulation parameters are as follows - '
      f'Thrust-to-Weight ratio = [{r_tw}], Profile drag coefficient = [{delta}], Air density = [{rho}] kg/m^3,'
      f'UAV weight = [{wgt_uav}] N, Max horz. vel. = [{v_h_max}] m/s, Max vert. vel. = [{v_v_max}] m/s, '
      f'Rotor radius = [{rtr_rad}] m, Incremental correction factor to induced power = [{inc_corr}],'
      f'Fuselage flat plate area = [{fp_area}] m^2, Number of blades = [{n_bld}], '
      f'Blade length = [{bld_len}] m, and Blade RPM = [{rpm}] rpm.')

''' Mobility Model Evaluations '''

e_vels = np.arange(start=v_min, stop=v_max + v_stp, step=v_stp)
e_trace = go.Scatter(x=e_vels, y=[energy_1(_e_vel) for _e_vel in e_vels], mode='lines+markers')

e_layout = dict(title='Rotary-Wing UAV 2D Mobility Power Analysis (Inertial Trajectories)',
                xaxis=dict(title='UAV Horizontal Flying Velocity in meters/second', autorange=True),
                yaxis=dict(title='UAV Mobility Power Consumption in Watts', type='log', autorange=True))

plotly.plotly.plot(dict(data=[e_trace], layout=e_layout))

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

# Forgy initialization
clusters = [{'id': _i, 'centroid': [gns[_j]['voxel']['x'], gns[_j]['voxel']['y'], gns[_j]['voxel']['z']],
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

''' IBF UAV Deployment '''

# UAV instances
uavs = [{'id': _cluster['id'], 'cumul_reward': 0, 'serv_voxel': voxels[0],
         'serv_voxels': [], 'bb_voxels': [], 'serv_nrg': 0, 'trans_nrg': 0,
         'start_time': 0, 'end_time': 0, 'serv_time': 0, 'trans_time': 0, 'serv_times': {},
         'gns': _cluster['obs_s'], 'start_voxel': voxels[0], 'end_voxel': voxels[0]} for _cluster in clusters]

for uav in uavs:
    voxels_z = [_ for _ in np.arange(start=z_d, stop=z_max, step=z_d)]

    voxels_x_ = [_gn['voxel']['x'] for _gn in uav['gns']]
    voxels_y_ = [_gn['voxel']['y'] for _gn in uav['gns']]

    voxels_x = [_ for _ in np.arange(start=min(voxels_x_), stop=max(voxels_x_) + x_d, step=x_d)]
    voxels_y = [_ for _ in np.arange(start=min(voxels_y_), stop=max(voxels_y_) + y_d, step=y_d)]

    bb_voxels = [{'id': int((_x / x_d) - 0.5) + int((_y / y_d) - 0.5) + int((_z / z_d) - 0.5),
                  'x': _x, 'y': _y, 'z': _z} for _z in voxels_z for _y in voxels_y for _x in voxels_x]

    uav['bb_voxels'] = bb_voxels  # Bounding-Box voxels
    uav['serv_voxels'] = [bb_voxels[_i] for _i in np.random.choice(len(bb_voxels), size=n_a_u)]

for uav in uavs:
    serv_voxels = []

    for gn in uav['gns']:
        p_voxel = gn['voxel']
        p_m, p_l, p_k = p_voxel, p_voxel, p_voxel

        p_vec = np.array([p_voxel['x'], p_voxel['y'], p_voxel['z']])
        p_m_vec, p_l_vec, p_k_vec = p_vec, p_vec, p_vec

        for it in range(it_max):
            uav['channel'] = np.array(ibf_channel_1(p_m, uav), dtype=np.complex128)

            if (a_bf * np.linalg.cond(uav['channel'])) < 1:
                break

            for a_u in range(n_a_u):
                q_m = uav['serv_voxels'][a_u]
                q_m_vec = np.array([q_m['x'], q_m['y'], q_m['z']])
                z_hat_vec = min([ex_vec, ey_vec, ez_vec], key=lambda _e_vec: ibf_obj(a_u, p_m, _e_vec, b_bf, uav))

                bf_mv_vec = np.add(q_m_vec, b_bf * z_hat_vec)

                uav['serv_voxels'][a_u] = {'x': bf_mv_vec[0], 'y': bf_mv_vec[1], 'z': bf_mv_vec[2], 'id': int(
                    (bf_mv_vec[0] / x_d) - 0.5) + int((bf_mv_vec[1] / y_d) - 0.5) + int((bf_mv_vec[2] / z_d) - 0.5)}

            b_bf *= df_bf  # Decay step-size...

        # 1: Distance minimization heuristic among distributed n_a_u Rx antennas
        serv_voxels.append(min(uav['bb_voxels'], key=lambda _bvx: sum([distance_3d(_bvx, _svx)
                                                                       for _svx in uav['serv_voxels']])))

    # 2: Distance minimization heuristic among the GNs in the UAV's designated cluster
    uav['serv_voxel'] = min(uav['bb_voxels'], key=lambda _bvx: sum([distance_3d(_bvx, _svx) for _svx in serv_voxels]))

''' MU-MIMO Channel Generation (Probabilistic LoS-NLoS) '''

for uav in uavs:
    uav['los_channel'] = np.array(ibf_channel_2(uav, True), dtype=np.complex128)
    uav['nlos_channel'] = np.array(ibf_channel_2(uav, False), dtype=np.complex128)

'''
COLLISION AVOIDANCE:

In ACCUSTOM, enforcing collision avoidance in an offline centralized setting is nearly impossible due to the 
scheduling/association that is to-be-determined by mTSP. So, we assume that the UAVs are equipped with LIDARs and 
other sensing mechanisms (along with UAV-UAV control communication) to avoid collisions with each other (and obstacles).

So, here in the IBF framework, to maintain consistency across comparisons, if a UAV nears a collision 
during its 'as-the-crow-flies' movement, it moves to the nearest 'collision-free' voxel.
'''

''' Service & Reward Computation '''

for uav in uavs:
    rewards, serv_times = [], []

    uav['trans_nrg'] = ((2 * energy_3(v_v_max, h_u / v_v_max)) +
                        (2 * energy_1(v_h_max, distance_3d(uav['start_voxel'], uav['serv_voxel']) / v_h_max)))
    uav['trans_time'] = 2 * ((distance_3d(uav['start_voxel'], uav['serv_voxel']) / v_h_max) + (h_u / v_v_max))

    avail_serv_time = t_max - uav['trans_time']
    assert avail_serv_time > max([_f['latency'] for _f in traffic.values()]), 'Not enough available service time!'

    for gn in uav['gns']:  # GNs served by 'uav'
        gn_traffic = gn['traffic_params']  # Traffic params for 'gn'
        f_pr, f_lt, f_df = gn_traffic['priority'], gn_traffic['latency'], gn_traffic['discount_factor']

        serv_time = comm_link(gn, uav)
        serv_times.append(serv_time if serv_time < avail_serv_time else avail_serv_time)

        if serv_time > avail_serv_time:
            print('[WARN] IBFEvaluations core_operations: GN service time exceeds the total available '
                  'mission (simulation) execution period for - UAV ID: {} | GN ID: {} | '
                  'GN Traffic Type: {}'.format(uav['id'], gn['id'], gn_traffic))
            continue

        rewards.append(f_pr * (f_df ** ((serv_time + (0.5 * uav['trans_time'])) - f_lt)))
        uav['cumul_reward'] += rewards[-1]

    uav['serv_time'] = max(serv_times)
    uav['end_time'] = uav['trans_time'] + uav['serv_time']
    uav['serv_nrg'] = uav['serv_time'] * energy_1(0, uav['serv_time'])  # Hover at cluster-position

# Report metrics
print('[INFO] IBFEvaluations core_operations: Average UAV Power Consumption = {} W | Fleet Reward = {}!'.format(
    np.mean(np.array([(_uav['trans_nrg'] + _uav['serv_nrg']) / _uav['end_time'] for _uav in uavs])),
    sum([_uav['cumul_reward'] for _uav in uavs])))

# Simulation ends...
print('[INFO] IBFEvaluations core_operations: Finished the simulation for the state-of-the-art IBF deployment!')
