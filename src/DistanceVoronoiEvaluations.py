"""
This script encapsulates the operations involved in evaluating the performance of the Iterative Euclidean Distance
based Voronoi Decomposition algorithm from Morocho-Cayamcela et al., 2021. These evaluations are conducted from the
perspective of our ACCUSTOM framework, i.e., employing the same modeling as that used in our ACCUSTOM solution approach.

REFERENCE PAPER:
    Manuel Eugenio Morocho-Cayamcela, Wansu Lim, Martin Maier,
    "An Optimal Location Strategy for Multiple Drone Base Stations in Massive MIMO,"
    ICT Express, Volume 8, Issue 2, 2022, Pages 230-234, ISSN 2405-9595, https://doi.org/10.1016/j.icte.2021.08.010.

DEPLOYMENT MECHANISM:
    a. 3D distance based Voronoi decomposition creates clusters of GNs, with a UAV serving each cluster;
    b. Horizontal transitions (takeoff-pad to service to landing-pad) at constant (max) horizontal velocity;
    c. Vertical transitions (ground-level to fixed-height to ground-level) at constant (max) vertical velocity.

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
temp, k_1, k_2, z_1, z_2, alpha, alpha_, kappa, bw = 300, 1, np.log(100) / 90, 9.61, 0.16, 2, 2.8, 0.2, 5e6
g, wgt_uav, tx_p, beta_0, w_var = constants.g, 80, dbm_watts(23), db_watts(20), constants.Boltzmann * temp * bw
r_tw, delta, rho, rtr_rad, inc_corr, fp_area, n_bld, bld_len, rpm = 1, 0.012, 1.225, 0.4, 0.1, 0.0302, 8, 0.0157, 5730
n_u, n_g, n_c, n_a_u, n_a_g, v_min, v_stp, v_max, v_p_min, v_h_max, v_v_max = 6, 36, 6, 16, 4, 0, 0.1, 50, 20.1, 50, 50
pi, eps, t_max, x_max, y_max, z_max, x_d, y_d, z_d, h_g, h_u = np.pi, 0.1, 3000, 3000, 3000, 150, 10, 10, 10, None, None
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
                                    (2 * wgt_uav * _a)) ** 2) / (4 * (wgt_uav ** 2)))) ** 0.5

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
    _dsc_area, _f_sl, _ang_vel = pi * (rtr_rad ** 2), (n_bld * bld_len) / (pi * rtr_rad), rpm * ((2 * pi) / 60)

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
                                    (2 * wgt_uav * _a)) ** 2) / (4 * (wgt_uav ** 2)))) ** 0.5

    ''' Split individual terms from the energy equation '''

    _term_0 = sum([_c_0 * (1 + (_c_1 * (__v ** 2))) for __v in _vs])

    _term_1 = sum([__c_2 * _kappa(_vs[__i], _as[__i]) * (((((_kappa(_vs[__i], _as[__i]) ** 2) + (
            (_vs[__i] ** 4) / (_c_3 ** 2))) ** 0.5) - ((_vs[__i] ** 2) / _c_3)) ** 0.5) for __i in range(len(_vs))])

    return _term_0 + _term_1


def voronoi_channel(_uav, _los):
    """
    Voronoi MU-MIMO Cluster-UAV channel generation considering both large- and small-scale fading statistics
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
print('[INFO] DistanceVoronoiEvaluations core_operations: Setting up the simulation for a dist. Voronoi deployment!')

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
print('[INFO] DistanceVoronoiEvaluations core_operations: Deployment model parameters in this simulation are - '
      f'Max simulation time = [{t_max}] s, Max site length = [{x_max}] m, Max site breadth = [{y_max}] m, '
      f'Max site height = [{z_max}] m, Voxel length = [{x_d}] m, Voxel breadth = [{y_d}] m, '
      f'Voxel height = [{z_d}] m, UAV height = [{h_u}] m, GN height = [{h_g}] m, '
      f'Number of UAV antennas = [{n_a_u}], Number of GN antennas = [{n_a_g}], '
      f'Number of UAVs = [{n_u}], Number of GNs = [{n_g}], and '
      f'Number of clusters = [{n_c}].')

# Channel model parameters
print('[INFO] DistanceVoronoiEvaluations core_operations: Rotary-wing UAV mobility model simulation parameters are - '
      f'Rx chain temperature = [{temp}] K, k1 = [{k_1}], k2 = [{k_2}], z1 = [{z_1}], z2 = [{z_2}], '
      f'NLoS attenuation factor = [{kappa}], GN-UAV channel bandwidth = [{bw}] Hz, '
      f'LoS pathloss exponent = [{alpha}], NLoS pathloss exponent = [{alpha_}].')

# Mobility model parameters
print('[INFO] DistanceVoronoiEvaluations core_operations: Rotary-wing UAV mobility model simulation parameters are - '
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

''' 3D Distance Voronoi Deployment '''

# UAV instances
uavs = [{'id': _u, 'cumul_reward': 0, 'serv_voxels': [], 'bb_voxels': [],
         'serv_voxel': voxels[np.random.choice(n_x * n_y)], 'serv_nrg': 0, 'trans_nrg': 0,
         'start_time': 0, 'end_time': 0, 'serv_time': 0, 'trans_time': 0, 'serv_times': {},
         'gns': [], 'start_voxel': voxels[0], 'end_voxel': voxels[0]} for _u in range(n_u)]

obj_fn = lambda __gn, __uav: distance_3d(__gn['voxel'], __uav['serv_voxel'])
prev_objs, curr_objs = {_g['id']: np.random.random() for _g in gns}, {_g['id']: np.random.random() for _g in gns}

while sum([abs(curr_objs[_gn] - prev_objs[_gn]) <= eps for _gn in range(n_g)]) < n_g:
    gns_vor = [_ for _ in gns]

    for uav in uavs:
        uav['gns'] = []

        for g, gn in enumerate(gns_vor):
            prev_objs[gn['id']] = curr_objs[gn['id']]

            obj = obj_fn(gn, uav)
            if obj <= min([obj_fn(gn, _uav) for _uav in uavs if uav['id'] != _uav['id']]):
                curr_objs[gn['id']] = obj
                uav['gns'].append(gn)
                del gns_vor[g]

        if len(uav['gns']) > 0:
            serv_voxel = [_ for _ in np.mean(np.array([[_g_u['voxel']['x'], _g_u['voxel']['y'],
                                                        _g_u['voxel']['z']] for _g_u in uav['gns']]), axis=0)]

            uav['serv_voxel'] = {'x': serv_voxel[0], 'y': serv_voxel[1], 'z': h_u}
            # uav['serv_voxel'] = {'x': serv_voxel[0], 'y': serv_voxel[1], 'z': serv_voxel[2]}

for uav in uavs:  # Update 'serv_voxel' ids to maintain DTO consistency...
    voxel_x, voxel_y, voxel_z = uav['serv_voxel']['x'], uav['serv_voxel']['y'], uav['serv_voxel']['z']
    uav['serv_voxel']['id'] = int((voxel_x / x_d) - 0.5) + int((voxel_y / y_d) - 0.5) + int((voxel_z / z_d) - 0.5)

''' MU-MIMO Channel Generation (Probabilistic LoS-NLoS) '''

for uav in uavs:
    uav['los_channel'] = np.array(voronoi_channel(uav, True), dtype=np.complex128)
    uav['nlos_channel'] = np.array(voronoi_channel(uav, False), dtype=np.complex128)

'''
COLLISION AVOIDANCE:

In ACCUSTOM, enforcing collision avoidance in an offline centralized setting is nearly impossible due to the 
scheduling/association that is to-be-determined by mTSP. So, we assume that the UAVs are equipped with LIDARs and 
other sensing mechanisms (along with UAV-UAV control communication) to avoid collisions with each other (and obstacles).

So, here in the dist. Voronoi framework, to maintain consistency across comparisons, if a UAV nears a collision 
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
            print('[WARN] DistanceVoronoiEvaluations core_operations: GN service time exceeds the total available '
                  'mission (simulation) execution period for - UAV ID: {} | GN ID: {} | '
                  'GN Traffic Type: {}'.format(uav['id'], gn['id'], gn_traffic))
            continue

        rewards.append(f_pr * (f_df ** ((serv_time + (0.5 * uav['trans_time'])) - f_lt)))
        uav['cumul_reward'] += rewards[-1]

    uav['serv_time'] = max(serv_times)
    uav['end_time'] = uav['trans_time'] + uav['serv_time']
    uav['serv_nrg'] = uav['serv_time'] * energy_1(0, uav['serv_time'])  # Hover at cluster-position

# Report metrics
print('[INFO] DistanceVoronoiEvaluations core_operations: Avg. UAV Pwr. Consumption = {} W | Fleet Reward = {}!'.format(
    np.mean(np.array([(_uav['trans_nrg'] + _uav['serv_nrg']) / _uav['end_time'] for _uav in uavs])),
    sum([_uav['cumul_reward'] for _uav in uavs])))

# Simulation ends...
print('[INFO] DistanceVoronoiEvaluations core_operations: Finished the simulation for a dist. Voronoi deployment!')
