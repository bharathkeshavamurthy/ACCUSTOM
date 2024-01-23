"""
The Iterative Gradient Descent (IGD) algorithm to optimize UAV positioning in a MU-MIMO prioritized data harvesting
application. The IGD algorithm determines the optimal receive antenna positions in a distributed MIMO setup to serve
each Ground Node (GN) in a UAV's designated cluster; then, a 3D (Euclidean) distance minimization heuristic determines
the cluster-wide optimal serving position of the UAV. This mechanism is adapted from the implementation in Hanna et al.

REFERENCE PAPER:
    S. Hanna, H. Yan and D. Cabric,
    "Distributed UAV Placement Optimization for Cooperative Line-of-sight MIMO Communications,"
    ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).

DEPLOYMENT MECHANISM:
    a. One UAV per GN cluster at the optimal position determined by the IGD algorithm;
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

# import plotly
import numpy as np
from scipy import constants
# import plotly.graph_objs as go

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
# TO-DO Configuration | Core analysis variables: Number of UAVs and Number of GNs
pi, ld, it_max, a_gd, b_gd, df_gd = np.pi, constants.speed_of_light / 1e9, 100, 0.99, 0.05, 0.99
t_max, x_max, y_max, z_max, x_d, y_d, z_d, h_g, h_u = 3000, 5000, 5000, 200, 5, 5, 5, None, None
temp, k_1, k_2, z_1, z_2, alpha, alpha_, kappa, bw = 300, 1, np.log(100) / 90, 9.61, 0.16, 2, 2.8, 0.2, 5e6
g, n_u, n_g, n_c, n_a_u, n_a_g, m_u, v_min, v_p_min, v_h_max, v_v_max = 9.8, 6, 36, 6, 16, 4, 3, 0, 22, 55, 55
tx_p, beta_0, w_var = dbm_watts(23), db_watts(20), constants.Boltzmann * temp * bw  # 23dBm, 20dB, Thermal noise
ld_omega, v_0, u_tip, p_0, p_1, p_2 = 1, 7.2, 200, 580.65, 790.6715, 0.0073  # p_2 = (d_0 * rho * eps * zeta) / 2

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


def igd_channel(_p_m, _uav):
    """
    IGD distributed SU-MIMO GN-UAV channel generation considering only large-scale fading statistics

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


def comm_link(_gn, _uav):
    """
    Render the GN-UAV link in the MU-MIMO paradigm (with ZF receive beam-forming and receiver thermal noise)
    """
    _h_matrix = _uav['channel']
    _payload_size = _gn['traffic']['size']
    _w_vector = np.random.multivariate_normal(np.zeros(2), 0.5 * np.eye(2), size=n_a_u).view(np.complex128)

    # noinspection PyUnresolvedReferences
    _w_hat_vector = np.linalg.pinv(_h_matrix.conj().T @ h_matrix) @ h_matrix.conj().T @ _w_vector

    return _payload_size / (bw * np.log2(1 + (tx_p / ((np.linalg.norm(_w_hat_vector) ** 2) / _w_hat_vector.shape[0]))))


"""
CORE OPERATIONS
"""

# Simulation begins...
print('[INFO] IGDEvaluations core_operations: Setting up the simulation for the state-of-the-art IGD deployment!')

# Heights of the UAVs and the GNs
h_g = z_d / 2 if h_g is None else h_g
h_u = z_max - (z_d / 2) if h_u is None else h_u

# Assertions for model validations...
assert x_max % x_d == 0 and y_max % y_d == 0 and z_max % z_d == 0, 'Potential error in given grid tessellation!'
assert int(energy_1(v_min)) == 1371 and int(energy_1(v_p_min)) == 937, 'Potential error in energy_1 computation!'
assert h_u != 0 and h_g != 0 and 0 < h_u < z_max and 0 < h_g < z_max, 'Unsupported or Impractical height values!'
assert sum([_f['n'] for _f in traffic.values()]) == n_g, 'Traffic QoS does not match the script simulation setup!'
assert h_u % (z_d / 2) == 0 and h_g % (z_d / 2) == 0, 'Height values do not adhere to the current grid tessellation!'
assert n_c == n_u, 'The number of UAVs should be equal to the number of GN clusters for this IGD static UAV deployment!'

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

''' IGD UAV Deployment '''

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
            uav['channel'] = np.array(igd_channel(p_m, uav), dtype=np.complex128)

            if (a_gd * np.linalg.cond(uav['channel'])) < 1:
                break

            for a_u in range(n_a_u):
                q_m = uav['serv_voxels'][a_u]
                q_m_vec = np.array([q_m['x'], q_m['y'], q_m['z']])

                lm_vec, km_vec = np.subtract(p_l_vec, q_m_vec), np.subtract(p_k_vec, q_m_vec)
                lm_uvec, km_uvec = lm_vec / np.linalg.norm(lm_vec), km_vec / np.linalg.norm(km_vec)

                gradient = np.array([0, 0, 0])
                dir_vec = np.subtract(lm_uvec, km_uvec)
                a_val = ((2 * pi) / ld) * (distance_3d(p_l, q_m) - distance_3d(p_k, q_m))

                for a_g_l in range(n_a_g):
                    for a_g_k in range(n_a_g):

                        if a_g_l == a_g_k:
                            continue

                        h_l, h_k = uav['channel'][:, a_g_l], uav['channel'][:, a_g_k]
                        h_m_l, h_m_k = np.delete(h_l, a_u), np.delete(h_k, a_u)

                        prod = sum([h_m_l[_r].conj() * h_m_k[_r] for _r in range(n_a_u)])
                        re, im = prod.real, prod.imag

                        gradient = np.add(gradient, (((4 * pi) / ld) * ((re * np.sin(a_val)) -
                                                                        (im * np.cos(a_val)))) * dir_vec)

                q_m_vec_ = np.add(q_m_vec, -b_gd * gradient)
                uav['serv_voxels'][a_u] = {'x': q_m_vec_[0], 'y': q_m_vec_[1], 'z': q_m_vec_[2], 'id': int(
                    (q_m_vec_[0] / x_d) - 0.5) + int((q_m_vec_[1] / y_d) - 0.5) + int((q_m_vec_[2] / z_d) - 0.5)}

            b_gd *= df_gd  # Decay step-size...

        # 1: Distance minimization heuristic among distributed n_a_u Rx antennas
        serv_voxels.append(min(uav['bb_voxels'], key=lambda _bvx: sum([distance_3d(_bvx, _svx)
                                                                       for _svx in uav['serv_voxels']])))

    # 2: Distance minimization heuristic among the GNs in the UAV's designated cluster
    uav['serv_voxel'] = min(uav['bb_voxels'], key=lambda _bvx: sum([distance_3d(_bvx, _svx) for _svx in serv_voxels]))

''' MU-MIMO Channel Generation '''

for uav in uavs:
    h_matrix = [[] for _ in range(n_a_u)]

    for a_u in range(n_a_u):
        for gn in uav['gns']:  # GNs served by 'uav'
            a_gu = angle(gn['voxel'], uav['serv_voxel'])
            d_gu = distance_3d(gn['voxel'], uav['serv_voxel'])
            k_factor, p_los = k_1 * np.exp(k_2 * a_gu), 1 / (z_1 * np.exp(-z_2 * (a_gu - z_1)))
            beta = (p_los * (beta_0 * (d_gu ** -alpha))) + ((1 - p_los) * (kappa * beta_0 * (d_gu ** -alpha_)))

            g_sigma = np.sqrt(1 / (2 * (k_factor + 1)))
            g_mu = np.sqrt(k_factor / (2 * (k_factor + 1)))

            # TO-DO: This difference in 'beta' and 'g' gen might warrant another look...
            [h_matrix[a_u].append(np.sqrt(beta) * complex(np.random.normal(g_mu, g_sigma),
                                                          np.random.normal(g_mu, g_sigma))) for _ in range(n_a_g)]

    uav['channel'] = np.array(h_matrix, dtype=np.complex128)

'''
COLLISION AVOIDANCE:

In ACCUSTOM, enforcing collision avoidance in an offline centralized setting is nearly impossible due to the 
scheduling/association that is to-be-determined by mTSP. So, we assume that the UAVs are equipped with LIDARs and 
other sensing mechanisms (along with UAV-UAV control communication) to avoid collisions with each other (and obstacles).

So, here in the IGD framework, to maintain consistency across comparisons, if a UAV nears a collision 
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
            print('[WARN] IGDEvaluations core_operations: GN service time exceeds the total available '
                  'mission (simulation) execution period for - UAV ID: {} | GN ID: {} | '
                  'GN Traffic Type: {}'.format(uav['id'], gn['id'], gn_traffic))
            continue

        rewards.append(f_pr * (f_df ** ((serv_time + (0.5 * uav['trans_time'])) - f_lt)))
        uav['cumul_reward'] += rewards[-1]

    uav['serv_time'] = max(serv_times)
    uav['end_time'] = uav['trans_time'] + uav['serv_time']
    uav['serv_nrg'] = uav['serv_time'] * energy_1(0, uav['serv_time'])  # Hover at cluster-position

# Report metrics
print('[INFO] IGDEvaluations core_operations: Average UAV Power Consumption = {} W | Fleet Reward = {}!'.format(
    np.mean(np.array([(_uav['trans_nrg'] + _uav['serv_nrg']) / _uav['end_time'] for _uav in uavs])),
    sum([_uav['cumul_reward'] for _uav in uavs])))

# Simulation ends...
print('[INFO] IGDEvaluations core_operations: Finished the simulation for the state-of-the-art IGD deployment!')
