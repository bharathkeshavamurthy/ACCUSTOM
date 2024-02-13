"""
This script encapsulates the operations involved in analyzing the mobility power consumption model of a rotary-wing UAV.

Author: Bharath Keshavamurthy <bkeshav1@asu.edu>
Organization: School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.
Copyright (c) 2023. All Rights Reserved.
"""

import warnings

# Ignore DeprecationWarnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import plotly
import numpy as np
import scipy.signal as sgnl
import scipy.stats as stats
from scipy import constants
import plotly.graph_objs as go

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

# Simulation setup (MKS/SI units)
pi, v_min, v_stp, v_max, v_h_min, v_h_max, v_v_min, v_v_max, v_std, sz = np.pi, 0, 0.1, 50, 0, 50, 0, 50, 2.5, 32
r_tw, delta, rho, rtr_rad, inc_corr, fp_area, n_bld, bld_len, rpm = 1, 0.012, 1.225, 0.4, 0.1, 0.0302, 8, 0.0157, 5730
g, wgt_uav, v_mul, m_sg, a_h_min, a_h_max, a_v_min, a_v_max, po = constants.g, 80, 1 / np.sqrt(2), 128, -5, 5, -5, 5, 4
# r_tw, delta, rho, rtr_rad, inc_corr, fp_area, n_bld, bld_len, rpm = 1, 0.012, 1.225, 0.4, 0.1, 0.0151, 4, 0.0157, 2865

"""
UTILITIES
"""


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


"""
CORE OPERATIONS
"""

e_vels = np.arange(start=v_min, stop=v_max + v_stp, step=v_stp)
vels_dict, horz_vels_dict, f_horz_vels_dict, f_vert_vels_dict = {}, {}, {}, {}
accs_dict, horz_accs_dict, f_horz_accs_dict, f_vert_accs_dict = {}, {}, {}, {}

for e_vel in e_vels:
    vels = stats.truncnorm((v_min - e_vel) / v_std, (v_max - e_vel) / v_std, loc=e_vel, scale=v_std)
    horz_vels = stats.truncnorm((v_h_min - e_vel) / v_std, (v_h_max - e_vel) / v_std, loc=e_vel, scale=v_std)

    vels_dict[e_vel] = [_ for _ in np.array(vels.rvs(m_sg))]
    horz_vels_dict[e_vel] = [_ for _ in np.array(horz_vels.rvs(m_sg))]

    f_horz_vels_dict[e_vel] = [_ for _ in np.clip(v_mul * np.array(vels.rvs(m_sg)), v_h_min, v_h_max)]
    f_vert_vels_dict[e_vel] = [_ for _ in np.clip(v_mul * np.array(vels.rvs(m_sg)), v_v_min, v_v_max)]

horz_accs_dict = {_k: np.clip(np.pad(np.diff(_v), (1, 0), mode='constant',
                                     constant_values=0), a_h_min, a_h_max) for _k, _v in horz_vels_dict.items()}

f_horz_accs_dict = {_k: np.clip(np.pad(np.diff(_v), (1, 0), mode='constant',
                                       constant_values=0), a_h_min, a_h_max) for _k, _v in f_horz_vels_dict.items()}

f_vert_accs_dict = {_k: np.clip(np.pad(np.diff(_v), (1, 0), mode='constant',
                                       constant_values=0), a_v_min, a_v_max) for _k, _v in f_vert_vels_dict.items()}

cnst_trace = go.Scatter(x=e_vels, y=[energy_1(_e_vel) for _e_vel in e_vels],
                        name='2D Inertial (Constant Velocity) Trajectory', mode='lines+markers')

horz_trace = go.Scatter(name='2D Non-Inertial (Horz. Accelerations) Trajectory', mode='lines+markers', x=e_vels,
                        y=[energy_2(horz_vels_dict[_e_vel], horz_accs_dict[_e_vel]) / m_sg for _e_vel in e_vels])

s_horz_trace = go.Scatter(mode='lines+markers', x=e_vels,
                          name='2D Non-Inertial (Horz. Accelerations) Trajectory (Smoothed)',
                          y=sgnl.savgol_filter([energy_2(horz_vels_dict[_e_vel],
                                                         horz_accs_dict[_e_vel]) / m_sg for _e_vel in e_vels], sz, po))

f_trace = go.Scatter(mode='lines+markers', x=e_vels,
                     y=[(energy_2(f_horz_vels_dict[_e_vel], f_horz_accs_dict[_e_vel])
                         + energy_3(f_vert_vels_dict[_e_vel], f_vert_accs_dict[_e_vel]))
                        / m_sg for _e_vel in e_vels], name='3D Non-Inertial (Horz. + Vert. Accelerations) Trajectory')

s_f_trace = go.Scatter(name='3D Non-Inertial (Horz. + Vert. Accelerations) Trajectory (Smoothed)',
                       y=sgnl.savgol_filter([(energy_2(f_horz_vels_dict[_e_vel], f_horz_accs_dict[_e_vel])
                                              + energy_3(f_vert_vels_dict[_e_vel], f_vert_accs_dict[_e_vel]))
                                             / m_sg for _e_vel in e_vels], sz, po), mode='lines+markers', x=e_vels)

e_layout = dict(title='Rotary-Wing UAV Mobility Power Analysis',
                xaxis=dict(title='UAV Flying Velocity in meters/second', autorange=True),
                yaxis=dict(title='UAV Mobility Power Consumption in Watts', type='log', autorange=True))

plotly.plotly.plot(dict(data=[cnst_trace, horz_trace, s_horz_trace, f_trace, s_f_trace], layout=e_layout))
