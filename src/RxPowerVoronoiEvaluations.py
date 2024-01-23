"""
This script encapsulates the operations involved in evaluating the performance of the Iterative Rx Signal Power
based Voronoi Decomposition algorithm from Morocho-Cayamcela et al., 2021. These evaluations are conducted from the
perspective of our ACCUSTOM framework, i.e., employing the same modeling as that used in our ACCUSTOM solution approach.

REFERENCE PAPER:
    Manuel Eugenio Morocho-Cayamcela, Wansu Lim, Martin Maier,
    "An optimal location strategy for multiple drone base stations in massive MIMO,"
    ICT Express, Volume 8, Issue 2, 2022, Pages 230-234, ISSN 2405-9595, https://doi.org/10.1016/j.icte.2021.08.010.

DEPLOYMENT MECHANISM:
    a. Rx power based Voronoi decomposition creates clusters of GNs, with a UAV serving each cluster;
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