# -*- coding: utf-8 -*-
"""
Created on Thu May 27 10:41:46 2021

@author: freeridingeo
"""

import numpy as np
import sys
sys.path.append("D:/Code/eotopia/core")
import constants

sentinel1_parameter = dict()
sentinel1_parameter["wavelength"] = 0.5547       # m
sentinel1_parameter["antenna_length"] = 12.3    # m
sentinel1_parameter["antenna_width"] = 0.821      # m
sentinel1_parameter["antenna_area"] = sentinel1_parameter["antenna_length"]\
                    * sentinel1_parameter["antenna_width"]
sentinel1_parameter["iw1_lookangle"] = 32.9
sentinel1_parameter["iw2_lookangle"] = 38.3
sentinel1_parameter["iw3_lookangle"] = 43.1
sentinel1_parameter["iw1_slantrange_resolution"] = 2.7
sentinel1_parameter["iw2_slantrange_resolution"] = 3.1
sentinel1_parameter["iw3_slantrange_resolution"] = 3.5
sentinel1_parameter["iw1_range_bandwidth"] = 56.5*10**6     # Hz
sentinel1_parameter["iw2_range_bandwidth"] = 48.2*10**6     # Hz
sentinel1_parameter["iw3_range_bandwidth"] = 42.8*10**6     # Hz
sentinel1_parameter["iw1_azimuth_resolution"] = 22.5     # m
sentinel1_parameter["iw2_azimuth_resolution"] = 22.7     # m
sentinel1_parameter["iw3_azimuth_resolution"] = 22.6     # m
sentinel1_parameter["iw1_processing_bandwidth"] = 327     # Hz
sentinel1_parameter["iw2_processing_bandwidth"] = 313     # Hz
sentinel1_parameter["iw3_processing_bandwidth"] = 314     # Hz
sentinel1_parameter["center_frequency"] = 5.405 *10**9     # Hz
sentinel1_parameter["max_rng_bandwidth"] = 100.*10**6          # Hz
sentinel1_parameter["slantrng_pixelspacing"] = 2.3   # m
sentinel1_parameter["rng_sampling_frequency"] = 64.35*10**6     # Hz
sentinel1_parameter["az_pixelspacing"] = 14.1   # m
sentinel1_parameter["az_sampling_frequency"] = 489.49     # Hz
sentinel1_parameter["burst_length"] = 2.75     # sec ~ 20km
sentinel1_parameter["pulse_width_min"] = 5.*10**-6
sentinel1_parameter["pulse_width_max"] = 1000.*10**-6
sentinel1_parameter["pulse_duration"] = 6.1996*10**-5     # s
sentinel1_parameter["prf_min"] = 1000 # Hz
sentinel1_parameter["prf_max"] = 3000 # Hz
sentinel1_parameter["prf"] = 486.49 # Hz
sentinel1_parameter["chirp_slope"] = 7.79 * 10**11
sentinel1_parameter["ground_swath_width"] = 250000    # m
sentinel1_parameter["slice_length"] = 170000    # m
sentinel1_parameter["satellite_velocity"] = 7500.    # m/s
sentinel1_parameter["satellite_height_min"] = 698000.    # m
sentinel1_parameter["satellite_height_max"] = 726000.    # m
sentinel1_parameter["satellite_height"] =\
            (sentinel1_parameter["satellite_height_min"]\
                + sentinel1_parameter["satellite_height_max"]) / 2
sentinel1_parameter["system_noise"] = 3 # dB
sentinel1_parameter["system_loss_epsilon"] = 10.**(-5./10.)  # assume 5 dB overall losses
sentinel1_parameter["half_power_bandwidth_l"] = .887 *\
                sentinel1_parameter["wavelength"]\
                    / sentinel1_parameter["antenna_length"]
sentinel1_parameter["half_power_bandwidth_w"] = .887 *\
                sentinel1_parameter["wavelength"]\
                    / sentinel1_parameter["antenna_width"]



# az_steering_angle_min = -0.9
# az_steering_angle_max = 0.9
# az_beam_width = 0.23                # Deg
# elevation_beam_width = 3.43                # Deg
# elevation_beam_steering_rng_min = -13.  # Deg
# elevation_beam_steering_rng_max = 12.3  # Deg


# radarnoise_temperature = 300.   # K # !!
# receiver_noise = k * radarnoise_temperature * max_rng_bandwidth
# receiver_noise_dB = 10.*np.log10(k * radarnoise_temperature * max_rng_bandwidth)

# range_bandwidth = max_rng_bandwidth     # Hz
# pulse_rate = 1600.          # Hz
# PRF = pulse_rate   
# peakpower_transmit = 4000.  # W # !!

# azimuth_sample_spacing = satellite_velocity/pulse_rate

# range_at_boresight = satellite_height / np.cos(offnadir_boresight) # m !!
# groundrange_at_boresight = satellite_height * np.sin(offnadir_boresight) # m !!
# range_at_nearbeam_edge = satellite_height /\
#     np.cos(offnadir_boresight - half_power_bandwidth_w/2) # !!
# groundrange_at_nearbeam_edge = satellite_height *\
#     np.sin(offnadir_boresight - half_power_bandwidth_w/2) # !!
# range_at_farbeam_edge = satellite_height /\
#     np.cos(offnadir_boresight + half_power_bandwidth_w/2)# !!
# groundrange_at_farbeam_edge = satellite_height *\
#     np.sin(offnadir_boresight + half_power_bandwidth_w/2)# !!
# range_swath = groundrange_at_farbeam_edge - groundrange_at_nearbeam_edge # m

# Delta_range = c / (2. * range_bandwidth)
# Delta_range_ng = Delta_range /\
#     np.sin(offnadir_boresight - half_power_bandwidth_w/2)
# Delta_range_fg = Delta_range /\
#     np.sin(offnadir_boresight + half_power_bandwidth_w/2)

# n_rs = int(np.round(range_swath/Delta_range))
# range_v = np.linspace(range_at_nearbeam_edge, range_at_farbeam_edge, n_rs)

# s_0 = 0.   # reference azimuth for defining calculations
# range_0 = range_at_boresight # Reference range for calculations



# offnadir_boresight = 30. * np.pi/180.   # !!
# azimuth_squint_of_boresight = 0. * np.pi/180. # !!


