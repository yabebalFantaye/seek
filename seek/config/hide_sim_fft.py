# SEEK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# SEEK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with SEEK.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Dec 18, 2015

author: jakeret

Config file that specifies parameters specific for the FFT spectrometer.
Includes the other two config files to avoid duplication.

'''
from __future__ import print_function, division, absolute_import, unicode_literals

from seek.config import process_survey

for name in [name for name in dir(process_survey) if not name.startswith("__")]:
    globals()[name] = getattr(process_survey, name)

plugins = ["seek.plugins.find_nested_files",
           "seek.plugins.calibration",
            "seek.plugins.initialize",
# comment out everything below when doing calibration-only analysis
           ParallelPluginCollection([
#                                     "seek.plugins.load_preprocessed_data",
                                    "seek.plugins.load_data",
                                    "seek.plugins.pre_process_tod",
                                    "seek.plugins.process_coords",
#                                    "seek.plugins.mask_objects",
#                                    "seek.plugins.mask_artefacts",
#                                    "seek.plugins.remove_RFI",
#                                    "seek.plugins.post_process_tod",
#                                    "seek.plugins.post_process_todrfi",
                                    "seek.plugins.post_process_todrfi_sdfits",
#                                    "seek.plugins.background_removal",
#                                    "seek.plugins.restructure_tod",
                                     ],
                                     "seek.plugins.map_file_paths",
#                                     "seek.plugins.reduce_map_indicies"
                                     ),
#            ParallelPluginCollection(["seek.plugins.create_maps"],
#                                     "seek.plugins.map_indicies",
#                                     "seek.plugins.reduce_maps"
#                                 ),
#            "seek.plugins.write_maps",
            "ivy.plugin.show_stats",
            ]

# ==================================================================
# D A T A  L O A D I N G
# ==================================================================
file_prefix = "./data/"

integration_time = 1                 # no of pixel to use for integration in time (axis=1)
integration_frequency = 1            # no of pixel to use for integration in freq (axis=0)
max_frequency = 1260
min_frequency = 990

# ==================================================================
# F I L E   I N P U T
# ==================================================================
strategy_start = "2016-10-26-00:00:00"      # survey start time. Format YYYY-mm-dd-HH:MM:SS
strategy_end   = "2016-10-27-23:59:00"      # survey end time. Format YYYY-mm-dd-HH:MM:SS

spectrometer = "M9703A"
data_file_prefix = "TEST_MP_PXX_"                 # First part of fit file name
data_file_suffix = ".h5"          # Suffix of file name
file_type = "hdf5"

coord_prefix = "coord7m"                    # prefix of coord file

chunk_size = 100                              # number of files to be grouped together

# ==================================================================
# F I L E   O U T P U T
# ==================================================================
post_processing_prefix='data/hide_simcalib_1year'

# ---------------------------
# F F T   s p e c t r o m e t e r
# ---------------------------
m9703a_mode = "phase_switch"                # FFT spectrometer accusition mode

rfi_gt_mask = True                        # True if RFI mask should be written to utput
rfi_mask_frac = 0.001                      # percentage of RFI fraction to mask
rfi_map=True

spectral_kurtosis = False                    # True if spectral kurtosis masking should be used
accumulations = 146484                      # number of spectrum accumulations
accumulation_offset = 10                    # SEL_Curtosis_Range (default: P_Select 0) 

# ==================================================================
# M A P  M A K I N G
# ==================================================================
nside = 64

# ==================================================================
# R F I  C L E A N I N G
# ==================================================================
cleaner = "seek.mitigation.sum_threshold"

# ---------------------------
# F R E Q U E N C Y   M A S K I N G 
# ---------------------------
mask_freqs = ()

# ---------------------------
# S u m   t h r e s h o l d 
# ---------------------------
chi_1 = 6                                 # First threshold value 
sm_kernel_m = 40                            # Smoothing, kernel window size in axis=1
sm_kernel_n = 20                            # Smoothing, kernel window size in axis=0
sm_sigma_m = 15                             # Smoothing, kernel sigma in axis=1
sm_sigma_n = 7.5                            # Smoothing, kernel sigma in axis=0

struct_size_0 = 3                           # size of struct for dilation on freq direction [pixels]
struct_size_1 = 7                           # size of struct for dilation on time direction [pixels]

eta_i = [0.5, 0.55, 0.62, 0.75, 1]

# ==================================================================
# C A L I B R A T I O N
# ==================================================================
flux_calibration = "default"                   # calibration mode, options: default, flat, data
gain_file_default = "data/gain_template_7m_FFT_phase_switch_ADU_K.dat"
gain_file_flat = "data/gain_template_7m_FFT_null_ADU_K.dat"
calibration_chi1 = 1e13 
