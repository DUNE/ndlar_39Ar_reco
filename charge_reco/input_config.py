# input_config.py
from consts import drift_distance, v_drift
full_drift_time = int(drift_distance / v_drift * 1e3)

class ModuleConfig:
    def __init__(self, module_name):
        self.module_name = module_name

        if self.module_name == 'module-0':
            self.detector = 'module-0'
            self.data_type = 'data'
            self.detector_dict_path = 'charge_layout/module0_multi_tile_layout-2.3.16.yaml'
            self.light_det_geom_path = 'light_layout/light_module_desc-0.0.0.yaml'
            self.detprop_path = 'detector_properties/module0.yaml'
            self.use_disabled_channels_list = False
            self.disabled_channels_list = 'disabled_channels/module0_disabled_channels_noise_cut.npz'
            self.pedestal_file = 'pedestal/module0_datalog_2021_04_02_19_00_46_CESTevd_ped.json'
            self.config_file = 'config/module0_evd_config_21-03-31_12-36-13.json'
            self.use_ped_config_files = True
            self.PACMAN_clock_correction1 = [-9.597, 3.7453e-06]
            self.PACMAN_clock_correction2 = [-9.329, 9.0283e-07]
            self.PACMAN_clock_correction = True
            self.timestamp_cut = True
            self.match_charge_to_ext_trig = True
            self.nBatches = 400
            self.batches_limit = 400
            self.ext_trig_matching_tolerance_unix = 1
            self.ext_trig_matching_tolerance_PPS = 2e3 # ns
            self.charge_light_matching_lower_PPS_window = 150000
            self.charge_light_matching_upper_PPS_window = full_drift_time + 150000
            self.charge_light_matching_unix_window = 0
            self.ext_trig_PPS_window = 1000
            self.samples = 256
            self.nchannels = 58

        elif self.module_name == 'module-0_MC':
            self.detector = 'module-0'
            self.data_type = 'MC'
            self.detector_dict_path = 'charge_layout/module0_multi_tile_layout-2.3.16.yaml'
            self.detprop_path = 'detector_properties/module0.yaml'
            self.use_disabled_channels_list = False
            self.disabled_channels_list = 'disabled_channels/module0_disabled_channels_noise_cut.npz'
            self.pedestal_file = 'pedestal/module0_datalog_2021_04_02_19_00_46_CESTevd_ped.json'
            self.config_file = 'config/module0_evd_config_21-03-31_12-36-13.json'
            self.use_ped_config_files = False
            self.PACMAN_clock_correction1 = [-9.597, 3.7453e-06]
            self.PACMAN_clock_correction2 = [-9.329, 9.0283e-07]
            self.PACMAN_clock_correction = False
            self.timestamp_cut = False
            self.match_charge_to_ext_trig = False
            self.nBatches = 10
            self.batches_limit = 10
            self.ext_trig_matching_tolerance_unix = 1
            self.ext_trig_matching_tolerance_PPS = 1.5e3 # ns
            self.charge_light_matching_lower_PPS_window = 150000
            self.charge_light_matching_upper_PPS_window = full_drift_time + 150000
            self.charge_light_matching_unix_window = 0
            self.ext_trig_PPS_window = 1000

        # You can add more elif conditions for different module names and their configurations
        elif self.module_name == 'module-1':
            self.detector = 'module-1'
            self.data_type = 'data'
            self.detector_dict_path = 'charge_layout/module1_multi_tile_layout-2.3.16.yaml'
            self.light_det_geom_path = 'light_layout/light_module_desc-0.1.0.yaml'
            self.detprop_path = 'detector_properties/module0.yaml'
            self.use_disabled_channels_list = False
            self.disabled_channels_list = 'disabled_channels/module1_disabled_channels_noise_cut.npz'
            self.pedestal_file = 'pedestal/module1_packet_2022_02_08_01_40_31_CETevd_ped.json'
            self.config_file = 'config/module1_config_22-02-08_13-37-39.json'
            self.use_ped_config_files = True
            self.PACMAN_clock_correction1 = [0., 0.]
            self.PACMAN_clock_correction2 = [0., 0.]
            self.PACMAN_clock_correction = True
            self.timestamp_cut = True
            self.match_charge_to_ext_trig = True
            self.nBatches = 400
            self.batches_limit = 400
            self.ext_trig_matching_tolerance_unix = 1
            self.ext_trig_matching_tolerance_PPS = 2.0e3 # ns
            self.charge_light_matching_lower_PPS_window = 150000
            self.charge_light_matching_upper_PPS_window = full_drift_time + 150000
            self.charge_light_matching_unix_window = 0
            self.ext_trig_PPS_window = 1000
            self.samples = 1000
            self.nchannels = 48
            
        elif self.module_name == 'module-2':
            self.detector = 'module-2'
            self.data_type = 'data'
            self.detector_dict_path = 'charge_layout/module_2_multi_tile_layout-2022_11_18_04_35_CET.yaml'
            self.detprop_path = 'detector_properties/module0.yaml'
            self.use_disabled_channels_list = False
            self.disabled_channels_list = None
            self.pedestal_file = 'pedestal/module2_ped-evd-2022_11_18_04_35_CET.json'
            self.config_file = 'config/module2_config-evd-2022_11_18_04_35_CET.json'
            self.use_ped_config_files = True
            self.PACMAN_clock_correction1 = [0., 0.]
            self.PACMAN_clock_correction2 = [0., 0.]
            self.PACMAN_clock_correction = True
            self.timestamp_cut = True
            self.match_charge_to_ext_trig = True
            self.nBatches = 400
            self.batches_limit = 400
            self.ext_trig_matching_tolerance_unix = 1
            self.ext_trig_matching_tolerance_PPS = 2.0e3 # ns
            self.charge_light_matching_lower_PPS_window = 150000
            self.charge_light_matching_upper_PPS_window = full_drift_time + 150000
            self.charge_light_matching_unix_window = 0
            self.ext_trig_PPS_window = 1000
            self.samples = 1024
            self.nchannels = 58
            
        elif self.module_name == 'module-X':
            self.detector = 'module-X'
            self.data_type = 'data'
            self.detector_dict_path = 'charge_layout/moduleX_multi_tile_layout.yaml'
            self.detprop_path = 'detector_properties/module0.yaml'
            self.use_disabled_channels_list = False
            self.disabled_channels_list = None
            self.pedestal_file = 'pedestal/moduleX_packet-2023_10_03_11_18_CESTevd_ped_corrected.json'
            self.config_file = 'config/moduleX_evd_config_23-10-04_16-37-01_corrected.json'
            self.use_ped_config_files = True
            self.PACMAN_clock_correction1 = [0., 0.]
            self.PACMAN_clock_correction2 = [0., 0.]
            self.PACMAN_clock_correction = True
            self.timestamp_cut = True
            self.match_charge_to_ext_trig = True
            self.nBatches = 200
            self.batches_limit = 200
            self.ext_trig_matching_tolerance_unix = 1
            self.ext_trig_matching_tolerance_PPS = 2.0e3 # ns
            self.charge_light_matching_lower_PPS_window = 150000
            self.charge_light_matching_upper_PPS_window = full_drift_time + 150000
            self.charge_light_matching_unix_window = 0
            self.ext_trig_PPS_window = 1000
            self.samples = 1024
        elif self.module_name == 'SingleCube':
            self.detector = 'SingleCube'
            self.data_type = 'data'
            self.detector_dict_path = 'charge_layout/single_tile_layout-2.0.1.yaml'
            self.detprop_path = 'detector_properties/SingleCube.yaml'
            self.use_disabled_channels_list = False
            self.disabled_channels_list = None
            self.pedestal_file = 'pedestal/Bern_SC_datalog_2020_10_12_16_45_30_PDT_evd_ped.json'
            self.config_file = 'config/Bern_SC_evd_config_20-10-26_10-48-37.json'
            self.use_ped_config_files = True
            self.PACMAN_clock_correction1 = [0., 0.]
            self.PACMAN_clock_correction2 = [0., 0.]
            self.PACMAN_clock_correction = True
            self.timestamp_cut = True
            self.match_charge_to_ext_trig = False
            self.nBatches = 10
            self.batches_limit = 10
            self.ext_trig_matching_tolerance_unix = 1
            self.ext_trig_matching_tolerance_PPS = 2.0e3 # ns
            self.charge_light_matching_lower_PPS_window = 150000
            self.charge_light_matching_upper_PPS_window = full_drift_time + 150000
            self.charge_light_matching_unix_window = 0
            self.ext_trig_PPS_window = 1000
            self.samples = 1024
        else:
            raise ValueError(f"Unsupported module name: {self.module_name}")
