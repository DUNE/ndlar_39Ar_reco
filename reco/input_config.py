# input_config.py
from consts import drift_distance, v_drift
full_drift_time = int(drift_distance / v_drift * 1e3)

class ModuleConfig:
    def __init__(self, module_name):
        self.module_name = module_name

        if self.module_name == 'module0_run1':
            self.detector = 'module0_run1'
            self.data_type = 'data'
            self.detector_dict_path = 'charge_layout/module0_multi_tile_layout-2.3.16.yaml'
            self.light_det_geom_path = 'light_layout/light_module_desc-0.0.0.yaml'
            self.detprop_path = 'detector_properties/module0.yaml'
            self.use_ped_config_files = True
            self.PACMAN_clock_correction1 = [-9.597, 3.7453e-06]
            self.PACMAN_clock_correction2 = [-9.329, 9.0283e-07]
            self.PACMAN_clock_correction = True
            self.timestamp_cut = True
            self.nBatches = 400
            self.batches_limit = 400
            self.charge_light_matching_lower_PPS_window = 150000
            self.charge_light_matching_upper_PPS_window = full_drift_time + 150000
            self.charge_light_matching_unix_window = 0
            self.f90_integration_start = 88 # ticks
            self.f90_integration_size = 150
            self.light_sampling_rate = 100e6
            self.vref_dac = 185
            self.vcm_dac = 41
            self.samples = 256
            self.nchannels = 58
            self.hit_threshold_LCM = 1500
            self.hit_threshold_ACL = 1e9
        elif self.module_name == 'module0_run2':
            self.detector = 'module0_run2'
            self.data_type = 'data'
            self.detector_dict_path = 'charge_layout/module0_multi_tile_layout-2.3.16.yaml'
            self.light_det_geom_path = 'light_layout/light_module_desc-0.0.0.yaml'
            self.detprop_path = 'detector_properties/module0.yaml'
            self.use_ped_config_files = True
            self.PACMAN_clock_correction1 = [-9.597, 3.7453e-06]
            self.PACMAN_clock_correction2 = [-9.329, 9.0283e-07]
            self.PACMAN_clock_correction = True
            self.timestamp_cut = True
            self.nBatches = 400
            self.batches_limit = 400
            self.charge_light_matching_lower_PPS_window = 150000
            self.charge_light_matching_upper_PPS_window = full_drift_time + 150000
            self.charge_light_matching_unix_window = 0
            self.f90_integration_start = 88 # ticks
            self.f90_integration_size = 625
            self.light_sampling_rate = 100e6
            self.samples = 1024
            self.nchannels = 58
            self.hit_threshold_LCM = 3000
            self.hit_threshold_ACL = 3000
        # You can add more elif conditions for different module names and their configurations
        elif self.module_name == 'module1':
            self.detector = 'module1'
            self.data_type = 'data'
            self.detector_dict_path = 'charge_layout/module1_multi_tile_layout-2.3.16.yaml'
            self.light_det_geom_path = 'light_layout/light_module_desc-0.1.0.yaml'
            self.detprop_path = 'detector_properties/module0.yaml'
            self.use_disabled_channels_list = False
            self.use_ped_config_files = True
            self.PACMAN_clock_correction1 = [0., 0.]
            self.PACMAN_clock_correction2 = [0., 0.]
            self.PACMAN_clock_correction = True
            self.timestamp_cut = True
            self.nBatches = 400
            self.batches_limit = 400
            self.charge_light_matching_lower_PPS_window = 150000
            self.charge_light_matching_upper_PPS_window = full_drift_time + 150000
            self.charge_light_matching_unix_window = 0
            self.f90_integration_start = 62 # ticks
            self.f90_integration_size = 625
            self.light_sampling_rate = 62.5e6
            self.vref_dac = 182
            self.vcm_dac = 40
            self.samples = 1000
            self.nchannels = 48
            self.hit_threshold_LCM = 4000
            self.hit_threshold_ACL = 4000
            
        elif self.module_name == 'module2':
            self.detector = 'module2'
            self.data_type = 'data'
            self.detector_dict_path = 'charge_layout/module_2_multi_tile_layout-2022_11_18_04_35_CET.yaml'
            self.detprop_path = 'detector_properties/module0.yaml'
            self.light_det_geom_path = 'light_layout/light_module_desc-0.1.0.yaml'
            self.use_disabled_channels_list = False
            self.use_ped_config_files = True
            self.PACMAN_clock_correction1 = [0., 0.]
            self.PACMAN_clock_correction2 = [0., 0.]
            self.PACMAN_clock_correction = True
            self.timestamp_cut = True
            self.nBatches = 400
            self.batches_limit = 400
            self.charge_light_matching_lower_PPS_window = 150000
            self.charge_light_matching_upper_PPS_window = full_drift_time + 150000
            self.charge_light_matching_unix_window = 0
            self.f90_integration_start = 74 # ticks
            self.f90_integration_size = 625
            self.light_sampling_rate = 62.5e6
            self.vref_dac = 223
            self.vcm_dac = 68
            self.samples = 1024
            self.nchannels = 48
            self.hit_threshold_LCM = 4800
            self.hit_threshold_ACL = 1500
        elif self.module_name == 'module3':
            self.detector = 'module3'
            self.data_type = 'data'
            self.detector_dict_path = 'charge_layout/multi_tile_layout-module3.yaml'
            self.detprop_path = 'detector_properties/module0.yaml'
            self.light_det_geom_path = 'light_layout/light_module_desc-0.1.0.yaml'
            self.use_disabled_channels_list = False
            self.use_ped_config_files = True
            self.PACMAN_clock_correction1 = [0., 0.]
            self.PACMAN_clock_correction2 = [0., 0.]
            self.PACMAN_clock_correction = True
            self.timestamp_cut = True
            self.nBatches = 400
            self.batches_limit = 400
            self.ext_trig_matching_tolerance_unix = 1
            self.ext_trig_matching_tolerance_PPS = 2.0e3 # ns
            self.charge_light_matching_lower_PPS_window = 150000
            self.charge_light_matching_upper_PPS_window = full_drift_time + 150000
            self.charge_light_matching_unix_window = 0
            self.f90_integration_start = 74 # ticks
            self.f90_integration_size = 625
            self.light_sampling_rate = 62.5e6
            self.vref_dac = 235
            self.vcm_dac = 68
            self.samples = 1024
            self.nchannels = 48
            self.hit_threshold_LCM = 4500
            self.hit_threshold_ACL = 1500
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
            self.charge_light_matching_lower_PPS_window = 150000
            self.charge_light_matching_upper_PPS_window = full_drift_time + 150000
            self.charge_light_matching_unix_window = 0
            self.samples = 1024
        else:
            raise ValueError(f"Unsupported module name: {self.module_name}")
