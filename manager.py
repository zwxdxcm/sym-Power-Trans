from types import SimpleNamespace
from enum import Enum

STORE_TRUE = None  # set alias


class ParamManager(object):
    def __init__(self, **kw):
        self._tag = "exp"
        self.p = SimpleNamespace()
        self._exp_name_list = []
        
        self._set_exp(**kw)
        
    
    def _set_special_case(self):
        # self.p.input_path = "./data/Kodak"
        # self.p.input_path = "./data/div2k/train_data"
        # self.p.not_zero_mean = STORE_TRUE
        # self.p.lr = 5e-4
        # self.p.input_path = "./data/div2k/test_data/00.png"
        
        # self.p.num_epochs = 100
        # self.p.lr = 1e-3
        # self.p.num_epochs = 5000
        # self.p.log_epoch = 1000
        pass
    
    def _set_default_parmas(self):
        self.p.method_type = "dev"

    def _set_exp(self, model="siren", enum=0):
        
        self.p.model_type = model
        self._set_default_parmas()
        # self._tag = "60.4"
        # self.p.up_folder_name = "60.31" 

        # self._set_debug_exp(enum)
        self._set_exp_900(enum)
        
        # self._set_exp_500(enum)
        # self._set_exp_700(enum)
        # self._set_exp_612(enum)
        # self._set_param_exp_norm(enum) # 不同normalization的方法
        # self._set_param_amp(enum) # 不同振幅
        # self._set_param_skew(enum) # 不同skewness_a
        
        # self.p.up_folder_name = "formal_exps"
        
        self.p.tag = f"{self._tag}_{self.p.model_type}_{self._get_exp_name(enum)}"

        # self.p.model_type = model
        self.p.lr = self._get_lr_by_model(self.p.model_type)
        print('self.p: ', self.p)
        
        # 优先级最高 暂时放弃 全部写到方法里面
        # self._set_special_case()
        self.p.lr = 1e-4 # audio
        

    def _get_exp_name(self, enum=0):
        if len(self._exp_name_list) > 0:
            return self._exp_name_list[enum]
        else:
            return enum

    def _get_lr_by_model(self, model):
        if model == "gauss" or model == "wire":
            return 5e-3
        elif model == "siren":
            return 1e-4 # 1e-4 | 5e-4
        elif model == "finer":
            return 1e-4 
        elif model == "pemlp":
            return 1e-3
        else:
            raise NotImplementedError

    def _convert_dict_args_list(self):
        args_dic = vars(self.p)
        args_list = []
        for key, val in args_dic.items():
            args_list.append(f"--{key}")
            if val is not STORE_TRUE:
                args_list.append(str(val))
        self._print_args_list(args_list)
        return args_list

    def export_args_list(self):
        return self._convert_dict_args_list()

    def export_cmd_str(self, use_cuda=[0]):
        args_list = self._convert_dict_args_list()
        script = "python main.py " + " ".join(args_list)
        script = self.add_cuda_visible_to_script(script, use_cuda)
        return script

    @staticmethod
    def add_cuda_visible_to_script(script, use_cuda=[0]):
        visible_devices: str = ",".join(map(str, use_cuda))
        return f"CUDA_VISIBLE_DEVICES={visible_devices} {script}"

    @staticmethod
    def _print_args_list(args_list):
        print("#" * 10 + "print for vscode debugger" + "#" * 10)
        for item in args_list:
            print(f'"{item}",')

    def _set_debug_exp(self, enum):
        self.p.debug = STORE_TRUE
        self.p.tag = "debug_audio"
        self.p.signal_type = "audio"
        self.p.method_type = "default"
        self.p.input_path = "/home/wxzhang/projects/coding4paper/projects/subband/data/wav_set_1/121-123859-0000.wav"

    def _set_exp_900(self, enum):
        #### 900 audio
        self._exp_name_list = [
            "siren",
            "siren-trans",
            "finer",
            "finer-trans",
            "gauss",
            "wire",
            "pemlp"
        ]
        self._tag = "90.0"
        self.p.up_folder_name = "902_audio"
        self.p.input_path = "/home/wxzhang/projects/coding4paper/projects/subband/data/121726"
        # self.p.input_path = "/home/wxzhang/projects/coding4paper/projects/subband/data/123859/121-123859-0002.flac"
        # self.p.input_path = "/home/wxzhang/projects/coding4paper/projects/subband/data/121726/121-121726-0002.flac"
        self.p.signal_type = "audio"
        self.p.method_type = "default"
        self.p.hidden_layers = 3
        
        self.p.log_epoch = 500
        self.p.num_epochs = 5000


        # tunning
        self.p.first_omega = 100
        self.p.hidden_omega = 100

        cur_exp = self._exp_name_list[enum]
        if cur_exp == "siren":
            self.p.model = "siren"
            self.p.normalization = "min_max"

        elif cur_exp == "siren-trans":
            self.p.model = "siren"
            self.p.normalization = "power"
            self.p.pn_beta = 0.05
            self.p.pn_buffer = -1  # adaptive

        elif cur_exp == "finer":
            self.p.model = "finer"
            self.p.normalization = "min_max"

        elif cur_exp == "finer-trans":
            self.p.model = "finer"
            self.p.normalization = "power"
            self.p.pn_beta = 0.05
            self.p.pn_buffer = -1  # adaptive
        
        elif cur_exp == "gauss": 
            self.p.model_type = "gauss"
        elif cur_exp == "wire": 
            self.p.model_type = "wire"
        elif cur_exp == "pemlp": 
            self.p.model_type = "pemlp"
        
        

    def _set_exp_800(self, enum):
        #### video
        self._exp_name_list = [
            "siren",
            "siren-trans",
            "finer",
            "finer-trans",
        ]
        self._tag = "80.0"
        self.p.up_folder_name = "80.03_video"
        self.p.input_path = "/home/wxzhang/projects/coding4paper/projects/subband/data/uvg/shakendry_1080"
        self.p.signal_type = "video"
        self.p.method_type = "default"
        self.p.hidden_layers = 6
        
        self.p.snap_epoch = 50
        self.p.num_epochs = 200

        cur_exp = self._exp_name_list[enum]
        if cur_exp == "siren":
            self.p.model = "siren"
            self.p.normalization = "min_max"

        elif cur_exp == "siren-trans":
            self.p.model = "siren"
            self.p.normalization = "power"
            self.p.pn_beta = 0.05
            self.p.pn_buffer = -1  # adaptive

        elif cur_exp == "finer":
            self.p.model = "finer"
            self.p.normalization = "min_max"

        elif cur_exp == "finer-trans":
            self.p.model = "finer"
            self.p.normalization = "power"
            self.p.pn_beta = 0.05
            self.p.pn_buffer = -1  # adaptive
        

    def _set_exp_500(self, enum):
        ## 正式版本的各种transform; 与60.1 比较近似
        self._tag = "50.0"
        self.p.up_folder_name = "50.0_transform" 
        self.p.input_path = "./data/div2k/test_data/"

        self._exp_name_list = [
            "gamma_0.5",
            "gamma_2",
            "scale_0.5",
            "scale_2",
            "inverse",
            "rpp",
        ]
        cur_exp = self._exp_name_list[enum]
        if cur_exp == "gamma_0.5":
            self.p.normalization = "power"
            self.p.pn_beta = 0
            self.p.pn_buffer = 0
            self.p.force_gamma = 0.5
        elif cur_exp == "gamma_2":
            self.p.normalization = "power"
            self.p.pn_beta = 0
            self.p.pn_buffer = 0
            self.p.force_gamma = 2
        elif cur_exp == "scale_0.5":
            self.p.normalization = "min_max"
            self.p.amp = 0.5
        elif cur_exp == "scale_2":
            self.p.normalization = "min_max"
            self.p.amp = 2
        elif cur_exp == "inverse":
            self.p.normalization = "min_max"
            self.p.inverse = STORE_TRUE
        elif cur_exp == "rpp":
            self.p.normalization = "min_max"
            self.p.rpp = STORE_TRUE


    # 60.1 Comparition with different normalization methods
    def _set_exp_601(self, enum):
        self._exp_name_list = [
            "min_max_01",
            "min_max_11",
            "z_score",
            "box_cox",
            "m_trans",
            "z_power",
        ]
        cur_exp = self._exp_name_list[enum]
        if cur_exp == "min_max_01":
            self.p.normalization = "min_max"
            self.p.not_zero_mean = STORE_TRUE
        elif cur_exp == "min_max_11":
            self.p.normalization = "min_max"
        elif cur_exp == "z_score":
            self.p.normalization = "instance"
        elif cur_exp == "box_cox":
            self.p.normalization = "box_cox"
        elif cur_exp == "m_trans": 
            self.p.normalization = "power"
            self.p.pn_beta = 0.05
            self.p.pn_buffer = -1  # adaptive
        elif cur_exp == "z_power": 
            self.p.normalization = "z_power"
            self.p.pn_beta = 0.05
            self.p.pn_buffer = -1  # adaptive

    # 60.2 test Kodak
    def _set_exp_602(self, enum):
        self.p.input_path = "./data/Kodak"
        self._exp_name_list = [
            "siren", # 0
            "siren-mtrans",
            "finer",
            "finer-mtrans",
            "gauss",
            "wire", 
            "pemlp", #6
            "gauss-mtrans",
            "wire-mtrans",
            "finer-z-std", # 9
        ]
        cur_exp = self._exp_name_list[enum]
        if cur_exp == "siren":
            self.p.model_type = "siren"
        elif cur_exp == "siren-mtrans":
            self.p.normalization = "power"
            self.p.pn_beta = 0.05
            self.p.pn_buffer = -1 
            self.p.model_type = "siren"
        elif cur_exp == "finer":
            self.p.model_type = "finer"
        elif cur_exp == "finer-mtrans":
            self.p.model_type = "finer"
            self.p.normalization = "power"
            self.p.pn_beta = 0.05
            self.p.pn_buffer = -1 
        elif cur_exp == "gauss": 
            self.p.model_type = "gauss"
        elif cur_exp == "wire": 
            self.p.model_type = "wire"
        elif cur_exp == "pemlp": 
            self.p.model_type = "pemlp"
        elif cur_exp == "gauss-mtrans": 
            self.p.model_type = "gauss"
            self.p.normalization = "power"
            self.p.pn_beta = 0.05
            self.p.pn_buffer = -1 
        elif cur_exp == "wire-mtrans":
            self.p.model_type = "wire"
            self.p.normalization = "power"
            self.p.pn_beta = 0.05
            self.p.pn_buffer = -1 
        elif cur_exp == "finer-z-std":
            self.p.model_type = "finer"
            self.p.normalization = "instance"

    def _set_exp_603(self, enum):
        ### test text image set
        self._set_exp_602(enum)
        self.p.input_path = "data/text/test_data"
        self.p.num_epochs = 500
        self.p.log_epoch = 100
        self.p.gamma_boundary = 1.25 # 因为是离散值 所以调低
        self._tag = "60.31"
        self.p.up_folder_name = "60.12"
    
    def _set_exp_604(self, enum):
        self._set_exp_602(enum)
        self.p.input_path = "data/div2k/test_data"
        self._tag = "60.4"
        self.p.up_folder_name = "60.4" 

    def _set_exp_605(self, enum):
        self._set_exp_602(enum)
        self.p.input_path = "data/div2k/test_data"
        self._tag = "60.5"
        self.p.up_folder_name = "60.5" 

    def _set_exp_612(self, enum):
        self._set_exp_602(enum)
        self.p.input_path = "/home/ubuntu/projects/coding4paper/data/uvg/hevc_output/ShakeNDry_3840x2160_120fps_420_8bit_HEVC_RA0002.jpg"
        self.p.input_path = "/home/ubuntu/projects/coding4paper/data/uvg/hevc_output/0001.jpg"
        self._tag = "61.2"
        self.p.up_folder_name = "61.2" 

    def _set_exp_700(self,enum):
        #### abaltion study
        self._tag = "70.0"
        self.p.up_folder_name = "70.0_ablation" 
        self.p.input_path = "./data/div2k/test_data"
        
        self.p.model_type = "siren"
        
        ### todo
        self._exp_name_list = [
            "baseline", # min_max
            "wo_basic",
            "wo_cali",
            "wo_soft",
            "wo_cali_soft",
            "sym_power",
        ]
        
        cur_exp = self._exp_name_list[enum]
        if cur_exp == "baseline":
            self.p.normalization = "min_max"
        elif cur_exp == "sym_power":
            self.p.normalization = "power"
            self.p.pn_beta = 0.05
            self.p.pn_buffer = -1  # adaptive
        elif cur_exp == "wo_basic":
            self.p.normalization = "power"
            self.p.pn_beta = 0
            self.p.pn_buffer = -1  # adaptive
            self.p.gamma_boundary = 1 # 使power无效 → 仍为minmax
        elif cur_exp == "wo_cali":
            self.p.normalization = "power"
            self.p.pn_beta = 0
            self.p.pn_buffer = -1  # adaptive
        elif cur_exp == "wo_soft":
            self.p.normalization = "power"
            self.p.pn_beta = 0.05
            self.p.pn_buffer = 0
        elif cur_exp == "wo_cali_soft":
            self.p.normalization = "power"
            self.p.pn_beta = 0
            self.p.pn_buffer = 0


            
        
    
        



    def _set_exp_143(self, enum):
        # mock different scaleness
        self._tag = "14.3"
        self.p.up_folder_name = "14.3" 
        self.p.normalization = "min_max"
        self.p.input_path = "./data/div2k/test_data/00.png" # 无所谓哪个 反正是模拟的
        self.p.pre_study = STORE_TRUE
        self.p.amp = enum
        # self.p.skew_w1 = enum

    
    # 实验: 不同norm的方式 同时跑6个
    def _set_param_exp_norm(self, enum):
        self._exp_name_list = [
            "min_max",
            "standard",
            "pn",
            "pn_cali",
            "pn_buffer",
            "pn_cali_buffer",
            "z_pn"
        ]
        cur_exp = self._exp_name_list[enum]

        if cur_exp == "min_max":
            self.p.normalization = "min_max"
        elif cur_exp == "standard":
            self.p.normalization = "instance"
        elif cur_exp == "pn":
            self.p.normalization = "power"
            self.p.pn_beta = 0
            self.p.pn_buffer = 0
        elif cur_exp == "pn_cali":
            self.p.normalization = "power"
            self.p.pn_beta = 0.05
            self.p.pn_buffer = 0
        elif cur_exp == "pn_buffer":
            self.p.normalization = "power"
            self.p.pn_beta = 0
            self.p.pn_buffer = -1  # adaptive
        # 完全版
        elif cur_exp == "pn_cali_buffer": # 5
            self.p.normalization = "power"
            self.p.pn_beta = 0.05
            self.p.pn_buffer = -1  # adaptive
        elif cur_exp == "z_pn": # 6
            self.p.normalization = "z_power"
            self.p.pn_beta = 0.05
            self.p.pn_buffer = -1  # adaptive

    def _set_param_amp(self, enum):
        self.p.normalization = "min_max"
        self.p.amp = enum
        self.p.num_epochs = 5000
        self.p.log_epoch = 500
    
    def _set_param_skew(self, enum):
        self.p.normalization = "min_max"
        self.p.input_path = "./data/div2k/test_data/00.png" # 无所谓哪个 反正是模拟的
        self.p.pre_study = STORE_TRUE
        self.p.skew_w1 = enum