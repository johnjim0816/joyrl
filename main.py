# import sys, os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoid "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
# curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
# parent_path = os.path.dirname(curr_path)  # parent path 
# sys.path.append(parent_path)  # add path to system path
import sys,os
import ray
import argparse,datetime,importlib,yaml,time 
import gymnasium as gym
from pathlib import Path
from joyrl.framework.config import GeneralConfig, MergedConfig, DefaultConfig
from joyrl.framework.collector import Collector
from joyrl.framework.tracker import Tracker
from joyrl.framework.interactor import InteractorMgr
from joyrl.framework.learner import LearnerMgr
from joyrl.framework.recorder import Logger, Recorder
from joyrl.framework.tester import OnlineTester
from joyrl.framework.trainer import Trainer
from joyrl.framework.model_mgr import ModelMgr
from joyrl.utils.utils import merge_class_attrs, all_seed,save_frames_as_gif

class Main(object):
    def __init__(self) -> None:
        self._get_default_cfg()  # get default config
        self._process_yaml_cfg()  # load yaml config
        self._merge_cfgs() # merge all configs
        self._config_dirs()  # create dirs
        self._save_cfgs({'general_cfg': self.general_cfg, 'algo_cfg': self.algo_cfg, 'env_cfg': self.env_cfg})
        all_seed(seed=self.general_cfg.seed)  # set seed == 0 means no seed
        
    def print_cfgs(self, logger = None):
        ''' print parameters
        '''
        def print_cfg(cfg, name = ''):
            cfg_dict = vars(cfg)
            logger.info(f"{name}:")
            logger.info(''.join(['='] * 80))
            tplt = "{:^20}\t{:^20}\t{:^20}"
            logger.info(tplt.format("Name", "Value", "Type"))
            for k, v in cfg_dict.items():
                if v.__class__.__name__ == 'list': # convert list to str
                    v = str(v)
                if v is None: # avoid NoneType
                    v = 'None'
                if "support" in k: # avoid ndarray
                    v = str(v[0])
                logger.info(tplt.format(k, v, str(type(v))))
            logger.info(''.join(['='] * 80))
        print_cfg(self.cfg.general_cfg, name = 'General Configs')
        print_cfg(self.cfg.algo_cfg, name = 'Algo Configs')
        print_cfg(self.cfg.env_cfg, name = 'Env Configs')

    def _get_default_cfg(self):
        ''' get default config
        '''
        self.general_cfg = GeneralConfig() # general config
        self.algo_name = self.general_cfg.algo_name
        self.algo_mod = importlib.import_module(f"joyrl.algos.{self.algo_name}.config") # import algo config
        self.algo_cfg = self.algo_mod.AlgoConfig()
        self.env_name = self.general_cfg.env_name
        self.env_mod = importlib.import_module(f"joyrl.envs.{self.env_name}.config") # import env config
        self.env_cfg = self.env_mod.EnvConfig()

    def _process_yaml_cfg(self):
        ''' load yaml config
        '''
        parser = argparse.ArgumentParser(description="hyperparameters")
        parser.add_argument('--yaml', default=None, type=str,
                            help='the path of config file')
        args = parser.parse_args()
        # load config from yaml file
        if args.yaml is not None:
            with open(args.yaml) as f:
                load_cfg = yaml.load(f, Loader=yaml.FullLoader)
                # load general config
                self.load_yaml_cfg(self.general_cfg,load_cfg,'general_cfg')
                # load algo config
                self.algo_name = self.general_cfg.algo_name
                self.algo_cfg = self.algo_mod.AlgoConfig()
                self.load_yaml_cfg(self.algo_cfg,load_cfg,'algo_cfg')
                # load env config
                self.env_name = self.general_cfg.env_name
                self.env_cfg = self.env_mod.EnvConfig()
                self.load_yaml_cfg(self.env_cfg, load_cfg, 'env_cfg')

    def _merge_cfgs(self):
        ''' merge all configs
        '''
        self.cfg = MergedConfig()
        setattr(self.cfg, 'general_cfg', self.general_cfg)
        setattr(self.cfg, 'algo_cfg', self.algo_cfg)
        setattr(self.cfg, 'env_cfg', self.env_cfg)
        self.cfg = merge_class_attrs(self.cfg, self.general_cfg)
        self.cfg = merge_class_attrs(self.cfg, self.algo_cfg)
        self.cfg = merge_class_attrs(self.cfg, self.env_cfg)
        
    def _save_cfgs(self, config_dict: dict):
        ''' save config
        '''
        with open(f"{self.cfg.task_dir}/config.yaml", 'w') as f:
            for cfg_type in config_dict:
                yaml.dump({cfg_type: config_dict[cfg_type].__dict__}, f, default_flow_style=False)

    def load_yaml_cfg(self,target_cfg: DefaultConfig,load_cfg,item):
        if load_cfg[item] is not None:
            for k, v in load_cfg[item].items():
                setattr(target_cfg, k, v)

    def _config_dirs(self):
        def config_dir(dir,name = None):
            Path(dir).mkdir(parents=True, exist_ok=True)
            setattr(self.cfg, name, dir)
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time
        env_name = self.env_cfg.id if self.env_cfg.id is not None else self.general_cfg.env_name
        task_dir = f"{os.getcwd()}/tasks/{self.general_cfg.mode.capitalize()}_{env_name}_{self.general_cfg.algo_name}_{curr_time}"
        dirs_dic = {
            'task_dir':task_dir,
            'model_dir':f"{task_dir}/models",
            'res_dir':f"{task_dir}/results",
            'fig_dir':f"{task_dir}/figs",
            'log_dir':f"{task_dir}/logs",
            'traj_dir':f"{task_dir}/traj",
            'video_dir':f"{task_dir}/videos",
            'tb_dir':f"{task_dir}/tb_logs"
        }
        for k,v in dirs_dic.items():
            config_dir(v,name=k)

    def env_config(self):
        ''' create single env
        '''
        env_cfg_dic = self.env_cfg.__dict__
        kwargs = {k: v for k, v in env_cfg_dic.items() if k not in env_cfg_dic['ignore_params']}
        env = gym.make(**kwargs)
        setattr(self.cfg, 'obs_space', env.observation_space)
        setattr(self.cfg, 'action_space', env.action_space)
        if self.env_cfg.wrapper is not None:
            wrapper_class_path = self.env_cfg.wrapper.split('.')[:-1]
            wrapper_class_name = self.env_cfg.wrapper.split('.')[-1]
            env_wapper = __import__('.'.join(wrapper_class_path), fromlist=[wrapper_class_name])
            env = getattr(env_wapper, wrapper_class_name)(env)
        return env
    
    def policy_config(self, cfg):
        ''' configure policy and data_handler
        '''
        policy_mod = importlib.import_module(f"joyrl.algos.{cfg.algo_name}.policy")
         # create agent
        data_handler_mod = importlib.import_module(f"joyrl.algos.{cfg.algo_name}.data_handler")
        policy = policy_mod.Policy(cfg) 
        if cfg.load_checkpoint:
            policy.load_model(f"tasks/{cfg.load_path}/models/{cfg.load_model_step}")
        data_handler = data_handler_mod.DataHandler(cfg)
        return policy, data_handler

    
    def _start(self, **kwargs):
        ''' start serial training
        '''
        env, policy, data_handler = kwargs['env'], kwargs['policy'], kwargs['data_handler']
        tracker = Tracker(self.cfg)
        logger = Logger(self.cfg)
        recorder = Recorder(self.cfg, logger = logger)
        online_tester = OnlineTester(self.cfg, env = env, policy = policy, logger = logger)
        collector = Collector(self.cfg, data_handler = data_handler, logger = logger)
        interactor_mgr = InteractorMgr(self.cfg, 
                                        env = env, 
                                        policy = policy,
                                        logger = logger
                                    )
        learner_mgr = LearnerMgr(self.cfg, 
                                policy = policy,
                                logger = logger
                            )
        model_mgr = ModelMgr(self.cfg, model_params = policy.get_model_params(),logger = logger)
        trainer = Trainer(  self.cfg,
                            tracker = tracker,
                            model_mgr = model_mgr,
                            collector = collector,
                            interactor_mgr = interactor_mgr,
                            learner_mgr = learner_mgr,
                            online_tester = online_tester,
                            recorder = recorder,
                            logger = logger
                        )
        trainer.run()

    def _ray_start(self, **kwargs):
        ''' start parallel training
        '''
        env, policy, data_handler = kwargs['env'], kwargs['policy'], kwargs['data_handler']
        ray.init()
        tracker = Tracker.remote(self.cfg)
        logger = Logger.remote(self.cfg)
        recorder = Recorder.remote(self.cfg, logger = logger)
        online_tester = OnlineTester.remote(self.cfg, env = env, policy = policy, logger = logger)
        collector = Collector.remote(self.cfg, data_handler = data_handler)
        interactor_mgr = InteractorMgr.remote(self.cfg, env = env, policy = policy)
        learner_mgr = LearnerMgr.remote(self.cfg, policy = policy)
        model_mgr = ModelMgr.remote(self.cfg, model_params = policy.get_model_params(),logger = logger)

        trainer = ray.remote(Trainer).remote(self.cfg,
                                tracker = tracker,
                                model_mgr = model_mgr,
                                collector = collector,
                                interactor_mgr = interactor_mgr,
                                learner_mgr = learner_mgr,
                                online_tester = online_tester,
                                recorder = recorder,
                                logger = logger).options(num_cpus = 0)
        
        # trainer = Trainer.remote(self.cfg,
        #                         tracker = tracker,
        #                         model_mgr = model_mgr,
        #                         collector = collector,
        #                         interactor_mgr = interactor_mgr,
        #                         learner_mgr = learner_mgr,
        #                         online_tester = online_tester,
        #                         recorder = recorder,
        #                         logger = logger)
        ray.get(trainer.run.remote())

    def run(self) -> None:
        env = self.env_config() # create single env
        policy, data_handler = self.policy_config(self.cfg) # configure policy and data_handler
        if self.cfg.learner_mode == 'serial':
            self._start(
                env = env,
                policy = policy,
                data_handler = data_handler
            )
        elif self.cfg.learner_mode == 'parallel':
            self._ray_start(
                env = env,
                policy = policy,
                data_handler = data_handler
            )
        else:
            raise ValueError(f"[Main.run] learner_mode must be 'serial' or 'parallel'!")

if __name__ == "__main__":
    main = Main()
    main.run()
