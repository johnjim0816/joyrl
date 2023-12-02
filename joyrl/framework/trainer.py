import time
import ray
from joyrl.framework.message import Msg, MsgType
from joyrl.framework.config import MergedConfig

class Trainer:
    def __init__(self, cfg : MergedConfig, *args, **kwargs) -> None:
        self.cfg = cfg
        self.model_mgr = kwargs['model_mgr']
        self.interactor_mgr = kwargs['interactor_mgr']
        self.learner_mgr = kwargs['learner_mgr']
        self.collector = kwargs['collector']
        self.online_tester = kwargs['online_tester']
        self.tracker = kwargs['tracker']
        self.recorder = kwargs['recorder']
        self.logger = kwargs['logger']

    def run(self):
        self.model_mgr.init()
        self.recorder.init()
        self.collector.init()
        self.interactor_mgr.init(
            model_mgr = self.model_mgr,
            tracker = self.tracker,
            collector = self.collector,
            recorder = self.recorder,
            logger = self.logger
        )
        self.learner_mgr.init(
            model_mgr = self.model_mgr,
            tracker = self.tracker,
            collector = self.collector,
            recorder = self.recorder,
            logger = self.logger
        )
        self.logger.info(f"[Trainer.run] Start {self.cfg.mode}ing!") # print info
        s_t = time.time()
        while True:
            self.interactor_mgr.run()
            self.learner_mgr.run()
            if self.tracker.pub_msg(Msg(type = MsgType.TRACKER_CHECK_TASK_END)):
                e_t = time.time()
                self.logger.info(f"[Trainer.run] Finish {self.cfg.mode}ing! Time cost: {e_t - s_t:.3f} s")
                break

    def ray_run(self):
        # self.online_tester.run.remote()
        self.model_mgr.run.remote()
        self.recorder.run.remote()
        self.collector.run.remote()
        self.interactor_mgr.start.remote(
            model_mgr = self.model_mgr,
            tracker = self.tracker,
            collector = self.collector,
            recorder = self.recorder,
            logger = self.logger
        )
        self.learner_mgr.run.remote(
            model_mgr = self.model_mgr,
            tracker = self.tracker,
            collector = self.collector,
            recorder = self.recorder,
            logger = self.logger
        )
        self.logger.info.remote(f"Start {self.cfg.mode}ing!") # print info
        s_t = time.time()
        while True:
            if ray.get(self.tracker.pub_msg.remote(Msg(type = MsgType.TRACKER_CHECK_TASK_END))):
                e_t = time.time()
                self.logger.info.remote(f"Finish {self.cfg.mode}ing! Time cost: {e_t - s_t:.3f} s")
                ray.shutdown()
                break 
