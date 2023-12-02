import ray
from ray.util.queue import Queue as RayQueue
from multiprocessing import Queue
import threading
from joyrl.framework.message import Msg, MsgType
from joyrl.framework.config import MergedConfig
from joyrl.algos.base.data_handler import BaseDataHandler
from joyrl.framework.base import Moduler

class Collector(Moduler):
    ''' Collector for collecting training data
    '''
    def __init__(self, cfg: MergedConfig, data_handler: BaseDataHandler, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.data_handler = data_handler
        

    
    def __init__(self, cfg: MergedConfig, data_handler: BaseDataHandler) -> None:
        self.cfg = cfg
        self.data_handler = data_handler
        

    def pub_msg(self, msg: Msg):
        ''' publish message
        '''
        msg_type, msg_data = msg.type, msg.data
        if msg_type == MsgType.COLLECTOR_PUT_EXPS:
            exps = msg_data
            self._put_exps(exps)
        elif msg_type == MsgType.COLLECTOR_GET_TRAINING_DATA:
            if self.training_data_queue.empty(): return None
            return self.training_data_queue.get()
        elif msg_type == MsgType.COLLECTOR_GET_BUFFER_LENGTH:
            return self.get_buffer_length()
        else:
            raise NotImplementedError
        
    def run(self):
        ''' run
        '''
        # use ray queue if ray is initialized
        self.training_data_queue = Queue(maxsize = 128) if not ray.is_initialized() else RayQueue(maxsize = 128)
        self._t_sample_training_data = threading.Thread(target=self._sample_training_data)
        self._t_sample_training_data.start()

    def _sample_training_data(self):
        '''  run
        '''
        while True:
            training_data = self._get_training_data()
            if training_data is None: continue
            while not self.training_data_queue.full():
                self.training_data_queue.put(training_data)
                break
        
    def _put_exps(self, exps):
        ''' add exps to data handler
        '''
        self.data_handler.add_exps(exps) # add exps to data handler

    def _get_training_data(self):
        training_data = self.data_handler.sample_training_data() # sample training data
        return training_data
    
    def handle_data_after_learn(self, policy_data_after_learn, *args, **kwargs):
        return 
    
    def get_buffer_length(self):
        return len(self.data_handler.buffer)


class BaseCollector:
    def __init__(self, cfg, data_handler = None) -> None:
        self.cfg = cfg
        if data_handler is None: raise NotImplementedError("data_handler must be specified!")
        self.data_handler = data_handler

    def pub_msg(self, msg: Msg):
        ''' publish message
        '''
        msg_type, msg_data = msg.type, msg.data
        if msg_type == MsgType.COLLECTOR_PUT_EXPS:
            exps = msg_data
            self._put_exps(exps)
        elif msg_type == MsgType.COLLECTOR_GET_TRAINING_DATA:
            return self._get_training_data()
        elif msg_type == MsgType.COLLECTOR_GET_BUFFER_LENGTH:
            return self.get_buffer_length()
        else:
            raise NotImplementedError
    def _put_exps(self, exps):
        ''' add exps to data handler
        '''
        self.data_handler.add_exps(exps) # add exps to data handler
    def _get_training_data(self):
        training_data = self.data_handler.sample_training_data() # sample training data
        return training_data
    def handle_data_after_learn(self, policy_data_after_learn, *args, **kwargs):
        return 
    def get_buffer_length(self):
        return len(self.data_handler.buffer)

class SimpleCollector(BaseCollector):
    def __init__(self, cfg, data_handler) -> None:
        super().__init__(cfg, data_handler)
    
