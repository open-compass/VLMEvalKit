import time
import random as rd
from abc import abstractmethod
from ..smp import get_logger

class BaseAPI:
    
    def __init__(self, 
                 retry=10, 
                 wait=3, 
                 system_prompt=None, 
                 verbose=True,
                 fail_msg='Failed to obtain answer via API.',
                 **kwargs):
        self.wait = wait 
        self.retry = retry
        self.system_prompt = system_prompt
        self.kwargs = kwargs
        self.verbose = verbose
        self.fail_msg = fail_msg
        self.logger = get_logger('ChatAPI')
        if len(kwargs):
            self.logger.info(f'BaseAPI received the following kwargs: {kwargs}')
            self.logger.info(f'Will try to use them as kwargs for `generate`. ')

    @abstractmethod
    def generate_inner(self, inputs, **kwargs):
        self.logger.warning(f'For APIBase, generate_inner is an abstract method. ')
        assert 0, 'generate_inner not defined'
        ret_code, answer, log = None, None, None
        # if ret_code is 0, means succeed
        return ret_code, answer, log

    def generate(self, inputs, **kwargs):
        input_type = None
        if isinstance(inputs, str):
            input_type = 'str'
        elif isinstance(inputs, list) and isinstance(inputs[0], str):
            input_type = 'strlist'
        elif isinstance(inputs, list) and isinstance(inputs[0], dict):
            input_type = 'dictlist'
        assert input_type is not None, input_type

        answer = None
        # a very small random delay [0s - 0.5s]
        T = rd.random() * 0.5
        time.sleep(T)
        
        for i in range(self.retry):
            try:
                ret_code, answer, log = self.generate_inner(inputs, **kwargs)
                if ret_code == 0 and self.fail_msg not in answer and answer != '':
                    if self.verbose:
                        print(answer)
                    return answer
                elif self.verbose:
                    self.logger.info(f"RetCode: {ret_code}\nAnswer: {answer}\nLog: {log}")
            except Exception as err:
                if self.verbose:
                    self.logger.error(f'An error occured during try {i}:')
                    self.logger.error(err)
            # delay before each retry
            T = rd.random() * self.wait * 2
            time.sleep(T)
        
        return self.fail_msg if answer in ['', None] else answer
