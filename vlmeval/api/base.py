import time
import random as rd
from abc import abstractmethod
import os.path as osp
from ..smp import get_logger, parse_file


class BaseAPI:

    allowed_types = ['text', 'image']

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
            self.logger.info('Will try to use them as kwargs for `generate`. ')

    @abstractmethod
    def generate_inner(self, inputs, **kwargs):
        self.logger.warning('For APIBase, generate_inner is an abstract method. ')
        assert 0, 'generate_inner not defined'
        ret_code, answer, log = None, None, None
        # if ret_code is 0, means succeed
        return ret_code, answer, log

    def working(self):
        retry = 3
        while retry > 0:
            ret = self.generate('hello')
            if ret is not None and ret != '' and self.fail_msg not in ret:
                return True
            retry -= 1
        return False

    def check_content(self, msgs):
        if isinstance(msgs, str):
            return 'str'
        if isinstance(msgs, dict):
            return 'dict'
        if isinstance(msgs, list):
            types = [self.check_content(m) for m in msgs]
            if all(t == 'str' for t in types):
                return 'liststr'
            if all(t == 'dict' for t in types):
                return 'listdict'
        return 'unknown'

    def preproc_content(self, inputs):
        if self.check_content(inputs) == 'str':
            return [dict(type='text', value=inputs)]
        elif self.check_content(inputs) == 'dict':
            assert 'type' in inputs and 'value' in inputs
            return [inputs]
        elif self.check_content(inputs) == 'liststr':
            res = []
            for s in inputs:
                mime, pth = parse_file(s)
                if mime is None or mime == 'unknown':
                    res.append(dict(type='text', value=s))
                else:
                    res.append(dict(type=mime.split('/')[0], value=pth))
            return res
        elif self.check_content(inputs) == 'listdict':
            for item in inputs:
                assert 'type' in item and 'value' in item
                mime, s = parse_file(item['value'])
                if mime is None:
                    assert item['type'] == 'text'
                else:
                    assert mime.split('/')[0] == item['type']
                    item['value'] = s
            return inputs
        else:
            return None

    def generate(self, inputs, **kwargs):
        assert self.check_content(inputs) in ['str', 'dict', 'liststr', 'listdict'], f'Invalid input type: {inputs}'
        inputs = self.preproc_content(inputs)
        assert inputs is not None and self.check_content(inputs) == 'listdict'
        for item in inputs:
            assert item['type'] in self.allowed_types, f'Invalid input type: {item["type"]}'

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
                    if not isinstance(log, str):
                        try:
                            log = log.text
                        except:
                            self.logger.warning(f'Failed to parse {log} as an http response. ')
                    self.logger.info(f'RetCode: {ret_code}\nAnswer: {answer}\nLog: {log}')
            except Exception as err:
                if self.verbose:
                    self.logger.error(f'An error occured during try {i}:')
                    self.logger.error(err)
            # delay before each retry
            T = rd.random() * self.wait * 2
            time.sleep(T)

        return self.fail_msg if answer in ['', None] else answer
