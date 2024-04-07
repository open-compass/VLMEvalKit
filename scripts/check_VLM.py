import sys
from vlmeval.smp import *
from vlmeval.config import transformer_ver, supported_VLM

msg1 = [
    'https://raw.githubusercontent.com/open-compass/VLMEvalKit/main/assets/apple.jpg', 
    'What is in this image?'
]
msg2 = [
    dict(type='image', value='https://raw.githubusercontent.com/open-compass/VLMEvalKit/main/assets/apple.jpg'),
    dict(type='text', value='What is in this image?')
]
msg3 = [
    'https://raw.githubusercontent.com/open-compass/VLMEvalKit/main/assets/apple.jpg', 
    'https://raw.githubusercontent.com/open-compass/VLMEvalKit/main/assets/apple.jpg', 
    'How many apples are there in these images?'
]
msg4 = [
    dict(type='image', value='https://raw.githubusercontent.com/open-compass/VLMEvalKit/main/assets/apple.jpg'),
    dict(type='image', value='https://raw.githubusercontent.com/open-compass/VLMEvalKit/main/assets/apple.jpg'),
    dict(type='text', value='How many apples are there in these images?')
]

def check_VLM():
    pth = osp.realpath(__file__)
    val = sys.argv[1]
    if val in supported_VLM:
        model = supported_VLM[val]()
        print(f'Model: {val}')
        for i, msg in enumerate([msg1, msg2, msg3, msg4]):
            if i > 1 and not model.INTERLEAVE:
                continue
            res = model.generate(msg)
            print(f'Test {i + 1}: {res}')
    elif val in transformer_ver:
        models = transformer_ver[val]
        for m in models:
            res = run_command(f'python {pth} {m}')

if __name__ == '__main__':
    check_VLM()