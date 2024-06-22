from vlmeval.smp import *
from vlmeval.evaluate.multiple_choice import multiple_choice_eval
import gradio as gr

HEADER = """
# Welcome to MMBench👏👏
We are delighted that you are willing to submit the evaluation results to the MMBench official website! The evaluation service currently can handle submissions of MMBench, MMBench-CN, and CCBench. We use `gpt-3.5-turbo-0125` to help answer matching. Evaluation Codes in VLMEvalKit: https://github.com/open-compass/VLMEvalKit. Please adopt / follow the implementation of VLMEvalKit to generate the submission files. 
Moreover, this is a temporary solution, which **does not support ChatGPT-based answer extraction**. So you need to make sure values in the `prediction` field of your submission files are single characters in A, B, C, D. Other free-form answers can not be recognized by the evaluation script and will be marked as **WRONG**. 

The evaluation script is available at https://github.com/open-compass/VLMEvalKit/tree/main/scripts/mmb_eval_gradio.py
Please contact `opencompass@pjlab.org.cn` for any inquirys about this script. 
"""

def upload_file(file):
    file_path = file.name
    return file_path

def prepare_file(file_name):
    file_md5 = md5(file_name)
    root = LMUDataRoot()
    root = osp.join(root, 'eval_server')
    os.makedirs(root, exist_ok=True)
    suffix = file_name.split('.')[-1]
    if suffix not in ['xlsx', 'tsv', 'csv']:
        return False, "Please submit a file that ends with `.xlsx`, `.tsv`, or `.csv`"
    new_file_name = osp.join(root, f'{file_md5}.{suffix}')
    shutil.move(file_name, new_file_name)
    eval_file = new_file_name
    try:
        data = load(eval_file)
    except:
        return False, "Your excel file can not be successfully loaded by `pd.read_excel`, please double check and submit again. "
    for k in data.keys():
        data[k.lower() if k not in 'ABCD' else k] = data.pop(k)
    if "index" not in data:
        return False, "Your excel file should have a column named `index`, please double check and submit again" , {}
    if "prediction" not in data:
        return False, "Your excel file should have a column named `prediction`, please double check and submit again" , {}
    for ch in 'ABCD':
        if ch not in data:
            return False, f"Your excel file should have a column named `{ch}`, please double check and submit again" , {}
    dump(data, eval_file)
    return True, eval_file

def determine_dataset(eval_file):
    data = load(eval_file)
    def cn_ratio(data):
        iscn = [cn_string(x) for x in data['question']]
        return np.mean(iscn)
    if len(data) < 2500 and 'l2-category' not in data:
        return 'CCBench' if cn_ratio(data) > 0.5 else "Unknown" 
    else:
        return 'MMBench_CN' if cn_ratio(data) > 0.5 else "MMBench"
    
def reformat_acc(acc):
    splits = set(acc['split'])
    keys = list(acc.keys())
    keys.remove('split')
    nacc = {'Category': []}
    for sp in splits:
        nacc[sp.upper()] = []
    for k in keys:
        nacc['Category'].append(k)
        for sp in splits:
            nacc[sp.upper()].append(acc[acc['split'] == sp].iloc[0][k] * 100)
    return pd.DataFrame(nacc)

def evaluate(file):
    file_name = file.name
    flag, eval_file = prepare_file(file_name)
    if not flag:
        return "Error: " + eval_file
    dataset = determine_dataset(eval_file)
    if dataset == 'Unknown':
        return "Error: Cannot determine the dataset given your submitted file. " 

    eval_id = eval_file.split('/')[-1].split('.')[0]
    ret = f"Evaluation ID: {eval_id}\n"
    timestamp = datetime.datetime.now().strftime('%Y.%m.%d  %H:%M:%S')
    ret += f'Evaluation Timestamp: {timestamp}\n'
    acc = multiple_choice_eval(eval_file, dataset=dataset, model='exact_matching')
    nacc = reformat_acc(acc).round(1)
    return ret, nacc

with gr.Blocks() as demo:
    gr.Markdown(HEADER)
    file_output = gr.File()
    upload_button = gr.UploadButton("Click to upload you prediction files for a supported benchmark")
    upload_button.upload(upload_file, upload_button, file_output)
    
    btn = gr.Button("🚀 Evaluate")
    eval_log = gr.Textbox(label="Evaluation Log", placeholder="Your evaluation log will be displayed here")
    df_empty = pd.DataFrame([], columns=['Evaluation Result'])
    eval_result = gr.components.DataFrame(value=df_empty)
    btn.click(evaluate, inputs=[file_output], outputs=[eval_log, eval_result])

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', debug=True, show_error=True)