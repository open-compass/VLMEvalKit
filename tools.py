import json
import os
import subprocess

import pandas as pd
from datasets import load_dataset

from vlmeval import load, decode_base64_to_image_file, MME_rating


def load_demo():
    os.environ["TSV_DATASET_LIMIT"] = "1"
    data = load('MME.tsv')
    print(len(data))


def write_req():
    lines = """
     1461  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple torch
     1464  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple pandas
     1466  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple validators
     1468  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple pillow
     1470  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple requests
     1472  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple tqdm
     1474  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple matplotlib
     1476  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple seaborn
     1478  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple tabulate
     1480  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple huggingface_hub
     1482  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple sty
     1484  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple rich
     1486  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple portalocker
     1488  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple pycocoevalcap
     1490  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple transformers
     1492  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple torchvision
     1494  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple omegaconf
     1496  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple tiktoken
     1498  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple einops
     1498  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple transformers_stream_generator
     1500  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple accelerate
     1509  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple pipreqs
     1520  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple xlsxwriter
     1522  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple openpyxl
     1528  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple timm
     1530  /home/jeeves/.virtualenvs/VLMEvalKit/bin/python -m pip install -i https://mirror.in.zhihu.com/simple sentencepiece
    """
    for line in lines.split("\n"):
        line = line.strip()
        if line:
            package = line.split()[-1]
            print(package)


def get_mme_result():
    # file = "/home/jeeves/logs2/MiniCPM-V/MiniCPM-V_MME_score.csv"
    # file = "/home/jeeves/logs2/MiniCPM-V-2/MiniCPM-V-2_MME_score.csv"
    file = "/home/jeeves/logs12/MiniCPM-V-2/MiniCPM-V-2_MME_score.csv"
    df = pd.read_csv(file)
    for idx, row in df.iterrows():
        row = dict(row)

        s = 0.0
        for key, val in row.items():
            if key in {"perception", "reasoning"}:
                continue
            print(key, val)
            s += float(val)
        print(s)
        print("{:.4f}".format(s / 2800 * 100))


def get_accuracy_from_log():
    """
2024-05-11 16:57:44,742 - Evaluation - INFO - VQA Eval Finished. Saved to /home/jeeves/logs3/MiniCPM-V-2/MiniCPM-V-2_TextVQA_VAL_acc.csv.
2024-05-11 16:57:44,742 - Evaluation - INFO -    Overall
0   71.238
jeeves@notebook-3650-vllm-256k-debug:~/VLMEvalKit$ cd /home/jeeves/VLMEvalKit && rm -rf /home/jeeves/logs3 && mkdir -p /home/jeeves/logs3 && LMUTSVData=/mnt/data/user/tc_agi/UltraEval/LMUData LOCAL_MODEL_ROOT=/mnt/data/user/tc_agi/UltraEval/open_models/modelscope/hub/ /home/jeeves/.virtualenvs/VLMEvalKit/bin/python run.py --work-dir /home/jeeves/logs3 --data TextVQA --model MiniCPM-V-2 --verbose
    """
    output_path = '/home/jeeves/logs3'
    model_name = 'MiniCPM-V-2'
    dataset_name = 'TextVQA_VAL'
    result_file = f'{output_path}/{model_name}/{model_name}_{dataset_name}_acc.csv'
    df = pd.read_csv(result_file)
    acc = float(df["Overall"][0])
    print(acc)


def cmp_score():
    MME_rating("/home/jeeves/logs12/MiniCPM-V-2/MiniCPM-V-2_MME_auxmatch.xlsx")

    df = pd.read_csv("/home/jeeves/logs12/MiniCPM-V-2/MiniCPM-V-2_MME_score.csv")

    total = df["perception"][0] + df["reasoning"][0]
    print(total)
    # 1813.1542617046819

    total = 0.0
    for idx, row in df.iterrows():
        row = dict(row)
        print(row)
        for key, val in row.items():
            if key not in {'perception', 'reasoning'}:
                total += val
    print(total)
    # 1813.1542617046819

    with open("id1077_results.673827/_all_results.json", 'r', encoding="utf-8") as fin:
        d = json.load(fin)

    for k, val in d['mme']['gen'].items():
        if k == 'mean_result':
            continue
        if k == "codereasoning":
            k = 'code_reasoning'
        if k == "commonsensereasoning":
            k = 'commonsense_reasoning'
        if k == 'numericalcalculation':
            k = 'numerical_calculation'
        if k == 'texttranslation':
            k = 'text_translation'

        val = val['accuracy']
        print(k, val * 200, row[k])

    count = 0
    total = 0
    df = pd.read_excel("/home/jeeves/logs12/MiniCPM-V-2/MiniCPM-V-2_MME_auxmatch.xlsx")
    for idx, row in df.iterrows():
        row = dict(row)
        if row['category'] == "OCR":
            total += 1
            if row['score']:
                count += 1
    print(count / total * 200)  # 0.675  # 135.0


def cmp_question(task_name):
    data = []
    # task_name = 'mme_texttranslation_gen'
    with open(f"id1077_results.673827/{task_name}/instances.jsonl", 'r', encoding='utf-8') as fin:
        for line in fin:
            d = json.loads(line)
            item = []
            item.append(d['doc']['question_id'])
            item.append(d['ground_truth'])
            item.append(d['processed_output'])
            item.append(int(d['eval_scores']['accuracy']))
            data.append(item)
    data.sort(key=lambda x: (x[0], x[1].lower() == "no"))

    # df = pd.read_excel("/home/jeeves/logs12/MiniCPM-V-2/MiniCPM-V-2_MME.xlsx")
    df = pd.read_excel("/home/jeeves/logs12/MiniCPM-V-2/MiniCPM-V-2_MME_auxmatch.xlsx")
    df_cat = df[df['category'] == task_name.split("_")[1]]
    for item, p, s in zip(data, df_cat["prediction"], df_cat['score']):
        s = int(s)
        # print(item, p, s)

        # if item[-2] != p:
        # if True:
        if item[-1] != s:
            print(item, p, s, "#########" if item[-1] != p else "")
    # ['artwork/10543.jpg', 'No', 'Yes, this artwork is displayed in Museo Civico, Pistoia.'] yes #########
    # ['artwork/11331.jpg', 'No', 'yes'] no #########
    # 61


def cmp_all_tasks():
    s = """mme_artwork_gen
mme_celebrity_gen
mme_codereasoning_gen
mme_color_gen
mme_commonsensereasoning_gen
mme_count_gen
mme_existence_gen
mme_landmark_gen
mme_numericalcalculation_gen
mme_OCR_gen
mme_position_gen
mme_posters_gen
mme_scene_gen
mme_texttranslation_gen"""
    for line in s.split("\n"):
        print(f"line: {line}")
        cmp_question(line)


def load_save_pil():
    # img_pil = Image.open("image_hardco.jpg")
    # img_pil.save("image_hardco_load_save.jpg")

    # json_data = json.dumps(np.array(image).tolist())
    # img_pil = Image.fromarray(np.array(json.loads(json_data), dtype='uint8'))
    # print(image == img_pil)
    # img_pil.save("image_hardco_load_save_np.jpg")

    dataset = load_dataset("lmms-lab/MME")["test"]
    dataset_ocr = dataset.filter(lambda x: x["question_id"] == "OCR/0017.jpg" and x["answer"] == "No")


def put_to_hdfs(local_path, remote_path):
    my_env = os.environ.copy()
    my_env["HADOOP_USER_NAME"] = "tc_agi"
    my_env["HADOOP_USER_PASSWORD"] = "IH2U3AS1D"

    mk_command = f"hdfs dfs -mkdir -p {os.path.dirname(remote_path)}"
    print(f"mk_command: {mk_command}")
    subprocess.run(mk_command.split(), env=my_env, check=True)

    put_command = f"hdfs dfs -put {local_path} {remote_path}"
    print(f"put_command: {put_command}")
    subprocess.run(put_command.split(), env=my_env, check=True)


def save_images():
    local_path = "/tmp/MME_VLMEvalKit_Images"
    remote_path = "/user/tc_agi/UltraEval/MME_VLMEvalKit_Images"
    tsv_path = '/mnt/data/user/tc_agi/UltraEval/LMUData/MME.tsv'
    df = pd.read_csv(tsv_path, sep='\t')
    for idx, row in df.iterrows():
        row = dict(row)
        print(row)
        if row["index"] % 2 == 0:
            image_path = row["image_path"].replace("/images", "")
            image_path = os.path.join(local_path, image_path)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            decode_base64_to_image_file(row["image"], image_path)
    put_to_hdfs(local_path, remote_path)


if __name__ == "__main__":
    # write_req()
    # get_mme_result()
    # get_accuracy_from_log()
    cmp_score()
    # cmp_all_tasks()
    # load_save_pil()
    # save_images()

    # cd /home/jeeves/VLMEvalKit && rm -rf /home/jeeves/logs && mkdir -p /home/jeeves/logs && LMUTSVData=/mnt/data/user/tc_agi/UltraEval/LMUData LOCAL_MODEL_ROOT=/mnt/data/user/tc_agi/UltraEval/open_models/modelscope/hub/ TSV_DATASET_LIMIT=1 /home/jeeves/.virtualenvs/VLMEvalKit/bin/python run.py --work-dir /home/jeeves/logs --data TextVQA --model MiniCPM-V-2 --verbose && hdfs dfs -put /home/jeeves/logs /user/tc_agi/UltraEval/logs
    # /home/jeeves/.virtualenvs/VLMEvalKit/bin/python tools.py
    # cd /home/jeeves/VLMEvalKit && rm -rf /home/jeeves/logs3 && mkdir -p /home/jeeves/logs3 && LMUTSVData=/mnt/data/user/tc_agi/UltraEval/LMUData LOCAL_MODEL_ROOT=/mnt/data/user/tc_agi/UltraEval/open_models/modelscope/hub/ /home/jeeves/.virtualenvs/VLMEvalKit/bin/python run.py --work-dir /home/jeeves/logs3 --data TextVQA --model MiniCPM-V-2 --verbose
    # cd /home/jeeves/VLMEvalKit && rm -rf /home/jeeves/logs10 && mkdir -p /home/jeeves/logs10 && LMUTSVData=/mnt/data/user/tc_agi/UltraEval/LMUData LOCAL_MODEL_ROOT=/mnt/data/user/tc_agi/UltraEval/open_models/modelscope/hub/ /home/jeeves/.virtualenvs/VLMEvalKit/bin/python run.py --work-dir /home/jeeves/logs10 --data MME --model MiniCPM-V-2 --verbose
    # cd /home/jeeves/VLMEvalKit && rm -rf /home/jeeves/logs11 && mkdir -p /home/jeeves/logs11 && LMUTSVData=/mnt/data/user/tc_agi/UltraEval/LMUData LOCAL_MODEL_ROOT=/mnt/data/user/tc_agi/UltraEval/open_models/modelscope/hub/ /home/jeeves/.virtualenvs/VLMEvalKit/bin/python run.py --work-dir /home/jeeves/logs11 --data MME --model MiniCPM-V-2 --verbose
    # cd /home/jeeves/VLMEvalKit && rm -rf /home/jeeves/logs12 && mkdir -p /home/jeeves/logs12 && LMUTSVData=/mnt/data/user/tc_agi/UltraEval/LMUData LOCAL_MODEL_ROOT=/mnt/data/user/tc_agi/UltraEval/open_models/modelscope/hub/ /home/jeeves/.virtualenvs/VLMEvalKit/bin/python run.py --work-dir /home/jeeves/logs12 --data MME --model MiniCPM-V-2 --verbose

    # Github进行fork后如何与原仓库同步
    # https://github.com/selfteaching/the-craft-of-selfteaching/issues/67

    # [{'type': 'text', 'value': "Study the image carefully and pick the option associated with the correct answer.                 Focus solely on selecting the option and avoid including any other content.\nDetermine the area in hectares between the line AB and a meandering stream for offsets taken at a regular interval of 20 m along the line AB (Fig. 12.5). Use both the trapezoidal rule and Simpson's rule."}, {'type': 'image', 'value': '/home/jeeves/LMUData/images/MMMU/1317_1.jpg'}, {'type': 'text', 'value': ''}, {'type': 'image', 'value': '/home/jeeves/LMUData/images/MMMU/1317_2.jpg'}, {'type': 'text', 'value': '\nOptions:\nA. Use trapezoidal rule,area = 0.5010 hectares;Use trapezoidal rule,area = 0.5560 hectares\nB. Use trapezoidal rule,area = 0.5010 hectares;Use trapezoidal rule,area = 0.5460 hectares\nC. Use trapezoidal rule,area = 0.5010 hectares;Use trapezoidal rule,area = 0.5260 hectares\nD. Use trapezoidal rule,area = 0.5010 hectares;Use trapezoidal rule,area = 0.5360 hectares\n'}]
