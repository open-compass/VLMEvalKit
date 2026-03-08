from vlmeval.smp import *
from vlmeval.dataset import SUPPORTED_DATASETS, build_dataset
import os.path as osp
import threading

LOCK = threading.Lock()

ESSENTIAL_MODELS = [
    'Claude4.5-Sonnet', 
    'Gemini2.5-Pro-Fix', 
    'Gemini3-Flash-Preview', 
    'Gemini3-Pro-Preview-Exp', 
    'GPT-5.1-High', 
    'GPT-5-High',
    'Qwen3-VL-235B', 
    'Seed1.5-VL', 
    'Seed1.6-1015', 
    'Seed1.8-Final', 
    'Seed1.8-woGUI',
    'Seed2.0-Pro-Preview',
    'Seed2.0-Pro-Preview-Min512-Max5120',
    'Seed2.0-Pro-Preview-MinMax1280',
]

SUMM_FILE = '/mnt/hdfs/kenny/outputs/summarize.json'
LOG_PATH = '/mnt/hdfs/kenny/outputs/summarize_log.json'
WORK_DIR = None

def load_summ(summ_file=SUMM_FILE):
    if not osp.exists(summ_file):
        return {}
    return load(summ_file)

def dump_summ(summ, summ_file=SUMM_FILE):
    dump(summ, summ_file)
    

DATASET_NAMES = set(k for k in SUPPORTED_DATASETS)
def get_default_work_dir():
    cur_path = osp.realpath(__file__)
    vlmeval_dir = osp.dirname(osp.dirname(cur_path))
    output_dir = osp.join(vlmeval_dir, 'outputs')
    return os.getenv('MMEVAL_ROOT', output_dir)


def parse_args(args):
    parser = argparse.ArgumentParser(description="Summarize VLMEvalKit results.")
    parser.add_argument(
        "--work-dir",
        type=str,
        default=get_default_work_dir(),
        help="Path to the VLMEvalKit outputs directory.",
    )
    args = parser.parse_args(args)
    return args


def dry_run(work_dir):
    fs = os.listdir(work_dir)
    tups = []
    for model_name in fs: 
        if ESSENTIAL_MODELS is not None and model_name not in ESSENTIAL_MODELS:
            continue
        model_dir = osp.join(work_dir, model_name)
        files = os.listdir(model_dir)
        for dataset_name in DATASET_NAMES:
            file_name = f'{model_name}_{dataset_name}.tsv'
            if file_name not in files:
                continue
            tups.append((model_name, dataset_name))
    return tups

def handle_tup_list(tups):
    assert len(tups), tups
    dataset_name = tups[0][1]
    for item in tups:
        assert item[1] == dataset_name, item
    with LOCK:
        dataset = build_dataset(dataset_name)
    summ = load_summ()
    logs = defaultdict(list)
    new_res = {}
    for item in tups:
        item_key = f'{item[0]};{item[1]}'
        if item_key in summ:
            prev_record = summ[item_key]
            assert 'timestamp' in prev_record, prev_record
            pred_file = f'{WORK_DIR}/{item[0]}/{item[0]}_{item[1]}.tsv'
            if osp.exists(pred_file) and osp.getmtime(pred_file) <= prev_record['timestamp']:
                logs['skip'].append(dict(tup=item))
                continue
        try:
            res = dataset.report(item[0], item[1], root=WORK_DIR, verbose=True)
            res['timestamp'] = time.time()
            new_res[item_key] = res
            max_err_rate = res['response_err_rate']
            if res.get('judge_err_rate', None) is not None:
                max_err_rate = max(max_err_rate, res['judge_err_rate'])
            # Thresholds: 0.005, 0.01
            if max_err_rate < 0.005:
                logs['pass'].append(dict(tup=item, max_err_rate=max_err_rate))
            elif 0.005 <= max_err_rate < 0.01:
                logs['warn'].append(dict(tup=item, max_err_rate=max_err_rate))
            else:
                logs['rerun'].append(dict(tup=item, max_err_rate=max_err_rate))
        except:
            logs['fail'].append(dict(tup=item))
    with LOCK:
        summ = load_summ()
        summ.update(new_res)
        dump_summ(summ)
    return logs

def get_logs_by_model(logs_by_bench):
    by_model = dict()
    for b, logs in logs_by_bench.items():
        for cate, v in logs.items():
            for item in v:
                model_name = item['tup'][0]
                if model_name not in by_model:
                    by_model[model_name] = defaultdict(list)
                by_model[model_name][cate].append(item)
    return by_model 


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    WORK_DIR = args.work_dir
    tups = dry_run(args.work_dir)
    tup_groups = defaultdict(list)
    for item in tups:
        tup_groups[item[1]].append(item)
    dataset_names = [k for k in tup_groups]
    groups = [dict(tups=tup_groups[dname]) for dname in dataset_names]
    # dump(tup_groups, './tup_groups.json')
    # exit(1)
    logs = track_progress_rich(handle_tup_list, groups, nproc=6, desc='Summarizing')
    logs_by_bench = {dname: log for dname, log in zip(dataset_names, logs)}
    logs_by_model = get_logs_by_model(logs_by_bench)
    logs = dict(by_bench=logs_by_bench, by_model=logs_by_model)
    log_dict = {} if not osp.exists(LOG_PATH) else load(LOG_PATH)
    log_dict[int(time.time())] = logs
    dump(log_dict, LOG_PATH)
    