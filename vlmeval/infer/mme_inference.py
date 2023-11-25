import torch
import torch.distributed as dist

torch.manual_seed(1234)
from vlmeval import *
from vlmeval.mllm.models import model_cls_map


def build_mme_prompt(dataset_dir, line):
    if "image" in line:
        dataset_dir = 'MME_images'
        os.makedirs(dataset_dir, exist_ok=True)
        tgt_path = osp.join(dataset_dir, f"{line['index']}.jpg")
        if not osp.exists(tgt_path):
            decode_base64_to_image_file(line['image'], tgt_path)
    elif "image_path" in line:
        img_path = line["image_path"]
        tgt_path = osp.join(dataset_dir, img_path)

    question = line["question"]
    return {"image": tgt_path, "text": question}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, nargs="+", required=True)
    parser.add_argument("--model", type=str, nargs='+', required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    return args


def infer_data(model_name, data, out_file, dataset_dir, verbose=False):
    res = {}
    if osp.exists(out_file):
        res = load(out_file)
    lt = len(data)

    # If finished, will exit without building the model
    finished, unfinished = [], []
    for i in range(lt):
        idx = data.iloc[i]["index"]
        if idx not in res:
            unfinished.append(idx)
        else:
            finished.append(idx)
    if len(unfinished) == 0:
        return
    else:
        print(f'For {out_file}, {len(finished)} finishied; {len(unfinished)} not finished')

    if isinstance(model_name, str):
        model = model_cls_map[model_name]()
    else:
        model = model_name

    for i in tqdm(range(lt)):
        idx = data.iloc[i]["index"]
        if idx in res:
            continue

        struct = build_mme_prompt(dataset_dir, data.iloc[i])
        response = model.generate(prompt=struct["text"], image_path=struct["image"])

        if verbose:
            print(response, flush=True)
        res[idx] = response
        if (i + 1) % 20 == 0:
            dump(res, out_file)
    dump(res, out_file)
    return model

def MME_rating(data):
    stats = defaultdict(dict)
    lt = len(data)
    for i in range(lt):
        item = data.iloc[i]
        category = item['category']
        image_path = item['image_path']
        score = item['score']
        if image_path not in stats[category]:
            stats[category][image_path] = []
        stats[category][image_path].append(score)
    
    def acc(key, mode='normal'):
        res = stats[key]
        values = []
        for val in res.values():
            if mode == 'normal':
                values.extend(val)
            elif mode == 'plus':
                values.append(val[0] * val[1])
        return np.mean(values) * 100
            
    scores = {}
    for k in stats:
        scores[k] = acc(k) + acc(k, 'plus')

    super_cates = dict(
        perception=['OCR', 'artwork', 'celebrity', 'color', 'count', 'existence', 'landmark', 'position', 'posters', 'scene'],
        reasoning=['code_reasoning', 'commonsense_reasoning', 'numerical_calculation', 'text_translation']
    )
    
    ret = {}
    for sc, cate_list in super_cates.items():
        base = 0
        for c in cate_list:
            base += scores[c]
        ret[sc] = base 
    ret.update(scores)
    return ret

def main():
    args = parse_args()
    assert len(args.data), "--data should be a list of data files"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    for model_name in args.model:
        for i, data_file in enumerate(args.data):
            DATASET_DIR = osp.dirname(data_file)

            data = load(data_file)
            data_base_name = osp.basename(data_file)
            output_filename = data_file.replace(data_base_name, f"{model_name}_" + data_base_name)
            score_filename = output_filename.replace(".tsv", "_score.tsv")
            if osp.exists(score_filename):
                continue

            tmpl = data_file.replace(data_base_name, f"{model_name}_" + "{}" + f"{world_size}_" + data_base_name)
            tmpl = tmpl.replace(".tsv", ".pkl")

            data = data.iloc[local_rank::world_size]
            out_file = tmpl.format(local_rank)

            # no need to build if we want to infer a second dataset
            infer_data(model_name, data, out_file, dataset_dir=DATASET_DIR, verbose=args.verbose)
            if world_size > 1:
                dist.barrier()

            if local_rank == 0:
                data_all = {}
                for i in range(world_size):
                    data_all.update(load(tmpl.format(i)))

                data = load(data_file)
                assert len(data_all) == len(data)
                
                data["prediction"] = [data_all[x] for x in data["index"]]
                if 'image' in data:
                    data.pop("image")
                data["yes"] = data["prediction"].str.contains("Yes", case=False)
                data["no"] = data["prediction"].str.contains("No", case=False)
                # Interesting
                data["prediction"] = data.apply(
                    lambda x: "Yes" if x["yes"] and not x["no"] else "No" if x["no"] and not x["yes"] else "Unknown", axis=1
                )
                data.drop(["yes", "no"], axis=1, inplace=True)
                data["score"] = (data["answer"] == data["prediction"])
                dump(data, output_filename)
                rating = MME_rating(data)
                for k, v in rating.items():
                    print(k, v)

                score = d2df(rating)
                dump(score, score_filename)

                for i in range(world_size):
                    os.remove(tmpl.format(i))


if __name__ == "__main__":
    main()
