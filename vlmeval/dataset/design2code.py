import contextlib

import joblib
from joblib import Parallel, delayed

from .image_base import ImageBaseDataset
from ..smp import *


def get_direct_prompt():
    return (
        "You are an expert web developer who specializes in HTML and CSS.\n"
        "A user will provide you with a screenshot of a webpage.\n"
        "You need to return a single html file that uses HTML and CSS to reproduce the given website.\n"
        "Include all CSS code in the HTML file itself.\n"
        'If it involves any images, use "rick.jpg" as the placeholder.\n'
        "Some images on the webpage are replaced with a blue rectangle as the placeholder, "
        'use "rick.jpg" for those as well.\n'
        "Do not hallucinate any dependencies to external files. "
        "You do not need to include JavaScript scripts for dynamic interactions.\n"
        "Pay attention to things like size, text, position, and color of all the elements, "
        "as well as the overall layout.\n"
        "Respond with the content of the HTML+CSS file:\n"
    )


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into a tqdm progress bar."""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def print_multi_score(multi_score):
    (
        _,
        final_size_score,
        final_matched_text_score,
        final_position_score,
        final_text_color_score,
        final_clip_score,
    ) = multi_score
    print()
    print("Block-Match: ", final_size_score)
    print("Text: ", final_matched_text_score)
    print("Position: ", final_position_score)
    print("Color: ", final_text_color_score)
    print("CLIP: ", final_clip_score)
    print("--------------------------------\n")


def cleanup_response(response):
    # Simple post-processing.
    if response[:3] == "```":
        response = response[3:].strip()
    if response[-3:] == "```":
        response = response[:-3].strip()
    if response[:4] == "html":
        response = response[4:].strip()

    # Strip anything before '<!DOCTYPE'.
    if "<!DOCTYPE" in response:
        response = response.split("<!DOCTYPE", 1)[1]
        response = "<!DOCTYPE" + response

    # Strip anything after '</html>'.
    if "</html>" in response:
        response = response.split("</html>")[0] + "</html>"
    return response


class Design2Code(ImageBaseDataset):
    TYPE = "VQA"

    DATASET_URL = {
        "Design2Code": "https://huggingface.co/datasets/yanghtr/Design2Code-VLMEvalKit/resolve/main/Design2Code.tsv",
    }

    DATASET_MD5 = {
        "Design2Code": "bab51290e57ae8dc611de7978d9f414b",
    }

    def __init__(self, dataset="Design2Code"):
        super().__init__(dataset=dataset)
        self.dataset_name = dataset

    @staticmethod
    def generate_tsv(lmudata_dir):
        data_file = osp.join(lmudata_dir, "Design2Code.tsv")
        if os.path.exists(data_file):
            print(f"{data_file} exists.")
            return

        img_dir = f"{lmudata_dir}/images/Design2Code/"
        img_fname_list = []
        for img_fname in os.listdir(img_dir):
            if img_fname.endswith(".png"):
                img_fname_list.append(img_fname)
        img_fname_list.sort()
        assert len(img_fname_list) == 484

        data_list = []
        for img_fname in tqdm(img_fname_list, total=len(img_fname_list)):
            data_list.append(
                {
                    "question": get_direct_prompt(),
                    "image": encode_image_file_to_base64(
                        f"{img_dir}/{img_fname}",
                        target_size=-1,
                        fmt="PNG",
                    ),
                    "img_fname": img_fname,
                }
            )
        data_df = pd.DataFrame(data_list)
        data_df = data_df.assign(index=range(len(data_df)))
        data_df.to_csv(data_file, sep="\t", index=False)
        print(f"Done: tsv file saved in {data_file}.")

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        msgs = []
        msgs.append(dict(type="text", value=line["question"]))
        msgs.append(dict(type="image", value=f"data:image/png;base64,{line['image']}"))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.design2code.visual_score import visual_eval_v3_multi

        infer_data_all = load(eval_file).to_dict(orient="records")

        save_root = os.path.join(os.path.dirname(os.path.abspath(eval_file)), "Design2Code")
        predictions_dir = f"{save_root}/predictions/"
        reference_dir = f"{save_root}/reference/"
        os.makedirs(predictions_dir, exist_ok=True)
        os.makedirs(reference_dir, exist_ok=True)
        root = LMUDataRoot()
        orig_reference_dir = os.path.join(root, "images", "Design2Code")

        for item in infer_data_all:
            code = cleanup_response(item["prediction"].strip())
            filename = item["img_fname"].replace(".png", ".html")
            predict_html = f"{predictions_dir}/{filename}"
            with open(predict_html, "w") as f:
                f.write(code)

        # Copy original html references because screenshot code writes generated pngs.
        for filename in os.listdir(orig_reference_dir):
            if filename.endswith(".html") or filename == "rick.jpg":
                src_path = os.path.join(orig_reference_dir, filename)
                dst_path = os.path.join(reference_dir, filename)
                shutil.copy(src_path, dst_path)
        print("copied original reference directory to ", reference_dir)

        shutil.copy(f"{orig_reference_dir}/rick.jpg", f"{predictions_dir}/rick.jpg")
        test_dirs = {
            "direct_prompting": predictions_dir,
        }

        file_name_list = []
        for filename in os.listdir(reference_dir):
            if not filename.endswith(".html"):
                continue
            if all(
                os.path.exists(os.path.join(test_dirs[key], filename))
                for key in test_dirs
            ):
                file_name_list.append(filename)

        print("total #egs: ", len(file_name_list))

        input_lists = []
        for filename in file_name_list:
            input_pred_list = [os.path.join(test_dirs[key], filename) for key in test_dirs]
            original = os.path.join(reference_dir, filename)
            input_lists.append([input_pred_list, original])

        nproc = judge_kwargs.pop("nproc", 1)
        print(nproc)
        multiprocessing = nproc != 1
        debug = False
        if multiprocessing:
            with tqdm_joblib(tqdm(total=len(input_lists))):
                return_score_lists = list(
                    tqdm(
                        Parallel(n_jobs=nproc)(
                            delayed(visual_eval_v3_multi)(input_list, debug=debug)
                            for input_list in input_lists
                        ),
                        total=len(input_lists),
                    )
                )
        else:
            return_score_lists = []
            for input_list in tqdm(input_lists):
                return_score_list = visual_eval_v3_multi(input_list, debug=debug)
                return_score_lists.append(return_score_list)

        res_dict = {}
        for key in test_dirs:
            res_dict[key] = {}

        for i, filename in enumerate(file_name_list):
            idx = 0
            return_score_list = return_score_lists[i]
            if return_score_list:
                for key in test_dirs:
                    if multiprocessing:
                        _, final_score, multi_score = return_score_list[idx]
                    else:
                        final_score = return_score_list[idx][1]
                        multi_score = return_score_list[idx][2]
                    idx += 1
                    current_score = [final_score] + [item for item in multi_score]
                    res_dict[key][filename] = current_score
            else:
                print(filename + " didn't get a score")
                for key in test_dirs:
                    res_dict[key][filename] = [0, 0, 0, 0, 0, 0]

        with open(f"{save_root}/res_dict.json", "w") as f:
            json.dump(res_dict, f, indent=4)

        for key in test_dirs:
            print(key)
            values = list(res_dict[key].values())
            current_res = np.mean(np.array(values), axis=0)
            print_multi_score(current_res)


if __name__ == "__main__":
    root = LMUDataRoot()
    Design2Code.generate_tsv(root)
