from .image_base import ImageBaseDataset
from ..smp import *
from ..dataset.utils.design2code.visual_score import visual_eval_v3_multi
from multiprocessing import Pool
import contextlib, joblib
from joblib import Parallel, delayed


def get_direct_prompt():
    ## the prompt 
    direct_prompt = ""
    direct_prompt += "You are an expert web developer who specializes in HTML and CSS.\n"
    direct_prompt += "A user will provide you with a screenshot of a webpage.\n"
    direct_prompt += "You need to return a single html file that uses HTML and CSS to reproduce the given website.\n"
    direct_prompt += "Include all CSS code in the HTML file itself.\n"
    direct_prompt += "If it involves any images, use \"rick.jpg\" as the placeholder.\n"
    direct_prompt += "Some images on the webpage are replaced with a blue rectangle as the placeholder, use \"rick.jpg\" for those as well.\n"
    direct_prompt += "Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions.\n"
    direct_prompt += "Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.\n"
    direct_prompt += "Respond with the content of the HTML+CSS file:\n"
    return direct_prompt


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
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
    _, final_size_score, final_matched_text_score, final_position_score, final_text_color_score, final_clip_score = multi_score
    print()
    print("Block-Match: ", final_size_score)
    print("Text: ", final_matched_text_score)
    print("Position: ", final_position_score)
    print("Color: ", final_text_color_score)
    print("CLIP: ", final_clip_score)
    print("--------------------------------\n")


def cleanup_response(response):
    ## simple post-processing
    if response[ : 3] == "```":
        response = response[3 :].strip()
    if response[-3 : ] == "```":
        response = response[ : -3].strip()
    if response[ : 4] == "html":
        response = response[4 : ].strip()

    ## strip anything before '<!DOCTYPE'
    if '<!DOCTYPE' in response:
        response = response.split('<!DOCTYPE', 1)[1]
        response = '<!DOCTYPE' + response
		
    ## strip anything after '</html>'
    if '</html>' in response:
        response = response.split('</html>')[0] + '</html>'
    return response 


class Design2Code(ImageBaseDataset):

    TYPE = "VQA"

    DATASET_URL = {
        "Design2Code": "https://huggingface.co/datasets/yanghtr/Design2Code-VLMEvalKit/resolve/main/Design2Code.tsv",
    }

    DATASET_MD5 = {
        "Design2Code": "bab51290e57ae8dc611de7978d9f414b",
    }

    def __init__(self, dataset='Design2Code'):
        super().__init__(dataset=dataset)
        self.dataset_name = dataset
    

    @staticmethod
    def generate_tsv(lmudata_dir):
        data_file = osp.join(lmudata_dir, 'Design2Code.tsv')
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
            data_list.append({
                'question': get_direct_prompt(),
                'image': encode_image_file_to_base64(f"{img_dir}/{img_fname}", target_size=-1, fmt='PNG'),
                'img_fname': img_fname,
            })
        data_df = pd.DataFrame(data_list)
        data_df = data_df.assign(index=range(len(data_df)))
        data_df.to_csv(data_file, sep='\t', index=False)
        print(f'Done: tsv file saved in {data_file}.')


    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        msgs = []
        msgs.append(dict(type='text', value=line['question']))
        msgs.append(dict(type='image', value=f"data:image/png;base64,{line['image']}"))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):

        infer_data_all = load(eval_file).to_dict(orient='records')

        save_root = os.path.join(os.path.dirname(os.path.abspath(eval_file)), 'Design2Code')
        predictions_dir = f"{save_root}/predictions/"
        reference_dir = f"{save_root}/reference/"
        os.makedirs(predictions_dir)
        os.makedirs(reference_dir)
        ROOT = LMUDataRoot()
        orig_reference_dir = os.path.join(ROOT, "images", "Design2Code")

        for item in infer_data_all:
            code = item["prediction"].strip()
            code = cleanup_response(code)
            # save code to html
            filename = item['img_fname'].replace('.png', '.html')
            predict_html = f"{predictions_dir}/{filename}"
            with open(predict_html, "w") as f:
                f.write(code)
        
        ## copy the original reference directory to a new directory
        ## because we will be creating new screenshots
        for filename in os.listdir(orig_reference_dir):
            if filename.endswith(".html") or filename == "rick.jpg":
                shutil.copy(os.path.join(orig_reference_dir, filename), os.path.join(reference_dir, filename))
        print ("copied original reference directory to ", reference_dir)

        shutil.copy(f"{orig_reference_dir}/rick.jpg", f"{predictions_dir}/rick.jpg")
        test_dirs = {
            "direct_prompting": predictions_dir,
        }

        file_name_list = []

        ## check if the file is in all prediction directories
        for filename in os.listdir(reference_dir):
            if filename.endswith(".html"):
                if all([os.path.exists(os.path.join(test_dirs[key], filename)) for key in test_dirs]):
                    file_name_list.append(filename)

        print ("total #egs: ", len(file_name_list))

        input_lists = []
        for filename in file_name_list:

            input_pred_list = [os.path.join(test_dirs[key], filename) for key in test_dirs]
            original = os.path.join(reference_dir, filename)

            input_list = [input_pred_list, original]
            input_lists.append(input_list)

        nproc = judge_kwargs.pop("nproc", 1)
        print(nproc)
        multiprocessing = False if nproc == 1 else True
        debug = False
        # print ("input_list: ", input_lists)
        if multiprocessing:
            with tqdm_joblib(tqdm(total=len(input_lists))) as progress_bar:
                return_score_lists = list(tqdm(Parallel(n_jobs=nproc)(delayed(visual_eval_v3_multi)(input_list, debug=debug) for input_list in input_lists), total=len(input_lists)))
        else:
            return_score_lists = []
            for input_list in tqdm(input_lists):
                return_score_list = visual_eval_v3_multi(input_list, debug=debug)
                return_score_lists.append(return_score_list)
            # print ("return lists: ", return_score_lists)
        
        res_dict = {}
        for key in test_dirs:
            res_dict[key] = {}

        for i, filename in enumerate(file_name_list):
            idx = 0
            return_score_list = return_score_lists[i]
            # print ("return score list: ", return_score_list)
            if return_score_list:
                for key in test_dirs:
                    if multiprocessing:
                        matched, final_score, multi_score = return_score_list[idx]
                    else:
                        matched = return_score_list[idx][0]
                        final_score = return_score_list[idx][1]
                        multi_score = return_score_list[idx][2]
                    idx += 1
                    current_score = [final_score] + [item for item in multi_score]
                    res_dict[key][filename] = current_score
            else:
                print (filename + " didn't get a score")
                for key in test_dirs:
                    res_dict[key][filename] = [0, 0, 0, 0, 0, 0]

        ## cache all scores 
        with open(f"{save_root}/res_dict.json", "w") as f:
            json.dump(res_dict, f, indent=4)

        for key in test_dirs:
            print(key)
            values = list(res_dict[key].values())
            # print (values)
            current_res = np.mean(np.array(values), axis=0)
            # print(current_res)
            print_multi_score(current_res)


if __name__ == '__main__':
    ROOT = LMUDataRoot()
    Design2Code.generate_tsv(ROOT)