# -*- coding: utf-8 -*-
# @Time    : 2024/8/19 下午4:01
# @Author  : zhaop-l(zhaop-l@glocon.com)
from functools import partial
from .image_base import ImageBaseDataset
from ..smp import *
import logging


def vllm_qwen15_32b(query):
	headers = {'Content-Type': 'application/json'}
	url = 'http://10.9.27.23:10086/v1/chat/completions'
	data = {"model": "Qwen2-7B-Instruct", "messages": [{"role": "user", "content": query}], "temperature": 0.1, "top_p": 0.8, "repetition_penalty": 1.05}
	json_data = json.dumps(data)
	response = requests.post(url, headers=headers, data=json_data)
	return response.json()["choices"][0]["message"]["content"]


tmpl = (
		'You are an artificial intelligence assistant who will help me compare two answers. '
		'The options are only Yes / No. '
		'You are provided with a question and two answer, '
		'and you need to compare the following two answers and determine if they are the same. '
		'Your should output a single word among the following 2 choices: Yes, No.\n'
		'Example 1: \n'
		'Question: 图中在纵向轴1上的物体有哪些？?\nAnswer 1: KZ4a、KZ8、KZ8.\nAnswer 2: 纵向轴1上标记了3个构件，它们分别是KZ4a、KZ8、KZ4b.\nYour output: No\n'
		'Example 2: \n'
		"Question: 图中在纵向轴1上的物体有哪些？?\nAnswer 1: KZ4a、KZ8、KZ8.\nAnswer 2: 纵向轴1上标记了3个构件，它们分别是KZ4a、KZ8、KZ8.\nYour output: Yes\n"
		'Example 3: \n'
		"Question: 以图纸中的轴网为参考，图中紫色边框内的构件位置是什么？\nAnswer 1: (H,2).\nAnswer 2: 以轴网为参考，图中紫色边框内的构件位置在(H, 2)，也就是横向轴线H和纵向轴线2的交点。\nYour output: Yes\n"
		'Example 4: \n'
		"Question: 图中红色边框内的构件截面尺寸是多少？\nAnswer 1: 构件尺寸为(300+300,100+200)，通过分段标注方式表示，横向尺寸分为300和300，总宽度为600，纵向尺寸分为100和200，总高度为300。.\nAnswer 2: 构件尺寸为(300+300,200+100)，通过分段标注方式表示，横向尺寸分为300和300，总宽度为600，纵向尺寸分为200和100，总高度为300。\nYour output: Yes\n"
		'Example 5: \n'
		'Question: {}?\nAnswer 1: {}\nAnswer 2: {}\nYour output: '
	)

def vqa_evaluate(question,answer,prediction):
	logger = logging.getLogger('LOAD_ENV')
	query1 = tmpl.format(question, answer, prediction)
	query2 = tmpl.format(question, prediction, answer)
	try:
		response1 = vllm_qwen15_32b(query1).lower()
		response2 = vllm_qwen15_32b(query2).lower()
		if 'yes' in response1 and 'no' not in response1:
			source1 = 1
		else:
			source1 = 0
		if 'yes' in response2 and 'no' not in response2:
			source2 = 1
		else:
			source2 = 0
	
		if source1 == source2:
			return source1
		else:
			return 1
	except Exception as e:
		logger.error('vllm qwen1.5_32b api error, Please contact Shi Dawei.')
		return 0

class DrawVQADataset(ImageBaseDataset):
	TYPE = 'DrawVQA'
	DATASET_URL = {'building_drawings_VQA_v1': '',
	               'building_drawings_OCR_v1': ''}

	def load_data(self, dataset):
		data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')

		if file_size(data_path, 'GB') > 1:
			local_path = data_path.replace('.tsv', '_local.tsv')
			if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
				from ..tools import LOCALIZE

				LOCALIZE(data_path, local_path)
			data_path = local_path
		return load(data_path)
	
	@classmethod
	def evaluate(self, eval_file, **judge_kwargs):
		data = load(eval_file)
		category_dict = dict(data['category'].value_counts())

		draw_number_level = {str(key): value for key, value in category_dict.items()}
		DrawBench_score = {str(key): 0 for key, value in category_dict.items()}
		
		lt = len(data)
		lines = [data.iloc[i] for i in range(lt)]
		for i in tqdm(range(len(lines))):
			line = lines[i]
			question = str(line['question'])
			predict = str(line['prediction'])
			answers = str(line['answer'])
			category = str(line['category'])
			
			answer = answers.lower().strip().replace(' ', '').replace('\n', ' ')
			predict = predict.lower().strip().replace(' ', '').replace('\n', ' ')
			
			if 'OCR' in eval_file or category == 'Corresponding image names':
				if answer in predict:
					DrawBench_score[category] += 1
			else:
				vqa_score = vqa_evaluate(question, answer, predict)
				if vqa_score == 1:
					DrawBench_score[category] += 1
		final_score_dict = {}
		for i in DrawBench_score:
			p = DrawBench_score[i] / draw_number_level[i]
			final_score_dict[i] = p
		final_score_dict['Average'] = sum(DrawBench_score.values()) / lt
		final_score_dict = d2df(final_score_dict)
		score_pth = eval_file.replace('.xlsx', '_acc.csv')
		dump(final_score_dict, score_pth)
		return final_score_dict
