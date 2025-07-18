from .model import Model
from .eval import Evaluator
import os


class VQAxVerifyEvaluator:
    def __init__(self, dataset_name, model_name='xVerify-9B-C'):
        self.dataset_name = dataset_name
        self.model_name = model_name 
        self.judge_model = Model(
            model_name = model_name,
            model_path_or_url = os.getenv('XVERIFY_API_URL'),
            inference_mode = 'api',
            api_key = "YOUR_API_KEY")
        self.eval_agent = Evaluator(model=self.judge_model)
        self.working()

    def working(self):
        response = self.judge_model.request("Hi!")
        if response:
            return True
        else:
            return False

    def score(self, predictions, references, questions):
        assert len(predictions) == len(references), (
            f"predictions({len(predictions)}), references({len(references)}) 长度不一致。")
        result_data = []
        for pred, ref, question in zip(predictions, references, questions):
            item = {
                'answer' : ref,
                'prediction' : pred,
                'question' : question,
            }
            result_data.append(item)

        result, info = self.eval_agent.evaluate(result_data, self.dataset_name)

        acc_count = 0
        results = []
        for i, data_point in enumerate(result):
            ans = data_point[f'{self.model_name}_judgment_result']
            if "Correct" in ans and "Incorrect" not in ans:
                results.append(1)
            else:
                results.append(0)
        
        return results



