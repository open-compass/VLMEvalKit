from vlmeval.smp import *
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.utils import build_judge_w_fallback


JUDGE_WORLDQA_PROMPT_EN = """
### Role
You are an expert judge specialized in evaluating the correctness of answers. Your task is to assess whether a model-generated answer is correct based on a given question, the model's response, and the ground truth answer.

### Task: Evaluate Answer Correctness
Please classify the model's response into one of the following three categories. Ignore differences in formatting, punctuation, language (Chinese vs. English), or abbreviations/full names. Focus strictly on the **core semantics** and the **level of detail (granularity)**:

1. **Correct**:
    - The model answer contains the core information of the ground truth.
    - The model answer is semantically consistent with the ground truth and contains no contradictions.
    - **The granularity of the model answer is equal to or finer than the ground truth.**
    - Extra irrelevant information is allowed as long as it does not conflict with the ground truth.

2. **Incorrect**:
    - The model answer provides information that contradicts the ground truth.
    - The model answer provides the wrong specific entity, value, or description.
    - **The granularity of the model answer is coarser than the ground truth**, leading to incomplete or insufficiently specific information.
    - Even if the model expresses uncertainty but follows up with a wrong answer (e.g., "I'm not sure, maybe it's B" when the truth is A), it is considered Incorrect.

3. **Unattempted**:
    - The model explicitly states it does not know the answer (e.g., "I don't know," "I cannot answer this question").
    - The model suggests the user search elsewhere (e.g., "Please search the internet").
    - The model answer contains no information from the ground truth but provides no incorrect or contradictory information.

### Output Format
Please strictly follow this two-line format for your output:
1. **Evaluation**: [A brief explanation of your reasoning]
2. **Label**: [Final classification: "Correct", "Incorrect", or "Unattempted"]

---
### Examples

**Example 1 (Incorrect - Granularity Mismatch/Too Coarse)**
Input:
'''
Question: 图片中属于什么类型的田地？
Model Answer: 图片中展示的是梯田。梯田是在山坡地上开垦并修筑的阶梯状农田。
Ground Truth Answer: 龙脊梯田
'''
Evaluation: 标准答案特指“龙脊梯田”，模型只回答了通用的“梯田”。模型答案层级比答案层级更粗略，未能提供标准答案所需的特指信息，属于层级不一致导致的回答错误。
Label: Incorrect

**Example 2 (Correct - Finer Granularity)**
Input:
'''
Question: What weather phenomenon is in the image?
Model Answer: Based on the visual evidence in the image, the weather phenomenon shown is a **severe storm with extremely high winds**, most likely a **tornado** or a very powerful **hurricane/typhoon**.
Ground Truth Answer: High winds
'''
Evaluation: The ground truth is "high winds," and a "tornado" is a more specific and granular type of high wind. The semantics are correct and the detail is finer.
Label: Correct

**Example 3 (Correct)**
Input:
'''
Question: 图中内容是什么品牌的logo？
Model Answer: via浏览器
Ground Truth Answer: via
'''
Evaluation: 模型答案“via浏览器”包含了标准答案“via”，核心语义一致，且“via浏览器”是更具体的描述，层级上是匹配的。
Label: Correct

**Example 4 (Unattempted)**
Input:
'''
Question: Which athlete is in the image?
Model Answer: I cannot answer this question as I do not have relevant sports data.
Ground Truth Answer: Wout Weghorst
'''
Evaluation: The model explicitly states its inability to answer and provides no incorrect information.
Label: Unattempted

**Example 5 (Incorrect)**
Input:
'''
Question: 图片中展示的是什么苹果品种？
Model Answer: 我觉得可能是阿克苏苹果。
Ground Truth Answer: 烟台苹果
'''
Evaluation: 虽然模型用了“可能”等词汇，但它给出的具体答案“阿克苏苹果”与标准答案“烟台苹果”不符，提供了错误信息。
Label: Incorrect

**Example 6 (Unattempted)**
Input:
'''
Question: What is the name of the insect in this image?
Model Answer: This is a photo of an insect. To find the species, consult an entomologist or use recognition software.
Ground Truth Answer: Japanese rhinoceros beetle
'''
Evaluation: The model does not attempt to name the insect and suggests the user search elsewhere, providing no incorrect information.
Label: Unattempted

---
### Current Task
Input:
'''
Question: {question}
Model Answer: {model_answer}
Ground Truth Answer: {ground_truth_answer}
'''

Evaluation:
"""  # noqa: E501


class WorldVQA(ImageBaseDataset):

    TYPE = 'VQA'
    DATASET_URL = {
        'WorldVQA': 'https://opencompass.openxlab.space/utils/VLMEval/WorldVQA.tsv',
    }
    DATASET_MD5 = {
        'WorldVQA': '3353b1151968179e5264190ece028fed',
    }

    JUDGE_FORMAT = '{model_name}_{dataset_name}_{judge_name}.tsv'
    RATING_FORMAT = '{model_name}_{dataset_name}_{judge_name}_score.json'
    DEFAULT_JUDGE = 'gpt-oss-120b'

    def build_prompt(self, line):
        """Build prompt for a single question."""
        if isinstance(line, int):
            line = self.data.iloc[line]
        elif isinstance(line, pd.Series):
            line = line.to_dict()

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line["question"]
        msgs = []

        for img in tgt_path:
            msgs.append(dict(type='image', value=img))

        # Add prompt text
        if cn_string(question):
            msgs.append(dict(type="text", value="请尽可能提供详细的回答。\n" + question))
        else:
            msgs.append(dict(
                type="text", value="Please provide as much detail as possible in your answer. \n" + question))

        return msgs

    def judge_single_response(self, line, judge_model):
        """Judge a single response."""
        prediction = line["prediction"]
        if self.is_response_err(prediction):
            return {
                "judge_result": None,
                "answer_category": "failed",
                "judge_reason": "Failed to obtain prediction"
            }

        # Build judge prompt
        judge_prompt = JUDGE_WORLDQA_PROMPT_EN.format(
            question=line["question"],
            model_answer=prediction,
            ground_truth_answer=line["answer"]
        )

        try:
            # Get judgment
            judge_result_str = judge_model.generate(
                [{"type": "text", "value": judge_prompt}],
                temperature=0.0
            )

            # Parse judgment
            if "Correct" in judge_result_str:
                judge_result = 1
                answer_category = "correct"
            elif "Unattempted" in judge_result_str:
                judge_result = 0
                answer_category = "unattempted"
            else:
                judge_result = 0
                answer_category = "incorrect"

            return {
                "judge_result": judge_result,
                "answer_category": answer_category,
                "judge_reason": judge_result_str
            }
        except Exception as e:
            print(f"Judge failed for question {line.get('index', 'unknown')}: {e}")
            return {
                "judge_result": None,
                "answer_category": "error",
                "judge_reason": str(e)
            }

    def calculate_overall_score(self, data):
        # Filter out failed judgments
        valid_results = data[~pd.isna(data['judge_result'])]
        failed_count = len(data) - len(valid_results)

        if failed_count / len(data) > 0.05:
            print(f"Warning: {failed_count}/{len(data)} results are missing ({failed_count/len(data)*100:.1f}%)")

        if len(valid_results) == 0:
            print("No valid results to calculate scores!")
            return {"Overall": 0.0}

        # Failed Prediction -> 0 Score
        data['judge_result'] = data['judge_result'].fillna(0)
        accuracies = {}
        accuracies['failure_rate'] = (failed_count / len(data))

        # Difficulty-based scores
        diff_scores = {"easy": [], "medium": [], "hard": []}
        for _, row in valid_results.iterrows():
            difficulty = row['difficulty']
            if difficulty in diff_scores:
                diff_scores[difficulty].append(row["judge_result"])

        for difficulty, scores in diff_scores.items():
            if scores:
                accuracies[difficulty] = np.mean(scores)
                print(f"{difficulty.capitalize()} Accuracy: {np.mean(scores)*100:.2f}% = {sum(scores)}/{len(scores)}")

        # Overall accuracy
        tot_correct_count = sum([r for r in valid_results['judge_result']])
        tot_accuracy = tot_correct_count / len(valid_results)
        print(f"Overall Accuracy: {tot_accuracy*100:.2f}% = {tot_correct_count}/{len(valid_results)}")
        accuracies["Overall"] = tot_accuracy

        # Answer category breakdown
        categories = {}
        for _, row in valid_results.iterrows():
            cat = row.get("answer_category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        if categories:
            total = len(valid_results)
            print("\nAnswer Category Breakdown:")
            for cat, count in categories.items():
                rate = count / total
                print(f"  {cat}: {count}/{total} ({rate*100:.2f}%)")
                accuracies[f"tot_{cat}"] = count
                accuracies[f"tot_{cat}_rate"] = rate

        # Category-wise scores (if category column exists)
        if len(valid_results) > 0 and "category" in valid_results:
            category_scores = {}
            for _, row in valid_results.iterrows():
                category = row.get("category")
                if category:
                    if category not in category_scores:
                        category_scores[category] = []
                    category_scores[category].append(row["judge_result"])

            if category_scores:
                print("\nCategory-wise Accuracy:")
                for category, scores in sorted(category_scores.items()):
                    if scores:
                        acc = np.mean(scores)
                        print(f"  {category}: {acc*100:.2f}% = {sum(scores)}/{len(scores)}")
                        accuracies[f"category_{category}"] = acc

        return accuracies

    def evaluate(self, eval_file, **judge_kwargs):
        nproc = judge_kwargs.pop("nproc", 16)
        model = judge_kwargs.get('model')

        judge_file = self.get_judge_file_path(eval_file, judge_name=model)
        rating_file = self.get_rating_file_path(eval_file, judge_name=model)

        # By default we use the openapi ver., modelcard too slow
        judge_model = build_judge_w_fallback(router='openapi', **judge_kwargs)

        if osp.exists(judge_file):
            data = load(judge_file)
        else:
            data = load(eval_file)
            jobs = [dict(line=line, judge_model=judge_model) for _, line in data.iterrows()]
            results = track_progress_rich(
                self.judge_single_response,
                jobs,
                nproc=nproc,
                desc="Judging WorldVQA Results"
            )
            for k in ['judge_result', 'answer_category', 'judge_reason']:
                data[k] = [item[k] for item in results]
            dump(data, judge_file)

        # Calculate final scores
        scores = self.calculate_overall_score(data)
        dump(scores, rating_file)
        return scores
