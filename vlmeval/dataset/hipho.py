# -*- coding: utf-8 -*-
import os
import re
import json
import pandas as pd
import numpy as np
import warnings
import time
import base64
from io import BytesIO

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from .utils.hipho_verifier import grade, extract_boxed_answer, get_answer_str, answer_tag_reward_fn_for_r1
from .utils.prompt_inference import SYSTEM_PROMPTS_EN, SYSTEM_PROMPTS_ZH, JUDGE_GRADING_PROMPT_TEMPLATE, TOTAL_SCORE_WARNING_TEMPLATE, RETRY_WARNING_TEMPLATE
from ..smp import *



class HiPhODataset(ImageBaseDataset):
    """
    HiPhO (High School Physics Olympiad) Benchmark Dataset
    
    支持13个物理奥林匹克竞赛数据集：
    - IPhO 2024/2025: 国际物理奥林匹克
    - EuPhO 2024/2025: 欧洲物理奥林匹克  
    - APhO 2025: 亚洲物理奥林匹克
    - PanPhO 2024/2025: 泛亚物理奥林匹克
    - NBPhO 2024/2025: 北欧-波罗的海物理奥林匹克
    - F_MA 2024/2025: 美国物理竞赛
    - PanMechanics 2024/2025: 泛亚力学竞赛
    
    集成了hipho_verifier验证器，支持粗细粒度评测
    """
    TYPE = 'VQA'  # 统一使用VQA类型
    
    # 数据集URL映射 - 指向HuggingFace数据集的不同split
    DATASET_URL = {
        'IPhO_2024': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'IPhO_2025': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'EuPhO_2024': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'EuPhO_2025': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'APhO_2025': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'PanPhO_2024': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'PanPhO_2025': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'NBPhO_2024': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'NBPhO_2025': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'F_MA_2024': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'F_MA_2025': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'PanMechanics_2024': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
        'PanMechanics_2025': 'https://huggingface.co/datasets/HY-Wan/HiPhO',
    }
    
    # MD5值暂时设为空，因为HuggingFace数据集是动态加载的
    DATASET_MD5 = {
        'IPhO_2024': '',
        'IPhO_2025': '',
        'EuPhO_2024': '',
        'EuPhO_2025': '',
        'APhO_2025': '',
        'PanPhO_2024': '',
        'PanPhO_2025': '',
        'NBPhO_2024': '',
        'NBPhO_2025': '',
        'F_MA_2024': '',
        'F_MA_2025': '',
        'PanMechanics_2024': '',
        'PanMechanics_2025': '',
    }

    def __init__(self, dataset='IPhO_2025', skip_noimg=False, language='en'):
        """初始化数据集"""
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)
        self.language = language

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL.keys())

    def load_data(self, dataset):
        """从HuggingFace加载数据集"""
        from datasets import load_dataset
        
        hf_dataset = load_dataset('HY-Wan/HiPhO', split=dataset)
        data = hf_dataset.to_pandas()
        
        if 'image_question' in data.columns:
            no_image_placeholder = 'NO_IMAGE_PLACEHOLDER_' + 'x' * 50
            
            def process_base64_image(base64_data):
                if pd.isna(base64_data) or not str(base64_data).strip() or len(str(base64_data).strip()) < 100:
                    return no_image_placeholder
                return str(base64_data)
            
            data['image'] = data['image_question'].apply(process_base64_image)
        
        return data

    def build_prompt(self, line):
        """构建物理竞赛prompt"""
        if isinstance(line, int):
            line = self.data.iloc[line]

        def safe_str(val):
            return "" if pd.isna(val) or val == '' else str(val)
        
        context = safe_str(line.get('context', ''))
        question = safe_str(line['question'])
        information = safe_str(line.get('information', ''))
        
        system_prompt = SYSTEM_PROMPTS_EN if self.language == 'en' else SYSTEM_PROMPTS_ZH
        formatted_prompt = system_prompt.replace('{context}', context).replace('{problem}', question).replace('{information}', information)
        
        msgs = []
        
        # 检查是否有真实的图像数据（排除占位符）
        image_val = str(line.get('image', '')).strip()
        
        if image_val and not image_val.startswith('NO_IMAGE_PLACEHOLDER_'):
            # 使用标准的VLMEvalKit图像处理流程
            if self.meta_only:
                tgt_path = toliststr(line['image_path']) if 'image_path' in line else []
            else:
                tgt_path = self.dump_image(line)
            
            if tgt_path and tgt_path != ['']:
                if isinstance(tgt_path, list):
                    msgs.extend([dict(type='image', value=p) for p in tgt_path])
                else:
                    msgs.append(dict(type='image', value=tgt_path))
        
        msgs.append(dict(type='text', value=formatted_prompt))
        
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        """评测函数"""
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data
        
        # 使用VLMEvalKit标准方式初始化judge模型
        judge_model = None
        if judge_kwargs.get('model') and judge_kwargs.get('model') != 'exact_matching':
            # 为物理题目设置合适的默认参数
            judge_kwargs.setdefault('timeout', 600)      # API级别超时时间（秒）
            judge_kwargs.setdefault('retry', 3)          # 重试次数
            judge_kwargs.setdefault('max_tokens', 4096)  # 限制输出长度，减少响应时间
            # judge_kwargs.setdefault('temperature', 0.0)  # 确保结果一致性
            judge_model = build_judge(**judge_kwargs)
            if judge_model and not judge_model.working():
                warnings.warn('Judge API不工作，跳过细粒度评测')
                judge_model = None
        
        fine_grained_total_score = 0.0
        coarse_grained_total_score = 0.0
        max_possible_score = 0.0
        detailed_results = []
        
        failed_count = 0
        for i in range(len(data)):
            row = data.iloc[i]
            
            result = self._evaluate_single_problem(judge_model, row, i, judge_kwargs)
            
            if result is None:
                failed_count += 1
                print(f"⚠️  题目 {i+1} 评测失败")
                continue
            
            fine_score = result['fine_grained_score']
            coarse_score = result['coarse_grained_score']
            item_points = result['item_total_points']
            
            fine_grained_total_score = round(fine_grained_total_score + fine_score, 2)
            coarse_grained_total_score = round(coarse_grained_total_score + coarse_score, 2)
            max_possible_score += item_points
            
            detailed_item = self._build_result_item(row, i, result)
            detailed_results.append(detailed_item)
        
        if failed_count > 0:
            print(f"⚠️  总计 {failed_count}/{len(data)} 题评测失败")
        
        max_possible_score = round(max_possible_score, 2)
        results = self._build_final_results(fine_grained_total_score, coarse_grained_total_score, max_possible_score)
        
        self._save_results(eval_file, results, detailed_results, data)
        self._print_summary(results)
        return results


    def _evaluate_single_problem(self, judge_model, row, index, judge_kwargs):
        """评测单个题目的函数"""
        # 提取字段
        prediction = str(row['prediction']).strip()
        ground_truth = self._safe_parse_json_field(row.get('answer', ''))
        answer_type = self._safe_parse_json_field(row.get('answer_type', 'Open-End'))
        unit = self._safe_parse_json_field(row.get('unit', ''))
        points = self._safe_parse_points_field(row.get('points', 0))
        marking = self._safe_parse_json_field(row.get('marking', ''))
        
        item_total_points = sum(points) if points else 0.0
        
        # 细粒度评测
        fine_grained_score, marking_detailed_scores = self._evaluate_fine_grained(
            prediction, marking, points, judge_model, row.get('question', '')
        )
        
        # 粗粒度评测
        coarse_grained_score, extracted_pred = self._evaluate_coarse_grained(
            prediction, ground_truth, answer_type, unit, points, 
            row.get('question', '')
        )
        
        # 最终得分取两者最大值
        final_score = max(fine_grained_score, coarse_grained_score)
        
        # 返回单题结果
        return {
            'index': index,
            'fine_grained_score': fine_grained_score,
            'coarse_grained_score': coarse_grained_score,
            'final_score': final_score,
            'extracted_pred': extracted_pred,
            'marking_detailed_scores': marking_detailed_scores,
            'item_total_points': item_total_points,
            'ground_truth': ground_truth,
            'answer_type': answer_type,
            'unit': unit,
            'points': points,
            'marking': marking,
            'prediction': prediction
        }

    def _evaluate_fine_grained(self, prediction, marking, points, judge_model, question):
        """细粒度评测 - 带重测机制"""
        if not marking or not judge_model:
            return 0.0, []
        
        # 检查是否有多套marking标准
        if self._has_multiple_marking_sets(marking):
            return self._evaluate_multiple_marking_sets(prediction, marking, points, judge_model, question)
            
        scoring_criteria = self._parse_marking_criteria(marking)
        max_possible_score = sum(points) if points else 0.0
        max_retries = 3  # 最大重测次数
        
        for attempt in range(max_retries + 1):
            scores = []
            detailed_scores = []
            
            for i, criterion in enumerate(scoring_criteria):
                score, response = self._evaluate_single_criterion(
                    prediction, criterion, judge_model, question, 
                    max_total_score=max_possible_score, 
                    current_attempt=attempt
                )
                scores.append(score)
                
                detailed_scores.append({
                    'marking_criterion': criterion['description'],
                    'score': round(score, 2),
                    'index': criterion['index'],
                    'attempt': attempt + 1,
                    'judge_response': response
                })
            
            total_score = sum(scores)
            
            if total_score <= max_possible_score or max_possible_score == 0:
                for detailed_score in detailed_scores:
                    detailed_score['retry_info'] = f"第{attempt + 1}次评测成功" if attempt > 0 else "首次评测成功"
                    detailed_score['final_success'] = True
                
                return round(total_score, 2), detailed_scores
            else:
                if attempt < max_retries:
                    continue  # 重试
                else:
                    # 强制调整
                    scale_factor = max_possible_score / total_score
                    adjusted_scores = [score * scale_factor for score in scores]
                    
                    for i, score in enumerate(adjusted_scores):
                        detailed_scores[i]['original_score'] = detailed_scores[i]['score']
                        detailed_scores[i]['score'] = round(score, 2)
                        detailed_scores[i]['forced_adjustment'] = True
                        detailed_scores[i]['scale_factor'] = round(scale_factor, 3)
                    
                    return round(sum(adjusted_scores), 2), detailed_scores
        
        return 0.0, []

    def _evaluate_coarse_grained(self, prediction, ground_truth, answer_type, unit, points, question):
        """粗粒度评测 - 基于physics_r1验证器的答案匹配"""
        extracted_pred = ""
        
        if ground_truth:
            # 使用physics_r1验证器
            total_score, total_point, extracted_preds, extracted_gts, scored_by_list = answer_tag_reward_fn_for_r1(
                prediction, ground_truth, problem=question, points=points, use_xverify=True, debug=False
            )
            
            extracted_pred = ", ".join([str(p) for p in extracted_preds if p])
            return round(total_point, 2), extracted_pred
        
        return 0.0, extracted_pred

    def _evaluate_single_criterion(self, prediction, criterion, judge_model, question, max_total_score=None, current_attempt=0):
        """使用judge模型评测单个marking标准"""
        
        # 构建总分限制提示
        total_score_warning = ""
        if max_total_score is not None and max_total_score > 0:
            total_score_warning = TOTAL_SCORE_WARNING_TEMPLATE.format(
                max_total_score=max_total_score, 
                current_attempt=current_attempt + 1
            )

        retry_warning = ""
        if current_attempt > 0:
            retry_warning = RETRY_WARNING_TEMPLATE

        # 使用统一的prompt模板
        prompt = JUDGE_GRADING_PROMPT_TEMPLATE.format(
            question=question,
            prediction=prediction,
            criterion_description=criterion['description'],
            total_score_warning=total_score_warning,
            retry_warning=retry_warning
        )
        
        start_time = time.time()
        response = judge_model.generate(prompt).strip()
        elapsed_time = time.time() - start_time
        
        score = self._extract_score_from_response(response)
        return score, response

    def _safe_parse_json_field(self, field_value):
        """安全解析JSON字段"""
        if pd.isna(field_value) or field_value == '':
            return []
        
        if isinstance(field_value, list):
            return field_value
        
        field_str = str(field_value).strip()
        if field_str.startswith('[') and field_str.endswith(']'):
            try:
                return json.loads(field_str)
            except json.JSONDecodeError:
                return [field_str]
        else:
            return [field_str] if field_str != 'nan' else []
    
    def _safe_parse_points_field(self, points_value):
        """安全解析points字段"""
        if pd.isna(points_value):
            return [0.0]
        
        if isinstance(points_value, list):
            return [float(p) for p in points_value if p is not None]
        
        if isinstance(points_value, (int, float)):
            return [float(points_value)]
        
        points_str = str(points_value).strip()
        if points_str.startswith('[') and points_str.endswith(']'):
            try:
                parsed = json.loads(points_str)
                return [float(p) for p in parsed if p is not None]
            except (json.JSONDecodeError, ValueError):
                pass
        
        try:
            return [float(points_str)]
        except ValueError:
            return [0.0]

    def _has_valid_marking(self, marking):
        """检查marking是否包含有效的评分标准"""
        if not marking:
            return False
        
        if not isinstance(marking, list):
            return False
        
        if len(marking) == 0:
            return False
        
        for item in marking:
            if item is None:
                continue
            
            if isinstance(item, list):
                if len(item) > 0:
                    return True
            elif isinstance(item, str):
                stripped = item.strip()
                if stripped and stripped.lower() not in ['', 'nan', 'none', 'null']:
                    return True
            else:
                return True
        
        return False

    def _has_multiple_marking_sets(self, marking):
        """检查是否有多套marking标准"""
        if not marking or len(marking) == 0:
            return False
        
        # 如果第一个元素是列表，则认为有多套标准
        return isinstance(marking[0], list)
    
    def _evaluate_multiple_marking_sets(self, prediction, marking_sets, points, judge_model, question):
        """评测多套marking标准，取最高分"""
        best_score = 0.0
        best_detailed_scores = []
        
        for set_idx, marking_set in enumerate(marking_sets):
            score, detailed_scores = self._evaluate_single_marking_set(
                prediction, marking_set, points, judge_model, question
            )
            
            # 更新最佳分数
            if score > best_score:
                best_score = score
                best_detailed_scores = detailed_scores
                # 在最佳详细得分中添加标记
                for detailed_score in best_detailed_scores:
                    detailed_score['best_marking_set'] = set_idx + 1
        
        return round(best_score, 2), best_detailed_scores
    
    def _evaluate_single_marking_set(self, prediction, marking, points, judge_model, question):
        """评测单套marking标准"""
        scoring_criteria = self._parse_marking_criteria(marking)
        max_possible_score = sum(points) if points else 0.0
        
        scores = []
        detailed_scores = []
        
        for criterion in scoring_criteria:
            score, response = self._evaluate_single_criterion(
                prediction, criterion, judge_model, question, 
                max_total_score=max_possible_score, 
                current_attempt=0
            )
            scores.append(score)
            
            # 保存每个marking的详细得分
            detailed_scores.append({
                'marking_criterion': criterion['description'],
                'score': round(score, 2),
                'index': criterion['index'],
                'judge_response': response
            })
        
        total_score = sum(scores)
        
        # 如果超过最大分数，按比例调整
        if total_score > max_possible_score and max_possible_score > 0:
            scale_factor = max_possible_score / total_score
            total_score = max_possible_score
            for detailed_score in detailed_scores:
                detailed_score['original_score'] = detailed_score['score']
                detailed_score['score'] = round(detailed_score['score'] * scale_factor, 2)
                detailed_score['scaled'] = True
        
        return round(total_score, 2), detailed_scores

    def _parse_marking_criteria(self, marking_list):
        """解析marking评分标准"""
        criteria = []
        if not marking_list:
            return criteria
        
        # 处理嵌套列表的情况
        flattened_marking = []
        for item in marking_list:
            if isinstance(item, list):
                flattened_marking.extend(item)
            else:
                flattened_marking.append(item)
        
        for i, marking_criterion in enumerate(flattened_marking):
            if marking_criterion and str(marking_criterion).strip():
                criteria.append({
                    'description': str(marking_criterion).strip(),
                    'index': i
                })
        
        return criteria

    def _extract_score_from_response(self, response):
        """从模型响应中提取分数"""
        if not response:
            return 0.0
            
        response = response.strip()
        
        # 使用boxed格式提取分数
        boxed_patterns = [
            r'\\boxed\{([^}]+)\}',
            r'boxed\{([^}]+)\}',
        ]
        
        for pattern in boxed_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in reversed(matches):
                match = match.strip()
                if match:
                    try:
                        score = float(match)
                        return round(score, 2)
                    except ValueError:
                        nums = re.findall(r'\d+\.?\d*', match)
                        if nums:
                            try:
                                score = float(nums[-1])
                                return round(score, 2)
                            except ValueError:
                                continue
        
        # 查找数字
        all_numbers = re.findall(r'[0-9]*\.?[0-9]+', response)
        if all_numbers:
            try:
                score = float(all_numbers[-1])
                return round(score, 2)
            except ValueError:
                pass
        
        return 0.0


    def _build_result_item(self, row, index, result):
        """构建详细结果项"""
        has_marking = result['marking'] and len(result['marking']) > 0 and self._has_valid_marking(result['marking'])
        earned_points = max(result['fine_grained_score'], result['coarse_grained_score'])
        
        return {
            "id": str(row.get('id', f"{self.dataset_name}_{index+1}")),
            "context": str(row.get('context', '')).strip(),
            "question": str(row.get('question', '')).strip(),
            "solution": str(row.get('solution', '')).strip(),
            "marking": result['marking'] if result['marking'] else [],
            "marking_detailed_scores": result['marking_detailed_scores'],
            "answer": [f"\\boxed{{{ans}}}" for ans in result['ground_truth']] if result['ground_truth'] else [''],
            "answer_type": result['answer_type'] if result['answer_type'] else ['Open-End'],
            "unit": result['unit'] if result['unit'] else [''],
            "points": result['points'] if result['points'] else [0.0],
            "modality": str(row.get('modality', 'text')).strip(),
            "field": str(row.get('field', '')).strip(),
            "source": self.dataset_name,
            "test_result": str(result['prediction']),
            "test_answer": [f"\\boxed{{{ans.strip()}}}" for ans in result['extracted_pred'].split(", ") if ans.strip()] if result['extracted_pred'] else [''],
            "fine_grained_score": result['fine_grained_score'],
            "coarse_grained_score": result['coarse_grained_score'],
            "earned_points": earned_points
        }

    def _build_final_results(self, fine_total, coarse_total, max_score):
        """构建最终结果"""
        fine_rate = round((fine_total / max_score * 100), 2) if max_score > 0 else 0.0
        coarse_rate = round((coarse_total / max_score * 100), 2) if max_score > 0 else 0.0
        
        return {
            'fine_grained_total_score': fine_total,
            'fine_grained_score_rate': fine_rate,
            'coarse_grained_total_score': coarse_total,
            'coarse_grained_score_rate': coarse_rate,
            'max_possible_score': max_score,
            'total_score': fine_total,
            'score_rate': fine_rate,
        }

    def _save_results(self, eval_file, results, detailed_results, data):
        """保存评测结果"""
        score_file = eval_file.replace('.xlsx', '_score.json')
        detailed_file = eval_file.replace('.xlsx', '_detailed_results.json')
        detailed_xlsx_file = eval_file.replace('.xlsx', '_detailed.xlsx')
        
        dump(results, score_file)
        dump(detailed_results, detailed_file)
        
        eval_data_with_results = data.copy()
        eval_data_with_results['fine_grained_score'] = [r['fine_grained_score'] for r in detailed_results]
        eval_data_with_results['coarse_grained_score'] = [r['coarse_grained_score'] for r in detailed_results]
        eval_data_with_results['earned_points'] = [r['earned_points'] for r in detailed_results]
        eval_data_with_results['marking_detailed_scores'] = [
            json.dumps(r['marking_detailed_scores'], ensure_ascii=False) if r['marking_detailed_scores'] else '[]' 
            for r in detailed_results
        ]
        dump(eval_data_with_results, detailed_xlsx_file)

    def _print_summary(self, results):
        """打印评测总结"""
        print(f"✅ {self.dataset_name} 评估完成！")
        print(f"🏆 总体得分: {results['total_score']:.2f} / {results['max_possible_score']:.2f} ({results['score_rate']:.2f}%)")
        print(f"📊 细粒度评测得分: {results['fine_grained_total_score']:.2f} ({results['fine_grained_score_rate']:.2f}%)")
        print(f"🎯 粗粒度评测得分: {results['coarse_grained_total_score']:.2f} ({results['coarse_grained_score_rate']:.2f}%)")
        print(f"💾 详细结果已保存")