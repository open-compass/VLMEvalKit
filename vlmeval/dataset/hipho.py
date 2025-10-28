# -*- coding: utf-8 -*-
import os
import re
import json
import pandas as pd
import numpy as np
import warnings
import time
import threading
import datetime
from functools import partial
import multiprocessing as mp

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from .utils.hipho_verifier import grade, extract_boxed_answer, get_answer_str, answer_tag_reward_fn_for_r1
from .utils.prompt_inference import SYSTEM_PROMPTS_EN, SYSTEM_PROMPTS_ZH
from ..smp import *
from ..smp.file import get_intermediate_file_path
from ..utils import track_progress_rich

# 线程锁用于同步输出
output_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """线程安全的打印函数"""
    with output_lock:
        print(*args, **kwargs)

class LogBuffer:
    """日志缓存类，用于收集单个任务的所有日志"""
    def __init__(self, task_id):
        self.task_id = task_id
        self.logs = []
        self.start_time = datetime.datetime.now()
    
    def log(self, message):
        """添加日志消息"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.logs.append(f"[{timestamp}] [{self.task_id}] {message}")
    
    def flush(self):
        """一次性输出所有缓存的日志"""
        with output_lock:
            for log in self.logs:
                print(log)
            print()


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
    
    # 数据集URL映射 - 指向HuggingFace数据集
    DATASET_URL = {
        'IPhO_2024': 'https://huggingface.co/datasets/haiyuanwan/HiPhO',
        'IPhO_2025': 'https://huggingface.co/datasets/haiyuanwan/HiPhO',
        'EuPhO_2024': 'https://huggingface.co/datasets/haiyuanwan/HiPhO',
        'EuPhO_2025': 'https://huggingface.co/datasets/haiyuanwan/HiPhO',
        'APhO_2025': 'https://huggingface.co/datasets/haiyuanwan/HiPhO',
        'PanPhO_2024': 'https://huggingface.co/datasets/haiyuanwan/HiPhO',
        'PanPhO_2025': 'https://huggingface.co/datasets/haiyuanwan/HiPhO',
        'NBPhO_2024': 'https://huggingface.co/datasets/haiyuanwan/HiPhO',
        'NBPhO_2025': 'https://huggingface.co/datasets/haiyuanwan/HiPhO',
        'F_MA_2024': 'https://huggingface.co/datasets/haiyuanwan/HiPhO',
        'F_MA_2025': 'https://huggingface.co/datasets/haiyuanwan/HiPhO',
        'PanMechanics_2024': 'https://huggingface.co/datasets/haiyuanwan/HiPhO',
        'PanMechanics_2025': 'https://huggingface.co/datasets/haiyuanwan/HiPhO',
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
        """从HuggingFace加载多split数据集"""
        from datasets import load_dataset
        
        safe_print(f"从HuggingFace加载数据集: haiyuanwan/HiPhO, split: {dataset}")
        
        # 从HuggingFace加载指定split的数据集
        hf_dataset = load_dataset('haiyuanwan/HiPhO', split=dataset)
        safe_print(f"✅ 成功加载数据集，共 {len(hf_dataset)} 行数据")
        
        # 转换为DataFrame
        data = hf_dataset.to_pandas()
        
        # 确保index列存在
        if 'index' not in data.columns:
            data['index'] = range(len(data))
        
        # 处理图像数据 - HuggingFace数据集中image_question包含base64数据
        if 'image_question' in data.columns:
            safe_print(f"🖼️  发现image_question列，处理base64图像数据")
            
            # 使用长度超过64的占位符来表示无图像
            no_image_placeholder = 'NO_IMAGE_PLACEHOLDER_' + 'x' * 50
            
            def process_hf_base64_image(base64_data):
                if pd.isna(base64_data) or not str(base64_data).strip() or len(str(base64_data).strip()) < 100:
                    return no_image_placeholder
                # HuggingFace中的image_question包含base64数据，直接返回用于VLMEvalKit处理
                return str(base64_data)
            
            # 创建image字段映射base64数据
            data['image'] = data['image_question'].apply(process_hf_base64_image)
            
            # 统计图像数量
            image_count = len(data[~data['image'].str.startswith('NO_IMAGE_PLACEHOLDER_')])
            safe_print(f"📈 图像数据统计: {image_count}/{len(data)} 条记录包含图像")
        
        safe_print(f"📊 数据列名: {list(data.columns)}")
        safe_print(f"✅ 数据加载完成")
        return data

    def build_prompt(self, line):
        """构建输入prompt，处理有图和无图两种情况，使用物理竞赛专业prompt"""
        if isinstance(line, int):
            line_idx = line
            line = self.data.iloc[line]
            safe_print(f"📝 构建第 {line_idx+1} 题的prompt")
        else:
            safe_print(f"📝 构建prompt (使用传入的line对象)")

        # 从数据中获取各个字段，安全处理可能为NaN的字段
        def safe_str(val):
            return "" if pd.isna(val) or val == '' else str(val)
        
        context = safe_str(line.get('context', ''))
        question = safe_str(line['question'])
        information = safe_str(line.get('information', ''))
        
        safe_print(f"   📋 题目信息:")
        safe_print(f"      - context长度: {len(context)} 字符")
        safe_print(f"      - question长度: {len(question)} 字符")
        safe_print(f"      - information长度: {len(information)} 字符")
        safe_print(f"      - 使用语言: {self.language}")
        
        # 选择语言对应的prompt模板
        system_prompt = SYSTEM_PROMPTS_EN if self.language == 'en' else SYSTEM_PROMPTS_ZH
        # 使用字符串替换而不是format，避免花括号冲突
        formatted_prompt = system_prompt.replace('{context}', context).replace('{problem}', question).replace('{information}', information)
        
        safe_print(f"   🔧 构建的prompt长度: {len(formatted_prompt)} 字符")
        
        msgs = []
        
        # 检查是否有图像数据（base64或路径）
        image_val = str(line.get('image', '')).strip()
        safe_print(f"   🖼️  图像检查: {'有图像' if image_val and not image_val.startswith('NO_IMAGE_PLACEHOLDER_') else '无图像'}")
        
        if image_val and not image_val.startswith('NO_IMAGE_PLACEHOLDER_'):
            # 检查是否是base64数据
            if len(image_val) > 1000 and not image_val.startswith('/'):  # base64数据通常很长且不以/开头
                safe_print(f"      - 检测到base64图像数据 (长度: {len(image_val)})")
                # 直接使用base64数据，VLMEvalKit框架会处理
                msgs.append(dict(type='image', value=image_val))
                safe_print(f"      - 添加了base64图像到消息列表")
            else:
                safe_print(f"      - 图像路径: {str(image_val)[:50]}{'...' if len(str(image_val)) > 50 else ''}")
                # 有图像路径的情况 - 使用框架的标准图像处理
                if self.meta_only:
                    tgt_path = toliststr(line['image_path']) if 'image_path' in line else []
                    safe_print(f"      - meta_only模式，图像路径: {tgt_path}")
                else:
                    safe_print(f"      - 开始dump图像...")
                    tgt_path = self.dump_image(line)
                    safe_print(f"      - dump结果: {tgt_path}")
                
                if tgt_path and tgt_path != ['']:
                    if isinstance(tgt_path, list):
                        msgs.extend([dict(type='image', value=p) for p in tgt_path])
                        safe_print(f"      - 添加了 {len(tgt_path)} 个图像到消息列表")
                    else:
                        msgs.append(dict(type='image', value=tgt_path))
                        safe_print(f"      - 添加了 1 个图像到消息列表")
        
        # 添加格式化的物理竞赛prompt
        msgs.append(dict(type='text', value=formatted_prompt))
        
        safe_print(f"   ✅ prompt构建完成，总消息数: {len(msgs)} (图像: {len([m for m in msgs if m['type'] == 'image'])}, 文本: {len([m for m in msgs if m['type'] == 'text'])})")
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        """评测函数 - 统一的粗细粒度评测"""
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data
        
        # 获取并行参数
        nproc = judge_kwargs.pop('nproc', 4)
        safe_print(f"🔧 设置并行进程数: {nproc}")
        
        # 初始化judge模型（用于细粒度评测）
        judge_model = self._init_judge_model(judge_kwargs)
        
        safe_print(f"📊 开始并行评测，共{len(data)}题...")
        
        # 构建任务列表
        tasks = []
        indices = []
        for i in range(len(data)):
            row = data.iloc[i]
            task_kwargs = judge_kwargs.copy()
            task = (judge_model, row, i, task_kwargs)
            tasks.append(task)
            indices.append(i)
        
        # 设置中间结果保存文件
        tmp_file = eval_file.replace('.xlsx', '_parallel_tmp.pkl')
        
        # 并行评测所有题目
        parallel_results = track_progress_rich(
            self._evaluate_single_problem,
            tasks,
            nproc=nproc,
            chunksize=max(1, nproc//2),
            keys=indices,
            save=tmp_file
        )
        
        safe_print(f"✅ 并行评测完成，开始汇总结果...")
        
        # 汇总并行结果
        fine_grained_total_score = 0.0
        coarse_grained_total_score = 0.0
        max_possible_score = 0.0
        detailed_results = []
        
        for i, result in enumerate(parallel_results):
            if result is None:
                safe_print(f"⚠️  题目 {i+1} 评测失败，跳过")
                continue
                
            row = data.iloc[i]
            fine_score = result['fine_grained_score']
            coarse_score = result['coarse_grained_score']
            item_points = result['item_total_points']
            
            # 累加得分
            fine_grained_total_score = round(fine_grained_total_score + fine_score, 2)
            coarse_grained_total_score = round(coarse_grained_total_score + coarse_score, 2)
            max_possible_score += item_points
            
            # 构建详细结果
            detailed_item = self._build_result_item_from_parallel_result(row, i, result)
            detailed_results.append(detailed_item)
            
            if (i + 1) % 10 == 0 or i == len(parallel_results) - 1:
                safe_print(f"📊 汇总进度 {i+1}/{len(parallel_results)}: 细粒度={fine_grained_total_score:.2f}, 粗粒度={coarse_grained_total_score:.2f}")
        
        # 计算最终结果
        max_possible_score = round(max_possible_score, 2)
        results = self._build_final_results(fine_grained_total_score, coarse_grained_total_score, max_possible_score, len(data))
        
        # 保存结果
        self._save_results(eval_file, results, detailed_results, data)
        
        # 清理临时文件
        try:
            if osp.exists(tmp_file):
                os.remove(tmp_file)
        except Exception as e:
            safe_print(f"⚠️  清理临时文件失败: {e}")
        
        # 打印总结并返回DataFrame格式结果
        self._print_summary(results)
        return results

    def _init_judge_model(self, judge_kwargs):
        """初始化judge模型"""
        judge_model_name = judge_kwargs.get('model', None)
        
        if judge_model_name and judge_model_name != 'exact_matching':
            if gpt_key_set():
                try:
                    model_kwargs = {
                        'model': judge_model_name,
                        'timeout': 600,  # 设置600秒API级别超时
                        'retry': 3,      # 设置重试次数
                        'max_tokens': 4096,  # 限制输出长度，减少响应时间
                        'verbose': False,  # 🔥 关键修复：关闭verbose模式，避免打印完整响应
                        **{k: v for k, v in judge_kwargs.items() if k not in ['model', 'nproc']}
                    }
                    test_model = build_judge(**model_kwargs)
                    if test_model.working():
                        safe_print(f"🤖 使用Judge模型: {judge_model_name} (timeout=600s, retry=3)")
                        return test_model
                    else:
                        warnings.warn('Judge API不工作，跳过细粒度评测')
                except Exception as e:
                    warnings.warn(f'模型初始化失败: {e}，跳过细粒度评测')
            else:
                warnings.warn('API_KEY无效，跳过细粒度评测')
        
        return None

    def _evaluate_single_problem(self, judge_model, row, index, judge_kwargs):
        """评测单个题目的函数（用于并行调用）"""
        task_id = f"题目{index + 1}"
        log_buffer = LogBuffer(task_id)
        
        try:
            log_buffer.log(f"📖 开始评测 - ID: {row.get('id', 'N/A')}")
            
            # 提取字段
            prediction = str(row['prediction']).strip()
            ground_truth = self._safe_parse_json_field(row.get('answer', ''))
            answer_type = self._safe_parse_json_field(row.get('answer_type', 'Open-End'))
            unit = self._safe_parse_json_field(row.get('unit', ''))
            points = self._safe_parse_points_field(row.get('points', 0))
            marking = self._safe_parse_json_field(row.get('marking', ''))
            
            item_total_points = sum(points) if points else 0.0
            log_buffer.log(f"   - 本题总分: {item_total_points}")
            
            # 细粒度评测
            log_buffer.log(f"🔍 开始细粒度评测...")
            fine_grained_score, marking_detailed_scores = self._evaluate_fine_grained_with_buffer(
                prediction, marking, points, judge_model, row.get('question', ''), log_buffer
            )
            log_buffer.log(f"✅ 细粒度得分: {fine_grained_score}")
            
            # 粗粒度评测
            log_buffer.log(f"🎯 开始粗粒度评测...")
            coarse_grained_score, extracted_pred = self._evaluate_coarse_grained_with_buffer(
                prediction, ground_truth, answer_type, unit, points, 
                row.get('question', ''), log_buffer
            )
            log_buffer.log(f"✅ 粗粒度得分: {coarse_grained_score}")
            log_buffer.log(f"📤 提取的预测答案: {extracted_pred}")
            
            # 最终得分取两者最大值
            final_score = max(fine_grained_score, coarse_grained_score)
            log_buffer.log(f"🏆 最终得分（取最大值）: {final_score} = max({fine_grained_score}, {coarse_grained_score})")
            
            # 返回单题结果
            result = {
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
            
            log_buffer.log(f"✅ 评测完成，最终得分: {final_score}")
            log_buffer.flush()
            return result
            
        except Exception as e:
            log_buffer.log(f"❌ 评测失败: {e}")
            import traceback
            log_buffer.log(f"📄 错误详情: {traceback.format_exc()}")
            log_buffer.flush()
            return None

    def _evaluate_fine_grained_with_buffer(self, prediction, marking, points, judge_model, question, log_buffer):
        """细粒度评测 - 带重测机制（带日志缓存版本）"""
        log_buffer.log(f"   🔍 细粒度评测开始")
        log_buffer.log(f"      - marking数量: {len(marking) if marking else 0}")
        log_buffer.log(f"      - judge_model: {'有' if judge_model else '无'}")
        
        if not marking or not judge_model:
            log_buffer.log(f"   ⚠️  跳过细粒度评测：{'无marking标准' if not marking else '无judge模型'}")
            return 0.0, []
        
        # 检查是否有多套marking标准
        if self._has_multiple_marking_sets(marking):
            log_buffer.log(f"   📋 发现多套marking标准，使用最佳得分策略")
            return self._evaluate_multiple_marking_sets_with_buffer(prediction, marking, points, judge_model, question, log_buffer)
            
        scoring_criteria = self._parse_marking_criteria(marking)
        max_possible_score = sum(points) if points else 0.0
        max_retries = 3  # 最大重测次数
        
        log_buffer.log(f"   📊 评测配置: {len(scoring_criteria)}个标准，最大总分: {max_possible_score}")
        
        for attempt in range(max_retries + 1):
            log_buffer.log(f"   🔄 开始第 {attempt + 1} 次评测")
            scores = []
            detailed_scores = []
            
            for i, criterion in enumerate(scoring_criteria):
                score, response = self._evaluate_single_criterion_with_buffer(
                    prediction, criterion, judge_model, question, 
                    max_total_score=max_possible_score, 
                    current_attempt=attempt,
                    log_buffer=log_buffer
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
            log_buffer.log(f"   📊 第 {attempt + 1} 次评测总分: {total_score}")
            
            if total_score <= max_possible_score or max_possible_score == 0:
                for detailed_score in detailed_scores:
                    detailed_score['retry_info'] = f"第{attempt + 1}次评测成功" if attempt > 0 else "首次评测成功"
                    detailed_score['final_success'] = True
                
                log_buffer.log(f"✅ 评测成功，总分 {total_score:.2f}")
                return round(total_score, 2), detailed_scores
            else:
                if attempt < max_retries:
                    log_buffer.log(f"⚠️  评测超分: {total_score:.2f} > {max_possible_score:.2f}，重测...")
                else:
                    # 强制调整
                    scale_factor = max_possible_score / total_score
                    adjusted_scores = [score * scale_factor for score in scores]
                    
                    for i, score in enumerate(adjusted_scores):
                        detailed_scores[i]['original_score'] = detailed_scores[i]['score']
                        detailed_scores[i]['score'] = round(score, 2)
                        detailed_scores[i]['forced_adjustment'] = True
                        detailed_scores[i]['scale_factor'] = round(scale_factor, 3)
                    
                    log_buffer.log(f"📊 强制调整分数，系数: {scale_factor:.3f}")
                    return round(sum(adjusted_scores), 2), detailed_scores
        
        return 0.0, []

    def _evaluate_coarse_grained_with_buffer(self, prediction, ground_truth, answer_type, unit, points, question, log_buffer):
        """粗粒度评测 - 基于physics_r1验证器的答案匹配"""
        log_buffer.log(f"   🎯 粗粒度评测开始")
        
        extracted_pred = ""
        
        if ground_truth:
            log_buffer.log(f"      ✅ 有标准答案，开始physics_r1验证")
            try:
                # 使用physics_r1验证器
                total_score, total_point, extracted_preds, extracted_gts, scored_by_list = answer_tag_reward_fn_for_r1(
                    prediction, ground_truth, problem=question, points=points, use_xverify=True, debug=False
                )
                
                extracted_pred = ", ".join([str(p) for p in extracted_preds if p])
                log_buffer.log(f"      📊 physics_r1验证得分: {total_point}")
                log_buffer.log(f"      📝 提取的答案: {extracted_pred}")
                
                return round(total_point, 2), extracted_pred
                
            except Exception as e:
                log_buffer.log(f"      ⚠️  physics_r1验证失败: {e}，使用简单匹配")
                # 回退到简单匹配
                simple_score = self._simple_answer_matching(prediction, ground_truth, points)
                extracted_pred = self._extract_prediction_for_display(prediction)
                return round(simple_score, 2), extracted_pred
        
        log_buffer.log(f"      ⚠️  无标准答案，返回0分")
        return 0.0, extracted_pred

    def _evaluate_single_criterion_with_buffer(self, prediction, criterion, judge_model, question, max_total_score=None, current_attempt=0, log_buffer=None):
        """使用judge模型评测单个marking标准"""
        log_buffer.log(f"         🤖 调用Judge模型评测标准")
        
        # 构建总分限制提示
        total_score_warning = ""
        if max_total_score is not None and max_total_score > 0:
            total_score_warning = f"""
⚠️  IMPORTANT TOTAL SCORE CONSTRAINT:
- This question has a maximum total score of {max_total_score} points
- ALL marking criteria scores combined MUST NOT exceed {max_total_score} points
- You are evaluating ONE criterion among multiple criteria for this question
- Be conservative in your scoring to ensure the total doesn't exceed the limit
- This is attempt #{current_attempt + 1} of evaluation"""

        retry_warning = ""
        if current_attempt > 0:
            retry_warning = f"""
🔄 RETRY NOTICE:
- Previous attempt(s) resulted in total score exceeding the maximum
- Please be more conservative in your scoring
- Focus on strict adherence to the criterion requirements"""

        prompt = f"""You are an expert physics competition grader. Evaluate the student's solution against the specific grading criterion.

PHYSICS PROBLEM:
{question}

STUDENT'S SOLUTION:
{prediction}

GRADING CRITERION:
{criterion['description']}{total_score_warning}{retry_warning}

INSTRUCTIONS:
1. Carefully analyze the student's solution for physics concepts, mathematical derivations, and calculations.
2. Compare the solution against the specific grading criterion provided.
3. Award points strictly according to the criterion, including partial credit when specified.
4. Consider both conceptual understanding and technical accuracy.
5. BE CONSERVATIVE - remember this is one of multiple criteria being evaluated simultaneously.

SCORING FORMAT:
- Read the grading criterion carefully to understand the maximum points and conditions for partial credit
- Evaluate whether the student's solution meets the full criteria, partial criteria, or no criteria
- Output your score using the exact format: \\boxed{{score}}
- The score should be a number (e.g., 0.4, 0.2, 0.1, 0.0)

CRITICAL REQUIREMENTS:
- You MUST output your final score in the format: \\boxed{{score}}
- The score must be a single number only (no text inside the boxed)
- Do not include explanations after the boxed score
- Ensure your score follows the point allocation in the grading criterion
- BE CONSERVATIVE to avoid exceeding the total score limit

Example outputs:
- \\boxed{{0.4}} (for full credit)
- \\boxed{{0.1}} (for partial credit)  
- \\boxed{{0.0}} (for no credit)

⚠️ CRITICAL INSTRUCTION: 
- Output ONLY: \\boxed{{score}}
- NO explanations, NO analysis, NO reasoning
- Just the number in the exact format \\boxed{{score}}
- Any other text will result in AUTOMATIC REJECTION

RESPOND WITH ONLY THE BOXED SCORE:"""
        
        try:
            start_time = time.time()
            response = judge_model.generate(prompt).strip()
            elapsed_time = time.time() - start_time
            
            log_buffer.log(f"         ⏱️  响应耗时: {elapsed_time:.2f}秒")
            
            score = self._extract_score_from_response(response)
            log_buffer.log(f"         🔍 提取的分数: {score}")
            return score, response
        except Exception as e:
            log_buffer.log(f"         ❌ Judge模型调用失败: {e}")
            return 0.0, f"Judge模型调用失败: {str(e)}"

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
    
    def _evaluate_multiple_marking_sets_with_buffer(self, prediction, marking_sets, points, judge_model, question, log_buffer):
        """评测多套marking标准，取最高分（带日志缓存版本）"""
        log_buffer.log(f"   📋 开始评测多套marking标准，共{len(marking_sets)}套")
        
        best_score = 0.0
        best_detailed_scores = []
        
        for set_idx, marking_set in enumerate(marking_sets):
            log_buffer.log(f"   🔄 评测第{set_idx + 1}套marking标准")
            score, detailed_scores = self._evaluate_single_marking_set_with_buffer(
                prediction, marking_set, points, judge_model, question, log_buffer
            )
            log_buffer.log(f"      📊 第{set_idx + 1}套得分: {score}")
            
            # 更新最佳分数
            if score > best_score:
                best_score = score
                best_detailed_scores = detailed_scores
                # 在最佳详细得分中添加标记
                for detailed_score in best_detailed_scores:
                    detailed_score['best_marking_set'] = set_idx + 1
                log_buffer.log(f"      ✅ 更新最佳得分: {best_score} (来自第{set_idx + 1}套)")
        
        log_buffer.log(f"   🏆 多套marking评测完成，最佳得分: {best_score}")
        return round(best_score, 2), best_detailed_scores
    
    def _evaluate_single_marking_set_with_buffer(self, prediction, marking, points, judge_model, question, log_buffer):
        """评测单套marking标准（带日志缓存版本）"""
        scoring_criteria = self._parse_marking_criteria(marking)
        max_possible_score = sum(points) if points else 0.0
        
        scores = []
        detailed_scores = []
        
        for criterion in scoring_criteria:
            score, response = self._evaluate_single_criterion_with_buffer(
                prediction, criterion, judge_model, question, 
                max_total_score=max_possible_score, 
                current_attempt=0,
                log_buffer=log_buffer
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

    def _simple_answer_matching(self, prediction, answer_list, points_list):
        """简单的答案匹配（回退方案）"""
        total_score = 0.0
        
        for gt, points in zip(answer_list, points_list):
            if gt and gt.strip():
                if str(gt).strip().lower() in prediction.lower():
                    total_score += points
        
        return total_score
    
    def _extract_prediction_for_display(self, prediction, num_answers=10):
        """提取预测答案用于显示"""
        try:
            extracted_answers = get_answer_str(prediction, return_origin=False, num_answers=num_answers)
            valid_answers = []
            
            for ans in extracted_answers:
                if ans and ans.strip():
                    cleaned_ans = ' '.join(ans.strip().replace('\n', ' ').replace('\r', ' ').split())
                    if cleaned_ans:
                        valid_answers.append(cleaned_ans)
            
            return ", ".join(valid_answers) if valid_answers else ""
        except Exception:
            try:
                extracted = extract_boxed_answer(prediction)
                if extracted and extracted.strip():
                    cleaned = ' '.join(extracted.strip().replace('\n', ' ').replace('\r', ' ').split())
                    return cleaned if cleaned else ""
            except Exception:
                pass
            return ""

    def _build_result_item_from_parallel_result(self, row, index, parallel_result):
        """从并行结果构建详细结果项"""
        has_marking = parallel_result['marking'] and len(parallel_result['marking']) > 0 and self._has_valid_marking(parallel_result['marking'])
        earned_points = max(parallel_result['fine_grained_score'], parallel_result['coarse_grained_score'])
        
        return {
            "id": str(row.get('id', f"{self.dataset_name}_{index+1}")),
            "context": str(row.get('context', '')).strip(),
            "question": str(row.get('question', '')).strip(),
            "solution": str(row.get('solution', '')).strip(),
            "marking": parallel_result['marking'] if parallel_result['marking'] else [],
            "marking_detailed_scores": parallel_result['marking_detailed_scores'],
            "answer": [f"\\boxed{{{ans}}}" for ans in parallel_result['ground_truth']] if parallel_result['ground_truth'] else [''],
            "answer_type": parallel_result['answer_type'] if parallel_result['answer_type'] else ['Open-End'],
            "unit": parallel_result['unit'] if parallel_result['unit'] else [''],
            "points": parallel_result['points'] if parallel_result['points'] else [0.0],
            "modality": str(row.get('modality', 'text')).strip(),
            "field": str(row.get('field', '')).strip(),
            "source": self.dataset_name,
            "test_result": str(parallel_result['prediction']),
            "test_answer": [f"\\boxed{{{ans.strip()}}}" for ans in parallel_result['extracted_pred'].split(", ") if ans.strip()] if parallel_result['extracted_pred'] else [''],
            "fine_grained_score": parallel_result['fine_grained_score'],
            "coarse_grained_score": parallel_result['coarse_grained_score'],
            "earned_points": earned_points
        }

    def _build_final_results(self, fine_total, coarse_total, max_score, total_count):
        """构建最终结果"""
        fine_rate = round((fine_total / max_score * 100), 2) if max_score > 0 else 0.0
        coarse_rate = round((coarse_total / max_score * 100), 2) if max_score > 0 else 0.0
        
        return {
            'fine_grained_total_score': fine_total,
            'fine_grained_score_rate': fine_rate,
            'fine_grained_count': total_count,  # 添加缺少的字段
            'coarse_grained_total_score': coarse_total,
            'coarse_grained_score_rate': coarse_rate,
            'coarse_grained_count': total_count,  # 添加缺少的字段
            'max_possible_score': max_score,
            'total_count': total_count,
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
        
        try:
            eval_data_with_results = data.copy()
            eval_data_with_results['fine_grained_score'] = [r['fine_grained_score'] for r in detailed_results]
            eval_data_with_results['coarse_grained_score'] = [r['coarse_grained_score'] for r in detailed_results]
            eval_data_with_results['earned_points'] = [r['earned_points'] for r in detailed_results]
            eval_data_with_results['marking_detailed_scores'] = [
                json.dumps(r['marking_detailed_scores'], ensure_ascii=False) if r['marking_detailed_scores'] else '[]' 
                for r in detailed_results
            ]
            dump(eval_data_with_results, detailed_xlsx_file)
        except Exception as e:
            safe_print(f"⚠️  保存详细Excel文件失败: {e}")

    def _print_summary(self, results):
        """打印评测总结"""
        safe_print(f"✅ HiPhO数据集评估完成！")
        safe_print(f"🏆 总体得分: {results['total_score']:.2f} / {results['max_possible_score']:.2f} ({results['score_rate']:.2f}%)")
        safe_print(f"📊 细粒度评测: {results['fine_grained_count']}题，得分 {results['fine_grained_total_score']:.2f} ({results['fine_grained_score_rate']:.2f}%)")
        safe_print(f"🎯 粗粒度评测: {results['coarse_grained_count']}题，得分 {results['coarse_grained_total_score']:.2f} ({results['coarse_grained_score_rate']:.2f}%)")
        safe_print(f"💾 详细结果已保存")