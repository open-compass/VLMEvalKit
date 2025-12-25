import asyncio
import json
import time
import threading
import pandas as pd
import math
from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Any, Literal
from pathlib import Path
from tabulate import tabulate
from functools import partial
from enum import Enum

from vlmeval.smp import load, dump, get_logger
from vlmeval.smp.log import setup_subprocess_logger

logger = get_logger(__name__)

FAIL_MSG = 'Failed to obtain answer via API.'
DatasetType = Literal["image", "video", "mt"]


def _eval_subprocess_target(
    dataset_obj,
    result_file: str,
    judge_kwargs: dict,
    log_file: str,
):
    """Evaluate function in child processes."""
    setup_subprocess_logger(log_file)
    logger.info(f"🔔 [Eval Start] {dataset_obj.dataset_name}")

    try:
        # 调用 evaluate 方法
        with open(log_file, 'a') as f:
            with redirect_stdout(f), redirect_stderr(f):
                result = dataset_obj.evaluate(result_file, **judge_kwargs)

        # 序列化结果（DataFrame 转 dict）
        if isinstance(result, pd.DataFrame):
            result = result.to_dict()

        return {'success': True, 'result': result, 'error': None}

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"\n❌ EVALUATION ERROR: {error_msg}")

        return {'success': False, 'result': None, 'error': error_msg}


def chat_mt(model, messages: List[dict], dataset_name: str) -> List[str]:
    """Infer function for multi-turn dataset."""
    assert len(messages) % 2 == 0, "Messages should be pairs of user and assistant"
    nturn = len(messages) // 2
    utter_stack = []
    predictions = []

    for i in range(nturn):
        utter = messages[2 * i]
        utter_stack.append(utter)
        try:
            resp = model.chat(utter_stack, dataset=dataset_name)
            utter_stack.append(dict(role='assistant', content=resp))
        except Exception as e:
            resp = FAIL_MSG + str(e)
            utter_stack.append(dict(role='assistant', content=resp))
        predictions.append(resp)

    return predictions

# ==========================================
# Core Data Structures
# ==========================================

class EvalStatus(Enum):
    Pending = 1
    Running = 2
    Done = 3
    Error = 4
    Skipped = 5


@dataclass
class InferenceTask:
    """单个推理任务"""
    dataset_name: str      # 数据集名称（用于命名）
    model_name: str        # 模型名称（用于命名）
    sample_index: str      # 样本ID
    prompt_struct: Any     # 已构建的prompt结构
    dataset_type: DatasetType = "image"  # 数据集类型

@dataclass
class DatasetConfig:
    """数据集配置和状态"""
    dataset_name: str      # 数据集名称
    dataset_obj: Any       # 数据集对象
    model_obj: Any         # 模型对象
    model_name: str        # 模型名称
    work_dir: str          # 工作目录
    result_file: str       # 结果文件路径
    judge_kwargs: dict     # judge参数
    verbose: bool = False  # 是否打印详细信息

    # 数据集类型: "image", "video", "mt"
    dataset_type: DatasetType = "image"

    # Video 特有配置
    video_llm: bool = False  # 是否使用视频模式（vs 多图模式）

    # Runtime state
    total_samples: int = 0
    processed: int = 0
    eval_status: EvalStatus = EvalStatus.Pending
    eval_start_time: float = 0.0
    eval_duration: float = 0.0

    # Inference timing statistics
    infer_start_time: float = 0.     # 推理开始时间(秒)
    infer_end_time: float = 0.       # 推理结束时间(秒)
    infer_total_time: float = 0.     # 推理累计耗时(秒)
    infer_count: int = 0             # 推理样本数

    # For final result assembly
    results_dict: Dict[str, Any] = field(default_factory=dict)

# ==========================================
# Main Pipeline Class
# ==========================================

class APIEvalPipeline:
    """
    API模型推理和评测流水线

    核心特性：
    1. 跨数据集的统一推理队列
    2. 推理和评测并行执行
    3. 断点续传
    4. 实时状态监控
    5. 线程安全的结果保存
    """

    def __init__(
        self,
        dataset_configs: List[DatasetConfig],
        concurrency: int = 32,
        monitor_interval: int = 30,
        run_infer: bool = True,
        run_eval: bool = True,
        debug: bool = False,
    ):
        """
        Args:
            dataset_configs: 数据集配置列表
            concurrency: 推理并发数
            monitor_interval: 状态监控间隔（秒）
            run_infer: 是否运行推理
            run_eval: 是否运行评测
            debug: 调试模式，在主进程中运行评测（支持 ipdb 断点）
        """
        self.dataset_configs = dataset_configs
        self.concurrency = concurrency
        self.monitor_interval = monitor_interval
        self.run_infer = run_infer
        self.run_eval = run_eval
        self.infer_executor = ThreadPoolExecutor(max_workers=concurrency)
        self.eval_executor = ProcessPoolExecutor(max_workers=4)
        self.producer_executor = ProcessPoolExecutor(max_workers=1)
        self.debug = debug

        # 核心组件
        self.queue = asyncio.Queue(maxsize=concurrency * 2)
        self.states: Dict[str, DatasetConfig] = {
            cfg.dataset_name: cfg for cfg in dataset_configs
        }
        self.file_locks: Dict[str, threading.Lock] = {
            cfg.dataset_name: threading.Lock() for cfg in dataset_configs
        }

        # 运行时状态
        self.start_time = 0.0
        self.active_workers = 0
        self.total_tasks_generated = 0

        # 创建工作目录
        for cfg in dataset_configs:
            Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
            if not self.run_eval:
                cfg.eval_status = EvalStatus.Skipped

    def _get_checkpoint_file(self, dataset_name: str) -> Path:
        """获取断点文件路径（推理中间结果）"""
        cfg = self.states[dataset_name]
        return Path(cfg.work_dir) / f"{cfg.model_name}_{dataset_name}_checkpoint.pkl"

    def _load_checkpoint(self, dataset_name: str) -> Dict[str, Any]:
        """
        加载断点：读取已完成的推理结果

        优先级：
        1. checkpoint.pkl（推理中间结果）
        2. result_file（最终结果文件）
        """
        cfg = self.states[dataset_name]
        results = {}

        # 1. 尝试加载checkpoint
        checkpoint_file = self._get_checkpoint_file(dataset_name)
        if checkpoint_file.exists():
            try:
                results = load(str(checkpoint_file))
                logger.info(f"   [{dataset_name}] Loaded {len(results)} results from checkpoint")
            except Exception as e:
                logger.warning(f"   [{dataset_name}] Failed to load checkpoint: {e}")

        # 2. 尝试从最终结果文件加载
        result_path = Path(cfg.result_file)
        if result_path.exists():
            try:
                data = load(str(result_path))
                if isinstance(data, pd.DataFrame):
                    # DataFrame格式：提取index和prediction列
                    existing_results = {
                        str(idx): pred
                        for idx, pred in zip(data['index'], data['prediction'])
                        if FAIL_MSG not in str(pred)
                    }
                    results.update(existing_results)
                    logger.info(f"   [{dataset_name}] Loaded {len(existing_results)} results from result file")
            except Exception as e:
                logger.warning(f"   [{dataset_name}] Failed to load result file: {e}")

        return results

    def _save_checkpoint(self, dataset_name: str, result: dict):
        """线程安全地保存单个结果到checkpoint"""
        cfg = self.states[dataset_name]
        checkpoint_file = self._get_checkpoint_file(dataset_name)

        with self.file_locks[dataset_name]:
            # 加载现有结果
            if checkpoint_file.exists():
                results = load(str(checkpoint_file))
            else:
                results = {}

            # 更新结果
            results[result['index']] = result['prediction']

            # 保存
            dump(results, str(checkpoint_file))

    def _save_final_result(self, dataset_name: str):
        """
        保存最终结果文件

        将所有推理结果整合为DataFrame格式，保存为result_file
        """
        cfg = self.states[dataset_name]

        with self.file_locks[dataset_name]:
            # 从数据集对象获取完整数据
            dataset_data = cfg.dataset_obj.data.copy()

            # 确保所有样本都有结果
            missing_indices = []
            for idx in dataset_data['index']:
                idx_str = str(idx)
                if idx_str not in cfg.results_dict:
                    missing_indices.append(idx_str)

            if missing_indices:
                logger.warning(
                    f"   [{dataset_name}] Missing results for {len(missing_indices)} samples: "
                    f"{missing_indices[:5]}{'...' if len(missing_indices) > 5 else ''}"
                )
                return False

            # 构建prediction列
            predictions = [cfg.results_dict[str(idx)] for idx in dataset_data['index']]
            dataset_data['prediction'] = predictions

            # 移除image列（如果存在）
            if 'image' in dataset_data:
                dataset_data.pop('image')

            # 保存最终结果
            dump(dataset_data, cfg.result_file)
            logger.info(f"   [{dataset_name}] Saved final results to {cfg.result_file}")

            # 删除checkpoint文件
            checkpoint_file = self._get_checkpoint_file(dataset_name)
            if checkpoint_file.exists():
                checkpoint_file.unlink()

            return True

    async def _producer(self):
        """
        任务生成器：扫描所有数据集，构建prompt，填充队列

        支持三种数据集类型：
        - image: 基础图像数据集
        - video: 视频数据集（支持 pack 模式）
        - mt: 多轮对话数据集
        """
        logger.info("📦 Initializing tasks and checking checkpoints...")

        for cfg in self.dataset_configs:
            dataset_name = cfg.dataset_name
            dataset_type = cfg.dataset_type

            logger.info(
                f"   [{dataset_name}] Start build prompt [type={dataset_type}]"
            )
            # 1. 加载断点
            existing_results = self._load_checkpoint(dataset_name)
            cfg.results_dict = existing_results
            cfg.processed = len(existing_results)
            cfg.total_samples = len(cfg.dataset_obj.data)

            # 2. 检查是否需要推理
            if (not self.run_infer) or cfg.processed == cfg.total_samples:
                logger.info(
                    f"   [{dataset_name}] All samples completed ({cfg.processed}/{cfg.total_samples}). "
                    "Will trigger eval directly."
                )
                # 如果没有最终结果文件，需要先保存
                if not Path(cfg.result_file).exists():
                    self._save_final_result(dataset_name)
                # 触发评测
                asyncio.create_task(self._trigger_eval(dataset_name))
                continue

            # 3. 根据数据集类型处理
            if dataset_type == "video":
                tasks_generated = await self._produce_video_tasks(cfg, existing_results)
            elif dataset_type == "mt":
                tasks_generated = await self._produce_mt_tasks(cfg, existing_results)
            else:  # image
                tasks_generated = await self._produce_image_tasks(cfg, existing_results)

            # 如果没能成功构造 prompt，说明数据集加载有问题，跳过评测
            if tasks_generated == 0:
                cfg.eval_status = EvalStatus.Skipped

            self.total_tasks_generated += tasks_generated
            logger.info(
                f"   [{dataset_name}] Generated {tasks_generated} tasks "
                f"(Skipped {cfg.processed}) [type={dataset_type}]"
            )

        # 放入结束标记
        for _ in range(self.concurrency):
            await self.queue.put(None)

        logger.info(f"🚀 Queue ready. Total pending tasks: {self.total_tasks_generated}")

    async def _produce_image_tasks(self, cfg: DatasetConfig, existing_results: dict) -> int:
        """生成图像数据集的推理任务"""
        model = cfg.model_obj
        dataset = cfg.dataset_obj
        dataset_name = cfg.dataset_name
        data = dataset.data

        if hasattr(model, 'set_dump_image'):
            model.set_dump_image(dataset.dump_image)

        # 检查是否使用自定义prompt
        use_custom_prompt = (
            hasattr(model, 'use_custom_prompt')
            and model.use_custom_prompt(dataset_name)
        )
        if use_custom_prompt:
            logger.info(f"   [{dataset_name}] Using model custom prompt")
        else:
            logger.info(f"   [{dataset_name}] Using vanilla dataset prompt")

        tasks_generated = 0
        for i in range(len(data)):
            item = data.iloc[i]
            idx_str = str(item['index'])

            if idx_str in existing_results:
                continue

            try:
                if use_custom_prompt:
                    prompt_struct = model.build_prompt(item, dataset=dataset_name)
                else:
                    prompt_struct = dataset.build_prompt(item)
            except Exception as e:
                logger.error(f"   [{dataset_name}] Failed to build prompt for sample {idx_str}: {e}")
                continue

            task = InferenceTask(
                dataset_name=dataset_name,
                model_name=cfg.model_name,
                sample_index=idx_str,
                prompt_struct=prompt_struct,
                dataset_type="image"
            )
            await self.queue.put(task)
            tasks_generated += 1

        return tasks_generated

    async def _produce_video_tasks(self, cfg: DatasetConfig, existing_results: dict) -> int:
        """
        生成视频数据集的推理任务

        视频数据集的特殊处理：
        1. 支持 video_llm 模式（原生视频输入）和多图模式
        2. 使用 dataset.build_prompt(sample, video_llm=...) 构建 prompt
        """
        loop = asyncio.get_running_loop()
        model = cfg.model_obj
        dataset = cfg.dataset_obj
        dataset_name = cfg.dataset_name
        data = dataset.data

        # 获取 video_llm 设置
        video_llm = cfg.video_llm

        logger.info(f"   [{dataset_name}] Video mode: video_llm={video_llm}")

        tasks_generated = 0
        for i in range(len(data)):
            item = data.iloc[i]
            idx_str = str(item['index'])

            if idx_str in existing_results:
                continue

            try:
                # 单独交给一个 CPU 处理，以避免 video 处理的高 CPU 占用影响主进程
                prompt_struct = await loop.run_in_executor(
                    self.producer_executor,
                    partial(dataset.build_prompt, i, video_llm=video_llm),
                )
                if prompt_struct is None:
                    continue
            except Exception as e:
                logger.error(f"   [{dataset_name}] Failed to build prompt for sample {idx_str}: {repr(e)}")
                continue

            task = InferenceTask(
                dataset_name=dataset_name,
                model_name=cfg.model_name,
                sample_index=idx_str,
                prompt_struct=prompt_struct,
                dataset_type="video"
            )
            await self.queue.put(task)
            tasks_generated += 1

        return tasks_generated

    async def _produce_mt_tasks(self, cfg: DatasetConfig, existing_results: dict) -> int:
        """
        生成多轮对话数据集的推理任务

        多轮对话数据集的特殊处理：
        1. prompt_struct 是 messages 列表
        2. 推理时使用 chat_mt 函数进行多轮对话
        """
        model = cfg.model_obj
        dataset = cfg.dataset_obj
        dataset_name = cfg.dataset_name
        data = dataset.data

        # 验证模型支持多轮对话
        if not hasattr(model, 'chat_inner'):
            logger.warning(f"   [{dataset_name}] Model does not support multi-turn chat (no chat_inner method)")

        logger.info(f"   [{dataset_name}] Multi-turn dialogue mode")

        tasks_generated = 0
        for i in range(len(data)):
            item = data.iloc[i]
            idx_str = str(item['index'])

            if idx_str in existing_results:
                continue

            try:
                # 多轮对话的 prompt 是 messages 列表
                prompt_struct = dataset.build_prompt(item)
            except Exception as e:
                logger.error(f"   [{dataset_name}] Failed to build prompt for sample {idx_str}: {e}")
                continue

            task = InferenceTask(
                dataset_name=dataset_name,
                model_name=cfg.model_name,
                sample_index=idx_str,
                prompt_struct=prompt_struct,
                dataset_type="mt"
            )
            await self.queue.put(task)
            tasks_generated += 1

        return tasks_generated

    async def _worker(self):
        """
        推理工作线程

        根据任务类型调用不同的推理方法：
        - image/video: model.generate()
        - mt: chat_mt() 多轮对话
        """
        loop = asyncio.get_running_loop()

        while True:
            task = await self.queue.get()
            if task is None:
                self.queue.task_done()
                break

            self.active_workers += 1
            cfg = self.states[task.dataset_name]
            try:
                model = cfg.model_obj
                start_time = time.time()
                if not cfg.infer_start_time:
                    cfg.infer_start_time = start_time

                # 根据数据集类型选择推理方式
                if task.dataset_type == "mt":
                    # 多轮对话推理
                    def mt_inference_call():
                        return chat_mt(model, task.prompt_struct, task.dataset_name)

                    output = await loop.run_in_executor(self.infer_executor, mt_inference_call)
                else:
                    # 图像/视频推理（使用 generate）
                    def inference_call():
                        return model.generate(
                            message=task.prompt_struct,
                            dataset=task.dataset_name
                        )

                    output = await loop.run_in_executor(self.infer_executor, inference_call)

                inference_time = time.time() - start_time

                # 保存结果
                result_item = {
                    "index": task.sample_index,
                    "prediction": output
                }
                self._save_checkpoint(task.dataset_name, result_item)

                # 更新状态
                cfg.results_dict[task.sample_index] = output
                cfg.infer_end_time = time.time()
                cfg.infer_total_time += inference_time
                cfg.infer_count += 1
                cfg.processed += 1

                # 打印详细信息
                if cfg.verbose:
                    output_preview = str(output)[:100] if output else ""
                    logger.info(
                        f"[{task.dataset_name}] Sample {task.sample_index}: "
                        f"{output_preview}... (took {inference_time:.2f}s)"
                    )

                # 检查是否触发评测
                if cfg.processed == cfg.total_samples and cfg.eval_status == EvalStatus.Pending:
                    if self._save_final_result(task.dataset_name):
                        asyncio.create_task(self._trigger_eval(task.dataset_name))

            except Exception as e:
                logger.error(
                    f"❌ Worker error on {task.dataset_name}/{task.sample_index}: {e}",
                    exc_info=True
                )
            finally:
                self.active_workers -= 1
                self.queue.task_done()

    async def _trigger_eval(self, dataset_name: str):
        """
        触发评测任务

        正常模式：在子进程中执行
        调试模式：在主进程中执行
        """
        cfg = self.states[dataset_name]

        # 防止重复触发
        if cfg.eval_status != EvalStatus.Pending:
            return

        cfg.eval_status = EvalStatus.Running
        cfg.eval_start_time = time.time()

        # 评测日志文件
        eval_log_dir = Path(cfg.work_dir) / 'eval_logs'
        eval_log_dir.mkdir(parents=True, exist_ok=True)
        eval_log_path = str(eval_log_dir / f"{cfg.model_name}_{dataset_name}_eval.log")

        if self.debug:
            # 调试模式：直接在主进程中运行评测
            await self._run_eval_in_main_process(dataset_name, eval_log_path)
        else:
            # 正常模式：使用子进程运行评测
            await self._run_eval_in_subprocess(dataset_name, eval_log_path)

    async def _run_eval_in_main_process(self, dataset_name: str, eval_log_path: str):
        """在主进程中运行评测"""
        cfg = self.states[dataset_name]
        logger.info(f"🔔 [Eval Start - Debug Mode] {dataset_name}")

        def run_eval():
            try:
                result = cfg.dataset_obj.evaluate(cfg.result_file, **cfg.judge_kwargs)

                # 序列化结果（DataFrame 转 dict）
                if isinstance(result, pd.DataFrame):
                    result = result.to_dict()

                return {'success': True, 'result': result, 'error': None}

            except Exception as e:
                import traceback
                error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                return {'success': False, 'result': None, 'error': error_msg}

        try:
            eval_result = run_eval()
            self._handle_eval_result(dataset_name, eval_result, eval_log_path)

        except Exception as e:
            cfg.eval_status = EvalStatus.Error
            cfg.eval_duration = time.time() - cfg.eval_start_time
            logger.error(f"❌ [Eval Failed] {dataset_name}: {e}")

    async def _run_eval_in_subprocess(self, dataset_name: str, eval_log_path: str):
        """在子进程中运行评测"""
        cfg = self.states[dataset_name]

        # 在后台线程中等待子进程完成（避免阻塞 asyncio 事件循环）
        loop = asyncio.get_running_loop()

        try:
            eval_result = await loop.run_in_executor(
                self.eval_executor,
                _eval_subprocess_target,
                cfg.dataset_obj,
                cfg.result_file,
                cfg.judge_kwargs,
                eval_log_path,
            )
            self._handle_eval_result(dataset_name, eval_result, eval_log_path)

        except Exception as e:
            cfg.eval_status = EvalStatus.Error
            cfg.eval_duration = time.time() - cfg.eval_start_time
            logger.error(
                f"❌ [Eval Failed] {dataset_name}: {e}\n"
                f"   Check log file: {eval_log_path}"
            )

    def _handle_eval_result(self, dataset_name: str, eval_result: dict, eval_log_path: str):
        """处理评测结果"""
        cfg = self.states[dataset_name]

        if eval_result['success']:
            cfg.eval_status = EvalStatus.Done
            cfg.eval_duration = time.time() - cfg.eval_start_time
            logger.info(
                f"✅ [Eval Finish] {dataset_name} (Took {cfg.eval_duration:.1f}s)"
            )

            # 打印评测结果（如果有）
            result = eval_result['result']
            if result is not None:
                if isinstance(result, dict):
                    # 可能是 DataFrame.to_dict() 的结果，尝试还原
                    try:
                        df = pd.DataFrame(result)
                        logger.info(f"   Results:\n{df}")
                    except Exception:
                        logger.info(f"   Results: {json.dumps(result, indent=2)}")
        else:
            cfg.eval_status = EvalStatus.Error
            cfg.eval_duration = time.time() - cfg.eval_start_time
            logger.error(
                f"❌ [Eval Failed] {dataset_name}: {eval_result['error']}\n"
                f"   Check log file: {eval_log_path}"
            )

    async def _monitor_loop(self):
        """后台监控：周期性打印状态快照"""
        self._log_snapshot()
        while True:
            await asyncio.sleep(self.monitor_interval)
            self._log_snapshot()

            # 退出条件：所有推理完成 且 所有评测完成
            all_infer_done = (
                self.queue.empty()
                and self.active_workers == 0
            )
            all_eval_done = all(
                cfg.eval_status in [EvalStatus.Done, EvalStatus.Error, EvalStatus.Skipped]
                for cfg in self.states.values()
            )

            if all_infer_done and all_eval_done:
                break

    def _log_snapshot(self):
        """生成清晰的状态日志"""
        elapsed = time.time() - self.start_time
        lines = []

        total_processed = 0
        total_samples = 0

        for cfg in self.states.values():
            total_processed += cfg.processed
            total_samples += cfg.total_samples

            # 推理进度
            if cfg.total_samples > 0:
                infer_pct = cfg.processed / cfg.total_samples
                # 保留一位小数，向下取整，100% 代表全部完成
                infer_pct = math.floor(infer_pct * 1000) / 10
            else:
                infer_pct = 0
            infer_str = f"{infer_pct:>5.1f}% ({cfg.processed:>4}/{cfg.total_samples:<4})"

            # 平均推理耗时
            if cfg.infer_count > 0:
                if cfg.processed == cfg.total_samples:
                    avg_time = (cfg.infer_end_time - cfg.infer_start_time) / cfg.infer_count
                else:
                    avg_time = (time.time() - cfg.infer_start_time) / cfg.infer_count
                if avg_time > 1:
                    infer_str += f" | {avg_time:.2f}s/it"
                elif avg_time > 0:
                    infer_str += f" | {1 / avg_time:.2f}it/s"

            # 评测状态
            eval_str = cfg.eval_status.name
            if cfg.eval_status == EvalStatus.Running:
                dur = time.time() - cfg.eval_start_time
                eval_str = f"Running ({dur:.0f}s)"
            elif cfg.eval_status == EvalStatus.Done:
                eval_str = f"Done ({cfg.eval_duration:.1f}s)"

            # 格式化输出
            name_str = cfg.dataset_name[:20].ljust(20)
            line = [name_str, f"Infer: {infer_str} | Eval: {eval_str:<20}"]
            lines.append(line)

        # 全局进度
        global_pct = (total_processed / total_samples * 100) if total_samples > 0 else 0
        global_str = f"Infer: {global_pct:>5.1f}% ({total_processed:>4}/{total_samples:<4})"

        logger.info('\n' + tabulate(lines, [f"Elapsed: {elapsed:.0f}s", global_str], tablefmt='simple_outline'))

    async def run(self):
        """启动流水线"""
        self.start_time = time.time()

        logger.info("  API Inference & Evaluation Pipeline")
        logger.info(f"  Datasets: {len(self.dataset_configs)}")
        logger.info(f"  Concurrency: {self.concurrency}")
        logger.info(f"  Infer: {self.run_infer}")
        logger.info(f"  Eval: {self.run_eval}")

        # 1. 启动生产者（不等待完成，让其与消费者并行）
        producer_task = asyncio.create_task(self._producer())

        # 2. 启动监控
        monitor_task = asyncio.create_task(self._monitor_loop())

        # 3. 启动消费者（与生产者并行运行）
        if self.run_infer:
            workers = [
                asyncio.create_task(self._worker())
                for _ in range(self.concurrency)
            ]
            # 等待生产者完成（队列填充结束标记）
            await producer_task
            # 等待所有消费者完成
            await asyncio.gather(*workers)
            logger.info("🎉 All inference tasks finished. Waiting for pending evaluations...")
        else:
            await producer_task
            logger.info("📊 Eval mode only. Skipping inference...")

        # 4. 等待监控循环结束（所有评测完成）
        await monitor_task

        # 5. 最终报告
        self._log_snapshot()
        total_time = time.time() - self.start_time
        logger.info(f"  🏁 Pipeline Completed in {total_time:.1f}s")
