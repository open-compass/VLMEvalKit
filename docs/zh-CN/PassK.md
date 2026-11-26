# 为 Benchmark 支持 pass@k

> 📐 如何让一个数据集报告 `pass@k` / `pass^k` / `avg@n`

有些 benchmark 的评分基于**对同一道题的多次尝试**：同一个 prompt 问模型 `N` 次，得分是
`pass@N`（至少一次答对）、`pass^N`（每次都答对）或各次的平均值。VLMEvalKit 用 `--n-runs N`
实现这件事，它会用 `RepeatedDataset`（`vlmeval/dataset/utils/repeat.py`）包装数据集。

**数据集本身不需要实现重复。** 包装器把每道题重复 `N` 次，每次调用数据集自己的 `evaluate`，
合并指标，再根据各次产生的逐样本明细文件计算 `pass@N` / `pass^N`。所以支持 `pass@k` 只需两件事：
告诉包装器默认跑几次，以及它能从哪里读到一个逐样本的正确性列。

## 方式一：从命令行把数据集跑 N 次

重复本身不需要改代码。任何有单次 `evaluate`（返回 `dict` 或 `pd.DataFrame`）的数据集都可以直接用：

```bash
python run.py --data YourBench --model YourModel --n-runs 5 --api-nproc 64
```

`evaluate` 返回的每个数值指标都会在各次之间取平均、并沿用原来的键——所以 `accuracy` 变成 5 次
尝试的平均准确率，也就是 `avg@5`；`pass@5` / `pass^5` **只有在**数据集提供了逐样本正确性列时才会
追加——见下文。`N = 1`（默认值）时数据集根本不会被包装，
所以正常跑一次的行为完全不变。

`--n-runs` 需要模型会采样。贪心解码下 `N` 次结果完全相同，`pass@N` 退化为单次得分，包装器
会为此打一条警告。

## 方式二：在数据集里声明

常见情况下，两个类属性就够了：

```python
class YourBench(ImageVQADataset):
    DEFAULT_N_RUNS = 5          # 命令行没给 --n-runs 时用这个
    PASS_K_HIT_COL = 'correct'  # evaluate 写出的逐样本明细里那列 0/1
```

`DEFAULT_N_RUNS` 只提供默认值，命令行上的 `--n-runs` 始终优先。`PASS_K_HIT_COL` 决定了
`pass@N` 和 `pass^N` 能不能算出来：没有它，包装器只做指标合并。

### evaluate 需要写出什么

包装器读取的是你的 `evaluate` 在单次运行中本来就会写的逐样本明细文件，从中取
`PASS_K_HIT_COL` 指定的列。默认在该文件位于

```python
get_intermediate_file_path(eval_file, '_results')
```

这个路径时被找到，并按 `csv`、`json`、`xlsx` 的顺序依次探测。所以一个已经会写逐样本结果表的
数据集通常完全不用改。有两点值得注意：

- **加上 `index` 列。** 明细里有 `index` 时，包装器按它把行 join 到题目上；没有时按位置对齐，
  而这只有在 evaluator 按 eval 文件的行序输出时才成立。一个会重排行序的 evaluator（例如按
  `self.data` 而不是按预测结果迭代）会把每道题的对错全部错配，而聚合指标看起来仍然"合理"——
  这种情况真实发生过。
- **让正确性列没有歧义。** 它按布尔解释：`True`/`1`、数值 `1`/`0`，或取值为 `1`、`1.0`、
  `true`、`yes` 的文本列。NaN（某个尝试根本没被评分）按**未通过**处理，这对 `pass@N` 是安全方向。

下面是 ZEROBench 的接入，除两个属性外什么都不需要：

```python
class ZEROBench(ImageVQADataset):
    DEFAULT_N_RUNS = 5
    PASS_K_HIT_COL = 'Correct?'

    def evaluate(self, eval_file, **judge_kwargs):
        ...
        dump(output_df, get_intermediate_file_path(eval_file, '_results', 'csv'))
        dump(d2df(result), get_intermediate_file_path(eval_file, '_acc', 'csv'))
        return result
```

## 全部可配置项

三个基类（`ImageBaseDataset`、`TextBaseDataset`、`VideoBaseDataset`）声明的属性和钩子完全一致。

| 属性 | 默认值 | 含义 |
|---|---|---|
| `DEFAULT_N_RUNS` | `1` | 命令行未给 `--n-runs` 时每题跑几次 |
| `PASS_K_HIT_COL` | `None` | 明细里的 0/1 列；为 `None` 时不计算 `pass@N` / `pass^N` |
| `PASS_K_DETAIL_SUFFIX` | `'_results'` | 该明细文件的后缀，传给 `get_intermediate_file_path` |
| `PASS_K_DETAIL_FORMAT` | `None` | 固定明细文件的格式；`None` 表示按序探测 |
| `PASS_K_DETAIL_PROBE` | `('csv', 'json', 'xlsx')` | 未固定格式时依次尝试的格式 |

不符合默认约定时覆写的方法：

| 方法 | 覆写场景 |
|---|---|
| `get_default_n_runs()` | 默认次数随数据集变体变化，而不是一个常量 |
| `get_pass_k_detail_paths(sub_file)` | 明细文件的命名不是固定后缀 |
| `merge_metrics(results)` | 指标不能取平均 |

### 明细文件在哪：`get_pass_k_detail_paths`

默认是 `<eval_file 文件名>_results.<格式>`。当文件名取决于基类无法知道的东西时需要覆写——
GDP.pdf 的明细文件名里带产出它的 judge：

```python
def get_pass_k_detail_paths(self, sub_file):
    return [get_judge_detail_file(sub_file, 'rubric_v1', self._judge_name, 'tsv')]
```

它返回候选列表，按顺序尝试，所以数据集可以给出兜底路径。

### 多次运行怎么合并：`merge_metrics`

默认实现（`merge_numeric_metrics`）对所有在各次中都是数值的指标取平均，非数值指标只在各次
完全一致时保留。当取平均是错的——需要取最大、求和，或必须从各次输出重新计算时，覆写它：

```python
def merge_metrics(self, results):
    return {'accuracy': max(r['accuracy'] for r in results)}
```

### 哪个指标作为主指标上报

`report_primary_metric` 决定汇总表里显示哪个键。重复运行之后指标里有 `accuracy`、`pass@N` 和
`pass^N`；如果这个 benchmark 的招牌数字是 `pass@N`，就要显式说明，否则通用的选键逻辑会报
`accuracy`：

```python
@classmethod
def report_primary_metric(cls, metrics):
    for key in metrics:
        if re.fullmatch(r'pass@\d+', str(key)):
            return {str(key): metrics[key]}
    return super().report_primary_metric(metrics)
```

注意这个钩子也必须能在**单次**运行下工作，那时没有 `pass@N` 这个键：要回落到父类实现，
而不是假定它一定存在。

## 评测与推理分开跑，以及复用

推理和评测可以分开，而且一次 `N` 次的推理可以按任意更小的 `k` 重复评分：

```bash
# 每题 5 个样本，判分一次
python run.py --data YourBench --model YourModel --n-runs 5 --mode all
# 用同一份预测按 k = 3 重新评分，不重新推理、不重新判分
python run.py --data YourBench --model YourModel --n-runs 3 --mode eval --reuse
```

预测文件里每题的样本数**多于** `n_runs` 时会被裁到前 `n_runs` 个——样本相互独立，前 `k` 个本身
就是一组合法的 `k` 次采样。样本数不足则报错。对由更小 `--n-runs` 写出的文件做 `--mode eval`
会以"推理结果不完整"跳过。

每次运行的文件放在预测文件旁边的 `<stem>_runs/` 目录里，`--reuse` 会把它和其他评测辅助文件
一起复制——所以按次写出的 judge 缓存不会被重复付费判分。**只要一次运行写出的东西都从你收到的
`eval_file` 派生，它就会被复用**；写到别处就不会。

## 限制

- 多轮对话数据集（`TYPE = 'MT'`）和 pack 模式视频拒绝 `N > 1`：它们的推理不走包装器依赖的
  逐行路径（pack 模式一个视频一次请求，重复出来的行会被忽略）。非 pack 视频支持。
- 同一道题的 `N` 次运行不共享 judge 缓存：每次一份、按内容寻址。模型会采样时各次回答本就不同；
  贪心解码下完全相同的回答会被判 `N` 次。
- `--n-runs 1` 从不包装数据集，所以对重复过的预测文件用 `--n-runs 1` 做 eval 会以"不完整"
  跳过，而不是正常评分。

## 验证接入是否正确

用 `--n-runs 2` 在小子集上跑一遍，检查 run 目录里的三个文件：

- `<model>_<alias>.<ext>` —— 预测，每个（题, 样本）一行，`index` 形如 `<原始 id>#<样本号>`；
  扩展名跟 `PRED_FORMAT` 走（`tsv`、`xlsx`——默认值——或 `json`），想直接当文本看就设 `PRED_FORMAT=tsv`
- `<model>_<alias>_acc.csv` —— 合并后的指标，必须包含 `pass@2` 和 `pass^2`
- `<model>_<alias>_results.csv` —— 合并后的逐样本明细，每个样本一行

如果 `pass@2` 缺失，有两种可能：`PASS_K_HIT_COL` 是 `None`，此时包装器只报合并后的指标、
不会报错，需要把它设上；或者明细文件没被找到，此时 `evaluate` 会抛 `FileNotFoundError`，列出
它尝试过的所有路径以及该怎么给文件命名。
