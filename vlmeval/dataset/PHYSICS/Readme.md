

## 安装步骤

`evaluate` 依赖以下两个自定义包：
- `latex2sympy2_extended`
- `math-verify`

#### 安装 `latex2sympy2_extended`

1. 进入包目录：
   ```bash
   cd ./latex2sympy2_extended
   ```

2. 安装包（推荐使用 ANTLR 4.11）：
   ```bash
   pip install -e .[antlr4_11_0]
   ```

   > **注意**：`latex2sympy2_extended` 使用 `omegaconf`，而 `omegaconf` 默认支持 ANTLR 4.9。若需支持 ANTLR 4.11，请参考 [Hydra Issue #2491](https://github.com/facebookresearch/hydra/issues/2491) 修改配置。

3. 如果 ANTLR 4.11 安装失败，可回退到 ANTLR 4.9：
   ```bash
   pip install -e .[antlr4_9_3]
   ```
#### 安装 `math-verify`
   ```bash
   pip install math-verify
   ```

## 情况说明
judge model实际在论文中为xverify-8b-sft，但该verifier暂无开源计划，因此可以使用xverify或其他LLM代替。