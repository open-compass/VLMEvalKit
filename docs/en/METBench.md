# MET-Bench

[Paper](https://arxiv.org/abs/2502.10886) · [Website](https://vanyacohen.com/MET-Bench/) · [Reference evaluator](https://github.com/vanyacohen/MET-Bench)

MET-Bench evaluates entity tracking in Minecraft, Chess, and Shell Game through parallel text and image inputs. Chess and Shell Game track ten actions from an initial state; Minecraft predicts the next state after an action.

## Datasets

| Domain | Hugging Face dataset | Text task | Image task |
|---|---|---|---|
| Minecraft | [🤗 Minecraft](https://huggingface.co/datasets/vanyacohen/MET-Bench-Minecraft) | `METBench_minecraft_text` | `METBench_minecraft_image` |
| Chess | [🤗 Chess](https://huggingface.co/datasets/vanyacohen/MET-Bench-Chess) | `METBench_chess_text` | `METBench_chess_image` |
| Shell Game | [🤗 Shell Game](https://huggingface.co/datasets/vanyacohen/MET-Bench-Shell) | `METBench_shell_text` | `METBench_shell_image` |

Each domain contains 500 unique test inputs. Dataset revisions are pinned, and the text and image tasks use the same ordered example IDs. Text tasks download `evaluation_text_only`; image tasks download `evaluation`. Chess and Shell Game are deduplicated by initial state and ten-action prefix. Minecraft is deduplicated by initial state, action, and ordered candidate states.

## Run

After installing VLMEvalKit, select a model that supports multiple interleaved images:

```bash
python run.py \
  --model YOUR_CONFIGURED_MODEL \
  --data METBench_minecraft_text METBench_minecraft_image \
         METBench_chess_text METBench_chess_image \
         METBench_shell_text METBench_shell_image \
  --work-dir results/metbench
```

The dataset supplies the benchmark's chain-of-thought prompts. Set temperature to zero and the output limit to 4,096 tokens in the model configuration for the reference evaluation. Credentials are read from the environment.

For a small check, select two examples:

```bash
python run.py \
  --model YOUR_CONFIGURED_MODEL \
  --data METBench_minecraft_image \
  --data-config '{"METBench_minecraft_image":{"class":"METBenchImage","dataset":"METBench_minecraft_image","limit":2}}' \
  --work-dir results/metbench-check
```

Each task produces an `_acc.csv` file. `Overall` reports accuracy in percent: correct answer choices for Minecraft and Shell Game, and correctly predicted board squares for Chess. `ci_lower` and `ci_upper` give 95% confidence bounds: a normal approximation using the standard error across board scores for Chess, and Wilson intervals across answer choices for Minecraft and Shell Game. Chess bounds are unavailable for a single example; `examples` gives the evaluated example count. Prompts and answer parsing match the reference evaluator and the lmms-eval integration.
