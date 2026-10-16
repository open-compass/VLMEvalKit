# Require a complete ZeroBench brace answer

## Context
ZEROBench.evaluate labels its scoring exact matching but accepts the gold answer as a substring of the last brace answer. This awards gold `2` for `12` and gold `red` for `redwood`, inflating correctness. The official ZeroBench example evaluator requires full content and equal length.

## Decision
Compare the entire parsed brace answer to the stripped gold answer after existing case normalization. Preserve final-brace extraction, ground-truth whitespace handling, missing-brace failures and prompt formatting.

## Alternatives
Word or numeric-token boundaries still accept additional content that the official evaluator rejects. New numeric normalization would also change the benchmark protocol. Updating the dataset version or images is separate from scoring and is not addressed here.

## Consequences
Longer numbers, words and answer alternatives containing the gold no longer count as correct. Exact case-insensitive final answers retain their result. This changes incorrect prior scores; saved old evaluations should not be interpreted as protocol-equivalent.

## Validation
Primary source: https://github.com/jonathan-roberts1/zerobench/blob/0debbe43deddfe54f9965a6fdfd4f70a20d8ab62/README.md. The official condition combines a matching gold prefix with equal answer length, equivalent to complete equality here. Frozen VLMEvalKit ec4a4ed412456a4d774e57974b21f90c80b2f519 has three false-positive regression failures. Regressions execute the actual class evaluator through isolated AST loading and real temporary prediction CSVs, with exact-answer, missing-brace and final-brace controls. This is not a downloaded benchmark corpus or model run; no provider calls were made. Local candidate/lint evidence is retained separately.

Candidate: ten impacted scoring tests pass. Flake8 has exactly the same 87 diagnostics as the frozen baseline (mostly existing long lines); no new lint diagnostics. An attempted adjacent LLaVABench suite had eleven passes and one missing-tabulate environment failure; this is preserved and not represented as a passing broader suite. Diff whitespace validation passes.
