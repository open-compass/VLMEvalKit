# Do not infer spatial MCQ labels from tagged words

## Context
The spatial benchmark rule scorer accepts tagged words such as cat, banana and apple as C, B and A because its tagged-answer parser captures only the first letter. Actual compute_mcq_score then awards incorrect credit when that accidental letter matches a valid gold option. This is a separate parser from matching_util.py, which existing PR #1751 changes.

## Decision
Require that the leading label not be followed by another ASCII letter in the tagged-answer block. Preserve single-letter labels, punctuation-prefixed explanations, case handling and all remaining extraction stages. Do not change score aggregation, option-range selection or failure-sentinel behavior.

## Alternatives
Dropping tagged-answer support would reject documented valid outputs. Changing the general matcher would leave this independent spatial parser unaffected. The active issue #1697 sentinel fix addresses a different failure and is not included here.

## Consequences
Tagged ASCII words no longer manufacture valid option labels from their first letter. Correctly tagged labels and label-plus-explanation outputs retain their scores. No provider or judge model calls are needed.

## Validation
Frozen upstream ec4a4ed412456a4d774e57974b21f90c80b2f519 has three false-positive failures and four passing controls. Regressions execute the actual matcher and compute_mcq_score functions via isolated AST loading with real pandas rows; they do not replace the matching or scoring logic. Baseline/candidate and lint receipts are retained locally. No complete hosted dataset or model run is claimed.

An initial harness revision accidentally loaded unrelated numeric initialization without its optional dependency and failed before evaluation; those logs are retained but excluded from regression evidence. Corrected frozen-baseline probe has three verdict failures and four valid controls passing; final candidate has eleven passing tests including four existing judge-score controls. New tests pass flake8 and source lint retains the same two pre-existing long-line findings, with no new diagnostics. Diff whitespace validation passes.
