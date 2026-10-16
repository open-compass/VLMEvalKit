# Assign MVBench scores by dataset ID

## Context

Issue #1186 reports missing category scores when evaluating selected MVBench subsets and identifies positional score assignment as the cause. The evaluator reads answers and predictions using the `index` column but writes scores using pandas row labels. These identifiers differ for subsets or reordered records. Both MVBench and MVBench_MP4 contain the same assignment pattern.

## Decision

Use the same `index` column selection for the success and failed-prediction score assignments in both evaluators. Preserve input row order, judge selection, option extraction and aggregation.

## Alternatives

Resetting the DataFrame index would still fail for noncontiguous dataset IDs. Setting the DataFrame index to dataset IDs would change the input shape and serialization. Selecting by the existing ID column keeps the change local.

## Consequences

Scores remain attached to their source records, and assigning scores does not add synthetic rows. Dataset IDs are expected to identify records uniquely, as already required by the evaluator's answer and prediction lookup. This change does not resolve missing predictions or alter judge behavior.

## Validation

At frozen upstream ec4a4ed412456a4d774e57974b21f90c80b2f519, six cases exercise the actual evaluate methods extracted through the AST, with pandas DataFrames and isolated file/judge boundaries: both variants, noncontiguous IDs, reordered IDs and ordinary sequential IDs. The baseline has four failures and two passes; the candidate has six passes. Four existing judge-score regressions also pass (ten total).

The regression checks row count/order, scores and preservation of answers. A deterministic judge boundary avoids provider requests; no GPU inference, dataset download, external judge or full model integration was executed. The test file passes flake8; source lint is compared against the unchanged upstream file and pre-existing findings are retained in private campaign logs. `git diff --check` passes. No provider spend was incurred; native execution cost is not measured.
