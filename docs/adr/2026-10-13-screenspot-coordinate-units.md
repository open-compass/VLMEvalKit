# State ScreenSpot coordinate units in the default prompt

## Context
Issue #1706 reports ScreenSpot predictions on a 0–1000 grid being interpreted in different units and receiving misleadingly low scores. The default prompt asks for x/y without a coordinate convention. Current ground-truth boxes are normalized by image width and height; the point scorer also supports a pixel fallback, which cannot distinguish a 1000-grid output from pixels.

## Decision
State that x and y are fractions of image width and height, between zero and one, in both functional and composite user instruction templates. Preserve action syntax, tasks, image messages and scoring behavior.

## Alternatives
Guessing the model's coordinate scale risks converting genuine pixel values incorrectly. An explicit per-wrapper scale declaration is a larger API change. Prompt clarification implements the first remedy proposed in the issue and matches the existing normalized ground-truth convention.

## Consequences
New default ScreenSpot prompts declare an unambiguous coordinate system. Existing saved predictions are not converted or rescored differently, and no model accuracy improvement is claimed. The existing pixel fallback remains unchanged. Older custom/model prompts remain their own path.

## Validation
Frozen upstream ec4a4ed412456a4d774e57974b21f90c80b2f519 has four prompt contract failures and one passing actual normalized-point control. Tests execute the actual builder for functional/composite and indexed/row cases, preserve screenshot messages/source tasks, and run actual bbox conversion/parser against a real temporary image. Existing open PRs #1719 and #1722 change weighting and parsing; their exact diffs do not change these instruction templates. No GPU, model or provider calls were made. Raw receipts remain local.

Candidate: seven impacted tests pass, including all four prompt cases, normalized-point control and two existing image helper tests. New test lint passes; source lint introduces no new diagnostic. Diff whitespace validation passes. No actual model response/accuracy improvement is measured.
