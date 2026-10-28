# Flatten transparent diagrams against white

## Context
Issue1752 reports transparent RGBA, grayscale-alpha and indexed-palette diagrams becoming black-on-black. Both dataset decoding and API encoding discarded alpha through RGB conversion, hiding black ink over transparent pixels.

## Decision
Flatten alpha-bearing images on a white RGB background before both conversions. Use Pillow RGBA conversion to honor palette transparency; preserve existing opaque RGB, grayscale and16-bit decode handling. Keep resize/budget logic unchanged.

## Alternatives
Preserving alpha through JPEG encoding is unsupported. Fixing only encoding leaves localized dataset images corrupted; fixing only decoding misses directly supplied API images. A configurable background expands scope beyond the reported document/diagram failure.

## Consequences
Transparent backgrounds appear white and opaque ink retains its color. Partially transparent pixels blend on white. Fully opaque inputs preserve their previous conversion behavior. No dataset claims or provider inference are made.

## Validation
Frozen upstream ec4a4ed412456a4d774e57974b21f90c80b2f519: six regression failures (three transparency modes across encode/decode), five existing/opaque controls pass. Tests execute the actual image helpers with isolated module loading, no model or network services. Candidate and impacted suite results are retained in local campaign logs.

Candidate:15 tests pass across image conversion and remote-media suites. An initial adjacent-suite run failed because local tqdm was absent; after adding that test dependency, all15 passed. Flake8 reports three pre-existing long lines in the existing test helper; the source and new regression lines pass, and unchanged baseline lint findings are retained locally. No GPU/private hosted CI run or model inference was performed.
