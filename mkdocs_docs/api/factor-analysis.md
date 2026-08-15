# Factor-analysis API

`fincore.alphalens` is the source-shaped strict façade. For new code, prefer
the enhanced `fincore.factor_analysis` API: prepare factor data once, analyze
the resulting clean table once, and render returned artifacts explicitly.

Plotting is optional and resolved only when a renderer runs. Install
`fincore[alphalens]` for rendering; compute-only enhanced workflows use
`fincore[factor-analysis]`.

## Enhanced API

::: fincore.factor_analysis

## Strict utilities

::: fincore.alphalens.utils

## Strict performance

::: fincore.alphalens.performance

## Strict tear sheets

::: fincore.alphalens.tears
