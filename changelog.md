# 2.0.18.dev

## Minor Changes

* Allow for input as pandas Dataframe, add pandas requirement, based on #95

# 2.0.17

## Bug fixes

* Fix #96, invert ticks and labels on categorical-categorical pairwise plot

# 2.0.16

## Minor Changes

* Prepare `visualizer.py` for `OrdinalHyperparameter`s
* Generally simplify code for maintainablitiy
* `generate_pairwise_marginal` does NOT swap values anymore

## Bug fixes

* Fix inverted axes-bug (#96)

# 2.0.15

## Bug fixes

* Fix Constant-Handling and "swap"-axis logic (#93)

# 2.0.14

## Minor Changes

* Add try-except to pickling pairwise plots now fails silently (on info-level)
* Add support for Constant hyperparameters
* Beautify code in visualizer-class

## Bug fixes

* Fix axis on pairwise mixed cat/non-cat plots
* Fix visualizer's `create_all_plots`

# 2.0.13

## Bug fixes

* Unassigned variable for marginal plot

# 2.0.12

## Major Changes

* Enforce log-scale in plots if specified in argument of plot-function
* Add option to plot incumbent(s) if it/they is/are passed

## Minor Changes

* Add legend to plots
* Add grid to plots

# 2.0.11

## Bug fixes

* Having a boolean in pairwise marginal plots does not lead to indice-crash anymore (#80)
