# Robust Bayesian Inference for Discrete Outcomes with the Total Variation Distance

This repository contains the code used to generate the results in the paper ``Robust Bayesian Inference for Discrete Outcomes with the Total Variation Distance''.

The `npl` folder contains two files implementing a (simplified) version of nonparametric Bayesian Learning due to [Lyddon/Holmes/Walker 2019](https://academic.oup.com/biomet/article/106/2/465/5385582) and [Fong/Lyddon/Holes 2019](https://arxiv.org/abs/1902.03175): `NPL.py` implements the algorithm; `likelihood_functions.py' provides a range of likelihood classes to use with NPL.

`run_simulations.py` recreates all simulation results from Section 6 of the paper, while `probit_experiments.py` and `NN_experiments.py` reproduce the real world data parts. The respective `plot_*.py` files visualize our results.