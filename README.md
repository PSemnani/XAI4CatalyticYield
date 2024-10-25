# ML & XAI for Catalytic Yield

This project implements all the experiments used for the paper "A Machine Learning and Explainable AI Framework Tailored for Unbalanced Experimental Catalyst Discovery" by Semnani et. al, which can be found [here](https://arxiv.org/abs/2407.18935).
This work focuses on the development of a framework that can be used to train, evaluate and explain machine learning models in a robust way, particularly when dealing with scarce and class-imbalanced experimental catalyst data.

## Installation

To install all dependencies and the core project code, Python 3.10 and pip is required. To create and activate a conda environment containing these prerequisites, use:
    
    conda create -n ml4cat python=3.10 pip
    conda activate ml4cat

The package and all its dependencies can then be installed using:
    
    pip install -e .

## Structure
The project is structures as follows:

* data/ - contains the OCM dataset used in the paper

* mlxai4cat/ - this directory contains the source code for this project
    * mlxai4cat/utils/ - contains various utility and helper functions
    * mlxai4cat/models/ - contains a simple feed forward neural network model, a neuralized SVM model used to extract explanations, as well as a generative model used to create promising catalyst candidates
* results/ - this is where any evaluation results or models created using the notebooks will be saved
* figures/ - this is where any figures created using the notebooks will be saves


## Reproducing the results
All the experiments and results in the paper can be reproduced using the jupyter notebooks found in the *notebooks/* directory. The notebooks should be executed in the following order:

1. notebooks/ML_catalytic_yield_DT.ipynb - Starting point for experiments, contains the training, evaluation, and feature importances for various decision tree models, including regular, pre-pruned and post-pruned decision trees.
2. notebooks/ML_catalytic_yield_RF.ipynb - Contains the training, evaluation, and feature importances for two tree ensamble models, Random Forests and XGBoost.
3. notebooks/ML_catalytic_yield_LR_SVM.ipynb - Contains training, evaluation, and feature importances for logistic regression and SVMs. The SVM models use Layerwise Relevance Propagation (LRP) for the explanations/feature importances.
4. notebooks/ML_catalytic_yield_neural_network.ipynb - Contains training, evaluation, and feature importances for a feedforward neural network model. The explanations/feature importances are extracted via LRP, and this notebook additionally includes a demonstration how a simple generative model can be used to produce new potentially high-yield catalyst candidates
based on the extracted feature importances.
5. notebooks/ML_model_performance_comparison.ipynb - Contains the comparison and visualization of the performance of the various ML models
6. notebooks/M_model_feature_importance_comparison.ipynb - Contains the comparison and visualization of the feature importances of various ML models
