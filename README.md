# Predict Bike Sharing Demand with AutoGluon

## Introduction to AWS Machine Learning Final Project

## Overview
In this project, students will apply the knowledge and methods they learned in the Introduction to Machine Learning course to compete in a Kaggle competition using the AutoGluon library.

Students will create a Kaggle account if they do not already have one, download the Bike Sharing Demand dataset, and train a model using AutoGluon. They will then submit their initial results for a ranking.

After they complete the first workflow, they will iterate on the process by trying to improve their score. This will be accomplished by adding more features to the dataset and tuning some of the hyperparameters available with AutoGluon.

Finally they will submit all their work and write a report detailing which methods provided the best score improvement and why. A template of the report can be found [here](report-template.md).

To meet specifications, the project will require at least these files:
* Jupyter notebook with code run to completion
* HTML export of the jupyter notebbook
* Markdown or PDF file of the report

Images or additional files needed to make your notebook or report complete can be also added.

## Getting Started
* Clone this template repository `git clone git@github.com:udacity/nd009t-c1-intro-to-ml-project-starter.git` into AWS Sagemaker Studio (or local development).

<img src="img/sagemaker-studio-git1.png" alt="sagemaker-studio-git1.png" width="500"/>
<img src="img/sagemaker-studio-git2.png" alt="sagemaker-studio-git2.png" width="500"/>

* Proceed with the project within the [jupyter notebook](project-template.ipynb).
* Visit the [Kaggle Bike Sharing Demand Competition](https://www.kaggle.com/c/bike-sharing-demand) page. There you will see the overall details about the competition including overview, data, code, discussion, leaderboard, and rules. You will primarily be focused on the data and ranking sections.

### Dependencies

```
Python 3.7
MXNet 1.8
Pandas >= 1.2.4
AutoGluon 0.2.0 
```

### Installation
For this project, it is highly recommended to use Sagemaker Studio from the course provided AWS workspace. This will simplify much of the installation needed to get started.

For local development, you will need to setup a jupyter lab instance.
* Follow the [jupyter install](https://jupyter.org/install.html) link for best practices to install and start a jupyter lab instance.
* If you have a python virtual environment already installed you can just `pip` install it.
```
pip install jupyterlab
```
* There are also docker containers containing jupyter lab from [Jupyter Docker Stacks](https://jupyter-docker-stacks.readthedocs.io/en/latest/index.html).

## Project Instructions

1. Create an account with Kaggle.
2. Download the Kaggle dataset using the kaggle python library.
3. Train a model using AutoGluon’s Tabular Prediction and submit predictions to Kaggle for ranking.
4. Use Pandas to do some exploratory analysis and create a new feature, saving new versions of the train and test dataset.
5. Rerun the model and submit the new predictions for ranking.
6. Tune at least 3 different hyperparameters from AutoGluon and resubmit predictions to rank higher on Kaggle.
7. Write up a report on how improvements (or not) were made by either creating additional features or tuning hyperparameters, and why you think one or the other is the best approach to invest more time in.

## License
[License](LICENSE.txt)


##  Directions

Paste the Github link of the repository. Use this link to clone the repository:

nd009t-c1-intro-to-ml-project-starter

It may ask you which kernel you want to use, please make sure you are using the Python 3 (MXNet 1.8 Python 3.7 CPU Optimized) Kernel.

It is recommended to use the ml.t3.medium (2v CPU, 4 GiB Memory) instance that is the default upon startup. While you can select a higher instance type, this instance should suffice.

Before leaving your Sagemaker Studio workspace, always be sure to shut down all running instances and kernels. You access the running instances on the left hand side tool bar.

Run command and get latest. 
### https://www.youtube.com/watch?v=K3ngZKF31mc

```
aws iam list-roles|grep SageMaker-Execution
```


```
conda activate base
```

cd into directory where project.ipynb
```
jupyter nbconvert --to html --execute notebook_name.ipynb --ExecutePreprocessor.kernel_name=python3

jupyter nbconvert --to html project.ipynb
```


  File "/opt/conda/lib/python3.10/site-packages/jupyter_client/kernelspec.py", line 285, in get_kernel_spec
    raise NoSuchKernel(kernel_name)
jupyter_client.kernelspec.NoSuchKernel: No such kernel named conda_python3

## Stand Out Suggestions
You completed the project notebook and wrote the report, but you are still looking for making your project even better. Here are some standout suggestions you might want to try. Remember these are optional, if you would rather submit your work as is that is perfectly fine, just skip the checklist below. Here are some suggestions that may be great themes for standout suggestions.

### Add more than one feature to the dataset and train models to see if it imporves the Kaggle score

### Perform mulptile rounds of hyperparameter tuning to see if it improves Kaggle's score

### Visualizations
- Time series of bike-sharing demand
- Plot correlation matrix of all features *Heatmap or scatter plot
- Plot model trining performnce with more than just the top model







