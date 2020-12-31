# Capstone Project - Azure Machine Learning Engineer 

In this project, a dataset i.e. bank note authentication has been used and fed to Azure ML services. Two approaches are taken on the dataset and they are as follows. 

* Automated ML: AutoML is an automation way to deal with time consuming tasks of machine learning model. As a result, it selects the best model after running through multiple algorithms. It saves time and resource. 

* Hyperdrive: Hyperdrive helps tune hyperparameters for the model. In this case, we need to select algorithm and hyperparameters to be tuned. 

## Project Set Up and Installation

In order to run notebooks in this project, the following needs to be met. 

* Access Microsoft Azure Portal.
* Create workspace in Azure ML Studio.
* Create a VM to run Jupyter Notebook.
* Register dataset that can be accessed from notebook.
* Upload .py an .ipynb files into workspace. 

## Dataset

### Overview

Data were extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.

1. variance of Wavelet Transformed image (continuous) 
2. skewness of Wavelet Transformed image (continuous) 
3. curtosis of Wavelet Transformed image (continuous) 
4. entropy of image (continuous) 
5. class (integer i.e. 0/1) 

Citation: dataset and above note are from [UCI](https://archive.ics.uci.edu/ml/datasets/banknote+authentication).

### Task

It is a classification problem to find out whether the bank note is genuine or forged.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
