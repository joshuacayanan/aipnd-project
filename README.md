<p float="left">
  <img src="https://github.com/joshuacayanan/aipnd-project/blob/master/assets/deep-learning.png" width="70" align="right"/> 
  <img src="https://github.com/joshuacayanan/aipnd-project/blob/master/assets/flower.png" width="70" align="right"/>
</p>

# AI Programming with Python Project
Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

The image classifier aims to recognize different species of flowers that commonly occur in the United Kingdom. The model has been trained using the [102 Cateogry Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) from the Visual Geometry Group at the University of Oxford. 

As part of the overall application architecture, pre-trained deep learning models from the torchvision models subpackage were used (vgg16 or densenet121). The classifier portions of the models were replaced with the student's work using beginner knowledge of fully connected network architecture. 

## Project Status
The model was trained over four epochs with a testing accuracy of 81%. 

A sample output of the classification model within a Jupyter Notebook is shown below. The top image shows the actual species and the bottom horizontal bar graph plots the model's predictions with associated probabilities.

<p float="left">
  <img src="https://github.com/joshuacayanan/aipnd-project/blob/master/assets/inference_example.png" width="300" align="middle"/>
</p>

## Dependencies
- All code is written in Python 3
- torch
- torchvision
- PIL
- matplotlib
- numpy
- json

## Description of Files
Python script files:
|filename     |description      |
|---          |---              |
|Image Classifier Project.ipynb| Jupyter Notebook of project code with explanatory text |
|get_input_args.py | Two functions to retrieve user command line inputs for training and prediction scripts |
|load_checkpoint.py | Function to load parameters of a trained network from checkpoint.pth file |
|load_data.py | Two functions, one that loads data for training and one that loads a single file to be fed into classifier model. Contains image transform code |
|predict.py | Calls the prediction_model.py function using user command line inputs from get_input_args.py |
|prediction_model.py | Function that feeds an image into classifier neural network and return prediction with associated probability |
|save_checkpoint.py | Function that saves parameters of a trained network for later use |
|train.py | Calls the train_model.py function using user command line inputs from get_input_args.py and generates checkpoint.pth output file using save_checkpoint.py function |
|train_model.py | Function that builds classifier neural network using PyTorch torchvision models subpackage and user defined layers. Training and validation functionality |
|workspace_utils.py | Keeps Udacity workspace active during training |

Input files:
|filename     |description      |
|---          |---              |
|cat_to_name.json | Dictionary of integers as keys and actual flower names as values |


Output files:
|filename     |description      |
|---          |---              |
|checkpoint.pth | PyTorch model checkpoint file, used to rebuild trained model |

## To-Do List
- [ ] Add model architecture section to README
- [ ] Continue training model, try to get higher accuracy

## Credits
Student: Joshua Cayanan
  
_Flower_ icon in readme header made by [Freepik](https://www.flaticon.com/free-icon/flower_2918004) from [www.flaticon.com](www.flaticon.com)  
_Deep Learning_ icon in readme header made by [Becris](https://www.flaticon.com/free-icon/deep-learning_2103787) from [www.flaticon.com](www.flaticon.com)

## Copyright
This project is licensed under the terms of the [MIT license](https://github.com/joshuacayanan/aipnd-project/blob/master/LICENSE) and protected by the [Udacity Honor Code](https://www.udacity.com/legal/en-us/honor-code) and [Community Code of Conduct](https://www.udacity.com/legal/en-us/honor-conduct).
