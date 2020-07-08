# AI Programming with Python Project

## About
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

## To-Do List
- [ ] Add 'Description of files' section to README
- [ ] Add model architecture section to README
- [ ] Continue training model, try to get higher accuracy

## Copyright
This project is licensed under the terms of the [MIT license](https://github.com/joshuacayanan/aipnd-project/blob/master/LICENSE) and protected by the [Udacity Honor Code](https://www.udacity.com/legal/en-us/honor-code) and [Community Code of Conduct](https://www.udacity.com/legal/en-us/honor-conduct).
