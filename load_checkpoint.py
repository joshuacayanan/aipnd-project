# PROGRAMMER:   Joshua Cayanan
# DATE CREATED: June 29, 2020                                   
# REVISED DATE: 
# PURPOSE: Load parameters of a trained network

#Import python modules
import torch
from torchvision import models

#Import pre-trained models
vgg16 = models.vgg16(pretrained = True)
densenet121 = models.densenet121(pretrained = True)

#Dictionary of models and the number of neurons in the last convolutional layer
models = {'vgg16': vgg16, 'densenet121': densenet121}

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    model = models[checkpoint['arch']]
    
    #Make sure to freeze densenet feature layer parameters, otherwise will throw cuda runtime error (2): out of memory
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = checkpoint['classifier_layers']
    model.load_state_dict(checkpoint['state_dict'])
    
    epochs = checkpoint['epochs']
    learning_rate = checkpoint['learnrate']
    
    return model, epochs, learning_rate
