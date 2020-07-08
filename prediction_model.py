# PROGRAMMER:   Joshua Cayanan
# DATE CREATED: June 29, 2020                                   
# REVISED DATE: 
# PURPOSE: Rebuild pre-trained network, return image classification probability

#Import python modules
import torch
from load_data import process_image
from load_checkpoint import load_checkpoint
import json

def predict(test_image, checkpoint, top_k, category_names, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    #Import image and process
    image = process_image(test_image)
    if gpu == 'cuda':
        image = torch.cuda.FloatTensor(image)
    else:
        image = torch.FloatTensor(image)
    image = image.unsqueeze(0)
    
    #Load image classification model
    model, _, _ = load_checkpoint(checkpoint)
    model.to(gpu)
    
    #Run image through the model
    model.eval()
    log_ps = model(image)
    ps = torch.exp(log_ps)
    probs, classes = ps.topk(top_k, dim = 1)
    
    #Convert classes tensor from integer encoding to actual flower names
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    idx_class = {int(i): v for i,v in cat_to_name.items()}
    flower_names = [idx_class[int(i)] for i in classes.squeeze()]
    
    return flower_names, probs
