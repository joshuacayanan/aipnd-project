# PROGRAMMER:   Joshua Cayanan
# DATE CREATED: June 29, 2020                                   
# REVISED DATE: 
# PURPOSE: Save parameters of a trained network for later use

#Import python modules
import torch
from datetime import datetime


#Write checkpoint for classifier portion of model, features portion of model not changed
def save_checkpoint(save_dir, arch, epochs, learning_rate, classifier_layers, model_state, optimizer_state):
    checkpoint = {'arch': arch,
                  'epochs': epochs,
                  'learnrate': learning_rate,
                  'classifier_layers': classifier_layers,
                  'state_dict': model_state,
                  'optimizer_dict': optimizer_state}
    #Generate filename with current timestamp
    now = datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    filename = save_dir + '/checkpoint_' + timestamp + '.pth'
    
    #Save to file
    torch.save(checkpoint, filename)