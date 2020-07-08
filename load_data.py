# PROGRAMMER:   Joshua Cayanan
# DATE CREATED: June 29, 2020                                   
# REVISED DATE: 
# PURPOSE: Two functions, one that loads data for training and one that loads a single file to be predicted

#Import python modules
import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

def load_data(data_dir):
    """
    Defines the training, validation, and testing image directories and applies the relevant transforms. Returns
    dataloader objects for each. Note that dataloader objects are generators.
    
    Parameters:
        data_dir - the location of the images
        
    Returns:
        training_loader, validation_loader, testing_loader
    """
    
    #Define training, validation, and testing directories, structured for use with ImageFolder Class
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #Define image transforms for training, validation, and testing
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(224),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    testing_transforms = validation_transforms


    #Load the datasets with ImageFolder
    training_data = datasets.ImageFolder(train_dir, transform = training_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform = validation_transforms)
    testing_data = datasets.ImageFolder(test_dir, transform = testing_transforms)

    #Using the image datasets and the trainforms, define the dataloaders
    training_loader = torch.utils.data.DataLoader(training_data, batch_size = 64, shuffle = True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size = 64, shuffle = False)
    testing_loader = torch.utils.data.DataLoader(testing_data, batch_size = 64, shuffle = False)
    
    return training_loader, validation_loader, testing_loader

def process_image(image, transpose = True):
    """
    Scales, crops, and normalizes a single PIL image for a PyTorch model, returns numpy array
    """
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)

    #Resize to 256x256 thumbnail 
    width, height = im.size
    if width < height:
        height = int(height * 256 / width)
        im = im.resize((256, height))
    else:
        width = int(width * 256 / height)
        im = im.resize((width, 256))
    
    #Set bounding box and crop
    width, height = im.size
    left = (width - 224) / 2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    
    im = im.crop((left, top, right, bottom))
    
    #Normalize color channel values
    im = np.array(im)
    im = im / 255
    
    #means = np.array([0.485, 0.456, 0.406])
    #sds = np.array([0.229, 0.224, 0.225])
        
    #im = (im - means) / sds 
    
    #Transpose array
    if transpose:
        im = im.transpose((2, 0, 1))
    else:
        return im
    
    return im
