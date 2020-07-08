# PROGRAMMER:   Joshua Cayanan
# DATE CREATED: June 29, 2020                                   
# REVISED DATE: 
# PURPOSE: Create a function that retrieves the following command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the inputs, then the default values are
#          used for the missing inputs. Note that there are two get_input functions,
#          one for the training program and one for the prediction program
#    Inputs for train.py:
#     1. Image Folder as data_dir 
#     2. Checkpoint Save Folder as --save_dir with default value 'saves'
#     3. CNN Model Architecture as --arch with default value 'vgg16' 
#     4. Learning Rate as --learning_rate with default value 0.01
#     5. Hidden Units as --hidden_units with default value of  1024
#     6. Epochs as --epochs with default value of 5
#     7. Use GPU for Training as --gpu with default value of 'cpu'
#
#    Inputs for predict.py       
#     1. Test Image Path as test_image
#     2. Checkpoint as checkpoint
#     3. Top K as --top_k with default value of 1
#     4. Category Mapping as --category_names with default value of cat_to_name.json
#     5. Use GPU for Training as --gpu with default value of 'cpu'


#Import python modules
import argparse

def get_input_args_train():
    #Create Parse using ArgumentParser
    parser = argparse.ArgumentParser() 
    
    #Create 7 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('data_dir', type = str, help = 'path to the folder of flower images')
    parser.add_argument('--save_dir', type = str, default = 'saves', help = 'path to checkpoint save folder')
    parser.add_argument('--arch', type = str, default = 'densenet121', help = 'model architecture')
    parser.add_argument('--learning_rate', type = float, default = 0.01, help = 'learning rate')
    parser.add_argument('--hidden_units', type = int, default = 1024, help = 'number of hidden units in first layer of classifiier portion of network')
    parser.add_argument('--epochs', type = int, default = 5, help = 'number of epochs')
    parser.add_argument('--gpu', type = str, default = 'cpu', help = 'device to use for training, choose "cuda" for gpu, else default is "cpu"')
    
    return parser.parse_args()

def get_input_args_predict():
    #Create Parse using ArgumentParser
    parser = argparse.ArgumentParser() 
    
    #Create 5 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('test_image', type = str, help = 'filepath to image you want a classification prediction for')
    parser.add_argument('checkpoint', type = str, help = 'filepath to model checkpoint file that you want to load')
    parser.add_argument('--top_k', type = int, default = 5, help = 'the number of top k results you want returned')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'maps actual names to image folders which are currently integers')
    parser.add_argument('--gpu', action = 'store_true', default = False, help = 'call --gpu flag if cuda is to be used, else will default to cpu')
    
    return parser.parse_args()
