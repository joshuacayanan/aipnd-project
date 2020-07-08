#PROGRAMMER:   Joshua Cayanan
# DATE CREATED: June 29, 2020                     
# REVISED DATE: 
# PURPOSE: Feed an image into classifier neural network and return prediction with associated probability

#   Example call:
#    python predict.py --image 07573.jpg --model checkpoint --mapping cat_to_name.json

#Import functions created for this program
from get_input_args import get_input_args_predict
from prediction_model import predict

#Funnel command line arguments into program
in_arg = get_input_args_predict()

#Assign input arguments to variables
test_image = in_arg.test_image
checkpoint = in_arg.checkpoint
top_k = in_arg.top_k
category_names = in_arg.category_names
gpu = in_arg.gpu
if gpu:
    gpu = 'cuda'
else:
    gpu = 'cpu'

#Call the prediction model function
flower_names, probs = predict(test_image, checkpoint, top_k, category_names, gpu)

print(flower_names)
print(probs)
