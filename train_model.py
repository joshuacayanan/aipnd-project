# PROGRAMMER:   Joshua Cayanan
# DATE CREATED: June 29, 2020                                   
# REVISED DATE: 
# PURPOSE: Contains code for classifer neural networks

#Import python modules
import torch
from torch import nn, optim
from torchvision import models
from load_data import load_data

#Import pre-trained models
vgg16 = models.vgg16(pretrained = True)
densenet121 = models.densenet121(pretrained = True)

#Dictionary of models and the number of neurons in the last convolutional layer
models = {'vgg16': [vgg16, 25088], 'densenet121': [densenet121, 1024]}


def train_model(data_dir, arch, learning_rate, hidden_units, epochs, gpu):
    #Create the generator objects for the training and validation data
    training_loader, validation_loader, _ = load_data(data_dir)
    
    #Choose model
    model = models[arch][0]
    conv_lastlayer_neurons = models[arch][1]
    
    #Freeze features layers
    for param in model.parameters():
        param.requires_grad = False
   
    #Create new classifier 
    classifier = nn.Sequential(nn.Linear(conv_lastlayer_neurons, hidden_units),
                           nn.ReLU(),
                           nn.Dropout(p = 0.5),
                           nn.Linear(hidden_units, 102),
                           nn.LogSoftmax(dim = 1))
    model.classifier = classifier
    
    #Define loss function
    criterion = nn.NLLLoss()

    #Define optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    
    #Use GPU if available
    
    model.to(gpu)

    #Train the model
    steps = 0
    training_loss = 0
    print_every = 5
    
    for epoch in range(epochs):
        for images, labels in training_loader:
            images, labels = images.to(gpu), labels.to(gpu)

            steps += 1 

            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            #Enter validation mode every five batches
            if steps % print_every == 0:
                model.eval()
                validation_loss = 0
                accuracy = 0

                for images, labels in validation_loader:
                    images, labels = images.to(gpu), labels.to(gpu)

                    log_ps = model(images)
                    loss = criterion(log_ps, labels)

                    validation_loss += loss.item()

                    #Calculate accuracy
                    ps = torch.exp(log_ps)
                    top_ps, top_class = ps.topk(1, dim = 1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor))

                #Print out model statistics for every five batches
                print(f'Epoch {epoch + 1} / {epochs}',
                        f'Training loss: {training_loss/print_every:.3f}',
                        f'Validation loss: {validation_loss/len(validation_loader):.3f}',
                        f'Accuracy: {accuracy/len(validation_loader):.3f}')

                #Return the model to training mode
                training_loss = 0
                model.train()
      
    return model.classifier, model.state_dict(), optimizer.state_dict()

                