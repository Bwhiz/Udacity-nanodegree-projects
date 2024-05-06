__author__ = "Ejelonu Benedict"

import torch
from torch import nn, optim
from torch.utils import data
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os, random, json
from collections import OrderedDict


# Defining functions for Transformations and training:

def data_transformation(args):

    train_dir = os.path.join(args.data_directory, "train")
    valid_dir = os.path.join(args.data_directory, "valid")

    if not os.path.exists(args.data_directory):
        print(f"Data Directory doesn't exist: {args.data_directory}")
        raise FileNotFoundError
    if not os.path.exists(args.save_directory):
        print(f"Save Directory doesn't exist: {args.save_directory}")
        raise FileNotFoundError

    if not os.path.exists(train_dir):
        print("Train folder doesn't exist: {train_dir}")
        raise FileNotFoundError
    if not os.path.exists(valid_dir):
        print("Valid folder doesn't exist: {valid_dir}")
        raise FileNotFoundError


    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])


    train_data = ImageFolder(root=train_dir, transform=train_transforms)
    valid_data = ImageFolder(root=valid_dir, transform=valid_transforms)

    train_data_loader = data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_data_loader = data.DataLoader(valid_data, batch_size=64, shuffle=True)

    return train_data_loader, valid_data_loader, train_data.class_to_idx

# Function to Train Model :
def train_model(args, train_data_loader, valid_data_loader, class_to_idx):
    # using options passed to model_arch
    if args.model_arch == "vgg11":
        model = torchvision.models.vgg11(pretrained=True)
    elif args.model_arch == "vgg13":
        model = torchvision.models.vgg13(pretrained=True)
    elif args.model_arch == "vgg16":
        model = torchvision.models.vgg16(pretrained=True)
    elif args.model_arch == "vgg19":
        model = torchvision.models.vgg19(pretrained=True)

    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    in_features_of_pretrained_model = model.classifier[0].in_features # dynamically get the number of input features

    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_features_of_pretrained_model, 2048, bias=True)),
                                        ('relu', nn.ReLU(inplace=True)),
                                        ('dropout', nn.Dropout(p=0.2)),
                                        ('fc2', nn.Linear(2048, 102, bias=True)),
                                        ('output', nn.LogSoftmax(dim=1))
                                           ]))


    model.classifier = classifier
    
    # deciding whether to use GPU or CPU based on User's Input
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    elif args.gpu and not(torch.cuda.is_available()):
        device = 'cpu'
        print("No GPU available, CPU would be used.")
    else:
        device = 'cpu'
    print(f"Using {device} to train model.")
    
    # define Criterion
    criterion = nn.NLLLoss()
    
    #define Optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
       
    # move model to selected device
    model.to(device)

    
    # --------- TRAINING PHASE ---------------------------
    print_every = 20

    # for each epoch
    for e in range(args.epochs):
        step = 0
        running_loss = 0
        
        # for each batch of images
        for inputs, labels in train_data_loader:
            step += 1

            # turn model to train mode
            model.train()

            # move images and model to device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0 or steps == len(train_data_loader) or steps == 1:
                        print(f"current epoch : {epoch+1}/{epochs}, Batch Completion % : {(steps)*100/len(train_data_loader)}")

        # validate
        valid_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            
            for inputs, labels in valid_data_loader:

                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)

                valid_loss += batch_loss.item()
                # accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Train loss: {running_loss/len(trainloader):.3f}.. "
          f"Test loss: {valid_loss/len(validloader):.3f}.. "
          f"Validation accuracy: {(accuracy/len(validloader))*100:.3f}")


    #  save model
    model.class_to_idx = class_to_idx
    checkpoint = {'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'epochs': args.epochs,
                  'optim_stat_dict': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'vgg_type': args.model_arch
                 }
    checkpoint_dir = os.path.join(args.save_directory, "checkpoint.pth")
    torch.save(checkpoint, checkpoint_dir)
    print(f"model saved to {checkpoint_dir}")
    return True
    
    if __name__ == '__main__':
        
        parser = argparse.ArgumentParser()

        # required arguments
        parser.add_argument(dest='data_directory')

        # optional arguments
        parser.add_argument('--save_directory', dest='save_directory', default='../saved_models')
        parser.add_argument('--learning_rate', dest='learning_rate', default=0.003, type=float)
        parser.add_argument('--epochs', dest='epochs', default=3, type=int)
        parser.add_argument('--gpu', dest='gpu', action='store_true')
        parser.add_argument('--model_arch', dest='model_arch', default="vgg19", type=str, choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'])

        args = parser.parse_args()

        # Using the data transformation Function
        train_data_loader, valid_data_loader, class_to_idx = data_transformation(args)

        # Training and saving the model:
        train_model(args, train_data_loader, valid_data_loader, class_to_idx)
