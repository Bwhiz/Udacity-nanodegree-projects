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


# Loading Model Checkpoint:
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['vgg_type'] == "vgg11":
        model = torchvision.models.vgg11(pretrained=True)
    elif checkpoint['vgg_type'] == "vgg13":
        model = torchvision.models.vgg13(pretrained=True)
    elif checkpoint['vgg_type'] == "vgg16":
        model = torchvision.models.vgg16(pretrained=True)
    elif checkpoint['vgg_type'] == "vgg19":
        model = torchvision.models.vgg19(pretrained=True)
    
    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    pil_image.resize((256,256))
    
    # centre crop
    width, height = pil_image.size   # Get dimensions
    new_width, new_height = 224, 224
    
    left = round((width - new_width)/2)
    top = round((height - new_height)/2)
    x_right = round(width - new_width) - left
    x_bottom = round(height - new_height) - top
    right = width - x_right
    bottom = height - x_bottom

    # Crop the center of the image
    pil_image = pil_image.crop((left, top, right, bottom))
    
    # convert colour channel from 0-255, to 0-1
    np_image = np.array(pil_image)/255
    
    # normalize for model
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    
    # tranpose color channge to 1st dim
    np_image = np_image.transpose((2 , 0, 1))
    
    # convert to Float Tensor
    tensor = torch.from_numpy(np_image)
    tensor = tensor.type(torch.FloatTensor)
   
    # return tensor
    return tensor

def predict(image_path, model, topk=5, device, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    image = process_image(image_path)
    image = image.unsqueeze(0)

    # move to device
    image = image.to(device)
    model.eval()
    with torch.no_grad():
        ps = torch.exp(model(image))
        
    idx_to_flower = {v:cat_to_name[k] for k, v in model.class_to_idx.items()}    
    
    top_ps, top_classes = ps.topk(topk, dim=1)
    predicted_flowers = [idx_to_flower[i] for i in top_classes.tolist()[0]]

    return top_ps.tolist()[0], predicted_flowers

def print_predictions(args):
    # load model
    model = load_checkpoint(args.model_filepath)

    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    elif args.gpu and not(torch.cuda.is_available()):
        device = 'cpu'
        print("No GPU available, CPU would be used.")
    else:
        device = 'cpu'

    model = model.to(device)

    with open(args.category_names_json_filepath, 'r') as f:
        cat_to_name = json.load(f)

    # predict image
    top_ps, top_classes = predict(args.image_filepath, model, args.top_k, device, cat_to_name)

    print("Predictions:")
    for i in range(args.top_k):
          print(f"#{i:<3} {top_classes[i]:<25} Prob: {top_ps[i]*100:.2f}%")


if __name__ == '__main__':
    
    # Create the parser and add arguments
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument(dest='image_filepath')
    parser.add_argument(dest='model_filepath')

    # optional arguments
    parser.add_argument('--category_names_json_filepath', dest='category_names_json_filepath', default='cat_to_name.json')
    parser.add_argument('--top_k', dest='top_k',  default=5, type=int)
    parser.add_argument('--gpu', dest='gpu',  action='store_true')

    # Parse and print the results
    args = parser.parse_args()

    print_predictions(args)