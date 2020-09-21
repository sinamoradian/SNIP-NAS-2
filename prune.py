import numpy as np

import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torchvision
from torch import autograd


#This script prunes neural network using sensitivity function described in SNIP
#for a given neural network (for now) this script prunes the X% weights and "alphas"
#with least sensitivity

#We wrote this version of the code for calculating sensitivity (using SNIP's criteria)
#and then pruning weights and biases
#this code was written for and tested with LeNet

#Goals for this script:
#1- This code was originally created for single input. Update it to handle a mini batch
#It can already handle minibatch. CrossEntropyLoss is designed with minibatch in mind.

#2- figure out how to prune weights, biases, and alphas at the same time. Can the alphas?
#how did DARTS do this?


# removed the code that Sina wrote for loading input and labels to snip functiona
# TODO: Check if we need to call this function for our minibatches or are the changes in train_search enough?
def load_data_for sinp():

    #I want to use CIFAR-10 with LeNET
    #download CIFAR-10 ; CIFAR-10 is downloaded to the data folder by line 110 in main() of train_search.py
    transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    # why don't we ever use trainLoader?
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

    #do we have to know the classes beforehand?
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataiter = iter(testloader)
    images, labels = dataiter.next() #I'm not sure if I need to have requires_grad=True
    return images, labels

# TODO: this function is based on LeNET must be changed for DARTS architecture
def print_sparsity_info(model):
    print(
        "Sparsity in conv1.weight: {:.2f}%".format(
            100. * float(torch.sum(model.conv1.weight == 0))
            / float(model.conv1.weight.nelement())
        )
    )
    print(
        "Sparsity in conv2.weight: {:.2f}%".format(
            100. * float(torch.sum(model.conv2.weight == 0))
            / float(model.conv2.weight.nelement())
        )
    )
    print(
        "Sparsity in fc1.weight: {:.2f}%".format(
            100. * float(torch.sum(model.fc1.weight == 0))
            / float(model.fc1.weight.nelement())
        )
    )
    print(
        "Sparsity in fc2.weight: {:.2f}%".format(
            100. * float(torch.sum(model.fc2.weight == 0))
            / float(model.fc2.weight.nelement())
        )
    )
    print(
        "Sparsity in fc3.weight: {:.2f}%".format(
            100. * float(torch.sum(model.fc3.weight == 0))
            / float(model.fc3.weight.nelement())
        )
    )
    print(
        "Global sparsity: {:.2f}%".format(
            100. * float(
                torch.sum(model.conv1.weight == 0)
                + torch.sum(model.conv2.weight == 0)
                + torch.sum(model.fc1.weight == 0)
                + torch.sum(model.fc2.weight == 0)
                + torch.sum(model.fc3.weight == 0)
            )
            / float(
                model.conv1.weight.nelement()
                + model.conv2.weight.nelement()
                + model.fc1.weight.nelement()
                + model.fc2.weight.nelement()
                + model.fc3.weight.nelement()
            )
        )
    )


def snip(model, inputs, labels):

    #alphas of the initial model with shape : two dimenssion matrix of size (k=14, num_ops=8)
    alphas = model.arch_parameters()
    model.zero_grad()
    criterion = nn.CrossEntropyLoss()
    #optimizer.zero_grad()

    outputs = model.forward(inputs)
    #should gradients be zeroed out here?
    loss = criterion(outputs, labels)

    #Sina: I think autograd.grad calculates the grads but doesn't save it in the model.grad attribute
    gradients = autograd.grad(loss, model.parameters())
    abs_gradients = [None] * len(gradients)

    #initialize sigma_gradients as a tesnor with correct size. values of the tensor don't matter
    sigma_gradients = torch.zeros(len(gradients))

    for i in range(0, len(gradients)):
      abs_gradients[i] = torch.abs(gradients[i])
      sigma_gradients[i] = torch.sum(abs_gradients[i])

    sigma_sigma_gradients = torch.sum(sigma_gradients) #calc sum of all layers

    #calculate sensitivity by dividing the gradients with sum of gradients
    sensitivity = [None] * len(gradients) #create an empty list with appropriate size
    for i in range(0, len(gradients)):
      sensitivity[i] = torch.div(torch.abs(abs_gradients[i]), sigma_sigma_gradients)

    #manually resets all weights, biases to zero.
    #the .grad attribute is empt
    for f in model.parameters():
      f.data.fill_(0)
    #manually update all grads to those calculated in gradients
    i=0
    for f in model.parameters():
      f.grad = abs_gradients[i] # grad[i]
      if i==1:
        print(f.grad)
        print(gradients[i])
      i = i + 1

    #update the weights using the updated grads above
    #this is basically the same as using the optimizer, but I just felt more comfortable this way
    learning_rate = -1
    for f in model.parameters():
      f.data.sub_(f.grad.data * learning_rate)
    # I wonder if I can just directly upload the gradients to weights instead of using this method

#model = LeNet()

    parameters_to_prune = (
        (model.conv1, 'weight'),
        (model.conv2, 'weight'),
        (model.fc1, 'weight'),
        (model.fc2, 'weight'),
        (model.fc3, 'weight'),
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.35, #what is amount exactly? is it the total portion of weights that are pruned?
    )

    print_sparsity_info(model)

    return model
