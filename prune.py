import numpy as np

import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torch import autograd

snip(model, inputs, labels):
    criterion = nn.CrossEntropyLoss()
    outputs = model.forward(inputs)
    loss = criterion(outputs, labels)

    gradients = autograd.grad(loss, model.parameters()) #Sina: I think autograd.grad calculates the grads but doesn't save it in the model.grad attribute
    abs_gradients = [None] * len(gradients)
    sigma_gradients = torch.zeros(len(gradients)) #initialize sigma_gradients as a tesnor with correct size. values of the tensor don't matter

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

    return model
