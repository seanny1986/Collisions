import numpy as np
import torch
import torch.nn.functional as Func
from torch.autograd import Variable

# class for creating and using dynamic encoder-decoder network. Instantiate an encoder and decoder network
# using a topology vector that describes the network architecture (e.g. [a, b, ...,n-1, n ] for a 
# network with an input of size a, hidden layers of size b, ..., n-1, and an output layer of dimension n).
# From there, instantiate the dynamic encoder-decoder by passing it the encoder and decoder models.

# Data handling and batching requires some thought, and you need to handle it yourself if you want to pass
# jagged arrays to the network (i.e. its main advantage). My suggestion is to have a jagged python list, and
# then to pass the necessary chunks to the dynamic encoder-decoder. A basic function to do this has been 
# provided below for the case of 1D interactions (e.g. an officer with variable violations). A training function
# for 2D interactions (e.g. balls bouncing off of one another) has been written, but still needs to be polished.

# -- Sean Morrison, 2017

class DynamicEncoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(DynamicEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, F, X):
        m = self.encoder.topology[-1]
        enc = Variable(torch.zeros(m))
        check = X.data.numpy().size
        if check != 0:
            for x in X:
                inp = torch.cat([F, x])
                effect = self.encoder(inp)
                enc = enc+effect
        h = torch.cat([F, enc])
        output = self.decoder(h)
        return output

# quick class to generate an MLP using a list to define network topology. An MLP is used for both
# encoding and decoding in the dynamic autoencoder used above. In theory, other types of network could
# be used (e.g. a variational autoencoder to recover uncertainty information by sampling z).
# Instantiate a fully connected network using net = npe.FullyConnectedNetwork(topology), where topology
# is a list of dimensions for each layer, e.g. topology = [10, 500, 500, 2] for a network with 10 inputs,
# 2 hidden layers of 500 neurons, and 2 outputs.

class FullyConnectedNetwork(torch.nn.Module):
    def __init__(self, topology):
        super(FullyConnectedNetwork,self).__init__()
        self.topology = topology
        m = len(topology)
        self.layers = torch.nn.ModuleList()
        for i in range(0,m-1):
            self.layers.append(torch.nn.Linear(topology[i],topology[i+1]))

    def forward(self, X):
        i = 0
        while i < len(self.layers)-1:
            X = Func.relu(self.layers[i](X))
            i += 1
        out = self.layers[-1](X)
        return out

# quick function for training the dynamic autoencoder. This function takes in a jagged list of
# interaction data, which is sequentially fed into the network. This is only for the 1D case.
# For the 2D interaction case, there is another method below.

def train(model, iterations, X, Y, criterion, optimizer):
    for i in range(0,iterations):
        total_loss = 0
        j = 0
        for x in X:
            focus = x[0]
            context = x[1]
            foc = Variable(torch.FloatTensor(focus))
            con = Variable(torch.FloatTensor(context))
            y = Y[j]
            y = Variable(torch.FloatTensor(y))
        
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(foc, con)

            # Compute and print loss
            loss = criterion(y_pred, y)
            total_loss = total_loss+loss.data[0]
        
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            j += 1
        print('Iteration: ' + str(i+1) + '/' + str(iterations) + ', Loss: ' + str(total_loss))