# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
import math
import pickle
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#######################################################################
class Layer():
    def __init__(self):
        self.input_x = None
        self.pre_layer = None
        self.next_layer = None
        self.batch_size= 1

    def reset_parameters(self):
        pass

    def forward(self, input, with_gradient=True):
        pass

    def back_prop(self, next_gradient, lr):
        pass


class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = np.random.normal(size=[in_features, out_features])
        if bias:
            self.bias = np.random.normal(size=out_features)
        else:
            self.bias = None

        self.reset_parameters()

        self.gradient_in = None
        self.gradient_weight = None
        self.gradient_bias = None

    def get_gradient(self):
        self.gradient_in = self.weight.copy()
        self.gradient_bias = np.eye(self.out_features)
        self.gradient_weight = np.zeros([self.batch_size, self.in_features * self.out_features, self.out_features])

        for idx in range(self.out_features):
            self.gradient_weight[:, (idx*self.out_features):((idx+1)*self.out_features), idx] = np.repeat(self.input_x[:, idx:idx+1], self.out_features, axis=1)

    def forward(self, input, with_gradient=True):
        self.input_x = input.copy()
        self.batch_size = input.shape[0]
        if len(self.input_x.shape) == 1:
            self.input_x = self.input_x.reshape(self.batch_size, -1)

        if with_gradient:
            self.get_gradient()

        if self.next_layer is None:
            if self.bias is not None:
                return np.dot(input, self.weight) + self.bias
            else:
                return np.dot(input, self.weight)
        else:
            if self.bias is not None:
                return self.next_layer.forward(np.dot(input, self.weight) + self.bias)
            else:
                return self.next_layer.forward(np.dot(input, self.weight))

    def back_prop(self, next_gradient, lr):
        if self.pre_layer is not None:
            pass_to_next = np.dot(next_gradient, self.gradient_in.T)

        # update weight
        update_weight = np.empty([self.batch_size, self.in_features, self.out_features])
        for ba in range(self.batch_size):
            update_weight[ba] = np.dot(next_gradient[ba], self.gradient_weight[ba].T).reshape(self.in_features, self.out_features)
        update_weight = update_weight.mean(0)
        self.weight -= lr * update_weight

        # update bias
        update_bias = np.dot(next_gradient, self.gradient_bias).mean(0)
        self.bias -= lr * update_bias

        if self.pre_layer is not None:
            self.pre_layer.back_prop(pass_to_next, lr)


class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, padding_mode=False, bias=True):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.bias = bias

        self.batch_size = 1
        self.input_m = 1
        self.input_n = 1

        # parameters
        self.weight = np.random.normal(size=[out_channels, in_channels, kernel_size, kernel_size])
        self.bias = np.random.normal(size=[out_channels])



    def reset_parameters(self):
        pass

    def forward(self, input, with_gradient=True):
        assert input.shape[-1]==self.in_channels, 'In channels wrong'
        self.batch_size, self.input_m, self.input_n, _ = input.shape
        self.input_x = input.copy()
        # Use multiprocessing Here!!
        output = np.empty([self.batch_size, self.input_m - self.kernel_size +1, self.input_n-self.kernel_size+1, self.out_channels])
        temp_out = np.empty([self.in_channels, self.input_m - self.kernel_size +1, self.input_n-self.kernel_size+1])
        for ba in range(self.batch_size):
            for out_id in range(self.out_channels):
                for in_id in range(self.in_channels):
                    temp_out[in_id] = signal.convolve2d(input[ba, :, :, in_id], self.weight[out_id, in_id, ::-1, ::-1], mode='valid')
                output[ba, :, :, out_id] = temp_out.sum(axis=0) + self.bias[out_id]

        if self.next_layer is None:
            return output
        else:
            return self.next_layer.forward(output)


    def back_prop(self, next_gradient, lr):
        gradient_weight = np.empty([self.batch_size, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size])
        gradient_in = np.empty([self.batch_size, self.input_m, self.input_n, self.in_channels, self.out_channels])
        gradient_bias = np.empty([self.out_channels])

        # Get gradient for pre layer
        if self.pre_layer is not None:
            for ba in range(self.batch_size):
                for in_id in range(self.in_channels):
                    for out_id in range(self.out_channels):
                        next_grad_pad = np.pad(next_gradient[ba, :, :, out_id], ((self.kernel_size-1,self.kernel_size-1),), 'constant', constant_values=0)
                        gradient_in[ba, :, :, in_id, out_id] = signal.convolve2d(next_grad_pad, self.weight[out_id, in_id, ::-1, ::-1], mode='valid')

            gradient_in = gradient_in.sum(axis=4)

        # update weight
        for ba in range(self.batch_size):
            for out_id in range(self.out_channels):
                for in_id in range(self.in_channels):
                    gradient_weight[ba, out_id, in_id] = signal.convolve2d(self.input_x[ba, :, :, in_id], next_gradient[ba, ::-1, ::-1, out_id], mode='valid')
        gradient_weight = gradient_weight.mean(axis=0)
        self.weight -= lr * gradient_weight

        # update bias
        gradient_bias = next_gradient.mean(axis=0).sum(axis=(0, 1))
        self.bias -= lr * gradient_bias

        if self.pre_layer is not None:
            self.pre_layer.back_prop(gradient_in, lr)


class Relu(Layer):
    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, input, with_gradient=True):
        self.input_x = input.copy()
        return self.next_layer.forward(np.maximum(input, 0))

    def back_prop(self, next_gradient, lr):
        self.pre_layer.back_prop(np.where(self.input_x > 0, next_gradient, 0), lr)


class MaxPooling(Layer):
    def __init__(self, kernel_size, stride, padding=False):
        super(MaxPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, input, with_gradient=True):
        self.batch_size, M, N, channels = input.shape

        out_M = math.ceil(M/self.kernel_size)
        out_N = math.ceil(N/self.kernel_size)
        input = np.pad(input, ((0,0), (0,out_M*self.kernel_size-M), (0,out_N*self.kernel_size-N), (0,0)), 'constant', constant_values=0)

        temp = input.reshape((self.batch_size, out_M, self.kernel_size, out_N, self.kernel_size, channels))
        output = temp.max(axis=(2,4))

        temp2 = np.repeat( np.repeat(output.reshape((self.batch_size, out_M, 1, out_N, 1, channels)), self.kernel_size, axis=2), self.kernel_size, axis=4)
        self.input_x = np.where(temp == temp2, 1, 0).reshape((self.batch_size, out_M*self.kernel_size, out_N*self.kernel_size, channels))[:, :M, :N, :]

        return self.next_layer.forward(output)

    def back_prop(self, next_gradient, lr):
        _, out_M, out_N, channels = next_gradient.shape

        next_gradient = np.repeat(np.repeat(next_gradient.reshape((self.batch_size, out_M, 1, out_N, 1, channels)), self.kernel_size, axis=2), self.kernel_size, axis=4)
        next_gradient.shape = (self.batch_size, out_M*self.kernel_size, out_N*self.kernel_size, channels)
        next_gradient = next_gradient[:, :self.input_x.shape[1], :self.input_x.shape[2], :]
        self.pre_layer.back_prop(self.input_x * next_gradient, lr)


class SoftMax(Layer):
    def __init__(self):
        super(SoftMax, self).__init__()
        self.gradient_in = None

    def forward(self, input, with_gradient=True):
        print('softmax input= \n {}'.format(input))
        self.batch_size = input.shape[0]
        feature_len = input.shape[1]
        self.input_x = input.copy()
        output = np.zeros_like(input)
        self.gradient_in = np.empty([self.batch_size, feature_len, feature_len])
        for ba in range(self.batch_size):
            exps = np.exp(input[ba] - np.max(input[ba]))  # shift x, make it stable
            output[ba] = exps / exps.sum()

            # avoid zeros
            if output[ba, 0] > 1 - 1e-10:
                output[ba, 0] -= 1e-10
                output[ba, 1] += 1e-10
            if output[ba, 0] < 1e-10:
                output[ba, 0] += 1e-10
                output[ba, 1] -= 1e-10

            if not with_gradient:
                continue

            for idx in range(feature_len):
                for idy in range(feature_len):
                    if idx == idy:
                        self.gradient_in[ba, idx, idy] = output[ba, idx] * (1-output[ba, idy])
                    else:
                        self.gradient_in[ba, idx, idy] = output[ba, idx] * (-output[ba, idy])
            self.gradient_in[ba] = self.gradient_in[ba].T

        # self.next_layer.forward(output)
        ### Should be the last layer
        return output

    def back_prop(self, next_gradient, lr):
        pre_grad = np.zeros_like(next_gradient)
        for ba in range(self.batch_size):
            pre_grad[ba] = np.dot(self.gradient_in[ba], next_gradient[ba])
        self.pre_layer.back_prop(pre_grad, lr)


class Flatten(Layer):
    def __init__(self):
        super(Flatten, self).__init__()
        self.in_shape = None
        self.out_shape = None

    def forward(self, input, with_gradient=True):
        self.in_shape = input.shape
        flat_num = 1
        for sh in self.in_shape:
            flat_num *= sh
        input.shape = (self.in_shape[0], flat_num//self.in_shape[0])
        self.out_shape = input.shape
        return self.next_layer.forward(input)

    def back_prop(self, next_gradient, lr):
        next_gradient.shape = self.in_shape
        self.pre_layer.back_prop(next_gradient, lr)

class MyNet():
    def __init__(self):
        self.layers = []
        self.make_layers()

    def make_layers(self):
        self.layers.append(Linear(32*32, 512))
        self.layers.append(Linear(512, 128))
        self.layers.append(Linear(128, 10))
        self.layers.append(SoftMax())

        for la in range(len(self.layers)):
            if la:
                self.layers[la].pre_layer = self.layers[la-1]
            if la < len(self.layers)-1:
                self.layers[la].next_layer = self.layers[la+1]

    def train(self, train_dict):
        out = self.layers[0].forward(train_dict['data'])

        # get cross entropy loss
        y0 = train_dict['labels'].reshape(len(train_dict['labels']), 1)
        y1 = 1-y0
        y = np.concatenate((y0, y1), axis=1)
        Loss = - y * np.log(out)
        Loss = Loss.sum()
        grad = -y / out
        self.layers[len(self.layers)-1].back_prop(grad, 0.001)
        return Loss

    def test(self, test_dict):
        out = self.layers[0].forward(test_dict['data'], with_gradient=False)

        return out

    def save_model(self, model_path):
        file = open(model_path, 'wb')
        pickle.dump(self, file)
        file.close()
        print('=======saved model to {}======'.format(model_path))

    def eval(self):
        pass


net = Net()
########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        train_data = {"data":inputs, "labels":labels}

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

########################################################################
# Let's quickly save our trained model:

# PATH = './cifar_net.pkl'
# net.save_model(PATH)

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

########################################################################
# See `here <https://pytorch.org/docs/stable/notes/serialization.html>`_
# for more details on saving PyTorch models.
#
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

# dataiter = iter(testloader)
# images, labels = dataiter.next()
#
# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# Next, let's load back in our saved model (note: saving and re-loading the model
# wasn't necessary here, we only did it to illustrate how to do so):

# net_file = open(PATH, 'rb')
# net = pickle.load(net_file)

net = Net()
net.load_state_dict(torch.load(PATH))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:

# outputs = net(images)

########################################################################
# The outputs are energies for the 10 classes.
# The higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
# _, predicted = torch.max(outputs, 1)
#
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                               for j in range(4)))

########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # outputs = net.test(images)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))