import os
import numpy as np
from PIL import Image
import math
from scipy import signal
import pickle
import torch.nn as nn
import torch.nn.functional as F


class LFWDataloader():
    def __init__(self, data_path, batch_size=1):
        self.data_path = data_path
        self.data = []
        self.labels = []
        self.batch_size = batch_size
        self.train_percents = 0.8
        self.get_data()
        self.data_index = np.array(range(len(self.data)))
        self.train_batch_num = int(len(self.data_index)*self.train_percents)//self.batch_size
        self.total_batch_num = len(self.data_index)//self.batch_size
        self.shuffle_all()

    def get_data(self):
        data_match = []
        data_mismatch = []
        path_match = os.path.join(self.data_path, 'match pairs')
        path_mismatch = os.path.join(self.data_path, 'mismatch pairs')

        for dir_name in os.listdir(path_match):
            if dir_name[0]=='.':
                continue
            path = os.path.join(path_match, dir_name)
            pair = []
            for img_name in os.listdir(path):
                img = Image.open(os.path.join(path, img_name)).convert('L')
                img = img.resize((150, 150))
                img = (np.array(img) / 255 - 0.5) / 0.5
                pair.append(img.reshape([150, 150, 1]))
            pair_array = np.concatenate((pair[0], pair[1]), axis=2)
            data_match.append(pair_array)
        print('#####Get match pairs done#####')

        for dir_name in os.listdir(path_mismatch):
            if dir_name[0]=='.':
                continue
            path = os.path.join(path_mismatch, dir_name)
            pair = []
            for img_name in os.listdir(path):
                img = Image.open(os.path.join(path, img_name)).convert('L')
                img = img.resize((150, 150))
                img = (np.array(img)/255-0.5)/0.5
                pair.append(img.reshape([150, 150, 1]))
            pair_array = np.concatenate((pair[0], pair[1]), axis=2)
            data_mismatch.append(pair_array)
        print('#####Get mis match pairs done#####')

        self.data = np.concatenate((np.array(data_match), np.array(data_mismatch)), axis=0)
        self.labels = np.concatenate((np.array([1]*len(data_match)), np.array([0]*len(data_mismatch))), axis=0)

    def shuffle_all(self):
        dir_list = os.listdir('./')
        if 'LFW_index.pkl' in dir_list:
            data_idx_file = open('./LFW_index.pkl', 'rb')
            self.data_index = pickle.load(data_idx_file)
        else:
            np.random.shuffle(self.data_index)
            data_idx_file = open('./LFW_index.pkl', 'wb')
            pickle.dump(self.data_index, data_idx_file)
            data_idx_file.close()

    def shuffle_train(self):
        np.random.shuffle(self.data_index[:self.train_batch_num])

    def data_transform(self, data_out):
        # potential data preprocess
        return (data_out/255 - 0.5)/0.5

    def __len__(self):
        return self.train_batch_num

    def __getitem__(self, item):
        item_list = np.array(range(self.batch_size))+self.batch_size*item
        data_idx = self.data_index[item_list]

        return {'data': self.data_transform(self.data[data_idx]), 'labels': self.labels[data_idx]}


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

        self.weight = np.random.normal(size=[in_features, out_features])*0.1
        if bias:
            self.bias = np.random.normal(size=out_features)*0.1
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
        self.weight = np.random.normal(size=[out_channels, in_channels, kernel_size, kernel_size])*0.1
        self.bias = np.random.normal(size=[out_channels])*0.1



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


class BatchNormalize(Layer):
    def __init__(self):
        super(BatchNormalize, self).__init__()

    def forward(self, input, with_gradient=True):
        pass

    def back_prop(self, next_gradient, lr):
        pass



class MyNet():
    def __init__(self):
        self.cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        }
        self.layers = []
        self.make_layers()

    def make_layers(self):
        self.layers.append(Conv2d(2, 8, 5))
        self.layers.append(Relu())
        self.layers.append(MaxPooling(2, 2))
        self.layers.append(Conv2d(8, 16, 5))
        self.layers.append(Relu())
        self.layers.append(MaxPooling(2, 2))
        self.layers.append(Conv2d(16, 16, 5))
        self.layers.append(Relu())
        self.layers.append(MaxPooling(2, 2))
        self.layers.append(Flatten())
        self.layers.append(Linear(16*16*16, 512))
        self.layers.append(Relu())
        self.layers.append(Linear(512, 128))
        self.layers.append(Relu())
        self.layers.append(Linear(128, 2))
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
        print('Labels= {}'.format(train_dict['labels']))
        print('get out \n {}'.format(out))
        print('====== Loss= {} ====='.format(Loss))
        grad = -y / out
        self.layers[len(self.layers)-1].back_prop(grad, Learning_rate)

    def test(self, test_dict):
        out = self.layers[0].forward(test_dict['data'], with_gradient=False)
        # get cross entropy loss
        y0 = test_dict['labels'].reshape(len(test_dict['labels']), 1)
        y1 = 1 - y0
        y = np.concatenate((y0, y1), axis=1)
        Loss = - y * np.log(out)
        out_choices = 1-np.argmax(out, axis=1)

        TP=0; TN=0; FP=0; FN=0
        for ba in range(out.shape[0]):
            if out_choices[ba] == test_dict['labels'][ba] and out_choices[ba] == 1:
                TP +=1
            if out_choices[ba] == test_dict['labels'][ba] and out_choices[ba] == 0:
                TN +=1
            if out_choices[ba] != test_dict['labels'][ba] and out_choices[ba] == 1:
                FP +=1
            if out_choices[ba] != test_dict['labels'][ba] and out_choices[ba] == 0:
                FN +=1

        return [Loss.sum(), TP, TN, FP, FN]

    def save_model(self, model_path):
        file = open(model_path, 'wb')
        pickle.dump(self, file)
        file.close()
        print('=======saved model to {}======'.format(model_path))

    def eval(self):
        pass




Load_Model = False
dataset_path = './LFW/'
model_path = './new_model_best.pkl'
model_save_path = './new_model_best.pkl'
model_temp_path = './new_model_temp.pkl'

Learning_rate = 0.01

loader = LFWDataloader(dataset_path, 4)

if Load_Model:
    file = open(model_path, 'rb')
    face_net = pickle.load(file)
    file.close()
else:
    face_net = MyNet()

best_f1 = 0

for ep in range(5):
    print("========start Epoch {} ======".format(ep))
    loader.shuffle_train()
    for iteration in range(len(loader)):
        iteration=0
        print('')
        print('========training {}/{}'.format(iteration+1, len(loader)))
        train_dict = loader[iteration]
        face_net.train(train_dict)

        if iteration % 150 == 149:
            print()
            print('==========start Testing========')
            loss=0; TP = 0; TN = 0; FP = 0; FN = 0
            for test_itr in range(loader.train_batch_num, loader.total_batch_num):
                print("=======Testing now   {}/{}".format(test_itr-loader.train_batch_num+1, -loader.train_batch_num+loader.total_batch_num))
                test_dict = loader[test_itr]
                loss_, TP_, TN_, FP_, FN_ = face_net.test(test_dict)
                loss+=loss_; TP+=TP_; TN+=TN_; FP+=FP_; FN=FN_
            print('loss per batch= {}'.format(loss/(loader.total_batch_num-loader.train_batch_num)))
            accu = (TP + TN)/(TP + FP + FN + TN)
            prec = TP /(TP + FP)
            f1_score = 2*(accu*prec)/(accu+prec)

            face_net.save_model(model_temp_path)
            if f1_score > best_f1:
                best_f1 = f1_score
                face_net.save_model(model_save_path)
            print("accu= {}, F1 score= {}".format(accu, f1_score))
