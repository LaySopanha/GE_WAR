import math
import random

import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, search_space,num_sample_pts, classes):
        super(MLP, self).__init__()
        self.num_layers = search_space["layers"]
        self.neurons = search_space["neurons"]
        self.activation = search_space["activation"]

        self.layers = nn.ModuleList()

        for layer_index in range(0, self.num_layers):
            if layer_index == 0:
                self.layers.append(nn.Linear(num_sample_pts, self.neurons))
            else:
                self.layers.append(nn.Linear(self.neurons, self.neurons))

            if self.activation == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activation == 'selu':
                self.layers.append(nn.SELU())
            elif self.activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif self.activation == 'elu':
                self.layers.append(nn.ELU())
        self.softmax_layer = nn.Linear(self.neurons, classes)

    def number_of_parameters(self):
        return (sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.softmax_layer(x) #F.softmax()
        x = x.squeeze(1)
        return x



class CNN(nn.Module):
    def __init__(self, search_space, num_sample_pts, classes):
        super(CNN, self).__init__()
        self.conv_base = nn.Sequential()
        self.dropout_rate = search_space.get("dropout_rate", 0.0)
        kernels, strides, filters, pooling_types, pooling_sizes, pooling_strides, paddings = create_cnn_hp(search_space)
        in_channels, current_len = 1, num_sample_pts
        if isinstance(pooling_types, str):
            pooling_types = [pooling_types] * search_space["conv_layers"]
        for i in range(search_space["conv_layers"]):
            if current_len <=1:
                print(f"Stopped adding layers at layer {i} due to small feature map.")
                break
            kernel_size = min(kernels[i], current_len) if current_len > 0 else 1
            stride = max(1, strides[i])
            padding = paddings[i]
            self.conv_base.add_module(f"conv_{i}", nn.Conv1d(in_channels, filters[i], kernel_size, stride, padding))
            current_len = math.floor(((current_len + (2 * padding) - (kernel_size - 1) - 1) / stride) + 1) if stride > 0 else 0
            activation = search_space.get("activation", "relu")
            if activation == 'selu': self.conv_base.add_module(f"act_{i}", nn.SELU())
            else: self.conv_base.add_module(f"act_{i}", nn.ReLU())
            pool_size = min(pooling_sizes[i], current_len) if current_len > 0 else 1
            pool_stride = max(1, min(pooling_strides[i], current_len) if current_len > 0 else 1)
            pooling_type = pooling_types[i]
            if pooling_type == "max_pool":
                self.conv_base.add_module(f"pool_{i}", nn.MaxPool1d(pool_size, pool_stride))
                current_len = math.floor(((current_len - pool_size) / pool_stride) + 1)
            else:
                self.conv_base.add_module(f"pool_{i}", nn.AvgPool1d(pool_size, pool_stride))
                current_len = math.floor(((current_len - pool_size) / pool_stride) + 1)
            self.conv_base.add_module(f"bn_{i}", nn.BatchNorm1d(filters[i]))
            in_channels = filters[i]
            if current_len <= 1:
                print(f"Stopped after pooling at layer {i} due to small feature map.")
                break
        self.mlp_head = nn.Sequential()
        self.mlp_head.add_module("flatten", nn.Flatten())
        in_features = int(max(1, current_len) * in_channels)
        num_dense_layers = search_space.get("layers", 1)
        neurons = search_space.get("neurons", 256)
        for i in range(num_dense_layers):
            self.mlp_head.add_module(f"dense_{i}", nn.Linear(in_features, neurons))
            self.mlp_head.add_module(f"bn_dense_{i}", nn.BatchNorm1d(neurons))
            
            activation = search_space.get("activation", "relu")
            if activation == 'selu':
                self.mlp_head.add_module(f"dense_act_{i}", nn.SELU())
            else:
                self.mlp_head.add_module(f"dense_act_{i}", nn.ReLU())
            if self.dropout_rate > 0:
                self.mlp_head.add_module(f"dropout_{i}", nn.Dropout(self.dropout_rate))
            in_features = neurons
        self.softmax_layer = nn.Linear(in_features, classes)
    def forward(self, x):
        x = self.conv_base(x)
        x = self.mlp_head(x)
        x = self.softmax_layer(x)
        return x.squeeze(1)



def cal_num_features_conv1d(n_sample_points,kernel_size, stride,padding = 0, dilation = 1):
        L_in = n_sample_points
        L_out = math.floor(((L_in +(2*padding) - dilation *(kernel_size -1 )-1)/stride )+1)
        return L_out


def cal_num_features_maxpool1d(n_sample_points, kernel_size, stride, padding=0, dilation=1):
    L_in = n_sample_points
    L_out = math.floor(((L_in + (2 * padding) - dilation * (kernel_size - 1) - 1) / stride) + 1)
    return L_out

def cal_num_features_avgpool1d(n_sample_points,kernel_size, stride, padding = 0):
    L_in = n_sample_points
    L_out = math.floor(((L_in + (2 * padding) - kernel_size ) / stride) + 1)
    return L_out


def create_cnn_hp(search_space):
    pooling_type = search_space["pooling_types"]
    pool_size = search_space["pooling_sizes"] #size == stride
    conv_layers = search_space["conv_layers"]
    init_filters = search_space["filters"]
    init_kernels = search_space["kernels"] #stride = kernel/2
    init_padding = search_space["padding"] #only for conv1d layers.
    kernels = []
    strides = []
    filters = []
    paddings = []
    pooling_types = []
    pooling_sizes = []
    pooling_strides = []
    for conv_layers_index in range(1, conv_layers + 1):
        if conv_layers_index == 1:
            filters.append(init_filters)
            kernels.append(init_kernels)
            strides.append(int(init_kernels / 2))
            paddings.append(init_padding)
        else:
            filters.append(filters[conv_layers_index - 2] * 2)
            kernels.append(kernels[conv_layers_index - 2] // 2)
            strides.append(int(kernels[conv_layers_index - 2] // 4))
            paddings.append(init_padding)
        pooling_sizes.append(pool_size)
        pooling_strides.append(pool_size)
        pooling_types.append(pooling_type)
    return kernels, strides, filters, pooling_types, pooling_sizes, pooling_strides, paddings




def weight_init(m, type = 'kaiming_uniform_'):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        if type == 'xavier_uniform_':
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('selu'))
        elif type == 'he_uniform':
            nn.init.kaiming_uniform_(m.weight)
        elif type == 'random_uniform':
            nn.init.uniform_(m.weight)
        if m.bias != None:
            nn.init.zeros_(m.bias)




def create_hyperparameter_space(model_type):
    if model_type == "mlp":
        search_space = {"batch_size": random.randrange(100, 1001, 100),
                                                   "lr": random.choice( [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),  # 1e-3, 5e-3, 1e-4, 5e-4
                                                    "optimizer": random.choice( ["RMSprop", "Adam"]),
                                                    "layers": random.randrange(1, 8, 1),
                                                    "neurons": random.choice( [10, 20, 50, 100, 200, 300, 400, 500]),
                                                    "activation": random.choice(  ["relu", "selu", "elu", "tanh"]),
                                                    "kernel_initializer": random.choice(["random_uniform", "glorot_uniform", "he_uniform"]),
                                                }
        return search_space
    elif model_type == "cnn":
        search_space = {"batch_size": random.randrange(100, 1001, 100),
                                              "lr":random.choice( [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),  # 1e-3, 5e-3, 1e-4, 5e-4
                                              "optimizer":random.choice(["RMSprop", "Adam"]),
                                              "layers": random.randrange(1, 8, 1),
                                              "neurons": random.choice( [10, 20, 50, 100, 200, 300, 400, 500]),
                                              "activation": random.choice( ["relu", "selu", "elu", "tanh"]),
                                              "kernel_initializer": random.choice( ["random_uniform", "glorot_uniform", "he_uniform"]),
                                              "pooling_types": random.choice(["max_pool", "average_pool"]),
                                              "pooling_sizes":random.choice(  [2,4,6,8,10]), #size == strides
                                              "conv_layers": random.choice( [1,2,3,4]),
                                              "filters": random.choice( [4,8,12,16]),
                                              "kernels": random.choice( [i for i in range(26,53,2)]), #strides = kernel/2
                                              "padding": random.choice(  [0,4,8,12,16]),
                                        }

        return search_space
