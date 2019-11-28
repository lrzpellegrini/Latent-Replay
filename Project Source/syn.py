####################################################################################################
# Copyright (c) 2019. Lorenzo Pellegrini, Gabriele Graffieti, Vincenzo Lomonaco, Davide Maltoni    #
#                                                                                                  #
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file #
####################################################################################################

import numpy as np

import sys
import os, time

import train_utils

SyN_use_simple_gradient = False
Syn_free_layers = {}


def create_syn_data(net):
    size = 0
    print('Creating Syn data for Optimal params and their Fisher info')
    for param in net.params.items():
        layer_name = param[0]
        layer_pos = list(net._layer_names).index(layer_name)  # Layer position (consider it an ID)
        free_layer = True if net.layers[layer_pos].type in ['BatchNorm', 'BatchReNorm', 'Scale'] else False  # ['BatchNorm','Scale','Convolution']
        Syn_free_layers[layer_name] = free_layer
        offset_start = size;
        num_weights = param[1][0].count
        size += num_weights  # First index is the blob name, second index = 0 denotes weight blob
        num_bias = param[1][1].count if len(param[1]) > 1 else 0
        size += num_bias  # First index is the blob name, second index = 1 denotes bias blob
        if len(param[1]) > 2:  # Does it have extra parameter blobs ?
            for i in range(2, len(param[1])):
                size += param[1][i].count  # BatchNorm layer has three blobs, BatchRenorm 4, etc!
        if free_layer:
            print('Layer {:s}: free!'.format(layer_name))
        else:
            print('Layer {:s}: Weight {:d}, Bias {:d}, OffsetStart {:d}'.format(layer_name, num_weights, num_bias, offset_start))
    print('Total size {:d}'.format(size))
    # The first array returned is a 2D array: the first component contains the params at loss minimum, the second the parameter importance
    # The second array is a dictionary with the synData
    synData = {}
    synData['old_theta'] = np.zeros(size, dtype=np.float32)
    synData['new_theta'] = np.zeros(size, dtype=np.float32)
    synData['grad'] = np.zeros(size, dtype=np.float32)
    synData['trajectory'] = np.zeros(size, dtype=np.float32)
    synData['cum_trajectory'] = np.zeros(size, dtype=np.float32)
    return np.zeros((2, size), dtype=np.float32), synData


def extract_weights(net, target):
    # Store the network weights into target
    offset = 0
    target[...] = 0
    for param in net.params.items():
        weights = param[1][0].data.flatten()
        # weights[...] = 0
        target[offset:offset + weights.size] = weights
        offset += weights.size
        if len(param[1]) > 1:
            bias = param[1][1].data.flatten()
            # bias[...] = 1
            target[offset:offset + bias.size] = bias
            offset += bias.size
        if len(param[1]) > 2:
            for i in range(2, len(param[1])):
                offset += param[1][i].count


def extract_grad(net, target):
    # Store the gradients into target
    offset = 0
    target[...] = 0
    for param in net.params.items():
        g_weights = param[1][0].diff.flatten()
        # weights[...] = 0
        target[offset:offset + g_weights.size] = g_weights
        offset += g_weights.size
        if len(param[1]) > 1:
            g_bias = param[1][1].diff.flatten()
            # bias[...] = 1
            target[offset:offset + g_bias.size] = g_bias
            offset += g_bias.size
        if len(param[1]) > 2:
            for i in range(2, len(param[1])):
                offset += param[1][i].count


def extract_grad_in_cwr(net, target, cwr_layers):
    # Store the gradients into target
    offset = 0
    target[...] = 0

    for layer, param in net.params.items():
        if (layer in cwr_layers) or Syn_free_layers[layer]:  # do not add constraints to fisher (leave at 0 -> free to move)
            g_weights = param[0].diff.flatten()
            target[offset:offset + g_weights.size] = 0
            offset += g_weights.size
            if len(param) > 1:
                g_bias = param[1].diff.flatten()
                target[offset:offset + g_bias.size] = 0
                offset += g_bias.size
            if len(param) > 2:
                for i in range(2, len(param)):
                    offset += param[i].count
        else:
            g_weights = param[0].diff.flatten()
            target[offset:offset + g_weights.size] = g_weights
            offset += g_weights.size
            if len(param) > 1:
                g_bias = param[1].diff.flatten()
                target[offset:offset + g_bias.size] = g_bias
                offset += g_bias.size
            if len(param) > 2:
                for i in range(2, len(param)):
                    offset += param[i].count


def init_batch(net, ewcData, synData):
    extract_weights(net, ewcData[0])  # Keep initial weights
    synData['trajectory'] = 0


def pre_update(net, ewcData, synData):
    extract_weights(net, synData['old_theta'])


def post_update(net, ewcData, synData, cwr_layers=None):
    extract_weights(net, synData['new_theta'])
    if cwr_layers is None:
        extract_grad(net, synData['grad'])
    else:
        extract_grad_in_cwr(net, synData['grad'], cwr_layers)

    if SyN_use_simple_gradient:
        synData['trajectory'] += np.abs(synData['grad']) * -0.000001
    else:
        # print(np.sum(synData['grad']), np.sum(synData['new_theta'] - synData['old_theta']))
        synData['trajectory'] += synData['grad'] * (synData['new_theta'] - synData['old_theta'])
    # synData['trajectory'] = 0.9 * synData['trajectory'] + 1.9 * (synData['grad'] * (synData['new_theta'] - synData['old_theta']))


def update_ewc_data(net, ewcData, synData, batch, clip_to, c=0.0015):
    extract_weights(net, synData['new_theta'])
    eps = 0.0000001  # 0.001 in few task - 0.1 used in a more complex setup
    
    if SyN_use_simple_gradient:
        synData['cum_trajectory'] += c/eps * synData['trajectory']
    else:
        synData['cum_trajectory'] +=  c * synData['trajectory'] / (np.square(synData['new_theta'] - ewcData[0]) + eps)

    ewcData[1] = np.copy(-synData['cum_trajectory'])   # change sign here because the Ewc regularization in Caffe (theta - thetaold) is inverted w.r.t. syn equation [4] (thetaold - theta)
    ewcData[1][ewcData[1]>clip_to] = clip_to

    # ewcData[1][ewcData[1]>0] = 5e-6  # use this to sligtly freeze all weights (except CWR)

    ewcData[0] = np.copy(synData['new_theta'])


def target_train_loss_accuracy_per_batch(batch):
    loss = [0, 1, 2, 3, 4.5, 5.0, 5.0, 5.0, 5.0]
    return loss[batch]


def weight_stats(net, batch, ewcData, clip_to):
    print('Average F saturation = %.3f%%' % (100*np.sum(ewcData[1])/(ewcData[1].size*clip_to)), ' Max = ', np.max(ewcData[1]), ' Size = ', ewcData[1].size)
    offset = 0
    checksum = 0
    level_sum={}
    for levname, param in net.params.items():
        sizew = param[0].data.size
        sizeb = param[1].data.size if len(param) > 1 else 0
        level_sum[levname, 0] = np.sum(ewcData[1][offset:offset+sizew])
        level_sum[levname, 1] = np.sum(ewcData[1][offset+sizew:offset+sizew+sizeb])
        print(levname, ' W = %.3f%%  B = %.3f%%' % (100*level_sum[levname, 0] / (sizew*clip_to), 100*level_sum[levname, 1] / (sizeb*clip_to)))
        checksum += level_sum[levname, 0] + level_sum[levname, 1]
        offset += sizew + sizeb
    # print('CheckSum Weights: ', checksum, ' Size = ', offset)
