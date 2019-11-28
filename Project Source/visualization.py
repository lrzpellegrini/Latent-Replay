####################################################################################################
# Copyright (c) 2019. Lorenzo Pellegrini, Gabriele Graffieti, Vincenzo Lomonaco, Davide Maltoni    #
#                                                                                                  #
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file #
####################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import time
# Set display defaults

import itertools

plt.rcParams['figure.figsize'] = (10, 10)        # Large images
plt.rcParams['image.interpolation'] = 'nearest'  # Don't interpolate
plt.rcParams['image.cmap'] = 'gray'  # Use grayscale output rather than a (potentially misleading) color heatmap

# From Inference ...

def ShowImage(image):
    plt.imshow(image)
    plt.show()


def PrintLayerShapes(net):
    for layer_name, blob in net.blobs.items():
        print (layer_name + '\t' + str(blob.data.shape))


def PrintFiltersBias(net):
    for layer_name, param in net.params.items():
        if len(param) == 2:
            print (layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape))
        else:
            print (layer_name + '\t' + str(param[0].data.shape))


def PrintNetworkArchitecture(net):
    print ('NETWORK ARCHITECTURE')
    print ('Layers')
    PrintLayerShapes(net)
    print ('Filter, Bias')
    PrintFiltersBias(net)


def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # Normalize data
    data = (data - data.min()) / (data.max() - data.min())

    # Force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),   # for each dimension you have a (before, after) padding pair
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # Tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))     # shape (n*n, height, width, 3) reshape-> (n, n, height, width, 3) transpose-> (n, height, n, width, 3)
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])    # reshape -> (n*height, n*width, 3)

    plt.figure()
    plt.imshow(data); plt.axis('off')
    plt.show()


def ShowFilters(net, conv_level_name):
    # The conv_level_name is a list of [weights, biases]
    filters = net.params[conv_level_name][0].data
    vis_square(filters.transpose(0, 2, 3, 1))  # Channels (RGB) must be at the end


def ShowFeatureMaps(net, conv_level_name, number = None):
    if (number is None):
        feat = net.blobs[conv_level_name].data[0]
    else:
        feat = net.blobs[conv_level_name].data[0, :number]  # Collapse batch size (only the first -> 0), for the second axis takes 0:number, for the rest take all
    vis_square(feat)


def ShowFlatLevelActivations(net, flat_level_name):
    feat = net.blobs[flat_level_name].data[0]
    plt.subplot(2, 1, 1)
    plt.plot(feat.flat)
    plt.subplot(2, 1, 2)
    plt.hist(feat.flat[feat.flat > 0], bins=100)
    plt.show()


def EwcHistograms(ewcData, bins_requested, save_as):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.hist(ewcData[0], range=[-1.0, 2.0], bins=bins_requested, log=True)
    plt.subplot(2, 1, 2)
    plt.hist(ewcData[1], range=[0, 0.0050], bins=bins_requested, log=True)
    if save_as != None:
        plt.savefig(save_as)
    plt.close()


def WeightHistograms(data, bins_requested, save_as):
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.hist(data, range=[-0.1, 0.1], bins=bins_requested, log=True)
    if save_as != None:
        plt.savefig(save_as)
    plt.close()

# From Training ...

# Show first_n images loaded in the net data blob (e.g., with forward step)
def ShowInputDataAsImages(net, first_n, color):
    data = net.blobs['data'].data
    data = (data - data.min()) / (data.max() - data.min())
    height = data.shape[-1]
    width = data.shape[-2]
    # print(data[:first_n, :].transpose(2, 0, 3, 1).shape)
    # print(data[:first_n, 0].transpose(1, 0, 2).shape)
    if (color):
        plt.imshow(data[:first_n, :].transpose(2, 0, 3, 1).reshape(height, first_n*width, 3)); plt.axis('off')
    else:
        plt.imshow(net.blobs['data'].data[:first_n, 0].transpose(1, 0, 2).reshape(height, first_n*width)); plt.axis('off')
    print ('Labels:', net.blobs['label'].data[:first_n])
    plt.show()


def ShowGradientUpdatesAsImages(net, conv_level_name):
    # print(net.params[conv_level_name][0].diff.transpose(0,2,3,1).shape);
    vis_square(net.params[conv_level_name][0].diff.transpose(0,2,3,1))


def Plot_TrainLoss_TestAccuracy(train_loss, test_acc, test_interval):
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(np.arange(train_loss.size), train_loss)
    ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('test accuracy')
    ax2.set_title('Custom Test Accuracy: {:.2f}'.format(test_acc[-1]))
    plt.show()


# Interactive Training plot
# train_loss - loss values (one for each iteration)
# test_acc - accuracy values (one for each test iteration)
# test_interval - the number of train iterations between tests
# loss_max, accuracy_max - plot upper bounds
# If prev_exist = true -> prev_train_loss, prev_test_acc are plotted for a comparison (dashed)
# train_iterations_per_epoch - number of iterations per epoch
def Plot_Training_Init(train_loss, test_acc, test_interval, loss_max, accuracy_max, prev_exist, prev_train_loss, prev_test_acc, train_iterations_per_epoch):
    global line1, line2
    global figure
    plt.ion()  # Interactive mode
    figure, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_ylim(0,loss_max)
    ax2.set_ylim(0,accuracy_max*100)
    line1, = ax1.plot(np.zeros(train_loss.size), train_loss, 'b')
    line2, = ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
    if prev_exist:
        ax1.plot(test_interval * np.arange(len(prev_train_loss)), prev_train_loss, 'b', linestyle = 'dashed')
        ax2.plot(test_interval * np.arange(len(prev_test_acc)), prev_test_acc, 'r', linestyle = 'dashed')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('test accuracy (%)')
    ax2.set_title('Training')
    iter = train_iterations_per_epoch
    while iter < train_loss.size:
        plt.axvline(x=iter, color = 'green')
        iter += train_iterations_per_epoch


def Plot_Training_Update(iter, train_loss, test_acc, test_interval):
    xdata = np.arange(train_loss.size)
    xdata[iter+1:]=iter
    line1.set_xdata(xdata)
    train_loss[iter+1:]= train_loss[iter]
    line1.set_ydata(train_loss)
    xdata = test_interval * np.arange(len(test_acc))
    xdata[iter//test_interval+1:]=iter
    line2.set_xdata(xdata)
    test_acc[iter//test_interval+1:]=test_acc[iter//test_interval]
    line2.set_ydata(test_acc)
    figure.canvas.flush_events()
    figure.canvas.flush_events()
    time.sleep(0.1)


def Plot_Training_End(close = False):
    plt.ioff()
    if close: plt.close()
    else: plt.show()


# Interactive Training Plot (init)
#  title - Plot title (es. "Incremental Training NI")
#  eval_iters - a list of iterations number in which the model will be evaluated
#  epochs_iter (size <= eval_iters) - a list of iterations numbers that identify the last iteration of epochs
#  btach_iter (size <= eval_iters) - a list of iterations numbers that identify the first iteration of epochs
#  train_loss (size eval_iters) - train accuracues
#  test_acc (size eval_iters) - test accuracies
#  loss_max, accuracy_max - plot upper bounds
#  if prev_exist = true -> prev_train_loss, prev_test_acc are plotted for a comparison (dashed)
def Plot_Incremental_Training_Init(title, eval_iters, epochs_iter, batch_iter, train_loss, test_acc, loss_max, accuracy_max, prev_exist, prev_train_loss, prev_test_acc, show_epochs_tick=True):
    global line1, line2
    global figure
    plt.ion()  # Interactive mode
    figure, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_ylim(0,loss_max)
    ax2.set_ylim(0,accuracy_max*100)
    line1, = ax1.plot(eval_iters, train_loss, 'b')
    line2, = ax2.plot(eval_iters, test_acc, 'r')
    if prev_exist:
        ax1.plot(eval_iters, prev_train_loss, 'b', linestyle = 'dashed')
        ax2.plot(eval_iters, prev_test_acc, 'r', linestyle = 'dashed')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('test accuracy (%)')
    ax2.set_title(title)
    if show_epochs_tick:
        for iter in epochs_iter:
            plt.axvline(x=iter, color = 'green', linestyle = 'dashed')
    if len(batch_iter) <= 150:
        for iter in batch_iter:
            plt.axvline(x=iter, linewidth=2, color = 'green')
    #plt.ioff()
    #plt.show()


def Plot_Incremental_Training_Update(eval_idx, eval_iters, train_loss, test_acc):
    global line1, line2
    global figure
    xdata = np.array(eval_iters)
    xdata[eval_idx+1:]=xdata[eval_idx]
    line1.set_xdata(xdata)
    train_loss[eval_idx+1:]= train_loss[eval_idx]
    line1.set_ydata(train_loss)
    line2.set_xdata(xdata)
    test_acc[eval_idx+1:]=test_acc[eval_idx]
    line2.set_ydata(test_acc)
    figure.canvas.flush_events()
    figure.canvas.flush_events()
    time.sleep(0.1)


def Plot_Incremental_Training_End(close = False):
    plt.ioff()
    if close: plt.close()
    else: plt.show()


def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          classes = None,
                          in_cell_accuracy = False,
                          save_as = None):

    plt.figure()

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if classes is not None:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    if in_cell_accuracy:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if save_as != None:
        plt.savefig(save_as)

    plt.close()
    # plt.show()


from mpl_toolkits.mplot3d import Axes3D


def Plot3d_param_stats(net, param_stats, batch_num, normalize):
    # netp = { 'lay1': 0, 'lay2': 0, 'lay3': 0 }
    # params = { (0,'lay1',0):1.5, (0,'lay2',0):2.5, (0,'lay3',0):1.0, (1,'lay1',0):2.5, (1,'lay2',0):3.5, (1,'lay3',0):2.0 }
    # layer_num = len(netp)
    # layers = sorted(netp.keys())
    # batch_num = 2

    layer_num = len(net.params)
    layers = net.params.keys()

    # Setup the figure and axes
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    # Fake data
    _x = np.arange(layer_num)
    _y = np.arange(batch_num)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = np.zeros(x.shape[0], dtype=float)
    i = 0
    for batch in range(batch_num):
        for layer in layers:
            top[i] = param_stats[batch, layer, 0]
            i += 1

    bottom = np.zeros_like(top)
    width = depth = 0.6

    ax1.w_xaxis.set_ticks(_x + width / 2.)
    ax1.w_xaxis.set_ticklabels(layers)
    ax1.w_yaxis.set_ticks(_y + width / 2.)
    ax1.w_yaxis.set_ticklabels(_y)

    ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
    ax1.set_title('Weights Changes')

    if not normalize:
        ax1.set_zlim(0, 0.00115)

    plt.show()

    # net.params.items()
    # layer_num = len(net.param)

    # for layer_name in net.params.items():

    # change[batch_num, layer_name, 0] = np.linalg.norm(param[0].data - prev_param[layer_name, 0])
    # change[batch_num, layer_name, 1] = np.linalg.norm(param[1].data - prev_param[layer_name, 1])
