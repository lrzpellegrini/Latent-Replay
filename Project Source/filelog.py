####################################################################################################
# Copyright (c) 2019. Lorenzo Pellegrini, Gabriele Graffieti, Vincenzo Lomonaco, Davide Maltoni    #
#                                                                                                  #
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file #
####################################################################################################

import os

def Train_Log_Init(train_log_file):
    if train_log_file == '':
        return
    global file
    file = open(train_log_file + 'Cur.txt','w')


def Train_Log_Update(train_log_file, iter, acc, loss, report_train_accuracy, train_acc):
    if train_log_file == '':
        return
    if report_train_accuracy:
        file.write('Iter = {:05d}'.format(iter) + '  TestAcc = {:6.3f}%'.format(acc*100) + '  TrainLoss = {:6.3f}'.format(loss) + '  TrainAcc = {:6.3f}%\n'.format(train_acc*100))
    else:
        file.write('Iter = {:05d}'.format(iter) + '  TestAcc = {:6.3f}%'.format(acc*100) + '  TrainLoss = {:6.3f}\n'.format(loss))
    file.flush()


def Train_Log_End(train_log_file):
    if train_log_file == '':
        return
    file.close()


def count_lines(fpath):

    lines = 0
    with open(fpath, 'r') as f:
        for line in f:
            lines += 1
    return lines

def TryLoadPrevTrainingLog(train_log_file, prev_train_loss, prev_test_acc, prev_train_acc):
    
    filename = train_log_file + 'Pre.txt'
    if not os.path.isfile(filename):
        return False
    prev_evals = count_lines(filename)
    if prev_evals > len(prev_test_acc):
        return False  # Cannot have more lines than in current test
    else:
        i = len(prev_test_acc) - prev_evals  # skip first evals (assume we skip initial batch). If same len start form 0.

    fpre = open(filename,'r')
    for line in fpre:
        if i >= len(prev_test_acc):
            break
        parts = line.split()
        prev_test_acc[i] = parts[5][:-1]  # Remove last char '%'
        prev_train_loss[i] = parts[8]
        prev_train_acc[i] = parts[11][:-1]
        # Train accuracy even if present is not used
        i+=1
    return True


def LoadAccuracyFromCurTrainingLog(train_log_file):
   
    filename = train_log_file + 'Cur.txt'
    accuracy = []
    fcur = open(filename,'r')
    for line in fcur:
        parts = line.split()
        accuracy.append(parts[5][:-1])    # Remove last char '%'
    return accuracy


def Train_LogDetails_Init(train_log_file):
    if train_log_file == '':
        return
    global file_detail
    file_detail = open(train_log_file + 'DetailsCur.txt', 'w')


def Train_LogDetails_Update(train_log_file, batch, mean_inbatch, std_inbatch, mean_outbatch, std_outbatch):
    if train_log_file == '':
        return
    file_detail.write(
        'Batch = {:03d}'.format(batch) + '  MeanIn = {:6.3f}%'.format(mean_inbatch * 100) + '  StdIn = {:6.3f}%'.format(
            std_inbatch * 100) + '  MeanOut = {:6.3f}%'.format(mean_outbatch * 100) + '  StdOut = {:6.3f}%\n'.format(
            std_outbatch * 100))
    file_detail.flush()


def Train_LogDetails_End(train_log_file):
    if train_log_file == '':
        return
    file_detail.close()
