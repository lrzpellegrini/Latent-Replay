####################################################################################################
# Copyright (c) 2019. Lorenzo Pellegrini, Gabriele Graffieti, Vincenzo Lomonaco, Davide Maltoni    #
#                                                                                                  #
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file #
####################################################################################################

import sys, os, time
import numpy as np

# Uncomment and customize the following lines if PyCaffe needs to be loaded dinamically
# caffe_root = 'D:/CaffeInstall/'
# sys.path.insert(0, caffe_root + 'python')
os.environ["GLOG_minloglevel"] = "1"  # limit logging (0 - debug, 1 - info (still a LOT of outputs), 2 - warnings, 3 - errors)
import caffe

# For prototxt parsing
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

from sklearn.metrics import confusion_matrix

import visualization
import filelog
import train_utils
import cwr, syn, rehearsal
import rehe_lat_surgery
from pathlib import Path

def main_Core50(conf, run, close_at_the_end=False):
    # Prepare configurations files
    conf['solver_file_first_batch'] = conf['solver_file_first_batch'].replace('X', conf['model'])
    conf['solver_file'] = conf['solver_file'].replace('X', conf['model'])
    conf['init_weights_file'] = conf['init_weights_file'].replace('X', conf['model'])
    conf['tmp_weights_file'] = conf['tmp_weights_file'].replace('X', conf['model'])
    train_filelists = conf['train_filelists'].replace('RUN_X', run)
    test_filelist = conf['test_filelist'].replace('RUN_X', run)
    run_on_the_fly = True  # If True, tells the train_utils.get_data(...) script not to cache batch data on disk

    (Path(conf['exp_path']) / 'CM').mkdir(exist_ok=True, parents=True)
    (Path(conf['exp_path']) / 'EwC').mkdir(exist_ok=True, parents=True)
    (Path(conf['exp_path']) / 'Syn').mkdir(exist_ok=True, parents=True)

    if 'brn_past_weight' not in conf or conf['brn_past_weight'] is None:
        if conf['rehearsal_is_latent']:
            conf['brn_past_weight'] = 20000
        else:
            conf['brn_past_weight'] = 10000

    # To change if needed the network prototxt
    if conf['rehearsal_is_latent']:
        solver_param = caffe_pb2.SolverParameter()
        with open(conf['solver_file']) as f:
            txtf.Merge(str(f.read()), solver_param)
        next_batches_net_prototxt_path = Path(solver_param.net)

        if not next_batches_net_prototxt_path.stem.endswith('b'):
            print('Error dealing with latent rehearsal: invalid net prototxt name!')
            exit(1)

        next_batches_net_prototxt_path_orig = next_batches_net_prototxt_path.parent / (next_batches_net_prototxt_path.stem[:-1] + next_batches_net_prototxt_path.suffix)
        moving_avg_fraction = 1.0 - (1.0/conf['brn_past_weight'])
        train_utils.modify_net_prototxt(str(next_batches_net_prototxt_path_orig), str(next_batches_net_prototxt_path), moving_average_fraction=moving_avg_fraction)

        if conf['model'] == 'MobileNetV1':
            rehearsal_layer_mapping_for_mobilenetv1 = {
                'data': ([-1, 3, 128, 128], 'conv1'),
                'conv2_1/dw': ([-1, 32, 64, 64], 'conv2_1/sep'),
                #conv2_1 / dw(128, 32, 64, 64)
                #    conv2_1 / sep(128, 64, 64, 64)
                'conv2_2/dw': ([-1, 64, 32, 32], 'conv2_2/sep'),
                #conv2_2 / dw(128, 64, 32, 32)
                #    conv2_2 / sep(128, 128, 32, 32)
                'conv3_1/dw': ([-1, 128, 32, 32], 'conv3_1/sep'),
                #conv3_1 / dw(128, 128, 32, 32)
                #    conv3_1 / sep(128, 128, 32, 32)
                'conv3_2/dw': ([-1, 128, 16, 16], 'conv3_2/sep'),
                #conv3_2 / dw(128, 128, 16, 16)
                #    conv3_2 / sep(128, 256, 16, 16)
                'conv4_1/dw': ([-1, 256, 16, 16], 'conv4_1/sep'),
                #conv4_1 / dw(128, 256, 16, 16)
                #    conv4_1 / sep(128, 256, 16, 16)
                'conv4_2/dw': ([-1, 256, 8, 8], 'conv4_2/sep'),
                #conv4_2 / dw(128, 256, 8, 8)
                #    conv4_2 / sep(128, 512, 8, 8)
                'conv5_1/dw': ([-1, 512, 8, 8], 'conv5_1/sep'),
                #conv5_1 / dw(512, 1, 3, 3)
                #    conv5_1 / sep(512, 512, 1, 1)
                'conv5_2/dw': ([-1, 512, 8, 8], 'conv5_2/sep'),
                #conv5_2 / dw(512, 1, 3, 3)
                #    conv5_2 / sep(512, 512, 1, 1)
                'conv5_3/dw': ([-1, 512, 8, 8], 'conv5_3/sep'),
                #conv5_3 / dw(512, 1, 3, 3)
                #    conv5_3 / sep(512, 512, 1, 1)
                'conv5_4/dw': ([-1, 512, 8, 8], 'conv5_4/sep'),
                #conv5_4 / dw(512, 1, 3, 3)
                #    conv5_4 / sep(512, 512, 1, 1)
                'conv5_5/dw': ([-1, 512, 8, 8], 'conv5_5/sep'),
                #conv5_5 / dw(512, 1, 3, 3)
                #    conv5_5 / sep(512, 512, 1, 1)
                'conv5_6/dw': ([-1, 512, 4, 4], 'conv5_6/sep'),
                #conv5_6 / dw(512, 1, 3, 3)
                #    conv5_6 / sep(1024, 512, 1, 1)
                'conv6/dw': ([-1, 1024, 4, 4], 'conv6/sep'),
                #conv6 / dw(1024, 1, 3, 3)
                #    conv6 / sep(1024, 1024, 1, 1)
                'pool6': ([-1, 1024, 1, 1], 'mid_fc7')
                #avg_pool(1024)
                #    mid_fc7(50, 1024, 1, 1)(50, )
            }

            current_mapping = rehearsal_layer_mapping_for_mobilenetv1[conf['rehearsal_layer']]
            if 'rehearsal_stop_layer' not in conf or conf['rehearsal_stop_layer'] is None:
                conf['rehearsal_stop_layer'] = current_mapping[1]

            rehe_lat_surgery.create_concat_layer_from_net_template(str(next_batches_net_prototxt_path),
                                                                   str(next_batches_net_prototxt_path),
                                                                   conf['rehearsal_layer'], current_mapping[0], current_mapping[1], original_input=21, rehearsal_input=107)
        else:
            raise RuntimeError('Unsupported model for latent rehearsal:', conf['model'])

    # Parse the solver prototxt
    #  for more details see - https://stackoverflow.com/questions/31823898/changing-the-solver-parameters-in-caffe-through-pycaffe
    if conf['initial_batch'] == 0:
        print('Solver proto: ', conf['solver_file_first_batch'])
        solver_param = caffe_pb2.SolverParameter()
        with open(conf['solver_file_first_batch']) as f:
            txtf.Merge(str(f.read()), solver_param)
        net_prototxt = solver_param.net  # Obtains the path to the net prototxt
        print('Net proto: ', net_prototxt)
    else:
        print('Solver proto: ', conf['solver_file'])
        solver_param = caffe_pb2.SolverParameter()
        with open(conf['solver_file']) as f:
            txtf.Merge(str(f.read()), solver_param)
        net_prototxt = solver_param.net  # Obtains the path to the net prototxt
        print('Net proto: ', net_prototxt)

    # Obtain class labels
    if conf['class_labels'] != '':
        # More complex than a simple loadtxt because of the unicode representation in python 3
        label_str = np.loadtxt(conf['class_labels'], dtype=bytes, delimiter="\n").astype(str)

    # Obtain minibatch size from net proto
    train_minibatch_size, test_minibatch_size = train_utils.extract_minibatch_size_from_prototxt_with_input_layers(net_prototxt)
    print(' test minibatch size: ', test_minibatch_size)
    print(' train minibatch size: ', train_minibatch_size)

    # Load test set
    print("Recovering Test Set: ", test_filelist, " ...")
    start = time.time()
    test_x, test_y = train_utils.get_data(test_filelist, conf['db_path'], conf['exp_path'], on_the_fly=run_on_the_fly, verbose=conf['verbose'])
    assert (test_x.shape[0] == test_y.shape[0])
    if conf['num_classes'] < 50:  # Checks if we are doing category-based classification
        test_y = test_y // 5
    test_y = test_y.astype(np.float32)
    test_patterns = test_x.shape[0]
    test_x, test_y, test_iterat = train_utils.pad_data(test_x, test_y, test_minibatch_size)
    print(' -> %d patterns of %d classes (%.2f sec.)' % (test_patterns, len(np.unique(test_y)), time.time() - start))
    print(' -> %.2f -> %d iterations for full evaluation' % (test_patterns / test_minibatch_size, test_iterat))

    # Load training patterns in batches (by now assume the same number in all batches)
    batch_count = conf['num_batches']
    train_patterns = train_utils.count_lines_in_batches(batch_count, train_filelists)
    train_iterations_per_epoch = np.zeros(batch_count, int)
    train_iterations = np.zeros(batch_count, int)
    test_interval_epochs = conf['test_interval_epochs']
    test_interval = np.zeros(batch_count, float)
    for batch in range(batch_count):
        if conf["rehearsal"] and batch > 0:
            train_patterns[batch] += conf["rehearsal_memory"]
        train_iterations_per_epoch[batch] = int(np.ceil(train_patterns[batch] / train_minibatch_size))
        test_interval[batch] = test_interval_epochs * train_iterations_per_epoch[batch]
        if (batch == 0):
            train_iterations[batch] = train_iterations_per_epoch[batch] * conf['num_epochs_first_batch']
        else:
            train_iterations[batch] = train_iterations_per_epoch[batch] * conf['num_epochs']
        print("Batch %2d: %d patterns, %d iterations (%d iter. per epochs - test every %.1f iter.)" \
              % (batch, train_patterns[batch], train_iterations[batch], train_iterations_per_epoch[batch], test_interval[batch]))

    # Create evaluation points
    # -> iterations which are boundaries of batches
    batch_iter = [0]
    iter = 0
    for batch in range(batch_count):
        iter += train_iterations[batch]
        batch_iter.append(iter)

    # Calculates the iterations where the network will be evaluated
    eval_iters = [1]  # Start with 1 (instead of 0) because the test net is aligned to the train one after solver.step(1)
    for batch in range(batch_count):
        start = batch_iter[batch]
        end = batch_iter[batch + 1]
        start += test_interval[batch]
        while start < end:
            eval_iters.append(int(start))
            start += test_interval[batch]
        eval_iters.append(end)

    # Iterations which are epochs in the evaluation range
    epochs_iter = []
    for batch in range(batch_count):
        start = batch_iter[batch]
        end = batch_iter[batch + 1]
        start += train_iterations_per_epoch[batch]
        while start <= end:
            epochs_iter.append(int(start))
            start += train_iterations_per_epoch[batch]

    prev_train_loss = np.zeros(len(eval_iters))
    prev_test_acc = np.zeros(len(eval_iters))
    prev_train_acc = np.zeros(len(eval_iters))
    prev_exist = filelog.TryLoadPrevTrainingLog(conf['train_log_file'], prev_train_loss, prev_test_acc, prev_train_acc)
    train_loss = np.copy(prev_train_loss)  # Copying allows to correctly visualize the graph in case we start from initial_batch > 0
    test_acc = np.copy(prev_test_acc)
    train_acc = np.copy(prev_train_acc)

    epochs_tick = False if batch_count > 30 else True  # For better visualization
    visualization.Plot_Incremental_Training_Init('Incremental Training', eval_iters, epochs_iter, batch_iter,
                                                train_loss, test_acc, 5, conf['accuracy_max'],
                                                prev_exist, prev_train_loss, prev_test_acc, show_epochs_tick=epochs_tick)
    filelog.Train_Log_Init(conf['train_log_file'])
    filelog.Train_LogDetails_Init(conf['train_log_file'])

    start_train = time.time()
    eval_idx = 0  # Evaluation iterations counter
    global_eval_iter = 0  # Global iterations counter
    first_round = True
    initial_batch = conf['initial_batch']
    if initial_batch > 0:  # Move forward by skipping unnecessary evaluation
        global_eval_iter = batch_iter[initial_batch]
        while eval_iters[eval_idx] < global_eval_iter:
            eval_idx += 1
        eval_idx += 1

    for batch in range(initial_batch, batch_count):

        print('\nBATCH = {:2d} ----------------------------------------------------'.format(batch))

        if batch == 0:
            solver = caffe.get_solver(conf['solver_file_first_batch'])  # Load the solver for the first batch and create net(s)
            if conf['init_weights_file'] != '':
                solver.net.copy_from(conf['init_weights_file'])
                print('Network created and Weights loaded from: ', conf['init_weights_file'])
                # Test
                solver.share_weights(solver.test_nets[0])
                print('Weights shared with Test Net')

                accuracy, _, pred_y = train_utils.test_network_with_accuracy_layer(solver, test_x, test_y, test_iterat,
                                                                                                   test_minibatch_size,
                                                                                                   prediction_level_Model[conf['model']],
                                                                                                   return_prediction=True)

            if conf['strategy'] in ['cwr+', 'ar1', 'ar1free']:
                cwr.zeros_cwr_layer_bias_lr(solver.net, cwr_layers_Model[conf['model']])
                class_updates = np.full(conf['num_classes'], conf['initial_class_updates_value'], dtype=np.float32)
                cons_w = cwr.init_consolidated_weights(solver.net, cwr_layers_Model[conf['model']], conf['num_classes'])  # allocate space for consolidated weights and initialze to 0
                cwr.reset_weights(solver.net, cwr_layers_Model[conf['model']], conf['num_classes'])  # reset weights to 0 (done here for the first batch to keep initial stats correct)

            # cwr.reset_weights(solver.net, cwr_layers_Model[conf['model']], conf['num_classes'])   # reset weights to 0 (done here for the first batch to keep initial stats correct)                 

            if conf['strategy'] in ['ar1', 'ar1free']:
                ewcData, synData = syn.create_syn_data(solver.net)  # ewcData stores optimal weights + normalized fisher; trajectory store unnormalized summed grad*deltaW

            if conf['rehearsal_is_latent']:
                reha_data_size = solver.net.blobs[conf['rehearsal_layer']].data[0].size
                rehearsal.allocate_memory(conf['rehearsal_memory'], reha_data_size, 1)
            else:
                rehearsal.allocate_memory(conf['rehearsal_memory'], test_x[0].size, 1)

        elif batch == 1:
            solver = caffe.get_solver(conf['solver_file'])  # load solver and create net
            if first_round:
                solver.net.copy_from(conf['init_weights_file'])
                print('Network created and Weights loaded from: ', conf['init_weights_file'])
            else:
                solver.net.copy_from(conf['tmp_weights_file'])
                print('Network created and Weights loaded from: ', conf['tmp_weights_file'])

            solver.share_weights(solver.test_nets[0])

            if first_round:
                print('Loading consolidated weights...')
                class_updates = np.full(conf['num_classes'], conf['initial_class_updates_value'], dtype=np.float32)
                rand_w, cons_w = cwr.copy_initial_weights(solver.net, cwr_layers_Model[conf['model']], conf['num_classes'])
                if conf['strategy'] in ['ar1']:
                    ewcData, synData = syn.create_syn_data(solver.net)  # ewcData stores optimal weights + normalized fisher; trajectory store unnormalized summed grad*deltaW

            if conf['strategy'] in ['cwr+']:
                cwr.zeros_non_cwr_layers_lr(solver.net, cwr_layers_Model[conf['model']])  # blocca livelli sotto

            if conf['strategy'] in ['cwr+', 'ar1', 'ar1free']:
                if 'cwr_lr_mult' in conf.keys() and conf['cwr_lr_mult'] != 1:
                    cwr.zeros_cwr_layer_bias_lr(solver.net, cwr_layers_Model[conf['model']], force_weights_lr_mult=conf['cwr_lr_mult'])
                else:
                    cwr.zeros_cwr_layer_bias_lr(solver.net, cwr_layers_Model[conf['model']])

            cwr.set_brn_past_weight(solver.net, conf['brn_past_weight'])

        # Initializes some data structures used for reporting stats. Executed once (in the first round)
        if first_round:
            if batch == 1 and (conf['strategy'] in ['cwr', 'cwr+', 'ar1', 'ar1free']):
                print('Cannot start from batch 1 in ', conf['strategy'], ' strategy!')
                sys.exit(0)
            visualization.PrintNetworkArchitecture(solver.net)
            # If accuracy layer is defined in the prototxt also in TRAIN mode -> log train accuracy too (not in the plot)
            try:
                report_train_accuracy = True
                err = solver.net.blobs['accuracy'].num  # Assume this is stable for prototxt of successive batches
            except:
                report_train_accuracy = False
            first_round = False
            if conf['compute_param_stats']:
                param_change = {}
                param_stats = train_utils.stats_initialize_param(solver.net)
                # nonzero_activations = train_utils.stats_activations_initialize(solver.net)

        # Load training data for the current batch
        # Note that the file lists are provided in the batch_filelists folder
        current_train_filelist = train_filelists.replace('XX', str(batch).zfill(2))
        print("Recovering training data: ", current_train_filelist, " ...")
        batch_x, batch_y = train_utils.get_data(current_train_filelist, conf['db_path'], conf['exp_path'], on_the_fly=run_on_the_fly, verbose=conf['verbose'])
        print("Done.")
        if conf['num_classes'] < 50:  # Category based classification
            batch_y = batch_y // 5

        batch_t = train_utils.compute_one_hot_vectors(batch_y, conf['num_classes'])

        # Load patterns from Rehearsal Memory
        rehe_x, rehe_y = rehearsal.get_samples()
        rehe_t = train_utils.compute_one_hot_vectors(rehe_y, conf['num_classes'])

        # Detects how many patterns per class are present in the current batch
        if batch == 0:
            classes_in_cur_train = batch_y.astype(np.int)
        else:
            classes_in_cur_train = np.concatenate((batch_y.astype(np.int), rehe_y.astype(np.int)))
        unique_y, y_freq = np.unique(classes_in_cur_train, return_counts=True)

        if conf['strategy'] in ['cwr+', 'ar1', 'ar1free'] and batch > initial_batch:
            cwr.reset_weights(solver.net, cwr_layers_Model[conf['model']], conf['num_classes'])  # Reset weights of CWR layers to 0

            # Loads previously consolidated weights
            # This procedure, explained in Fine-Grained Continual Learning (https://arxiv.org/pdf/1907.03799.pdf),
            # is necessary in the NIC scenario
            if 'cwr_nic_load_weight' in conf.keys() and conf['cwr_nic_load_weight']:
                cwr.load_weights_nic(solver.net, cwr_layers_Model[conf['model']], unique_y, cons_w)

        if conf['strategy'] in ['ar1'] and batch > initial_batch:
            syn.weight_stats(solver.net, batch, ewcData, conf['ewc_clip_to'])
            solver.net.blobs['ewc'].data[...] = ewcData

        # Convert labels to float32
        batch_y = batch_y.astype(np.float32)
        assert (batch_x.shape[0] == batch_y.shape[0])
        rehe_y = rehe_y.astype(np.float32)

        avg_train_loss = 0
        avg_train_accuracy = 0
        avg_count = 0

        if conf['strategy'] in ['syn', 'ar1']:
            syn.init_batch(solver.net, ewcData, synData)

        reharshal_size = conf["rehearsal_memory"] if batch > initial_batch else 0
        orig_in_minibatch = np.round(train_minibatch_size * batch_x.shape[0] / (batch_x.shape[0] + reharshal_size)).astype(np.int)
        reha_in_minibatch = train_minibatch_size - orig_in_minibatch

        print(' -> Current Batch: %d patterns, External Memory: %d patterns' % (batch_x.shape[0], reharshal_size))
        print(' ->   per minibatch (size %d): %d from current batch and %d from external memory' % (train_minibatch_size, orig_in_minibatch, reha_in_minibatch))

        # Padding and shuffling
        batch_x, orig_iters_per_epoch = train_utils.pad_data_single(batch_x, orig_in_minibatch)
        batch_y, _ = train_utils.pad_data_single(batch_y, orig_in_minibatch)
        batch_t, _ = train_utils.pad_data_single(batch_t, orig_in_minibatch)
        batch_x, batch_y, batch_t = train_utils.shuffle_in_unison((batch_x, batch_y, batch_t), 0)

        if conf['rehearsal_is_latent']:
            req_shape = (batch_x.shape[0],) + solver.net.blobs[conf['rehearsal_layer']].data.shape[1:]
            latent_batch_x = np.zeros(req_shape, dtype=np.float32)

        # Padding and shuffling of rehasal patterns
        reha_iters_per_epoch = 0
        if reharshal_size > 0:
            rehe_x, reha_iters_per_epoch = train_utils.pad_data_single(rehe_x, reha_in_minibatch)
            rehe_y, _ = train_utils.pad_data_single(rehe_y, reha_in_minibatch)
            rehe_t, _ = train_utils.pad_data_single(rehe_t, reha_in_minibatch)
            rehe_x, rehe_y, rehe_t = train_utils.shuffle_in_unison((rehe_x, rehe_y, rehe_t), 0)  # shuffle

        print(' ->   iterations per epoch (with padding): %d, %d (initial %d)' % (orig_iters_per_epoch, reha_iters_per_epoch, train_iterations_per_epoch[batch]))

        # The main solver loop (per batch)
        it = 0
        while it < train_iterations[batch]:
            # The following part is pretty much straight-forward
            # The current batch is split in minibatches (which size was previously detected by looking at the net prototxt)
            # The minibatch is loaded in blobs 'data', 'data_reha', 'label' and 'target'
            it_mod_orig = it % orig_iters_per_epoch
            orig_start = it_mod_orig * orig_in_minibatch
            orig_end = (it_mod_orig + 1) * orig_in_minibatch

            if conf['rehearsal_is_latent']:
                solver.net.blobs['data'].data[...] = batch_x[orig_start:orig_end]
            else:
                solver.net.blobs['data'].data[:orig_in_minibatch] = batch_x[orig_start:orig_end]

            # Provide data to input layers (new patterns)
            solver.net.blobs['label'].data[:orig_in_minibatch] = batch_y[orig_start:orig_end]
            solver.net.blobs['target'].data[:orig_in_minibatch] = batch_t[orig_start:orig_end]

            # Provide data to input layers (reharsal patterns)
            if reharshal_size > 0:
                it_mod_reha = it % reha_iters_per_epoch
                reha_start = it_mod_reha * reha_in_minibatch
                reha_end = (it_mod_reha + 1) * reha_in_minibatch

                if conf['rehearsal_is_latent']:
                    solver.net.blobs['data_reha'].data[...] = rehe_x[reha_start:reha_end]
                else:
                    solver.net.blobs['data'].data[orig_in_minibatch:] = rehe_x[reha_start:reha_end]

                solver.net.blobs['label'].data[orig_in_minibatch:] = rehe_y[reha_start:reha_end]
                solver.net.blobs['target'].data[orig_in_minibatch:] = rehe_t[reha_start:reha_end]

            if conf['strategy'] in ['ar1']:
                syn.pre_update(solver.net, ewcData, synData)

            # Explicit (net.step(1))
            solver.net.clear_param_diffs()
            solver.net.forward()  # start=None, end=None
            if batch > 0 and conf['strategy'] in ['cwr+', 'cwr']:
                solver.net.backward(end='mid_fc7')  # In CWR+ we stop the backward step at the CWR layer
            else:
                if batch > 0 and 'rehearsal_stop_layer' in conf.keys() and conf['rehearsal_stop_layer'] is not None:
                    # When using latent replay we stop the backward step at the latent rehearsal layer
                    solver.net.backward(end=conf['rehearsal_stop_layer'])
                else:
                    solver.net.backward()

            if conf['rehearsal_is_latent']:
                # Save latent features of new patterns (only during the first epoch)
                if batch > 0 and it < orig_iters_per_epoch:
                    latent_batch_x[orig_start:orig_end] = solver.net.blobs[conf['rehearsal_layer']].data

            # Weights update
            solver.apply_update()

            if conf['strategy'] == 'ar1':
                syn.post_update(solver.net, ewcData, synData, cwr_layers_Model[conf['model']])

            print('+', end='', flush=True)

            global_eval_iter += 1
            avg_count += 1

            avg_train_loss += solver.net.blobs['loss'].data
            if report_train_accuracy:
                avg_train_accuracy += solver.net.blobs['accuracy'].data

            if global_eval_iter == eval_iters[eval_idx]:
                # Evaluation point
                if avg_count > 0:
                    avg_train_loss /= avg_count
                    avg_train_accuracy /= avg_count
                train_loss[eval_idx] = avg_train_loss
                print('\nIter {:>4}'.format(it + 1), '({:>4})'.format(global_eval_iter), ': Train Loss = {:.5f}'.format(avg_train_loss), end='', flush=True)
                if report_train_accuracy:
                    train_acc[eval_idx] = avg_train_accuracy
                    print('  Train Accuracy = {:.5f}%'.format(avg_train_accuracy * 100), end='', flush=True)

                compute_confusion_matrix = True if (conf['confusion_matrix'] and it == train_iterations[batch] - 1) else False  # last batch iter

                # The following lines are executed only if this is the last iteration for the current batch
                if conf['strategy'] in ['cwr+', 'ar1', 'ar1free'] and it == train_iterations[batch] - 1:
                    cwr.consolidate_weights_cwr_plus(solver.net, cwr_layers_Model[conf['model']], unique_y, y_freq, class_updates, cons_w)
                    class_updates[unique_y] += y_freq
                    print(class_updates)
                    cwr.load_weights(solver.net, cwr_layers_Model[conf['model']], conf['num_classes'], cons_w)  # Load consolidated weights for testing

                accuracy, _, pred_y = train_utils.test_network_with_accuracy_layer(solver, test_x, test_y, test_iterat, test_minibatch_size, prediction_level_Model[conf['model']],
                                                                                   return_prediction=compute_confusion_matrix)
                test_acc[eval_idx] = accuracy * 100
                print('  Test Accuracy = {:.5f}%'.format(accuracy * 100))

                # Batch(Re)Norm Stats
                train_utils.print_bn_stats(solver.net)

                visualization.Plot_Incremental_Training_Update(eval_idx, eval_iters, train_loss, test_acc)

                filelog.Train_Log_Update(conf['train_log_file'], eval_iters[eval_idx], accuracy, avg_train_loss, report_train_accuracy, avg_train_accuracy)

                avg_train_loss = 0
                avg_train_accuracy = 0
                avg_count = 0
                eval_idx += 1  # Next eval

            it += 1  # Next iter

        # Current batch training concluded
        if conf['strategy'] in ['ar1']:
            syn.update_ewc_data(solver.net, ewcData, synData, batch, conf['ewc_clip_to'], c=conf['ewc_w'])
            if conf['save_ewc_histograms']:
                visualization.EwcHistograms(ewcData, 100, save_as=conf['exp_path'] + 'Syn/F_' + str(batch) + '.png')

        if conf['rehearsal_is_latent']:
            if batch == 0:
                reha_it = 0
                while reha_it < orig_iters_per_epoch:
                    orig_start = reha_it * orig_in_minibatch
                    orig_end = (reha_it + 1) * orig_in_minibatch
                    solver.net.blobs['data'].data[...] = batch_x[orig_start:orig_end]
                    solver.net.forward()
                    latent_batch_x[orig_start:orig_end] = solver.net.blobs[conf['rehearsal_layer']].data
                    reha_it+=1

            rehearsal.update_memory(latent_batch_x, batch_y.astype(np.int), batch)
        else:
            rehearsal.update_memory(batch_x, batch_y.astype(np.int), batch)

        if compute_confusion_matrix:
            # Computes the confusion matrix and logs + plots it
            cnf_matrix = confusion_matrix(test_y, pred_y, range(conf['num_classes']))
            if batch == 0:
                prev_class_accuracies = np.zeros(conf['num_classes'])
            else:
                prev_class_accuracies = current_class_accuracies
            current_class_accuracies = np.diagonal(cnf_matrix) / cnf_matrix.sum(axis=1)
            deltas = current_class_accuracies - prev_class_accuracies
            classes_in_batch = set(batch_y.astype(np.int))
            classes_non_in_batch = set(range(conf['num_classes'])) - classes_in_batch
            mean_class_in_batch = np.mean(deltas[list(classes_in_batch)])
            std_class_in_batch = np.std(deltas[list(classes_in_batch)])
            mean_class_non_in_batch = np.mean(deltas[list(classes_non_in_batch)])
            std_class_non_in_batch = np.std(deltas[list(classes_non_in_batch)])
            print('InBatch -> mean =  %.2f%% std =  %.2f%%, OutBatch -> mean =  %.2f%% std =  %.2f%%' % (
                mean_class_in_batch * 100, std_class_in_batch * 100, mean_class_non_in_batch * 100, std_class_non_in_batch * 100))
            filelog.Train_LogDetails_Update(conf['train_log_file'], batch, mean_class_in_batch, std_class_in_batch, mean_class_non_in_batch, std_class_non_in_batch)
            visualization.plot_confusion_matrix(cnf_matrix, normalize=True, title='CM after batch: ' + str(batch), save_as=conf['exp_path'] + 'CM/CM_' + str(batch) + '.png')

        if conf['compute_param_stats']:
            train_utils.stats_compute_param_change_and_update_prev(solver.net, param_stats, batch, param_change)

        if batch == 0:
            solver.net.save(conf['tmp_weights_file'])
            print('Weights saved to: ', conf['tmp_weights_file'])
            del solver

    print('Training Time: %.2f sec' % (time.time() - start_train))

    if conf['compute_param_stats']:
        stats_normalization = True
        train_utils.stats_normalize(solver.net, param_stats, batch_count, param_change, stats_normalization)
        visualization.Plot3d_param_stats(solver.net, param_change, batch_count, stats_normalization)

    filelog.Train_Log_End(conf['train_log_file'])
    filelog.Train_LogDetails_End(conf['train_log_file'])

    visualization.Plot_Incremental_Training_End(close=close_at_the_end)


def main_Core50_multiRun(conf, runs):
    conf['confusion_matrix'] = True
    conf['save_ewc_histograms'] = False
    conf['compute_param_stats'] = False
    allfile = open(conf['train_log_file'] + 'All.txt', 'w')
    for r in range(runs):
        run = 'run' + str(r)
        main_Core50(conf, run, close_at_the_end=True)
        runres = filelog.LoadAccuracyFromCurTrainingLog(conf['train_log_file'])
        for item in runres:
            allfile.write("%s " % item)
        allfile.write("\n")
        allfile.flush()
    allfile.close()


if __name__ == "__main__":

    import nicv2_configuration

    ## --------------------------------------------------------------------

    prediction_level_Model = {
        'CaffeNet': 'mid_fc8',
        'Nin': 'pool4',
        'GoogleNet': 'loss3_50/classifier',
        'MobileNetV1': 'mid_fc7'
    }

    cwr_layers_Model = {
        'CaffeNet': ['mid_fc8'],
        'GoogleNet': ['loss1_50/classifier', 'loss2_50/classifier', 'loss3_50/classifier'],
        'MobileNetV1': ['mid_fc7'],
    }


    conf = nicv2_configuration.conf_NIC

    # Setting hardware
    if conf['backend'] == 'GPU':
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    if conf['strategy'] not in ['cwr+', 'ar1', 'ar1free']:
        print("Undefined strategy!")
        sys.exit(1)

    # Single Run
    sys.exit(int(main_Core50(conf, 'run0') or 0))

    # Multi Run
    # runs = 5
    # sys.exit(int(main_Core50_multiRun(conf, runs) or 0))