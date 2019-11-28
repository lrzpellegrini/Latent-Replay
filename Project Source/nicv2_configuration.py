####################################################################################################
# Copyright (c) 2019. Lorenzo Pellegrini, Gabriele Graffieti, Vincenzo Lomonaco, Davide Maltoni    #
#                                                                                                  #
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file #
####################################################################################################

# --------------------------------------------------------------------

import json
from pathlib import Path

conf_NIC = {
    'model': 'MobileNetV1',
    'db_path': 'C:/DB/Core50/128/',   # Location of patterns and filelists
    'class_labels': 'C:/DB/Core50/core50_labels.txt',
    'exp_path': 'C:/Temp/Core50/NIC/NIC_v2_391/',   # Location of snapshots, temp binary database, logfiles, etc.
    'solver_file_first_batch': './NIC_v2/NIC_v2_79/NIC_solver_X_first_batch.prototxt',
    'solver_file': './NIC_v2/NIC_v2_79/NIC/NIC_solver_X.prototxt',
    'init_weights_file': './models//X.caffemodel',
    'tmp_weights_file': 'E:/Temp/Core50/NIC/NIC/X.caffemodel',
    'train_filelists': './batch_filelists/sIII_v2_79/RUN_X/train_batch_XX_filelist.txt',
    'test_filelist': './batch_filelists/test_filelist_20.txt',
    'train_log_file': 'C:/PRJ/CNN/Experiments/Core50/NIC/trainLog', # 'Cur.txt' is appended to create the file. 'Pre.txt' is appended when searching an old file for (optional) comparison.
    'num_classes': 50,  # Number of classes. There are 50 classes in Core50
    'num_batches': 79,  # including first one (all -> 79)
    #'num_batches': 196,  # including first one (all -> 196)
    #'num_batches': 391,  # including first one (all -> 391)
    'initial_batch': 0, # Valid values: (0, 1). 0 = include initial tuning from ImageNet, 1 = start from previously saved model after 0
    'num_epochs_first_batch': 4.0,  # Training epochs for first batch
    'num_epochs': 4.0,  # Training epochs for all the other batches
    'strategy': 'cwr+',  # Valid values: 'cwr+','naive','lwf','ewc','ar1'
    'lwf_weight': 0.1,  # Weight of previous training. If -1, computed according to pattern proportions in batches
    'ewc_clip_to': 0.001,  # Max value for fisher matrix elements (clip)
    'ewc_w': 1,  # Additional premultiply constant for the Fisher matrix elements
    'cwr_batch0_weight': 1.0,
    'cwr_nic_load_weight': True,
    'cwr_lr_mult': 1,  # Multiplies CWR layers LR by this value
    'rehearsal': False,  # If True, rehearsal is enabled
    'rehearsal_is_latent': False,  # If True, latent rehearsal is enabled
    'rehearsal_memory': 0,  # External memory size
    'rehearsal_layer': '',  # Reharsal layer
    'rehearsal_stop_layer': '',  # Latent extraction layer used in latent rehearsal
    'brn_past_weight': 10000,  # 10000 for pure or no rehearsal, 20000 for latent rehearsal
    'backend': 'GPU',  # Valid values: 'GPU', 'CPU'
    'accuracy_max': 1.0,   # For plotting
    'test_interval_epochs': 4.0,  # Evaluation (and graphical plot) every (fraction of) batch epochs
    'dynamic_head_expansion': False,  # Usually false
    'confusion_matrix': True, # If True, a confusion matrix will be saved after each batch
    'save_ewc_histograms': True, # If True, EWC histogrames will be saved after each batch
    'compute_param_stats': True,
    'verbose': False,
    'initial_class_updates_value': 0.0  # Initial class update value, usually 0.0
}

if (Path.cwd() / 'exp_configuration.json').exists():
    with open(str(Path.cwd() / 'exp_configuration.json')) as f:
        conf_NIC = json.load(f)