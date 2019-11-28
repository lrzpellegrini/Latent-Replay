####################################################################################################
# Copyright (c) 2019. Lorenzo Pellegrini, Gabriele Graffieti, Vincenzo Lomonaco, Davide Maltoni    #
#                                                                                                  #
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file #
####################################################################################################

import numpy as np

# ------ REHEARSAL ------
ExtMem = [[], []]
ExtMemSize = 0


def allocate_memory(num_patterns, size_x, size_y):
    global ExtMem
    global ExtMemSize

    assert size_y == 1, "Rehearsal Memory: size_y must be 1!"
    ExtMem = [np.zeros((num_patterns, size_x), dtype=np.float32),
              np.zeros(num_patterns, dtype=np.int32)]
    ExtMemSize = num_patterns

    print("Rehearsal Memory created for %d pattern and size %d" % (num_patterns, num_patterns*(size_x+size_y)))


# Selects random patterns to replace in the external memory
def update_memory(train_x, train_y, batch):
    global ExtMem
    global ExtMemSize

    n_cur_batch = ExtMemSize // (batch + 1)
    if n_cur_batch > train_x.shape[0]:
        n_cur_batch = train_x.shape[0]
    n_ext_mem = ExtMemSize - n_cur_batch

    assert n_ext_mem >= 0, "Rehearsal: n_ext_mem should never be less than 0!"
    assert n_cur_batch <= train_x.shape[0], "Rehearsal: non enough pattern to get in current batch!"

    idxs_cur = np.random.choice(train_x.shape[0], n_cur_batch, replace=False)

    if n_ext_mem == 0:
        ExtMem = [train_x[idxs_cur], train_y[idxs_cur]]
    else:
        idxs_ext = np.random.choice(ExtMemSize, n_ext_mem, replace=False)
        ExtMem = [np.concatenate((train_x[idxs_cur], ExtMem[0][idxs_ext])),
                  np.concatenate((train_y[idxs_cur], ExtMem[1][idxs_ext]))]


def get_samples():
    global ExtMem

    return ExtMem[0], ExtMem[1] 
