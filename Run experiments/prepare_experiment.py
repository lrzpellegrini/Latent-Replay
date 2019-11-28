####################################################################################################
# Copyright (c) 2019. Lorenzo Pellegrini, Gabriele Graffieti, Vincenzo Lomonaco, Davide Maltoni    #
#                                                                                                  #
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file #
####################################################################################################

import sys
import argparse
from pathlib import Path
import json
import experiments_configuration

PROJECT_SOURCE_PATH = Path('../Project Source/')

AVAILABLE_LATENT_LAYERS = ['data', 'conv2_1/dw', 'conv2_2/dw', 'conv3_1/dw', 'conv3_2/dw', 'conv4_1/dw', 'conv4_2/dw', 'conv5_1/dw', 'conv5_2/dw', 'conv5_3/dw',
                           'conv5_4/dw', 'conv5_5/dw', 'conv5_6/dw', 'conv6/dw', 'pool6']

def print_layers():
    print('Here is the list of available layers for latent replay:')
    print(AVAILABLE_LATENT_LAYERS)


def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
                            

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == 'list_layers':
            print_layers()
            exit(0)

    parser = argparse.ArgumentParser(description='Configures the desidered experiment. Run this script with argument list_layers for a complete list of valid latent layer values.', 
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('cl_method', help='The CL method. Valid values: CWR (for CWR*), AR1S (for AR1*), AR1F (for AR1*-free)')
    parser.add_argument('scenario', help='The scenario. Valid values: 79, 196, 391 (for NICv2-79/196/391)')
    parser.add_argument('replay', help='The replay strategy. Valid values: no (no replay), pure (replay from full images), latent (latent replay)')
    parser.add_argument('db_path', help='The path to the Core50 dataset. Check experiments_configuration.py script for more details on the expected directory structure!')
    parser.add_argument('--nvidia_docker', help='Nvidia Docker version. Valid values: "nvidia-docker2" (or "old"), "nvidia-container-toolkit" (or "new"). Defaults to "old" nvidia-docker2\nvidia-docker2: <https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)>\nvidia-container-toolkit: <https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(Native-GPU-Support)#usage>')
    parser.add_argument('--latent_layer', help='Latent layer name. Defaults to conv5_4/dw. Run this script with argument list_layers for a complete list of valid values.')
    parser.add_argument('--replay_memory', help='External replay memory size. Defaults to 1500.')

    args = parser.parse_args()

    cl_method = args.cl_method
    if cl_method not in ['CWR', 'AR1S', 'AR1F']:
        print('Error: invalid CL method', cl_method)
        exit(1)
    
    scenario = args.scenario
    if scenario not in ['79', '196', '391']:
        print('Error: invalid scenario', scenario)
        exit(1)

    replay = args.replay
    if replay not in ['no', 'pure', 'latent']:
        print('Error: invalid replay strategy', cl_method)
        exit(1)

    db_path = Path(args.db_path)
    if not db_path.exists():
        print('Error: invalid dataset path', str(db_path))
        exit(1)
    
    if not (db_path / 'core50_labels.txt').exists():
        print('Error: core50_labels.txt not found at', str(db_path))
        exit(1)
    
    if not (db_path / 'core50_128x128').exists():
        print('Error: core50_128x128 not found at', str(db_path))
        exit(1)

    nvidia_docker = 'old'
    if args.nvidia_docker is not None:
        nvidia_docker = args.nvidia_docker

    if nvidia_docker not in ['old', 'new', 'nvidia-docker2', 'nvidia-container-toolkit']:
        print('Error: invalid nvidia docker argument. Must be "old" or "new".')
        exit(1)
    
    if nvidia_docker == 'nvidia-docker2':
        nvidia_docker = 'old'
    elif nvidia_docker == 'nvidia-container-toolkit':
        nvidia_docker = 'new'
    
    latent_layer_set = False
    latent_layer = 'conv5_4/dw'
    if args.latent_layer is not None:
        latent_layer = args.latent_layer
        latent_layer_set = True
    
    if latent_layer not in AVAILABLE_LATENT_LAYERS:
        print('Error: invalid latent replay layer:', latent_layer)
        exit(1)
    
    if latent_layer_set and replay in ['no', 'pure']:
        print('Error: ixplicit latent layer set, but latent strategy is "', replay, '"', sep='')
        exit(1)
    
    if cl_method == 'CWR' and replay == 'latent' and latent_layer != 'pool6' and latent_layer_set:
        print('Error: CWR strategy is selected but latent layer is not pool6!')
        exit(1)

    if cl_method == 'CWR' and replay == 'latent':
        latent_layer = 'pool6'
    
    replay_memory_set = False
    replay_memory = '1500'
    if args.replay_memory is not None:
        replay_memory = args.replay_memory
        replay_memory_set = True
    
    if not RepresentsInt(replay_memory):
        print('Error: external memory is not a valid integer')
        exit(1)

    replay_memory = int(replay_memory)

    if replay_memory <= 0:
        print('Error: external memory must be greater than 0')
        exit(1)
    
    if replay_memory_set and replay == 'no':
        print('Error: explicit replay memory size set, but replay is disabled')
        exit(1)
    
    if not replay_memory_set:
        print('Info: will use a default replay memory size of 1500')

    scripts_path = Path.cwd()
    experiments_configuration.configure_experiment(cl_method, scenario, replay, db_path, PROJECT_SOURCE_PATH, scripts_path, nvidia_docker, latent_layer, replay_memory)
    