# Latent Replay for Real-Time Continual Learning

This is a Caffe implementation of Latent Replay: a Continual Learning technique for Real Time and On The Edge applications.

A custom Caffe distribution packaged as a Docker image is used. More info and source code can be found [here](https://github.com/lrzpellegrini/CI-Customized-BVLC-caffe-docker).

## Reference

Our article is now available [here](https://arxiv.org/abs/1912.01100)!

    @article{pellegrini2019latent,
        title={Latent Replay for Real-Time Continual Learning},
        author={Lorenzo Pellegrini and Gabriele Graffieti and Vincenzo Lomonaco and Davide Maltoni},
        year={2019},
        eprint={1912.01100},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }

## Running the experiments
You can run an experiment by following the steps below:
    
1. Install the Nvidia Docker Toolkit from [here](https://github.com/NVIDIA/nvidia-docker)

2. Move inside the `Run experiments` folder:

```bash
cd "Run experiments"
```

3. Prepare the project source and create the bash script. This can be achieved by issuing the following command:

```bash
python prepare_experiment.py method scenario replay path-to-core50 [--nvidia_docker x] [--latent_layer layer] [--replay_memory size]
```

where *method* can be "CWR", "AR1F" (for AR1\* free) or "AR1S" (for AR1\*), *scenario* can be "79", "196" or "391" (for NICv2-79/196/391) and *replay* can be "no", "pure" or "latent". You can also execute the script with a single argument "-h" to view a description of the expected parameters.

You can set the desidered Nvidia Docker run method by passing either:
  - "--nvidia\_docker nvidia-docker2" [more info here](https://github.com/nvidia/nvidia-docker/wiki/Installation-\(version-2.0\))
  - "--nvidia\_docker nvidia-container-toolkit" [more info here](https://github.com/nvidia/nvidia-docker/wiki/Installation-\(Native-GPU-Support\))

as an argument. Defaults to nvidia-docker2.

When passing the "path-to-core50" argument, make sure that the selected folder contains the following content:
  - A file named "*core50_labels.txt*", containing the Core50 labels. Can be downloaded [here](https://vlomonaco.github.io/core50/data/core50_class_names.txt)
  - A folder named "*core50_128x128*" containing the 128x128 version of the CORe50 dataset. Can be downloaded [here](http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip)

You can set the replay memory size by passing the parameter "\-\-replay\_memory N" where N must be greater than 0. Defaults to 1500.

You can set the latent replay layer by passing the parameter "\-\-latent\_layer layer\_name" where layer\_name defaults to "conv5\_4/dw". For a full list of available replay layers execute the provided script with the single argument "list\_layers".

Here is an example of a valid command:
```bash
python prepare_experiment.py AR1F 391 latent /home/x/datasets/core50 --latent_layer pool6
```
    
4. Running the aforementioned python script will have the following effects:
    - Copy the correct prototxt files inside the `Project Source/NIC_v2/NIC_v2_X/REPLAY_TYPE/` folder;
    - Create a proper `exp_configuration.json` file inside the `Project Source` folder;
    - Create a `run_experiment.sh` file inside the `Run experiments` folder. Should be already executable when created;
  
5. Execute the `run_experiment.sh` script as follows:

```bash
./run_experiment.sh
```

This will run the experiment on "run0" inside our docker image in interactive mode (issuing the CTRL+C command or closing the terminal will terminate the experiment).

## Content

The content of this repository can be summarized as follows:

- The project source code (`Project Source` folder)
    -  `inc_training_Core50.py` contains the entry point;
    -  `nicv2_configuration.py` contains the experiment configuration loader;
    - The filelists used (`batch_filelists` subdirectory): for the proposed scenarios (NICv2 79, 196, 391) we provide a separate folder, each containing a sub-directory for each run. We used the first 5 runs in order to obtain the average test accuracy curves reported in our paper;
    - A MobileNetV1 pretrained with ImageNet (`models/MobileNetV1.caffemodel`);
    - The prototxt(s) describing the solvers and nets (`NIC_v2` folder);
- A set of configuration scripts required to run the experiments (`Run experiments` folder)

## Core50 Dataset
The Core50 Dataset can be downloaded from <https://vlomonaco.github.io/core50/index.html#download>
In our test we used the 128x128 version, zip archive.
