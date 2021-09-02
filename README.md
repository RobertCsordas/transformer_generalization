# Codebase for training transformers on systematic generalization datasets.

The official repository for our EMNLP 2021 paper [The Devil is in the Detail: Simple Tricks Improve Systematic Generalization of Transformers](https://arxiv.org/abs/2108.12284).

Please note that this repository is a cleaned-up version of the internal research repository we use. In case you encounter any problems with it, please don't hesitate to contact me.

## Setup

This project requires Python 3 (tested with Python 3.8 and 3.9) and PyTorch 1.8.

```bash
pip3 install -r requirements.txt
```

Create a Weights and Biases account and run 
```bash
wandb login
```

More information on setting up Weights and Biases can be found on
https://docs.wandb.com/quickstart.

For plotting, LaTeX is required (to avoid Type 3 fonts and to render symbols). Installation is OS specific.

### Downloading data

All datasets are downloaded automatically except the Mathematics Dataset and CFQ which is hosted in Google Cloud and one has to log in with his/her Google account to be able to access it.

#### Math dataset
Download the .tar.gz file manually from here:

https://console.cloud.google.com/storage/browser/mathematics-dataset?pli=1

Copy it to the ``cache/dm_math/`` folder. You should have a ``cache/dm_math/mathematics_dataset-v1.0.tar.gz`` file in the project folder if you did everyhing correctly. 

#### CFQ
Download the .tar.gz file manually from here:

https://storage.cloud.google.com/cfq_dataset/cfq1.1.tar.gz

Copy it to the ``cache/CFQ/`` folder. You should have a ``cache/CFQ/cfq1.1.tar.gz`` file in the project folder if you did everyhing correctly. 


## Usage

### Running the experiments from the paper on a cluster

The code makes use of Weights and Biases for experiment tracking. In the ```sweeps``` directory, we provide sweep configurations for all experiments we have performed. The sweeps are officially meant for hyperparameter optimization, but we use them to run multiple configurations and seeds.

To reproduce our results, start a sweep for each of the YAML files in the ```sweeps``` directory. Run wandb agent for each of them in the _root directory of the project_. This will run all the experiments, and they will be displayed on the W&B dashboard. The name of the sweeps must match the name of the files in ```sweeps``` directory, except the ```.yaml``` ending. More details on how to run W&B sweeps can be found at https://docs.wandb.com/sweeps/quickstart.

For example, if you want to run Math Dataset experiments, run ```wandb sweep --name dm_math sweeps/dm_math.yaml```. This creates the sweep and prints out its ID. Then run ```wandb agent <ID>``` with that ID.

#### Re-creating plots from the paper

Edit config file ```paper/config.json```. Enter your project name in the field "wandb_project" (e.g. "username/project").

Run the scripts in the ```paper``` directory. For example:

```bash
cd paper
./run_all.sh
```

The output will be generated in the ```paper/out/``` directory. Tables will be printed to stdout in latex format.

If you want to reproduce individual plots, it can be done by running individial python files in the ```paper``` directory.

### Running experiments locally

It is possible to run single experiments with Tensorboard without using Weights and Biases. This is intended to be used for debugging the code locally.
  
If you want to run experiments locally, you can use ```run.py```:

```bash
./run.py sweeps/tuple_rnn.yaml
```

If the sweep in question has multiple parameter choices, ```run.py``` will interactively prompt choices of each of them.

The experiment also starts a Tensorboard instance automatically on port 7000. If the port is already occupied, it will incrementally search for the next free port.

Note that the plotting scripts work only with Weights and Biases.

### Reducing memory usage

In case some tasks won't fit on your GPU, play around with "-max_length_per_batch <number>" argument. It can trade off memory usage/speed by slicing batches and executing them in multiple passes. Reduce it until the model fits.
  
### BibTex
```
@inproceedings{csordas2021devil,
      title={The Devil is in the Detail: Simple Tricks Improve Systematic Generalization of Transformers}, 
      author={R\'obert Csord\'as and Kazuki Irie and J\"urgen Schmidhuber},
      booktitle={Proc. Conf. on Empirical Methods in Natural Language Processing (EMNLP)},
      year={2021},
      month={November},
      address={Punta Cana, Dominican Republic}
}
```
