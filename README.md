# ARC-AGI-Challenge-2024

This repo contains my solution to the ARC AGI Challenge 2024 hosted on Kaggle. See [kaggle site](https://www.kaggle.com/competitions/arc-prize-2024/overview) for a description of the competition.

## Directory structure

The structure of the code is rather simple:

- /cfg contains configuration
- /code contains the python code. 
- /input. kaggle competition data should be downloaded in the input directory
- /notebooks contains data prep and mdodel training notebooks
- /checkpoints contains saved checkpoints

The code is not decomposed into logical units but has some structure init. First there is dataset creation, then data loader, then model class implementation, then training and inference code. The model can be trained using Pytorch distributed data parallel.

the configuration file contains all optiosn that can be overriden at runtime.

There are two versions of code and configuration. Version 53 is the one used ot train the model used for the competition. Version 64 contains the test timne training code used by the submisison notebook. Verison 64 is mostly a superset of version 53.

There are three notebooks. 

- `data_gen.ipynb` is a data generfation notebook to be used with the re-arc code, see below.
- `train.ipynb` is the training ruin we used for the competition
- `ttt.ipynb` is a test time training code we ran with the competition evaluation data.

Dependencies are minimal. Code was run using NVIDIA ngc/pytorch container v 24.09-py3 . The only additional install is:

`pip install rotary_embedding_torch`

If running in a fresh python environment, installing numpy, pytorch, einops and the above should get you up and running.

## Solution description

The pdf paper in main directory is a short paper describing our solution.

## Reproducing the solution

Steps to reproduce my solution:
- clone Michael Hodel data generator: https://github.com/michaelhodel/re-arc
- copy the noebook 'gen1000.ipynb' from the notebook directory and paste it in the re-arc local directlry. Then run the notebook. It should generate a directory gen1000 containing 10k exmaples per task. Running it took 6 hours on my workstation. That code scould be made parallel, but I did not bother.
- run the training notebook (or run the otrchrun command directly in a terminal) by using the path to the generated data as option `--data_path`. Notebook `train.ipynb` shows the run we used for the competition.Thjsi should save a ,model checkpoint. That checkpoint can then be uploaded to Kaggle as a dataset and be use din our submisison notebook.








