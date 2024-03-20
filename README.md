# DeepMoFL
This repository stores our experimental codes and results.

## Dataset
The `Dataset` folder contains 121 model errors used in our experiment, with each error corresponding to a Stack Overflow post ID. In the respective folders, there are source codes required to train the bug model.

## Source Code
The `Code` folder contains the source code for our method. You can run it from the command line as follows:
```
python main.py <id> <is_classification> <selected_neuron_num> <activation_threshold> <seed>
```
There are five parameters:
 - `id`: bug id in our dataset
 - `is_classification`: whether the buggy model is a classification model
 - `selected_neuron_num`: the number of selected neurons
 - `activation_threshold`: the activation threshold used to determine whether a neuron is activated
 - `seed`: random number seeds when randomly selecting neurons

## Results
The results folder contains the full results of our experiment.