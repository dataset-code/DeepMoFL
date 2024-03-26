# GUIDER
This repository stores our experimental codes and results.

## Dataset
### Collection

Our dataset is acquired from two aspects.

First, we revisited the dataset from the previous studies ([Deepmufl](https://github.com/ali-ghanbari/deepmufl-ase-2023), [DeepFD](https://github.com/ArabelaTso/DeepFD), [Defect4ML](https://github.com/mohmehmo/defects4ml)). After removing duplicate bugs, the three datasets contain 160 StackOverflow bugs. Among them, 49 bugs were excluded because they could not be reproduced or were not model bugs, etc., so we selected the remaining 111 bugs.

Second, we used the SQL query provided by [Deepmufl](https://github.com/ali-ghanbari/deepmufl-ase-2023) and added some rules to obtain StackOverflow posts from Stack Exchange Data Explorer and then filtered model bugs from them. The query is in `Dataset/query.sql` and the file `Dataset/QueryResults.csv` contains the raw result of running the SQL query as of Januray 2024, which contains 240 eligible StackOverflow posts. We further manually filtered model bugs and collected a total of 10 bugs.

### Bugs
The `Dataset/bugs` folder contains 121 model bugs used in our experiment, with each bug corresponding to a Stack Overflow post ID. In the respective folders, there are source codes required to train the buggy model.

The `dataset.csv` file contains basic information about each bug, including Stack Overflow post ID, whether it is a classification model, the number of error layers, and error type.


## Source Code
We've tested our code on Windows 10 and Ubuntu 22.04 with Python 3.10.

### Requirments
 - tensorflow == 2.9.0
 - pandas == 2.2.1

### Run
The `Code` folder contains the source code for our method. You can run it from the command line as follows:
```
python run.py -m MODEL -i INPUT -o OUTPUT -c CLASSIFICATION -n SELECTED_NEURON_NUM -t THRESHOLD -s SEED
```
There are seven parameters:
 - `MODEL`: The file path of the model, which needs to be .h5 file
 - `INPUT`: The file path to the test case inputs, which needs to be a .npy file
 - `OUTPUT`: The file path to the test case outputs, which needs to be a .npy file
 - `CLASSIFICATION`: whether the buggy model is a classification model (If it's a classification model, pass 1. If it's a regression model, pass 0)
 - `SELECTED_NEURON_NUM`: the number of selected neurons
 - `SELECTED_NEURON_NUM`: the activation threshold used to determine whether a neuron is activated
 - `SEED`: random number seeds when randomly selecting neurons

## Results
The `results` folder contains the full results of our experiment.
 - The folder `baseline` contains the experimental results for the baseline methods. The result file for each method contains the bug id, running time, output, and whether the bug was detected. Due to the lengthy output of Deepmufl, we put its results separately in a folder.
 - The folder `raw_data` contains the raw results of our method. Since we trained each buggy model three times, we ran our method on three models for each bug separately. The results for each bug are stored under the folder named with the corresponding post ID, which contains 54 files and is named `res_{threshold}_{selected_neuron_num}_{seed}.json`. The files with the gini suffix are the results obtained using the mean&gini aggregation method, and the files without the gini suffix are the results obtained using the mean aggregation method. The raw result file is in json format and contains the number of neurons per layer for each buggy model and the number of selected neurons, five weighted suspicious scores (sorted by layer number), and the running time.
 - The folders `gini` and `mean` store the summarized results of the corresponding aggregation methods, respectively, and each folder contains 10 csv files, which are the results of 9 repeated experiments and the average of the results, respectively.
 - The `time_cost.csv` file records the time to train each buggy model. 
 - The `res_all.csv` file records the number of bugs detected by all methods combined. 
 - The `res_overlap.csv` file records the overlap of bug detection between our method and baseline methods. 
 - The files `RQ3_1.csv` and `RQ3_2.csv` record the results of our method for different configurations.