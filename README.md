# Pooling-for-MTS-classification
Multivariate time series classification using graph pooling.

This repository is the official implementation of [Multivariate time-series classification with hierarchical variational graph pooling](https://www.sciencedirect.com/science/article/abs/pii/S0893608022002970). 

## Requirements
- Python 3.6.2
- PyTorch 1.4.0

To install requirements:

```setup
pip install -r requirements.txt
```

In this study, all experiments were carried on a computer equipped with GPU NVIDIA GeForce RTX 2080 Ti with 8 Gb GRAM and 32 Gb of RAM. 

## Overview

#### Dataset

We conduct experiments on ten benchmark datasets for multivariate time series classification tasks, this table shows dataset statistics:

| Dataset               | Train Size | Test Size | Num Series | Series Length | Classes |
| ----------------------|------------|-----------|------------|---------------|---------|
| AtrialFibrillation    | 15         | 15        | 2          | 640           | 3       |
| FingerMovements       | 316        | 100       | 28         | 50            | 2       |
| HandMovementDirection | 160        | 74        | 10         | 400           | 4       |
| Heartbeat             | 205        | 205       | 61         | 405           | 2       |
| Libras                | 180        | 180       | 2          | 45            | 15      |
| MotorImagery          | 278        | 100       | 64         | 3000          | 2       |
| NATOPS                | 180        | 1180      | 24         | 51            | 6       |
| PenDigits             | 7494       | 3498      | 2          | 8             | 10      |
| SelfRegulationSCP2    | 200        | 180       | 7          | 1152          | 2       |
| StandWalkJump         | 12         | 15        | 4          | 2500          | 3       |

Dataset can be downloaded from http://timeseriesclassification.com.

#### Preprocessing
We convert the original data into numpy array format and use the original split test and training set.

## Training and testing

To train and test the model(s) in the paper, run this code (an example):

```
python train.py
```

## Results

We train MTPool for 10000 epochs for each train option, and record the model that has the best performance on test set. 

We use accuracy evaluation metrics to evaluate the performance of MTPool model. For specific results, please refer to our paper.

## Modify and Repeat

Once you've successfully run the baseline system, you'll likely want to improve on it and measure the effect of your improvements.

To help you get started in modifying the baseline system to incorporate your new idea -- or incorporating parts of the baseline system's code into your own system -- we provide an overview of how the code is organized:

1. [Model.py] - The core PyTorch model code. If you want to change the overall structure of MTPool, it is recommended to start with this file.

2. [utils.py] - Code containing data preprocessing and other operations.

3. [layer.py] - Code that defines different pooling layers.

4. [gnn_layer.py] - Code related to the implementation of the GNNs.

6. [train.py] - The main driver script that uses all of the above and calls PyTorch to do the main training and testing loops.
