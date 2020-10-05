# Sequence classification with Neural Networks: a primer

This repository demonstrates the advantages of RNNs and CNNs over traditional ML models on time-series data **with outliers**.

## Task description
We're going to use **transport mode detection task** as our running example.
Given a time series of sensor data, the goal is to classify each time step with one of the predefined transport modes: walk, car, bike, etc.

For demonstration purposes, we will use only two modes, "walk" and "train", so that our task becomes a binary classification task.

## Data
The data is generated synthetically based on common sense assumptions. Outliers in the data represent faulty sensor readings, which often happens in real life (wrong geo-positions, acceleration).
For simplicity, a single feature representing the speed of a device is used.
The [Data generation](rnnprimer/Data%20generation.ipynb) notebook describes the data generation methodology in detail.

## List of notebooks
1. [Data generation](https://nbviewer.jupyter.org/github/dragoon/rnn-primer/blob/master/tmdprimer/Data%20generation.ipynb): Describes the data and outlier generation methodology with examples.
2. [Basic Tree model](https://nbviewer.jupyter.org/github/dragoon/rnn-primer/blob/master/tmdprimer/Tree%20model.ipynb): Modelling the task using decision trees.
3. [Tree model with multiple time steps: TODO](https://nbviewer.jupyter.org/github/dragoon/rnn-primer/blob/master/tmdprimer/Tree%20model%20advanced.ipynb): Decision tree that has access to past timesteps.
4. [Basic CNN model](https://nbviewer.jupyter.org/github/dragoon/rnn-primer/blob/master/tmdprimer/CNN%20Basics.ipynb):  Modelling the task using a Convolutional NN model.
5. [Basic RNN model](https://nbviewer.jupyter.org/github/dragoon/rnn-primer/blob/master/tmdprimer/RNN%20Basics.ipynb): Modelling the task using a simple Recurrent NN model (with GRU).
6. [RNN padding and masking](https://nbviewer.jupyter.org/github/dragoon/rnn-primer/blob/master/tmdprimer/RNN%20padding%20and%20masking.ipynb): Generating data samples of different sizes. Padding samples in RNN model.
7. [RNN class weights: TODO](https://nbviewer.jupyter.org/github/dragoon/rnn-primer/blob/master/tmdprimer/RNN%20class%20weights.ipynb): Generating data samples with different class proportions. Class weights in the RNN model.
8. [RNN truncated back-propagation: TODO](https://nbviewer.jupyter.org/github/dragoon/rnn-primer/blob/master/tmdprimer/RNN%20TBTT.ipynb): TODO
