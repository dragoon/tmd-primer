# Sequence classification with Neural Networks: a primer

The examples in this repository demonstrate the advantages of RNNs and CNNs over traditional ML models on time-series data **with outliers**.

## Task description
We're going to use **transport mode detection task** as our running example.
Given a time series of sensor data, the goal is to classify each time step with one of the predefined transport modes: walk, car, bike, etc.

For demonstration purposes, we will use only two modes, "walk" and "train", so that our task becomes a binary classification task.

## Data
The data is generated synthetically based on common sense assumptions. Outliers in the data represent faulty sensor readings, which often happens in real life (wrong geo-positions, acceleration).
Again, for simplicity, we are going to use a single variable representing the speed of a device.
The [Data generation](rnnprimer/Data%20generation.ipynb) notebook describes the data generation methodology in detail.

## List of notebooks
1. [Data generation](rnnprimer/Data%20generation.ipynb): Describes the data and outlier generation methodology with examples.
2. [Basic Tree model](rnnprimer/Tree%20model.ipynb): Modelling the task using decision trees.
3. [Basic RNN model](rnnprimer/RNN%20Basics.ipynb): Modelling the task using a simple RNN model (GRU).
4. [RNN padding and masking](rnnprimer/RNN%20padding%20and%20masking.ipynb): Generating data samples of different sizes. Padding samples in RNN model.
5. [RNN weights](rnnprimer/RNN%20weights.ipynb): Generating data samples with different class proportions. Weights in the RNN model.
6. [RNN truncated back-propagation](rnnprimer/RNN%20TBTT.ipynb): XXX
7. [Basic CNN model](rnnprimer/CNN%20Basics.ipynb):  Modelling the task using a 1D Convolutional NN model.
