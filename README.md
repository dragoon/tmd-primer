# RNN Primer

The examples in this repository demonstrate the advantages of RNNs over traditional ML models on time-series data **with outliers**.

## Task description
I'm using the following classification task throughout the examples:

**General**: classify the transport mode of a user given device sensor data.

**Specific**: binary classification between "walk" and "train" modes, which correspond to a person walking or riding a train.

## Data
The data is generated synthetically based on the common sense assumptions (see first notebook for details).

## List of notebooks
* [RNN Basics](rnnprimer/RNN%20Basics.ipynb): Introduction to how data are generated. Showcases the advantages of an RNN model (GRU) over a Random Forest model on equal size samples.
* [RNN padding and masking](rnnprimer/RNN%20padding%20and%20masking.ipynb): Generating data samples of different sizes. Padding samples in RNN model.
* Generating data samples with different class proportions. Weights in the RNN model. 
