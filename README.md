# Venkatesh Tentu-Ma23m025

# Transliteration System using RNN, GRU & LSTM with Encoder-Decoder Seq2Seq Models (with and without Attention)

Wandb Link:

## Fileinfo

Dataset-This folder contains data which we used in this Project

Codes-This folder contains codes(da6401_ass3_a is without attention and da6401_ass3_b is with attention)

Test data Predicted Datasets-This folder has two csv files,predictions_vanilla.csv(i.e Predictions of test dataset without attention) and predictions_attentions.csv(Predictions of test dataset with attention

## Project Description

This project aims to build a transliteration system that converts text from one script to another while preserving its phonetic structure. The system leverages advanced neural network architectures, including Recurrent Neural Networks (RNNs), Gated Recurrent Units (GRUs), and Long Short-Term Memory networks (LSTMs). Using encoder-decoder sequence-to-sequence (Seq2Seq) models, the project explores both the conventional approach and enhanced versions incorporating attention mechanisms to improve accuracy.Hyperparameter tuning is done using wandb to find the best performing configurations.

## Dependencies

python,numpy,wandb,torch,matplotlib,pandas

## Model Architecture

### Encoder
- **Embedding layer:** Converts input tokens into dense vectors of fixed size.
- **Bidirectional RNN/GRU/LSTM layers:** Captures bidirectional dependencies in the input sequence.
- **Dropout:** Regularization technique to prevent overfitting.
- **Context vector:** Aggregates information from the entire input sequence into a fixed-size vector.

### Decoder
- **Embedding layer:** Converts input tokens into dense vectors of fixed size.
- **RNN/GRU/LSTM layers:** Generates the output sequence based on the context vector and previous decoder states.
- **Attention mechanism (optional):** Allows the decoder to focus on different parts of the input sequence while generating the output.

## Training

The model is trained using the Adam optimizer with a cross-entropy loss function. During training, the model learns to minimize the difference between the predicted translations and the ground truth translations in the training set.

## Hyperparameter Tuning

Hyperparameters such as embedding size, number of layers, hidden layer size, learning rate, and dropout rate are tuned using Bayesian optimization. Wandb is used for experiment tracking and hyperparameter search.

| Command                   | Description                                   | Accepted Values                   | Value                                                                               |
|---------------------------|-----------------------------------------------|-----------------------------------|-------------------------------------------------------------------------------------|
| `--train_dataset_path`,<br> `-ptrn` | Path to the training dataset            | String                            | `/kaggle/input/dakshina-dataset/te.translit.sampled.train.tsv`                 |
| `--test_dataset_path`,<br> `-ptst`  | Path to the testing dataset             | String                            | `/kaggle/input/dakshina-dataset/te.translit.sampled.test.tsv`                  |
| `--epochs`,<br> `-ep`               | Number of epochs for training          | Integer                           | `10`                                                                                |
| `--optimizer`,<br> `-opt`           | Optimizer for training                 | `'adam'`                          | `adam`                                                                              |
| `--batch_size`,<br> `-bs`           | Batch size for training                | Integer                           | `64`                                                                                |
| `--input_embed_size`,<br> `-ies`    | Size of the input embedding            | Integer                           | `64`                                                                                |
| `--num_enc_layers`,<br> `-nel`      | Number of layers in the encoder        | Integer                           | `3`                                                                                 |
| `--num_dec_layers`,<br> `-ndl`      | Number of layers in the decoder        | Integer                           | `3`                                                                                 |
| `--hid_layer_size`,<br> `-hls`      | Size of the hidden layer               | Integer                           | `512`                                                                               |
| `--cell_type`,<br> `-ct`            | Type of RNN cell for encoder and decoder | `'lstm'`                        | `lstm`                                                                              |
| `--bidirectional`,<br> `-bd`        | Whether to use bidirectional RNN layers  | Boolean                           | `True`                                                                              |
| `--dropout`,<br> `-dp`              | Dropout rate for regularization        | Float                             | `0.2`                                                                               |
| `--new_learning_rate`,<br> `-lr`    | Learning rate for the optimizer        | Float                             | `0.001`                                                                             |


# Sweep Configuration(Without Attention)

Below are the Sweep Configuration which i used for Question 1 to Question 4

input_embed_size: 16,32,64,256,512

num_enc_layers: 1,2,3

num_dec_layers: 1,2,3

hid_layer_size: 16,32,64,256,512

cell_type: 'rnn','gru','lstm'

bidirectional: True, False

dropout: 0.2, 0.3

new_learning_rate:0.001,0.01,0.1

# Best Hyperparameters(Without Attention):

input_embed_size: 64

num_enc_layers: 2

num_dec_layers: 2

hid_layer_size: 512

cell_type: lstm

bidirectional: False

dropout: 0.3

new_learning_rate:0.001

With above hyperparameters best validation accuracy is 56.802%,and when i tried with the test dataset,i got test accuracy:57%

# Sweep Configuration(With Attention)

Below are the Sweep Configuration which i used for Question 5

input_embed_size: 16,32,64,256,512

num_enc_layers: 1

num_dec_layers: 1

hid_layer_size: 16,32,64,256,512

cell_type: lstm

bidirectional: True ,False

dropout: 0.2, 0.3

new_learning_rate:0.001,0.01,0.1

# Best Hyperparameters(With Attention):

input_embed_size: 64

num_enc_layers: 1

num_dec_layers: 1

hid_layer_size: 512

cell_type: lstm

bidirectional: True

dropout: 0.2

new_learning_rate:0.001

With above hyperparameters best validation accuracy is 61.75%,and when i tried with the test dataset,i got test accuracy:61.73%

## Testing

After training, the model is evaluated on the test set to assess its performance on unseen data. The test accuracy and loss are reported to measure the effectiveness of the model.

# Conclusion

With above reults,we can conclude that after adding  Attention to our network,,our model accuracy has been increased.
