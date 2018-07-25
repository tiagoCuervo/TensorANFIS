# FuzzyTensor
This is a very simple implementation of the [Adaptive Neuro-Based Fuzzy Inference System (ANFIS)](https://www.dca.ufrn.br/~meneghet/FTP/anfis%2093.pdf) on Tensorflow.

## Code Structure

- `anfis.py`: contains the ANFIS implementation.
- `mackey.py`: contains an example that uses ANFIS for the prediction of the Mackey Glass series. This example trains the system on 1500 points of the series and plots the real vs. predicted series, the learning curves, and the resulting membership functions after training.

## Requirements
Known dependencies:
- Python (3.5.5)
- Tensorflow (1.7.0)
- Numpy (1.14.2)
- Matplotlib (2.2.2)

## TODO:
- Implement membership functions other than Gaussians.