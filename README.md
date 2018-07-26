# FuzzyTensor
This is a very simple implementation of the [Adaptive Neuro-Based Fuzzy Inference System (ANFIS)](https://www.dca.ufrn.br/~meneghet/FTP/anfis%2093.pdf) on Tensorflow.

## Requirements
Known dependencies:
- Python (3.5.5)
- Tensorflow (1.7.0)
- Numpy (1.14.2)
- Matplotlib (2.2.2)

To install dependencies, `cd` to the directory of the repository and run `pip install -r requirements.txt`

## Code Structure

- `anfis.py`: contains the ANFIS implementation.
- `mackey.py`: contains an example that uses ANFIS for the prediction of the Mackey Glass series. This example trains the system on 1500 points of the series and plots the real vs. predicted series, the learning curves, and the resulting membership functions after training.

To run the example, `cd` to the directory of the repository and run `python mackey.py`

## TODO:
- Implement membership functions other than Gaussians.