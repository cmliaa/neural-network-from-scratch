# Neural Network from Scratch
This project builds a fully connected neural network from scratch using NumPy to perform  bianry classification on a synthetic dataset.
It demenostrates core deep-learning operations without using high-level frameworks.

# Dataset
a dataset is generated using:

(x, y, z) = sin((x − 3)² + (y − 5)²) + z 

Labels are assigned as:

y = 1 if f > 20 else 0

The training set contains 10,000 samples, and the test set contains 2,000 samples.

All features are normalized using training‑set mean and standard deviation.

# Model
A four‑layer neural network is implemented manually:

- Hidden layers: 3 × 10‑neuron fully connected layers
- Activation: Sigmoid
- Output: Sigmoid (binary classification)


Forward pass, backpropagation, and gradient descent are implemented directly with NumPy.

# Training
The model is trained for 20 epochs using:

- Binary cross‑entropy loss
- Learning rate: 0.1
- Full‑batch gradient descent


Training loss and accuracy are recorded and visualized.

# Evaluation
After training, the script reports:

- Final training loss and accuracy
- Test loss and accuracy
- Loss and accuracy curves across epochs
