import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt

def generate_data(samples):
  X = np.random.uniform(-10, 20, (samples, 3))
  x = X[:, 0]
  y = X[:, 1]
  z = X[:, 2]
  f = np.sin((x - 3) ** 2 + (y - 5) ** 2 ) + z
  y_label = (f > 20)
  y_label = y_label.astype(float)
  y_label = y_label.reshape(samples, 1)

  return X, y_label

X_train, y_train = generate_data(10000)
X_test, y_test = generate_data(2000)

mean = np.mean(X_train, axis = 0)
std = np.std(X_train, axis = 0)

X_train_norm = (X_train - mean) / std
X_test_norm = (X_test - mean) / std

def cost_function(a, y):
    eps = 1e-10
    return -np.mean(y * np.log(a + eps) + (1 - y) * np.log(1 - a + eps))

def accuracy(y_pred, y_true):
  y_class = (y_pred > 0.5).astype(float)
  acc = np.mean(y_true == y_class)
  return acc
  
parameters = {
  'w1': np.random.normal(0, 0.01, (3, 10)),
  'b1' :  np.zeros((1, 10)),

  'w2': np.random.normal(0, 0.01, (10, 10)),
  'b2' : np.zeros((1, 10)),

  'w3': np.random.normal(0, 0.01, (10, 10)),
  'b3' : np.zeros((1, 10)),

  'w4': np.random.normal(0, 0.01, (10, 1)),
  'b4' : np.zeros((1, 1))
}

def sigmoid(z):
  return 1 / ( 1 + np.exp(-z))

def feed_forward(A0, parameters):
  w1, b1 = parameters['w1'], parameters['b1']
  w2, b2 = parameters['w2'], parameters['b2']
  w3, b3 = parameters['w3'], parameters['b3']
  w4, b4 = parameters['w4'], parameters['b4']
  
  Z1 = np.dot(A0, w1) + b1
  A1 = sigmoid(Z1)
  
  Z2 = np.dot(A1, w2) + b2
  A2 = sigmoid(Z2)
  
  Z3 = np.dot(A2, w3) + b3
  A3 = sigmoid(Z3)
  
  Z4 = np.dot(A3, w4) + b4
  A4 = sigmoid(Z4)
  
  mem_feed = {
     'A0': A0,
     'A1': A1,
     'A2': A2,
     'A3': A3,
     'A4': A4
  }
  return mem_feed

# mem_feed = feed_forward(X_train_norm, parameters)
# print(mem['A1'])

M = X_train_norm.shape[0]
def backprop(y_true, mem):
  A0 = mem['A0']
  dZ4 = mem['A4'] - y_true
  dw4= np.dot(mem['A3'].T, dZ4) / M
  db4 = np.sum(dZ4, axis=0).reshape(1,1) / M
  
  dZ3 = np.dot(dZ4, parameters['w4'].T) * (mem['A3'] * (1 - mem['A3']))
  dw3 = np.dot(mem['A2'].T, dZ3) / M
  db3 = np.sum(dZ3, axis=0).reshape(1, 10) / M
  
  dZ2 = np.dot(dZ3, parameters['w3'].T) * (mem['A2'] * (1 - mem['A2']))
  dw2 = np.dot(mem['A1'].T, dZ2) /M
  db2 = np.sum(dZ2, axis=0).reshape(1, 10) / M
  
  dZ1 = np.dot(dZ2, parameters['w2'].T) * (mem['A1'] * (1 - mem['A1']))
  dw1 = np.dot(A0.T, dZ1) / M
  db1 = np.sum(dZ1, axis=0).reshape(1, 10) / M

  mem_back = {
    'dw1': dw1, 'db1': db1,
    'dw2': dw2, 'db2': db2,
    'dw3': dw3, 'db3': db3,
    'dw4': dw4, 'db4': db4
  }
  return mem_back

# gradients = backprop(y_train, mem_feed)
alpha = 0.1

def gradient_descent(parameters, gradients):
  parameters['w1'] -= gradients['dw1'] * alpha
  parameters['b1'] -= gradients['db1'] * alpha
  
  parameters['w2'] -= gradients['dw2'] * alpha
  parameters['b2'] -= gradients['db2'] * alpha
  
  parameters['w3'] -= gradients['dw3'] * alpha
  parameters['b3'] -= gradients['db3'] * alpha
  
  parameters['w4'] -= gradients['dw4'] * alpha
  parameters['b4'] -= gradients['db4'] * alpha
  return parameters

losses = []
accs = []

for epoch in range(20):
  mem = feed_forward(X_train_norm, parameters)
  grads = backprop(y_train, mem)
  parameters = gradient_descent(parameters, grads)
  
  loss = cost_function(mem['A4'], y_train)
  acc = accuracy(mem['A4'], y_train)
  print(acc,epoch)
  losses.append(loss)
  accs.append(acc)
print(f'Loss Train: {loss}')
print(f'Accuracy Train: {acc}')

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over epoches")
plt.show()

plt.plot(accs)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over epoches")
plt.show()

mem_feed_test = feed_forward(X_test_norm, parameters)
loss = cost_function(mem_feed_test['A4'], y_test)
acc = accuracy(mem_feed_test['A4'], y_test)
print(f'Loss Test: {loss}')
print(f'Accuracy Test: {acc}')
