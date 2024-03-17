import numpy as np
import matplotlib.pyplot as plt 
class Linear_regression:
  """Implementation of gradient descent and its variants (GD, SGD, MBGD, all with momentum)"""

  def __init__(self):
    self.losses = []


  def mse(self, y, ypred):
    """The loss function"""
    return np.mean((ypred - y) ** 2)

  def linear_function(self, X, theta):
    """Compute X: NxD @ theta: Dx1 """
    return np.dot(X,theta)

  def batch_gradient(self, X, y, theta):
    """The derivative of mse"""
    return np.divide(2 * np.dot(X.T, self.linear_function(X,theta) - y),y.shape[0]) #Dx1

  def initialise_theta(self,X):
    """Initialization of theta with shape Dx1"""
    N, D = X.shape
    return np.zeros([D,1])

  def update_fonction(self, theta, lr, grad):
    """Updating the target"""
    return theta - lr * grad

  def plot_loss(self):
    """Plotting Losses"""
    plt.plot(self.losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")

  def sGD(self, xi, yi, theta):
    """The gradient for a single sample"""
    return 2 * xi.T * (self.linear_function(xi,theta) - yi)

  def shuffle(self,X,y):
    """Shuffle X and Y index"""
    N, _ = X.shape
    return X[np.random.permutation(N)],y[np.random.permutation(N)]

  def momentum(self, momentum, beta, grad):
    """Get the momentum"""
    return beta * momentum + (1 - beta) * grad

  def train_batch_gradient_descent(self, X, y, num_epochs, lr=0.1):
    """ Trainning using GD """
    theta = self.initialise_theta(X)
    for epoch in range(num_epochs):
      ypred = self.linear_function(X,theta)
      loss = self.mse(y,ypred)
      grads = self.batch_gradient(X, y, theta)
      theta = self.update_fonction(theta, grads, lr)
      self.losses.append(loss)

  def train_batch_gradient_descent_with_momentum(self, X, y, num_epochs, beta=.99, lr=0.1):
    """ Trainning using GD """
    theta = self.initialise_theta(X)
    self.losses = []
    momentum = np.zeros([X.shape[1],1])
    for epoch in range(num_epochs):
      ypred = self.linear_function(X,theta)
      loss = self.mse(y,ypred)
      grads = self.batch_gradient(X, y, theta)
      momentum = self.momentum(momentum, beta, grads)
      theta = self.update_fonction(theta, momentum, lr)
      self.losses.append(loss)

  def train_sGD(self, X, y, num_epochs, lr=0.1):
    """ Trainning using SGD """
    theta = self.initialise_theta(X)
    self.losses = []
    epoch = 0
    loss_tolerance = 0.001
    avg = float("inf")
    while epoch < num_epochs and avg > loss_tolerance:
      runLoss = 0.0
      xi, yi = self.shuffle(X,y)
      for idx in range(xi.shape[0]):
        sample_x = xi[idx].reshape(-1, X.shape[1])
        sample_y = yi[idx].reshape(-1, 1)
        ypred = self.linear_function(sample_x, theta)
        loss = self.mse(sample_y,ypred)
        grad = self.sGD(sample_x, sample_y, theta)
        theta = self.update_fonction(theta, grad, lr)
        runLoss += loss
        avg = runLoss / X.shape[0]
        epoch +=1
        self.losses.append(avg)

  def train_sGD_with_momentum(self, X, y, num_epochs, beta=.99, lr=0.1):
    """ Trainning using SGD """
    theta = self.initialise_theta(X)
    self.losses = []
    epoch = 0
    loss_tolerance = 0.001
    avg = float("inf")
    while epoch < num_epochs and avg > loss_tolerance:
      runLoss = 0.0
      momentum = 0.0
      xi, yi = self.shuffle(X,y)
      for idx in range(xi.shape[0]):
        sample_x = xi[idx].reshape(-1, X.shape[1])
        sample_y = yi[idx].reshape(-1, 1)
        ypred = self.linear_function(sample_x, theta)
        loss = self.mse(sample_y,ypred)
        grad= self.sGD(sample_x, sample_y, theta)
        momentum = self.momentum(momentum, beta, grad)
        theta = self.update_fonction(theta, momentum, lr)
        runLoss += loss      #loss
        avg = runLoss / X.shape[0] #avearage loss
        epoch +=1
        self.losses.append(avg)


  def train_mbGD(self,  X, y, num_epochs, batch_size, lr=0.1):
    nb_batch = X.shape[0] / batch_size
    theta = self.initialise_theta(X)
    self.losses = []
    avg = float("inf")
    xi, yi = self.shuffle(X,y)
    for epoch in range(num_epochs):
      runLoss = 0.0
      for idx in range(0,X.shape[0],batch_size):
        sample_x = xi[idx: idx+batch_size]  # select a batch of xi
        sample_y = yi[idx: idx+batch_size].reshape(-1, 1)  # select a batch of yi
        ypred = self.linear_function(sample_x, theta)
        loss = self.mse(sample_y,ypred)
        grads = self.batch_gradient(sample_x, y, theta)
        theta = self.update_fonction(theta, grads, lr)
        runLoss += loss / sample_x.shape[0] # loss here is mean for batch
        avg = runLoss / X.shape[0]
        self.losses.append(avg)


  def train_mbGD_with_momentum(self,  X, y, num_epochs, batch_size, beta=.99, lr=.1):
    nb_batch = X.shape[0] / batch_size
    theta = self.initialise_theta(X)
    self.losses = []
    momentum = np.zeros([X.shape[1],1])
    avg = float("inf")
    xi, yi = self.shuffle(X,y)
    for epoch in range(num_epochs):
      runLoss = 0.0
      for idx in range(0,X.shape[0],batch_size):
        sample_x = xi[idx: idx+batch_size]  # select a batch of xi
        sample_y = yi[idx: idx+batch_size].reshape(-1, 1)  # select a batch of yi
        ypred = self.linear_function(sample_x, theta)
        loss = self.mse(sample_y,ypred)
        grads = self.batch_gradient(sample_x, y, theta)
        momentum = self.momentum(momentum, beta, grads)
        theta = self.update_fonction(theta, momentum, lr)
        runLoss += loss / sample_x.shape[0] # loss here is mean for batch
        avg = runLoss / X.shape[0]
        self.losses.append(avg)
