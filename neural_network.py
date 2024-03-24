import numpy as np
import matplotlib.pyplot as plt

class Neural_network:
    def __init__(self):
        self.alpha = 0.01
        self.n_epochs = 1000
        self.h0, self.h1, self.h2 = 2, 10, 1                # Number of input unit(h0), hiden Unit(h1), output unit(h2)
        self.W1, self.W2, self.b1, self.b2 =[],[],[],[]
        self.dW1, self.dW2, self.db1, self.db2 = [],[],[],[]
        self.Z1, self.A1, self.Z2, self.A2 = [],[],[],[]    
        self.train_loss = []
        self.test_loss = []

    def init_params(self):
        '''Initialisation of parametters using Xavier initialisation'''
        # self.W1 = np.random.normal(0,np.sqrt(2/self.h0+self.h1),(self.h1,self.h0))
        # self.W2 = np.random.normal(0,np.sqrt(2/self.h2+self.h1),(self.h2,self.h1))
        # self.b1 = np.random.normal(0,np.sqrt(2/self.h0+self.h1),(self.h1,1))
        # self.b2 = np.random.normal(0,np.sqrt(2/self.h2+self.h1),(self.h2,1))
        self.W1 = np.random.randn(self.h1,self.h0)
        self.W2 = np.random.randn(self.h2,self.h1)
        self.b1 = np.random.randn(self.h1,self.h2)
        self.b2 = np.random.randn(self.h2,self.h2)

    def sigmoid(self,z):
       return 1 / (1 + np.exp(-z))

    def d_sigmoid(self,z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def loss(self,y_pred, Y):
        return np.divide(-(np.sum(Y * np.log(y_pred) + (1-Y) * np.log(1 - y_pred))),Y.shape[1])
    
    def accuracy(self,y_pred, y):
        return np.mean(y == y_pred) * 100

    def predict(self,X):
        return self.sigmoid(self.W2.dot(self.sigmoid(self.W1.dot(X) + self.b1)) + self.b2) 

    def forward_pass(self,X):
        self.Z1 = self.W1.dot(X) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = self.sigmoid(self.Z2)

    def backward_pass(self, X, Y):
        temp = self.A2-Y
        self.dW1 = self.W2.T * temp * self.d_sigmoid(self.Z1) @ X.T 
        self.dW2 = temp @ self.A1.T
        self.db1 = self.d_sigmoid(self.Z1) @ temp.T * self.W2.T
        self.db2 = np.sum(temp, axis=1, keepdims=True)

    def update(self, alpha ):
        self.W1 -= alpha * self.dW1
        self.W2 -= alpha * self.dW2
        self.b1 -= alpha * self.db1
        self.b2 -= alpha * self.db2

    def fit(self,X_train, Y_train, X_test, Y_test):
        self.init_params() 
        for i in range(self.n_epochs):
            ## forward pass
            self.forward_pass(X_train)
            ## backward pass
            self.backward_pass(X_train, Y_train)
            ## update parameters
            self.update(self.alpha )
            ## save the train loss
            self.train_loss.append(self.loss(self.A2, Y_train))
            ## compute test loss
            self.forward_pass(X_test)
            self.test_loss.append(self.loss(self.A2, Y_test))

    def plot(self):
        plt.title(f"Train and Test loss")
        plt.plot(self.train_loss)
        plt.plot(self.test_loss)
        plt.show()


