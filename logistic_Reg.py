import numpy as np
import matplotlib.pyplot as plt

class Logistic_Regression:
    def __init__(self,lr,nb_epoch):
        self.lr = lr
        self.nb_epoch = nb_epoch
        self.w = None
        self.weight = []
        self.train_losses = []

    def add_ones(self,x):
        return np.hstack((np.ones((x.shape[0],1)),x))

    def sigmoid(self,x):
        z = np.dot(x, self.w)
        return np.divide(1, 1 + np.exp(-z))

    def cross_entropy(self, x, y):
        ypred = self.sigmoid(x)
        return np.divide(-(np.sum(y * np.log(ypred) + (1 - y) * np.log(1 - ypred))), x.shape[0])

    def predict(self,x):
        x = self.add_ones(x)
        return (self.sigmoid(x) >= .5).astype(int)

    def predict_proba(self, x):
        return self.sigmoid(x)

    def fit(self,x,y):
        x = self.add_ones(x)
        y = y.reshape((x.shape[0],1))

        self.w = np.zeros((x.shape[1],1))

        for epoch in range(self.nb_epoch):
            ypredict = self.predict_proba(x)
            grad = np.divide(-(np.transpose(x) @ (y - ypredict)), x.shape[0])
            self.w = self.w - self.lr * grad
            loss = self.cross_entropy(x, y)
        self.train_losses.append(loss)
        
    def accuracy(self,y_true, y_pred):
        acc = np.mean(y_true == y_pred) * 100
        return acc

    def plot_decision(self, X, w, b, y):
        # z = w1x1 + w2x2 + w0
        # one can think of the decision boundary as the line x2=mx1+c
        # Solving we find m and c
        x1 = [X[:,0].min(), X[:,0].max()]
        m = -w[1]/w[2]
        c = -b/w[2]
        x2 = m*x1 + c

        # Plotting
        fig = plt.figure(figsize=(10,8))
        plt.scatter(X[:, 0], X[:, 1],c=y)
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.xlim([-2, 3])
        plt.ylim([0, 2.2])
        plt.xlabel("feature 1")
        plt.ylabel("feature 2")
        plt.title('Decision Boundary')
        plt.plot(x1, x2, 'y-')
