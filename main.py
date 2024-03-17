from linear_Reg import Linear_regression
from logistic_Reg import Logistic_Regression
from PCA import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.datasets import make_classification


# Generate linear regression data
def generate_linear_data(n=1000):
    np.random.seed(0)
    x = np.linspace(-5.0, 5.0, n).reshape(-1, 1)
    y = (29 * x + 30 * np.random.rand(n, 1)).squeeze()
    x = np.hstack((np.ones_like(x), x))
    return x, y

# Generate classification data 1
def generate_classification_data1(n=100):
    # Read CSV file
    data = pd.read_csv('moon_data.csv')

    # Extract features and target
    X = data[['feature_1', 'feature_2']].values
    y = data['target'].values
    return X, y

# Generate classification data2
def generate_classification_data2(n=100):
    # Read CSV file
    data = pd.read_csv('make_classification.csv')

    # Extract features and target
    X = data[['feature_1', 'feature_2']].values
    y = data['target'].values
    return X, y

# Test linear regression
def test_linear_regression():
    print("Testing Linear Regression...")
    x, y = generate_linear_data()
    model = Linear_regression()
    model.train_batch_gradient_descent(x, y, num_epochs=1000)
    plt.plot(model.losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Linear Regression Loss")
    plt.show()

# Test1 logistic regression
def test1_logistic_regression():
    print("Fisrt Testing Logistic Regression...")
    X, y = generate_classification_data1()
    model = Logistic_Regression(lr=0.1, nb_epoch=1000)
    model.fit(X, y)
    plt.plot(model.train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Logistic Regression Loss")
    plt.show()
    
# Test2 logistic regression
def test2_logistic_regression():
    print("Second Testing Logistic Regression...")
    X, y = generate_classification_data2()
    model = Logistic_Regression(lr=0.1, nb_epoch=1000)
    model.fit(X, y)
    plt.plot(model.train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Logistic Regression Loss")
    plt.show()

# Test PCA
    
if __name__ == "__main__":
    test_linear_regression()
    test1_logistic_regression()
    test2_logistic_regression