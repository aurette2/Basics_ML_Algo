from linear_Reg import LinearRegression
from logistic_Reg import LogisticRegression
from PCA import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


# Generate linear regression data
def generate_linear_data(n=1000):
    np.random.seed(0)
    x = np.linspace(-5.0, 5.0, n).reshape(-1, 1)
    y = (29 * x + 30 * np.random.rand(n, 1)).squeeze()
    x = np.hstack((np.ones_like(x), x))
    return x, y

# Generate classification data
def generate_classification_data(n=100):
    X, y = make_classification(n_samples=n, n_features=2, n_redundant=0,
                               n_informative=2, random_state=1,
                               n_clusters_per_class=1)
    return X, y

# Test linear regression
def test_linear_regression():
    print("Testing Linear Regression...")
    x, y = generate_linear_data()
    model = LinearRegression()
    model.train_batch_gradient_descent(x, y, num_epochs=1000)
    plt.plot(model.losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Linear Regression Loss")
    plt.show()

# Test logistic regression
def test_logistic_regression():
    print("Testing Logistic Regression...")
    X, y = generate_classification_data()
    model = LogisticRegression(lr=0.1, nb_epoch=1000)
    model.fit(X, y)
    plt.plot(model.train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Logistic Regression Loss")
    plt.show()
# Test PCA
    
if __name__ == "__main__":
    test_linear_regression()
    test_logistic_regression()