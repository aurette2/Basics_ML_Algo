from linear_Reg import Linear_regression
from logistic_Reg import Logistic_Regression
from neural_network import Neural_network
from PCA import PCA
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd

# Split data into train/test data
def train_test_split(X,y):
  '''
  this function takes as input the sample X and the corresponding features y
  and output the training and test set
  '''
  np.random.seed(0) # To demonstrate that if we use the same seed value twice, we will get the same random number twice

  train_size = 0.8
  n = int(len(X)*train_size)
  indices = np.arange(len(X))
  np.random.shuffle(indices)
  train_idx = indices[: n]
  test_idx = indices[n:]
  X_train, y_train = X[train_idx], y[train_idx]
  X_test, y_test = X[test_idx], y[test_idx]

  return X_train, y_train, X_test, y_test

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
    xtrain, ytrain, _, _= train_test_split(x,y)
    model = Linear_regression()
    model.train_batch_gradient_descent(xtrain, ytrain, num_epochs=1000)
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
    X_train, y_train, X_test, y_test = train_test_split(X,y)
    print(f" the training shape is: {X_train.shape}")
    print(f" the test shape is: {X_test.shape}")
    model.fit(X_train,y_train)
    ypred_train = model.predict(X_train)
    acc = model.accuracy(y_train,ypred_train)
    print(f"The training accuracy is: {acc}")
    print(" ")

    ypred_test = model.predict(X_test)
    acc = model.accuracy(y_test,ypred_test)
    print(f"The test accuracy is: {acc}")
    model.plot_decision(X_train,model.w,model.w[0],y_train)
    plt.show()
    
# Test2 logistic regression
def test2_logistic_regression():
    print("Second Testing Logistic Regression...")
    X, y = generate_classification_data2()
    model = Logistic_Regression(lr=0.1, nb_epoch=1000)
    X_train, y_train, X_test, y_test = train_test_split(X,y)
    print(f" the training shape is: {X_train.shape}")
    print(f" the test shape is: {X_test.shape}")
    model.fit(X_train,y_train)
    ypred_train = model.predict(X_train)
    acc = model.accuracy(y_train,ypred_train)
    print(f"The training accuracy is: {acc}")
    print(" ")

    ypred_test = model.predict(X_test)
    acc = model.accuracy(y_test,ypred_test)
    print(f"The test accuracy is: {acc}")
    model.plot_decision(X_train,model.w,model.w[0],y_train)
    plt.show()

# Test PCA
    
def test_PCA():
    print("Testing PCA............")
    iris = load_iris()
    X = iris['data']
    y = iris['target']
    n_samples, n_features = X.shape

    print('Number of samples:', n_samples)
    print('Number of features:', n_features)

    my_pca = PCA(n_component=2)
    my_pca.fit(X)
    new_X = my_pca.transform(X)
    my_pca.plot(new_X,y)
    plt.show()


# Test NN
def test_NN():
    print("Testing Simple layer Neural Network from scrach............")
    # generate data
    var = 0.2
    n = 800
    class_0_a = var * np.random.randn(n//4,2)
    class_0_b =var * np.random.randn(n//4,2) + (2,2)

    class_1_a = var* np.random.randn(n//4,2) + (0,2)
    class_1_b = var * np.random.randn(n//4,2) +  (2,0)

    X = np.concatenate([class_0_a, class_0_b,class_1_a,class_1_b], axis =0)
    Y = np.concatenate([np.zeros((n//2,1)), np.ones((n//2,1))])

    # shuffle the data
    rand_perm = np.random.permutation(n)

    X = X[rand_perm, :]
    Y = Y[rand_perm, :]
    X = X.T
    Y = Y.T
    # train test split
    ratio = 0.8
    X_train = X [:, :int (n*ratio)]
    Y_train = Y [:, :int (n*ratio)]

    X_test = X [:, int (n*ratio):]
    Y_test = Y [:, int (n*ratio):]
    model = Neural_network()
    model.fit(X_train, Y_train, X_test, Y_test)

    plt.plot(model.train_loss)
    plt.plot(model.test_loss)
    plt.show()

    y_pred = model.predict(X_train)
    train_accuracy = model.accuracy(y_pred, Y_train)
    print ("train accuracy :", train_accuracy)

    y_pred = model.predict(X_test)
    test_accuracy = model.accuracy(y_pred, Y_test)
    print ("test accuracy :", test_accuracy)



if __name__ == "__main__":
    test_linear_regression()
    test1_logistic_regression()
    test2_logistic_regression()
    test_PCA()
    test_NN()