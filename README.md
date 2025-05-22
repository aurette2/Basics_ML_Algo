# Basics Machine Learning Algorithms

This repository contains implementations of gradient descent and its variants (GD, SGD, MBGD, all with momentum) for regression and classification, along with some utility functions such as `split_data()`, `add_ones()`, `mse_loss()`, `grad_mse_loss()`, `cross_entropy()`, `grad_cross_entropy()`, `update_weights()`.

## Usage

### Requirements

To run the code in this repository, you need:

- Python (>=3.6)
- NumPy
- Matplotlib

### Installation

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/aurette2/Basics_ML_Algo.git
    ```

2. Navigate to the repository directory:

    ```bash
    cd Basics_ML_Algo
    ```

3. Create virtual environnement
    ```bash
    python3 -m venv my_env
    source my_env/bin/activate
    ```
4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Examples
#### Main Script
The main.py script provides an interface to generate data and test the implemented algorithms. You can use it as follows:
```
python main.py
```
### How to Use the Repository
Linear_Reg.py: Contains the implementation of linear regression algorithms.
Logistic_Reg.py: Contains the implementation of logistic regression algorithms.
PCA.py: Placeholder for future implementation of PCA algorithm.
main.py: Script to generate data and test the implemented algorithms.
LICENSE: MIT License file.

### Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

### License
This project is licensed under the terms of the MIT license.
