from typing import Tuple, Union, List
import numpy as np
from json_tricks import dump, load
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

Params = Union[XY, Tuple[np.ndarray]]

def find_label(name): 
        if('Benign' in name): return 0
        elif('Ransomware' in name): return 1
        elif('Spyware' in name): return 2
        elif('Trojan' in name): return 3
        
# def get_model_parameters(model: Sequential) -> LogRegParams: #trả về tham số mô hình cục bộ
#     """Returns the paramters of a sklearn LogisticRegression model."""
#     params = [model.coef_,]
#     return params

# def set_model_params(model: Sequential, params: LogRegParams) -> Sequential:
#     """Sets the parameters of a sklean LogisticRegression model."""
#     model.coef_ = params[0]
#     return model

# def set_initial_params(model: Sequential): #cập nhật tham số
#     """Sets initial parameters as zeros Required since model params are
#     uninitialized until model.fit is called.
#     But server asks for initial parameters from clients at launch. Refer
#     to sklearn.linear_model.LogisticRegression documentation for more
#     information.
#     """
#     n_classes = 4  # MNIST has 10 classes
#     n_features = 55  # Number of features in dataset
#     model.classes_ = np.array([i for i in range(4)])
#     model.coef_ = np.zeros((n_classes, n_features))

def get_model_parameters(model: Sequential) -> list:
    """Returns the parameters of a Keras Sequential DNN model."""
    return model.get_weights()

def set_model_params(model: Sequential, params: list) -> Sequential:
    """Sets the parameters of a Keras Sequential DNN model."""
    model.set_weights(params)
    return model

def set_initial_params(model: Sequential):
    n_classes = 4  # MNIST has 10 classes
    n_features = 55  # Number of features in dataset
    model.classes_ = np.array([i for i in range(4)])
    input_example = np.zeros((n_classes, n_features))  # Đầu vào ví dụ
    _ = model.predict(input_example)

def load_mnist() -> Dataset:
    data = pd.read_csv('D:\Federated learning\Dataset\Obfuscated-MalMem2022.csv')
    X= data.iloc[:, 1:56]
    y= data ['Category']
    # Tách label từ cột category và lưu thành cột mới
    labels=[]
    for i in y:
        labels.append( i.split('-')[0])
    data ['label'] = labels
    y = []
    for i in data ['label']:
        y.append(find_label(i))
    X = np.array(X)
    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return (x_train, y_train), (x_test, y_test)

def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList: #phân chia dữu liệu
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )
