import flwr as fl
import utilss
from keras.models import Sequential 
from typing import Dict
import numpy as np
import tensorflow as tf
from sklearn.metrics import log_loss
import utilss
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from tensorflow import keras
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

def plot_confusion_matrix(confusion_matrix, labels):
    plt.figure(figsize=(4, 4))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(np.arange(len(labels)) + 0.5, labels)
    plt.yticks(np.arange(len(labels)) + 0.5, labels)
    plt.show()

def fit_round(server_round: int) -> Dict:  # kết quả fit tổng hợp tính bằng trung bình có trọng số 
    """Send round number to client."""
    return {"server_round": server_round}

def get_evaluate_fn(model: Sequential): #hàm lấy loss của trung bình trọng số
    """Return an evaluation function for server-side evaluation."""
    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, (X_test, y_test) = utilss.load_mnist()
    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config): 
        # Update model with the latest parameters
        utilss.set_model_params(model, parameters)
        loss, accuracy = model.evaluate(X_test, y_test)
        print (loss, accuracy)
        return loss, {"Log accuracy": accuracy}
    
    return evaluate 


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    # Create LogisticRegression Model
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[55,1]))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(32, activation="relu"))
    model.add(keras.layers.Dense(4, activation="softmax"))
    utilss.set_initial_params(model)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    strategy = fl.server.strategy.FedAvg( #chiến lược triểm khai lớp cơ sở trừu tượng 
        min_available_clients=2, 
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round, #định cấu hình đào tạo 
    )

    fl.server.start_server(
        server_address="[::]:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=5), #giá trị num_round
    )

    accuracy_list = []
    loss_list = []
    for round_num in range(1, 6):
        loss, metrics = get_evaluate_fn(model)(round_num, model.get_weights(), {})
        accuracy = metrics["Log accuracy"]
        accuracy_list.append(accuracy)
        loss_list.append(loss)

    
    #Import svm model