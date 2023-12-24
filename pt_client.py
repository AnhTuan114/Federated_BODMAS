import warnings
import keras
import flwr as fl
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

# Define Flower client
class Client(fl.client.NumPyClient):
    def get_parameters(self, config):  # type: ignore #config các tham số cấu honhfmáy chủ yêu cầu. Báo cho client :tham số cần thiết ,thuộc tính vô hướng
        return utilss.get_model_parameters(model)#Tham số mô hình cục bộ dưới dạng danh sách nd.arrays
#config: tham số cấu hình cho phép của máy chủ: truyền giá trị tùy ý từ máy chủ đến máy khách VD: epoch: nút

    def fit(self, parameters, config):  # parameter: tham số mô hình toàn cầu hiện tại, 
        utilss.set_model_params(model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
            model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=300)
        print(f"Training finished for round {config['server_round']}")
        return utilss.get_model_parameters(model), len(X_train), {}
        
    def evaluate(self, parameters, config):  # đánh giá mô hình
            model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
            utilss.set_model_params(model, parameters)
            loss, accuracy = model.evaluate(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}
    

if __name__ == "__main__":
    # Load MNIST dataset from https://www.openml.org/d/554
    (X_train, y_train), (X_test, y_test) = utilss.load_mnist()

    # Split train set into 10 partitions and randomly use one for training.
    partition_id = np.random.choice(10)
    (X_train, y_train) = utilss.partition(X_train, y_train, 10)[partition_id]
    tf.convert_to_tensor(X_train, dtype=tf.float32)
    # Create LogisticRegression Model
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=X_train.shape[1:3]))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(32, activation="relu"))
    model.add(keras.layers.Dense(4, activation="softmax"))

    # Setting initial parameters, akin to model.compile for keras models
    #cập nhật tham số vào mô hìnhq
    utilss.set_initial_params(model)

    # Start Flower client
    fl.client.start_numpy_client(server_address="localhost:8080", client=Client())#khởi động flower NumpyClient/ MnistClient: lớp cơ sở trừu tượng


