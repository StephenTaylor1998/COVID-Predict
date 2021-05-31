import matplotlib.pyplot as plt
import numpy as np

from data_process.data_visualization import data_handle


def plot_output(data):
    plt.plot(data)
    plt.show()
    pass


if __name__ == '__main__':
    from models import cnn
    import torch
    train_origin, test_origin = data_handle.origin("../../datasets/china.csv")

    model = cnn()
    model.load_state_dict(torch.load("../../weights/cnn/epoch30.pth"))
    train_predict, test_predict = data_handle.predict("../../datasets/china.csv", model)

    origin = np.expand_dims(test_origin[:, 2], axis=-1)
    predict = np.expand_dims(test_predict[:, 2], axis=-1)
    print(test_predict.shape)
    test = np.concatenate([origin, predict], axis=-1)
    print(test.shape)
    plot_output(test)
