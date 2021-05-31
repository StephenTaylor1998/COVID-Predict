import numpy

from data_process import covid_dataloader, covid_dataset


def origin(path: str):
    train_data = covid_dataset.COVIDDataset(path, mode="train")
    test_data = covid_dataset.COVIDDataset(path, mode="test")
    return train_data.target.numpy(), test_data.target.numpy()


def predict(path: str, model):
    train_loader = covid_dataloader.train_loader(path)
    test_loader = covid_dataloader.test_loader(path)
    model.eval()
    train_output, test_output = [], []
    for inputs, _ in train_loader:
        train_output.append(model(inputs).detach().numpy())
    train_output = numpy.concatenate(train_output, axis=0)
    for inputs, _ in test_loader:
        test_output.append(model(inputs).detach().numpy())
    test_output = numpy.concatenate(test_output, axis=0)
    return train_output, test_output


def get_info(path: str):
    dataset = covid_dataset.COVIDDataset(path)
    return dataset.mean, dataset.std, dataset.min, dataset.max


if __name__ == '__main__':
    from models import cnn
    import torch
    train, test = origin("../../datasets/china.csv")
    print(train.shape, test.shape)

    model = cnn()
    model.load_state_dict(torch.load("../../weights/cnn/epoch30.pth"))
    train, test = predict("../../datasets/china.csv", model)
    print(train.shape, test.shape)

    print(get_info("../../datasets/china.csv"))
