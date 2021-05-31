import numpy
import pandas as pd
import torch
from torch.utils.data import Dataset


class COVIDDataset(Dataset):
    def __init__(self, data_dir: str, window_size: int = 30, select: tuple = None,
                 mode='train', split=0.2, normal_type='div_max'):
        csv = pd.read_csv(data_dir).values[:, : 3]
        csv = numpy.array(csv, dtype=numpy.float32).squeeze()
        if mode == "train":
            csv = csv[:int(csv.shape[0] * (1-split))]
        else:
            csv = csv[int(csv.shape[0] * (1-split)):]
        if select is not None:
            tmp = []
            for index in select:
                tmp.append(numpy.expand_dims(csv[:, index], axis=-1))
                csv = numpy.array(csv, dtype=numpy.float32)

            csv = numpy.concatenate(tmp, 1)

        csv = numpy.array(csv, dtype=numpy.float32)
        self.mean = numpy.mean(csv, axis=0, keepdims=True)
        self.std = numpy.std(csv, axis=0, keepdims=True)
        self.min = numpy.min(csv, axis=0)
        self.max = numpy.max(csv, axis=0)
        if normal_type == "div_max":
            csv = csv / self.max

        elif normal_type == "norm":
            csv = (csv - self.mean) / self.std

        else:
            raise NotImplemented

        self.inputs, self.target = [], []
        for i in range(window_size + 1, csv.shape[0]):
            self.inputs.append(csv[i - window_size:i, :])
            self.target.append(csv[i:i + 1, :])

        self.inputs = torch.from_numpy(numpy.array(self.inputs))
        self.target = torch.from_numpy(numpy.array(self.target))

    def __len__(self):
        assert self.inputs.shape[0] == self.target.shape[0], "shape miss matched"
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.target[idx]


if __name__ == '__main__':
    dataset = COVIDDataset("../datasets/china.csv", select=(0, 2), mode='train')
    # dataset = COVIDDataset("../datasets/china.csv", normal_type="norm")

    for item in dataset:
        print(item[0].shape)
        print(item[1].shape)
        print(item[0])
        print(item[1])
        break

    print(dataset.__len__())
    print(dataset.mean)
    print(dataset.std)
    print(dataset.min)
    print(dataset.max)
