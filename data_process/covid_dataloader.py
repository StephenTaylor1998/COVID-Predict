import torch
from data_process.covid_dataset import COVIDDataset
from torch.utils.data import DataLoader


def train_loader(data_dir: str, batch_size, num_workers=1, window_size: int = 30, select: tuple = None,
                 mode='train', split=0.2, normal_type='div_max'):
    dataset = COVIDDataset(data_dir, window_size, select, mode, split, normal_type)
    return DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)


def test_loader(data_dir: str, batch_size, num_workers=1, window_size: int = 30, select: tuple = None,
                mode='test', split=0.2, normal_type='div_max'):
    dataset = COVIDDataset(data_dir, window_size, select, mode, split, normal_type)
    return DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)


if __name__ == '__main__':
    dataloader = train_loader("../datasets/china.csv", batch_size=2)
    for i in dataloader:
        print(i)
