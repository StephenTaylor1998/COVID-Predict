import tqdm
import torch
import models
from data_process import covid_dataloader
from trainer.utils import adjust_learning_rate, save_weight, load_weight


def train(train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    total_loss = 0.0
    model.train()
    train_loader = tqdm.tqdm(train_loader)
    for inputs, target in train_loader:
        output = model(inputs)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        train_loader.set_description("Epoch {}: total_loss={:4f}".format(epoch+1, total_loss))

    train_loader.close()


def test(test_loader, model, criterion):
    # switch to eval mode
    total_loss = 0.0
    model.eval()
    test_loader = tqdm.tqdm(test_loader)
    for inputs, target in test_loader:
        output = model(inputs)
        loss = criterion(output, target)
        total_loss += loss.item()
        test_loader.set_description("Test: total_loss={:4f}".format(total_loss))

    test_loader.close()


def main_worker(args):
    model = models.__dict__[args.model]()
    train_loader = covid_dataloader.train_loader(
        args.data_path, batch_size=args.batch_size, mode='train', num_workers=args.workers
    )
    test_loader = covid_dataloader.train_loader(
        args.data_path, batch_size=args.batch_size, mode='test', num_workers=args.workers
    )
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=args.learn_rate, momentum=args.momentum, weight_decay=args.weight_decay
    # )
    if args.resume is not None:
        load_weight(model, args)
        pass

    if args.test:
        test(test_loader, model, criterion)
        return

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        train(train_loader, model, criterion, optimizer, epoch)
        save_weight(model, args, epoch)

    test(test_loader, model, criterion)


# if __name__ == '__main__':
#     import torch
#     from data_process import covid_dataloader
#
#     model = models.__dict__['cnn']()
#     train_loader = covid_dataloader.train_loader("../datasets/china.csv", batch_size=4, mode='train')
#     test_loader = covid_dataloader.train_loader("../datasets/china.csv", batch_size=4, mode='test')
#     criterion = torch.nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.1)
#     for epoch in range(20):
#         train(train_loader, model, criterion, optimizer, epoch)
#     test(test_loader, model, criterion)
