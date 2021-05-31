import torch
from torch import nn


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.l1_cn = nn.Conv1d(3, 3, kernel_size=3, stride=2, groups=3)
        self.l1_ln = nn.LayerNorm([3, 14])
        self.l2_cn = nn.Conv1d(3, 3, kernel_size=3, stride=2, groups=3)
        self.l2_ln = nn.LayerNorm([3, 6])
        self.l3_cn = nn.Conv1d(3, 3, kernel_size=5, stride=2, groups=3)
        self.l3_ln = nn.LayerNorm([3, 1])
        # self.fc = nn.Linear(128, 32)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.l1_ln(self.l1_cn(x))
        x = self.l2_ln(self.l2_cn(x))
        x = self.l3_ln(self.l3_cn(x)).transpose(-1, -2)
        # x = self.fc(x)
        # x = torch.mean(x, dim=1, keepdim=True)
        return x


def cnn(**kwargs):
    return CNN()


if __name__ == '__main__':
    import os
    inp = torch.ones((2, 30, 3))
    model = cnn()
    out = model(inp)
    print(out.shape)
    state = model.state_dict()
    dir_name = os.path.join("../weights/", 'cnn')
    os.makedirs(dir_name, exist_ok=True)
    path_name = os.path.join(dir_name, "model.pth")
    torch.save(state, path_name)
    model.load_state_dict(torch.load(path_name))



