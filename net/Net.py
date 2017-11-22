from multiprocessing import cpu_count
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Module, DataParallel, CrossEntropyLoss, ModuleList, Conv2d, Dropout, Linear
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset.TwitterDataset import TwitterDataset


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class Net(Module):
    def __init__(self, kernels_size):
        super().__init__()
        types = 50
        self.convs = ModuleList([Conv2d(1, types, (k, 300)) for k in kernels_size])
        self.dropout = Dropout(0.5)
        self.fc = Linear(len(kernels_size) * types, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc(x)
        return logit


if __name__ == '__main__':
    root_path = "../data"
    dataset = TwitterDataset(root_path)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=cpu_count())
    net = Net([3, 4, 5])
    net = DataParallel(net).cuda()
    criterion = CrossEntropyLoss().cuda()
    optimizer = Adam(net.parameters(), lr=0.001)
    epochs = 1000
    print("begin training")
    for epoch in range(epochs):
        losses = []
        for batch, (words_vectors, labels) in enumerate(dataloader, 0):
            words_vectors, labels = Variable(words_vectors.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = net(words_vectors)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean())
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        })
        print('[{}/{}] Loss: {}'.format(epoch + 1, epochs, np.mean(losses)))
