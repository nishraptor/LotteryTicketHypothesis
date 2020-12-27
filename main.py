import torch
import torchvision.models as models
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
from models import LeNet
from torch import nn
from collections import defaultdict
import matplotlib.pyplot as plt
from config import cfg


def train(model,train_loader,test_loader,optimizer,criterion):

    training_loss = defaultdict(int)
    #training_accuracy = defaultdict(int)
    testing_loss = defaultdict(int)
    testing_accuracy = defaultdict(int)

    for i in range(cfg['epochs']):
        print('Epoch', i)

        #training
        for idx, (images,labels) in enumerate(train_loader):
            print('Index: %d/%d'%(idx, len(train_loader)))

            optimizer.zero_grad()

            images = images.cuda()
            labels = labels.cuda()


            output = model(images)
            loss = criterion(output,labels)

            training_loss[i] += loss



            loss.backward()
            optimizer.step()

        training_loss[i] /= len(train_loader)
        print(training_loss[i])

        #testing
        correct = 0
        with torch.no_grad():
            for images,labels in test_loader:
                images = images.cuda()
                labels = labels.cuda()

                output = model(images)
                loss = criterion(output,labels)
                testing_loss[i] += loss.item()

                _, pred = torch.max(output.data,1)
                testing_accuracy[i] += (pred == labels).sum().item()

        testing_loss[i] /= len(test_loader)


    return model, training_loss, testing_loss, testing_accuracy

def main():
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    train_MNIST = torchvision.datasets.MNIST('data/', train = True, download = True, transform = transform)
    test_MNIST =  torchvision.datasets.MNIST('data/', train = True, download = True, transform = transform)

    train_loader = torch.utils.data.DataLoader(train_MNIST,
                                              batch_size=cfg['batch_size'],
                                              shuffle=True,
                                              num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_MNIST,
                                              batch_size=cfg['batch_size'],
                                              shuffle=True,
                                              num_workers=0)

    model = LeNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), cfg['learning_rate'])


    model,training_loss, testing_loss, testing_accuracy = train(model, train_loader,test_loader, optimizer, criterion)

    print(training_loss)
    print(testing_loss)
    print(testing_accuracy)

    torch.save(model.state_dict(), './LeNet_%s_epochs_prune_percent_%s.pth'%(str(cfg['epochs']), str(cfg['prune_percent'])))


if __name__ == '__main__':
    main()