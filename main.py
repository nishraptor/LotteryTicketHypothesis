import torch
import torchvision.models as models
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
from models import LeNet
from torch import nn
import matplotlib.pyplot as plt
from config import cfg
import numpy as np

def create_mask(model):

    step = sum([1 for name, param in model.named_parameters if 'weight' in name])
    mask = [None] * step
    step = 0

    for name, param in model.named_parameters():
        if 'weight' in name:
            mask[step] = np.ones_like(param.data.cpu().numpy())
            step = step + 1

    return mask

def prune(percent, model, mask):
    step = 0

    for name, param in model.named_parameters():

        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            nonzero = tensor[np.nonzero[tensor]]
            percentile = np.percentile(abs(nonzero), percent) # Get the value where 'percent' values are less than

            new_mask = np.where(abs(tensor) < percentile, 0, mask[step])

            param.data = torch.from_numpy(tensor * new_mask).to('cuda')
            mask[step] = new_mask
            step += 1

    return model, mask

def original_init(model, mask, state_dict):
    step = 0

    for name,param in model.named_parameters():
        if 'weight' in name:
            param.data = torch.from_numpy(mask[step] * state_dict[name].cpu(().numpy())).to('cuda')
            step += 1
        if 'bias' in name:
            param.data = state_dict[name]
    return model


def train(model,train_loader,test_loader,optimizer,criterion):

    mask = create_mask(model)
    init_state = model.state_dict().copy()

    training_loss = np.zeros(cfg['epochs'])
    #training_accuracy = defaultdict(int)
    testing_loss = np.zeros(cfg['epochs'])
    testing_accuracy = np.zeros(cfg['epochs'])

    for prune_iteration in range(cfg['prune_iterations']):
        if prune_iteration != 0:
            model, mask = prune(cfg['prune_percent'], model, mask)
            model = original_init(model, mask, init_state)

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
        total = 0
        with torch.no_grad():
            for images,labels in test_loader:
                images = images.cuda()
                labels = labels.cuda()

                output = model(images)

                loss = criterion(output,labels)
                testing_loss[i] += loss.item()

                _, pred = torch.max(output.data,1)
                total += labels.size(0)
                testing_accuracy[i] += (pred == labels).sum().item()

        print('Accuracy: ', testing_accuracy[i]/total)
        testing_loss[i] /= len(test_loader)


    return model, training_loss, testing_loss, testing_accuracy

def main():
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    train_MNIST = torchvision.datasets.MNIST('data/', train = True, download = True, transform = transform)
    test_MNIST =  torchvision.datasets.MNIST('data/', train = False, download = True, transform = transform)

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