import torch
import torch.nn as nn
import torch.optim as optim

class Train_test():
    def __init__(self,net,optimizer,criterion,device,train_loader=None,test_loader=None):
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self,train_loader,epoch):
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss().to(self.device)
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = criterion(outputs, torch.argmax(targets, dim=1))

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(torch.argmax(targets, dim=1)).sum().item()

        print(epoch, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss / len(train_loader.dataset), 100. * correct / total, correct, total))
        # if (epoch % 10) == 0:
        #     print(epoch, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #                  % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        return 100.*correct/total

    def test(self,test_loader,epoch):
        global best_acc
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
    #            inputs = inputs.long()
                targets = targets.long()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)

                loss = self.criterion(outputs, torch.argmax(targets, dim=1))

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(torch.argmax(targets, dim=1)).sum().item()
        # Save checkpoint.
        acc = 100.*correct/total
        print(" Epoch" , epoch , " Test accuray ", acc)
