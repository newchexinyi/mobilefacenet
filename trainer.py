import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from loss import MyLoss
import utils


class Trainer:
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.net = kwargs['net']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = self.args.writer
        self.logger = self.args.logger
        self.criterion = MyLoss().to(self.args.device)

    def train(self, train_loader, valid_loader1, valid_loader2):
        model_dir = os.path.join(self.writer.log_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        epochs = self.args.epochs
        valid_every_epochs = self.args.valid_every_epochs
        early_stop_epochs = self.args.early_stop_epochs
        num_steps = len(train_loader)
        self.sum_train_steps = 0
        self.sum_valid_steps = 0
        best_metric = np.float32('inf')
        no_better_epoch = 0
        for epoch in range(0, epochs + 1):
            # train
            sum_loss = 0
            self.net.train()
            train_bar = tqdm(train_loader, total=num_steps, desc=f'Epoch-{epoch}')
            for (x, target) in train_bar:
                # forward
                x, target = x.float().to(self.args.device), target.squeeze().long().to(self.args.device)
                out, _ = self.net(x, target)
                loss = self.criterion(out, target)
                train_bar.set_postfix(loss=f'{loss.item():.5f}')
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                # visualization
                self.writer.add_scalar('loss/train_loss', loss.item(), global_step=self.sum_train_steps)
                sum_loss += loss.item()
                self.sum_train_steps += 1
            avg_loss = sum_loss / num_steps
            self.logger.info(f'Epoch-{epoch}\tloss:{avg_loss:.3f}')
            self.writer.add_scalar('step-epoch', epoch, global_step=self.sum_train_steps)
            # valid
            if epoch % valid_every_epochs == 0:
                metric = self.valid(valid_loader1, valid_loader2)
                eer = metric['eer']
                self.writer.add_scalar('EER', eer, global_step=epoch)
                if eer <= best_metric:
                    no_better_epoch = 0
                    best_metric = eer
                    best_model_path = os.path.join(model_dir, 'best_checkpoint.pth.tar')
                    utils.save_model_state_dict(best_model_path, epoch=epoch,
                                                net=self.net.module if self.args.dp else self.net,
                                                optimizer=self.optimizer)
                else:
                    # early stop
                    no_better_epoch += 1
                    if no_better_epoch > early_stop_epochs > 0: break
            # save model
            if epoch >= self.args.start_save_model_epochs:
                if (epoch - self.args.start_save_model_epochs) % self.args.save_model_interval_epochs == 0:
                    model_path = os.path.join(model_dir, f'{epoch}_checkpoint.pth.tar')
                    utils.save_model_state_dict(model_path, epoch=epoch,
                                                net=self.net.module if self.args.dp else self.net,
                                                optimizer=self.optimizer)

    def valid(self, valid_loader1, valid_loader2):
        metric = {}
        sum_loss = 0
        num_steps = len(valid_loader1)
        self.net.eval()
        valid_bar1 = tqdm(valid_loader1, total=num_steps, desc=f'Valid1')
        valid_bar2 = tqdm(valid_loader2, total=num_steps, desc=f'Valid2')
        fea_list1, label_list1 = [], []
        for (x, target) in valid_bar1:
            x, target = x.float().to(self.args.device), target.squeeze().long().to(self.args.device)
            label_list1.append(target.cpu())
            with torch.no_grad():
                _, features = self.net(x, target)
            fea_list1.append(features.cpu())
        fea_list2, label_list2 = [], []
        for (x, target) in valid_bar2:
            x, target = x.float().to(self.args.device), target.squeeze().long().to(self.args.device)
            label_list2.append(target.cpu())
            with torch.no_grad():
                _, features = self.net(x, target)
            fea_list2.append(features.cpu())
        scores = []
        label_list1 = torch.cat(label_list1, dim=0).numpy()
        label_list2 = torch.cat(label_list2, dim=0).numpy()
        features1 = F.normalize(torch.cat(fea_list1, dim=0))
        features2 = F.normalize(torch.cat(fea_list2, dim=0))
        for i in range(features1.shape[0]):
            scores.append(torch.matmul(features1[i].unsqueeze(0), features2[i].unsqueeze(1))[0])
        scores = np.array(scores)
        labels = label_list1 == label_list2
        eer, thresh = utils.solve_eer(scores, labels)
        metric['eer'] = eer
        metric['thresh'] = thresh
        self.logger.info(f'EER: {eer:.2f}%\tThresh: {thresh:.3f}')
        return metric

    # def test(self, test_loader, valid_loader):
    #     metric = self.valid(valid_loader)
    #     thresh = metric['thresh']
    #     num_steps = len(valid_loader)
    #     self.net.eval()
    #     test_bar = tqdm(test_loader, total=num_steps, desc=f'Test')
    #     fea_list, label_list = [], []
    #     for (x, target) in test_bar:
    #         x, target = x.float().to(self.args.device), target.squeeze().long().to(self.args.device)
    #         label_list.append(target.cpu())
    #         with torch.no_grad():
    #             _, features = self.net(x, target)
    #         fea_list.append(features.cpu())
    #     labels = torch.cat(label_list, dim=0).reshape(-1, 1)
    #     match_labels = torch.eq(labels, labels.T)[~torch.eye(labels.shape[0], dtype=torch.bool)].reshape(-1)
    #     features = F.normalize(torch.cat(fea_list, dim=0))
    #     scores = torch.matmul(features, features.T)[~torch.eye(labels.shape[0], dtype=torch.bool)].reshape(-1)
    #     eer, thresh = utils.solve_eer(scores.numpy(), match_labels.numpy())
    #     acc = utils.solve_accuarcy(scores, labels, thresh)
    #     self.logger.info(f'Acc: {acc:.2f}%\tEER: {eer:.2f}%\tThresh: {thresh:.3f}')
