import os
import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as tfs
from torchvision.datasets.cifar import CIFAR10
from tqdm import tqdm
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from distutils.version import LooseVersion
import torch.nn.functional as F

class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = self.setup_device()
        self.model = self.setup_network()
        self.optimizer = self.setup_optimizer()
        self.train_loader = self.get_dataloader()
        self.val_loader = self.get_val_dataloader()
        self.loss = self.setup_loss()
        self.scheduler = self.setup_scheduler()
        self.global_step = 0
        self.save_path = self.cfg['model']['save_dir']
        self.writer = SummaryWriter(log_dir=self.save_path)

    def setup_device(self):
        if self.cfg['args']['gpu_id'] is not None:
            device = torch.device("cuda:{}".format(self.cfg['args']['gpu_id']) if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        return device

    def setup_network(self):
        if self.cfg['args']['network_name'] == 'vit':
            from model.VIT import ViT
            model = ViT()
        return model.to(self.device)

    def get_dataloader(self):
        if self.cfg['dataset']['name'] == 'cifar10':
            transform_cifar = tfs.Compose([
                tfs.RandomCrop(32, padding=4),
                tfs.RandomHorizontalFlip(),
                tfs.ToTensor(),
                tfs.Normalize(mean=(0.4914, 0.4822, 0.4465), std= (0.2023, 0.1994, 0.2010))
            ])
            train_set = CIFAR10(root= './CIFAR10', train= True, download= True, transform=transform_cifar)

        else:
            raise ValueError("Invalid dataset name...")
        loader = torch.utils.data.DataLoader(train_set, batch_size=self.cfg['args']['batch_size'], shuffle=True,
                                             num_workers=self.cfg['args']['num_workers'])

        return loader

    def get_val_dataloader(self):
        if self.cfg['dataset']['name'] == 'cifar10':
            test_transform_cifar = tfs.Compose([
                tfs.ToTensor(),
                tfs.Normalize(mean=(0.4914, 0.4822, 0.4465), std= (0.2023, 0.1994, 0.2010))
            ])
            test_set = CIFAR10(root='./CIFAR10', train= False, download=True, transform= test_transform_cifar)
        else:
            raise ValueError("Invalid dataset name...")
        loader = torch.utils.data.DataLoader(test_set, batch_size=self.cfg['args']['batch_size'], shuffle=True,
                                             num_workers=self.cfg['args']['num_workers'])

    def setup_optimizer(self):
        if self.cfg['solver']['optimizer'] == "adam":
            optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.cfg['solver']['lr'],
                                         weight_decay=self.cfg['solver']['weight_decay'])

        return optimizer

    def setup_loss(self):
        if self.cfg['solver']['loss'] == 'crossentropy':
            loss =torch.nn.CrossEntropyLoss()

        return loss

    def setup_scheduler(self):
        if self.cfg['solver']['scheduler']:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg['args']['epochs'], eta_min=1e-5)

        return scheduler

    def save_model(self, save_path):
        save_file = 'fcn_epochs:{}_optimizer:{}_lr:{}.pth'.format(self.cfg['args']['epochs'],
                                                                  self.cfg['solver']['optimizer'],
                                                                  self.cfg['solver']['lr'])
        path = os.path.join(save_path, save_file)
        torch.save({'model': deepcopy(self.model), 'optimizer': self.optimizer.state_dict()}, path)
        print("Success save")



    def training(self):
        print("Start Training...")
        train_loss = 0

        for epoch in range(self.cfg['args']['epochs']):
            self.model.train()
            if (epoch + 1) % 3 == 0:
                acc = self.validation()
                self.writer.add_scalar(tag='Accuracy', scalar_value=acc, global_step=self.global_step)

            for batch_idx, (data, target) in enumerate(self.train_loader):

                self.global_step += 1
                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += float(loss)

                for param_group in self.optimizer.param_groups:
                    self.cfg['solver']['lr'] = param_group['lr']


                if self.global_step % self.cfg['solver']['print_freq'] == 0:
                    self.writer.add_scalar(tag='train_loss', scalar_value=loss, global_step=self.global_step)
                if self.global_step % (10 * self.cfg['solver']['print_freq']) == 0:
                    self.writer.add_image('train/train_image', self.matplotlib_imshow(data[0].to('cpu')),
                                          dataformats='HWC', global_step=self.global_step)
                    self.writer.add_image('train/predict_image',
                                          self.pred_to_rgb(output[0]),
                                          dataformats='HWC', global_step=self.global_step)
                    self.writer.add_image('train/target_image',
                                          self.trg_to_rgb(target[0]),
                                          dataformats='HWC', global_step=self.global_step)


            self.save_model(self.cfg['model']['checkpoint'])
            self.scheduler.step()


    def validation(self):
        print("Start Validation...")
        self.model.eval()
        correct = 0
        total = 0
        for iter, (data, target, label) in (enumerate(self.val_loader)):
            data = data.to(self.device)
            target = target.to(self.device)

            output = self.model(data)
            pred, idx_ = output.max(-1)
            correct += torch.eq(target, idx_).sum().item()
            total += target.size(0)

        accuracy = correct / total

        return accuracy








