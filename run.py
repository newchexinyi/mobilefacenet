import os
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from net import MobileFacenet
from trainer import Trainer
from dataset import MyDataset
import utils

sep = os.sep

def main(args):
    # set random seed
    utils.setup_seed(args.random_seed)
    # set device
    cuda = args.cuda
    device_ids = args.device_ids
    args.dp = False
    if not cuda or device_ids is None:
        args.device = torch.device('cpu')
    else:
        args.device_ids = [int(i) for i in args.device_ids.split(',')]
        args.device = torch.device(f'cuda:{device_ids[0]}')
        if len(device_ids) > 1: args.dp = True
    #
    # load data
    num_valid_class = 400
    if not args.load_epoch:
        args.logger.info('Model Training...')
        train_file_list = utils.get_filename_list(args.train_dirs)
        train_name2label = utils.name2label(train_file_list)
        train_file_list, valid_file_list = utils.split_train_valid(train_file_list, list(train_name2label.keys()), num_valid_class=num_valid_class)
        args.logger.info(f'Select Match Samples for valid...')
        valid_file_list1, valid_file_list2 = utils.get_match_files(valid_file_list, train_name2label, factor=int(args.match_factor))
        train_dataset = MyDataset(train_file_list, train_name2label, transform_flag=True)
        valid_dataset1 = MyDataset(valid_file_list1, train_name2label, transform_flag=False)
        valid_dataset2 = MyDataset(valid_file_list2, train_name2label, transform_flag=False)
        train_dataloader = DataLoader(train_dataset, batch_size=int(args.batch_size),
                                      shuffle=True, num_workers=int(args.num_workers))
        valid_dataloader1 = DataLoader(valid_dataset1, batch_size=int(args.batch_size),
                                      shuffle=False, num_workers=int(args.num_workers), drop_last=True)
        valid_dataloader2 = DataLoader(valid_dataset2, batch_size=int(args.batch_size),
                                       shuffle=False, num_workers=int(args.num_workers), drop_last=True)
        # set model
        net = MobileFacenet(num_class=len(train_name2label) - num_valid_class, arcface=args.arcface, m=int(args.m), s=int(args.s))
        if args.dp:
            net = nn.DataParallel(net, device_ids=args.device_ids)
        net = net.to(args.device)
        # optimizer & scheduler
        optimizer = torch.optim.Adam(net.parameters(), lr=float(args.lr))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                               eta_min=0.1 * float(args.lr))
        # trainer
        trainer = Trainer(args=args,
                          net=net,
                          optimizer=optimizer,
                          scheduler=scheduler)
        trainer.train(train_dataloader, valid_dataloader1, valid_dataloader2)
    else:
        args.logger.info('Model Testing...')
        train_file_list = utils.get_filename_list(args.train_dirs)
        train_name2label = utils.name2label(train_file_list)
        test_file_list = utils.get_filename_list(args.test_dirs)
        test_name2label = utils.name2label(test_file_list)
        args.logger.info(f'Select Match Samples for test...')
        test_file_list1, test_file_list2 = utils.get_match_files(test_file_list, test_name2label, factor=int(args.match_factor))
        test_dataset1 = MyDataset(test_file_list1, test_name2label, transform_flag=False)
        test_dataset2 = MyDataset(test_file_list2, test_name2label, transform_flag=False)
        test_dataloader1 = DataLoader(test_dataset1, batch_size=int(args.batch_size),
                                     shuffle=False, num_workers=int(args.num_workers), drop_last=True)
        test_dataloader2 = DataLoader(test_dataset2, batch_size=int(args.batch_size),
                                      shuffle=False, num_workers=int(args.num_workers), drop_last=True)
        # set model
        net = MobileFacenet(num_class=len(train_name2label) - num_valid_class, arcface=args.arcface, m=args.m, s=args.s)
        if args.dp:
            net = nn.DataParallel(net, device_ids=args.device_ids)
        net = net.to(args.device)
        # trainer
        trainer = Trainer(args=args,
                          net=net,
                          optimizer=None,
                          scheduler=None)
        # test model
        load_epoch = args.load_epoch if args.load_epoch else 'best'
        model_path = os.path.join(args.writer.log_dir, 'model', f'{load_epoch}_checkpoint.pth.tar')
        if args.dp:
            trainer.net.module.load_state_dict(torch.load(model_path)['model'])
        else:
            trainer.net.load_state_dict(torch.load(model_path)['model'])
        trainer.valid(test_dataloader1, test_dataloader2)


def run():
    # init config parameters
    params = utils.load_yaml(file_path='./config.yaml')
    parser = argparse.ArgumentParser(description=params['description'])
    for key, value in params.items():
        parser.add_argument(f'--{key}', default=value, type=utils.set_type)
    args = parser.parse_args()
    # init logger and writer
    time_str = time.strftime('%Y-%m-%d-%H', time.localtime(time.time()))
    args.version = f'{time_str}-{args.version}' if not args.load_epoch and args.time_version else args.version
    log_dir = f'runs/{args.version}'
    writer = SummaryWriter(log_dir=log_dir)
    logger = utils.get_logger(filename=os.path.join(log_dir, 'running.log'))
    # save version files
    if args.save_version_files: utils.save_load_version_files(log_dir, args.save_version_file_patterns, args.pass_dirs)
    # run
    args.writer, args.logger = writer, logger
    args.logger.info(args.version)
    main(args)


if __name__ == '__main__':
    run()