import torch
import numpy as np
import random

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
set_seed(2023)
import sys
sys.path.append('..')
from parser import get_config
from pretrainTrainer import PretrainTrainer
from pretrainDataset import PretrainDataloader
from horae import Horae
import warnings
import logging
import os
warnings.filterwarnings('ignore')


def print_config(config):
    for item in config.keys():
        print(str(item) + ':' + str(config[item]))


def train_one_model():
    config = get_config()
    config['device'] = torch.device('cuda:' + str(config['gpu_id'])) if config['cuda'] else torch.device('cpu')
    dataloader = PretrainDataloader(config)
    train_dataloader, valid_dataloader, test_dataloader = dataloader.generate_dataloader(config)
    print_config(config)

    logging.basicConfig(format='[%(asctime)s] - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.StreamHandler()])
    torch.set_printoptions(linewidth=2000)
    logging.info(config)
    model = Horae(config)
    model = model.to(config['device'])

    if config['stage'] != 'pretrain' and not config['load_model_path']:
        raise NotImplementedError   # downstream stage needs pretrained model

    if config['load_model_path']:
        model.load_state_dict(torch.load(config['load_model_path'], map_location=config['device']), strict=False)
        if config['stage'] != 'pretrain' and config['freeze']:
            model.downstream_freeze_parameter()

    trainer = PretrainTrainer(model, config, train_dataloader, valid_dataloader, test_dataloader)
    trainer.valid(0)
    return trainer.train()

if __name__ == '__main__':
    train_one_model()
