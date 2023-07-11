import os
import json
from collections import defaultdict

import torch
import random

import numpy as np
from torchvision.models import resnet18

import datasets.ss_transforms as sstr

from torch import nn
from client import Client
from server import Server
from utils.args import get_parser
from datasets.idda import IDDADataset
from datasets.gta5 import GTA5
from serverGTA import ServerGTA
from models.deeplabv3 import deeplabv3_mobilenetv2
from utils.stream_metrics import StreamSegMetrics, StreamClsMetrics

import sys
import yaml
import wandb


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataset_num_classes(dataset):
    if dataset == 'idda':
        return 16
    if dataset == 'femnist':
        return 62
    if dataset == 'gta5':
        return 16
    raise NotImplementedError


def model_init(args):
    if args.model == 'deeplabv3_mobilenetv2':
        return deeplabv3_mobilenetv2(num_classes=get_dataset_num_classes(args.dataset))
    if args.model == 'resnet18':
        model = resnet18()
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=get_dataset_num_classes(args.dataset))
        return model
    if args.model == 'cnn':
        # TODO: missing code here!
        raise NotImplementedError
    raise NotImplementedError


def get_transforms(args):
    # TODO: test your data augmentation by changing the transforms here!
    if args.model == 'deeplabv3_mobilenetv2':
        train_transforms = sstr.Compose([
            sstr.RandomResizedCrop((512, 928), scale=(0.5, 2.0)),
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transforms = sstr.Compose([
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif args.model == 'cnn' or args.model == 'resnet18':
        train_transforms = nptr.Compose([
            nptr.ToTensor(),
            nptr.Normalize((0.5,), (0.5,)),
        ])
        test_transforms = nptr.Compose([
            nptr.ToTensor(),
            nptr.Normalize((0.5,), (0.5,)),
        ])
    else:
        raise NotImplementedError
    return train_transforms, test_transforms


def read_femnist_dir(data_dir):
    data = defaultdict(lambda: {})
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        data.update(cdata['user_data'])
    return data


def read_femnist_data(train_data_dir, test_data_dir):
    return read_femnist_dir(train_data_dir), read_femnist_dir(test_data_dir)


def get_datasets(args, train_transforms = None, test_transforms = None):

    train_datasets = []
    test_train_datasets = []

    if train_transforms == None or test_transforms == None:
        train_transforms, test_transforms = get_transforms(args)

    if args.dataset == 'idda':
        root = 'data/idda'
        if args.framework == 'federated':
            with open(os.path.join(root, 'train.json'), 'r') as f:
                all_data = json.load(f)
            for client_id in all_data.keys():
                train_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], transform=train_transforms,
                                                client_name=client_id))
                
                test_train_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], transform = test_transforms,
                                                client_name=client_id))
        
        elif args.framework == 'centralized':
            with open(os.path.join(root, 'train.txt'), 'r') as f:
                train_data = f.read().splitlines()
                train_datasets.append(IDDADataset(root=root, list_samples=train_data, transform=train_transforms,
                                                    client_name="iddaTrain"))
                
                test_train_datasets.append(IDDADataset(root=root, list_samples=train_data, transform=test_transforms,
                                                    client_name="iddaTrain"))
                       
        with open(os.path.join(root, 'test_same_dom.txt'), 'r') as f:
            test_same_dom_data = f.read().splitlines()
            test_same_dom_dataset = IDDADataset(root=root, list_samples=test_same_dom_data, transform=test_transforms,
                                                client_name='test_same_dom')
            
        with open(os.path.join(root, 'test_diff_dom.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()
            test_diff_dom_dataset = IDDADataset(root=root, list_samples=test_diff_dom_data, transform=test_transforms,
                                                client_name='test_diff_dom')
        test_datasets = [test_same_dom_dataset, test_diff_dom_dataset]


    elif args.dataset == 'femnist':
        niid = args.niid
        train_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'train')
        test_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'test')
        train_data, test_data = read_femnist_data(train_data_dir, test_data_dir)

        train_transforms, test_transforms = get_transforms(args)

        train_datasets, test_datasets = [], []

        for user, data in train_data.items():
            train_datasets.append(Femnist(data, train_transforms, user))
        for user, data in test_data.items():
            test_datasets.append(Femnist(data, test_transforms, user))

    elif args.dataset == 'gta5':
        train_datasets.append(GTA5(transform=train_transforms, test_transform=test_transforms, client_name = 'train_gta5', target_dataset='idda'))
        
        idda_root = 'data/idda'
        with open(os.path.join(idda_root, 'train.txt'), 'r') as f:
            idda_train_data = f.read().splitlines()
            idda_train = IDDADataset(root=idda_root, list_samples=idda_train_data, transform=train_transforms,
                                                client_name="idda_test")
            
            
        with open(os.path.join(idda_root, 'test_same_dom.txt'), 'r') as f:
            test_same_dom_data = f.read().splitlines()
            test_same_dom_dataset = IDDADataset(root=idda_root, list_samples=test_same_dom_data, transform=test_transforms,
                                                client_name='test_same_dom')
            
        with open(os.path.join(idda_root, 'test_diff_dom.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()
            test_diff_dom_dataset = IDDADataset(root=idda_root, list_samples=test_diff_dom_data, transform=test_transforms,
                                                client_name='test_diff_dom')
            
        test_datasets = [idda_train, test_same_dom_dataset, test_diff_dom_dataset]

    else:
        raise NotImplementedError

    return train_datasets, test_train_datasets, test_datasets

    #[train_dataset], [test_train_dataset], [test_same_dom_data, test_diff__dom]
    

def set_metrics(args):
    num_classes = get_dataset_num_classes(args.dataset)
    if args.model == 'deeplabv3_mobilenetv2' and args.dataset == 'idda':
        metrics = {
            'eval_train': StreamSegMetrics(num_classes, 'eval_train'),
            'test_same_dom': StreamSegMetrics(num_classes, 'test_same_dom'),
            'test_diff_dom': StreamSegMetrics(num_classes, 'test_diff_dom')
        }
    elif args.model == 'deeplabv3_mobilenetv2' and args.dataset == 'gta5':
        metrics = {
            'idda_test' : StreamSegMetrics(num_classes, 'idda_test'),
            'test_same_dom': StreamSegMetrics(num_classes, 'test_same_dom'),
            'test_diff_dom': StreamSegMetrics(num_classes, 'test_diff_dom')
        }

    elif args.model == 'resnet18' or args.model == 'cnn':
        metrics = {
            'eval_train': StreamClsMetrics(num_classes, 'eval_train'),
            'test': StreamClsMetrics(num_classes, 'test')
        }
    else:
        raise NotImplementedError
    return metrics


def gen_clients(args, train_datasets, test_train_datasets, test_datasets, model):
    clients = [[], []]

    for train_dataset, test_train_dataset in zip(train_datasets, test_train_datasets):
        clients[0].append(Client(args, train_dataset = train_dataset, test_dataset = test_train_dataset, model = model))
    
    for test_dataset in test_datasets:
        clients[1].append(Client(args, train_dataset=None, test_dataset = test_dataset, model = model, test_client=True))

    """
    clients = [[], []]
    for i, datasets in enumerate([train_datasets, test_datasets]):
        for ds in datasets:
            clients[i].append(Client(args, ds, model, test_client=i == 1))
    """

    return clients[0], clients[1]


def gen_clients_dom_adapt(args, test_datasets, model):
    clients = []
    
    #Creates 3 test clients: idda_test(file train.txt), idda_same_dom, idda_diff_dom
    for test_dataset in test_datasets:
        clients.append(Client(args, train_dataset=None, test_dataset = test_dataset, model = model, test_client=True))
    
    #Creates the various clients, each one having a partition of the idda dataset
    #for idda_client_dataset in idda_clients_datasets:

    return clients

def get_sweep_transforms2(args, config):
    # TODO: test your data augmentation by changing the transforms here!
    if args.model == 'deeplabv3_mobilenetv2':
        rnd_transforms = []
        configColorJitter = config.get('colorJitter')
        configRndRot = config.get("rndRot")
        if configColorJitter.get("active"):
            rnd_transforms.append(sstr.ColorJitter(brightness=configColorJitter.get("brightness"),
                                                   contrast=configColorJitter.get("contrast"),
                                                   saturation=configColorJitter.get("saturation"),
                                                   hue=configColorJitter.get("hue")))
        
        if config.get("rndHzFlip"):
            rnd_transforms.append(sstr.RandomHorizontalFlip())
        if config.get("rndVertFlip"):
            rnd_transforms.append(sstr.RandomVerticalFlip())
        
        if configRndRot.get("active"):
            rnd_transforms.append(sstr.RandomRotation(configRndRot.get("maxDeg"), expand=False))

        base_train_transforms = [
            sstr.RandomResizedCrop((512, 928), scale=(0.5, 2.0)),
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        
        test_transforms = [
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        
        train_transforms = sstr.Compose(rnd_transforms + base_train_transforms)
        test_transforms = sstr.Compose(test_transforms)

    else:
        raise NotImplementedError
    
    return train_transforms, test_transforms

def yaml_to_dict(path):
    with open(path, "r") as file:
        try:
            return yaml.load(file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

def sweeping(args):
    wandb.login()

    if args.wandb ==  'hypTuning':
        with open('configs/' + args.sweep_config, 'r') as f:
                dict_sweep = yaml.safe_load(f)
    
    elif args.wandb == 'transformTuning':
        with open('configs/' + args.sweep_config, 'r') as f:
                dict_sweep = yaml.safe_load(f)


    project_name = args.wb_project_name
    if args.sweep_id == None:
        sweep_id = wandb.sweep(dict_sweep, project = project_name)
    else:
        sweep_id = args.sweep_id

    train_func = lambda: sweep_train(args=args)
    wandb.agent(sweep_id = sweep_id, function = train_func, count = 10, project = project_name)


def sweep_train(args, config = None):
    
    with wandb.init(config = config):
        config = wandb.config
        print(f'Initializing model...')
        model = model_init(args)
        model.cuda()
        print('Done.')

        if args.wandb == 'transformTuning':
            train_transforms, test_transforms = get_sweep_transforms2(args, config)
            print(train_transforms)
            print(test_transforms)
            train_datasets, test_train_datasets, test_datasets = get_datasets(args=args, train_transforms = train_transforms , test_transforms = test_transforms)
            train_clients, test_clients = gen_clients(args, train_datasets, test_train_datasets, test_datasets, model)
            metrics = set_metrics(args)
            server = Server(args, train_clients, test_clients, model, metrics)
            path = 'configs/' + args.config
            configHyp = yaml_to_dict(path)
            server.distribute_config_dict(configHyp)


        elif args.wandb == 'hypTuning':
            train_datasets, test_train_datasets, test_datasets = get_datasets(args=args)
            train_clients, test_clients = gen_clients(args, train_datasets, test_train_datasets, test_datasets, model)
            metrics = set_metrics(args)
            server = Server(args, train_clients, test_clients, model, metrics)
            server.distribute_config_dict(config)
        
        server.train()
        server.eval_train()
        server.test()


def main():
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    if args.wandb == 'hypTuning' or args.wandb == 'transformTuning':
        if args.sweep_config == None:
            sys.exit('An error occurred: you must specify a sweep_config file in the args!')

        sweeping(args)
    
    else:
        #get the configuration from command line
        with open('configs/' + args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        if args.wandb == 'singleRun':
            wandb.login()
            wandb.init(
                project = args.wb_project_name,
                config = config)
            config = wandb.config
        
        print(f'Initializing model...')
        model = model_init(args)
        model.cuda()
        print('Done.')
        print('Generate datasets...')
        train_datasets, test_train_datasets, test_datasets = get_datasets(args)
        print('Done.')

        metrics = set_metrics(args)
        
        if args.dataset == 'gta5':
            test_clients = gen_clients_dom_adapt(args, test_datasets, model)
            server = ServerGTA(args, source_dataset=train_datasets[0], test_clients=test_clients, model=model, metrics=metrics)
            server.create_opt_sch(config=config)
            server.train()
            server.test()

        else:
            train_clients, test_clients = gen_clients(args, train_datasets, test_train_datasets, test_datasets, model)
            server = Server(args, train_clients, test_clients, model, metrics)
            server.distribute_config_dict(config)
            server.train()
            server.eval_train()
            server.test()

if __name__ == '__main__':
    main()
