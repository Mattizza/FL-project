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
from models.deeplabv3 import deeplabv3_mobilenetv2
from utils.stream_metrics import StreamSegMetrics, StreamClsMetrics
from centralized import Centralized
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
    if dataset == 'idda' or dataset == 'iddaCB':
        return 16
    if dataset == 'femnist':
        return 62
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

    if train_transforms == None or test_transforms == None:
        train_transforms, test_transforms = get_transforms(args)

    if args.dataset == 'idda' or args.dataset == 'iddaCB':
        root = 'data/idda'
        if args.dataset == 'idda':
            with open(os.path.join(root, 'train.json'), 'r') as f:
                all_data = json.load(f)
            for client_id in all_data.keys():
                train_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], transform=train_transforms,
                                                client_name=client_id))
        
        elif args.dataset == 'iddaCB':
            with open(os.path.join(root, 'train.txt'), 'r') as f:
                train_data = f.read().splitlines()
                train_datasets.append(IDDADataset(root=root, list_samples=train_data, transform=train_transforms,
                                                    client_name="Unique"))
            
            
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
        train_datasets.append(GTA5(transform=train_transforms, test_transform=test_transforms, target_dataset='idda'))

    else:
        raise NotImplementedError

    return train_datasets, test_datasets


def set_metrics(args):
    num_classes = get_dataset_num_classes(args.dataset)
    if args.model == 'deeplabv3_mobilenetv2':
        metrics = {
            'eval_train': StreamSegMetrics(num_classes, 'eval_train'),
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


def gen_clients(args, train_datasets, test_datasets, model):
    if (args.dataset == 'idda') :
        clients = [[], []]
        for i, datasets in enumerate([train_datasets, test_datasets]):
            for ds in datasets:
                clients[i].append(Centralized(args, ds, model, test_client=i == 1)) #Chiamare centralized Client

    elif args.dataset == 'iddaCB':
        clients = [[], []]
        for i, datasets in enumerate([train_datasets, test_datasets]):
            for ds in datasets:
                clients[i].append(Centralized(args, ds, model, test_client=i == 1))

    return clients[0], clients[1]

def get_sweep_transforms(args, config):
    # TODO: test your data augmentation by changing the transforms here!
    if args.model == 'deeplabv3_mobilenetv2':
        rnd_transforms = []

        # Select only the transforms of interest. We take the string and we build the method.
        # WARNING: omitted '(10)' as argument like in the previous version.
        # WARNING: not tested due to problems with WandB API.
        keep = [value for key, value in config.transforms.items() if value is not np.nan] 
        
        for i in range(len(keep)):
            rnd_transforms.append(getattr(sstr, keep[i])) 

        base_transforms = [
            sstr.RandomResizedCrop((512, 928), scale=(0.5, 2.0)),
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        
        train_transforms = sstr.Compose(rnd_transforms + base_transforms)
        test_transforms = sstr.Compose(base_transforms)

    else:
        raise NotImplementedError
    
    return train_transforms, test_transforms

def get_sweep_transforms2(args, config):
    # TODO: test your data augmentation by changing the transforms here!
    if args.model == 'deeplabv3_mobilenetv2':
        rnd_transforms = []

        if config.rndRot:
            rnd_transforms.append(sstr.RandomRotation(10))
        if config.rndHzFlip:
            rnd_transforms.append(sstr.RandomHorizontalFlip())
        if config.rndVertFlip:
            rnd_transforms.append(sstr.RandomVerticalFlip())
        if config.colorJitter:
            rnd_transforms.append(sstr.ColorJitter())


        base_transforms = [
            sstr.RandomResizedCrop((512, 928), scale=(0.5, 2.0)),
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        
        train_transforms = sstr.Compose(rnd_transforms + base_transforms)
        test_transforms = sstr.Compose(base_transforms)

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

    dict_sweep = {'method' : 'grid'}
    metric = {
        'name' : 'loss',
        'goal' : 'minimize'
    }
    dict_sweep['metric'] = metric
    
    if args.wandb ==  'hypTuning':
        #!togliere commenti per usare yaml
        #with open('configs/sweep_config.yaml', 'r') as f:
        #        sweep_config = yaml.safe_load(f)

        parameters = { 
            'optimizer' : { 'values' : ['Adam', 'SGD', 'Adagrad'],
                           'distribution' : 'categorical'},
            'learning_rate' : {'values': [0.01, 0.001, 0.0001, 0.0005, 0.00001]},
            
            'weight_decay': {'distribution': 'uniform',
                                'min': 0.0,
                                'max': 1.0},
            'momentum' : {'distribution': 'uniform',
                                'min': 0.0,
                                'max': 1.0},
            'scheduler' : {'values' : ['ExponentialLR', 'StepLR', None],
                           'distribution' : 'categorical'
                           },

            'gamma' : {'values':[0.01, 0.1, 0.33, 0.5, 0.7, 1.0]}
            
        }

        #parameters = { 
        #    'optimizer' : { 'value' : 'Adam'},
        #
        #    'learning_rate' : {'values': [0.0005, 0.00045],
        #                       'distribution': 'categorical'}
        #}

        # First we define our configuration
        parameters = {        
                'optimizer': {
                    'parameters': {
                                'name': {'values': ['Adam', 'SGD']},
                                'lr'      : {'value': 0.0005}#,
                                #'momentum': {'max': 1, 'min': 0}
                                    }
                            },

                'scheduler': {
                    'parameters':{
                                'name'  : {'value': 'ReduceLROnPlateau'},
                                'patience': {'value': 2}
                                #'gamma' : {'max': 1, 'min': 0.1},
                                #'factor': {'max': 1, 'min': 0.1}
                                }
                            }
                        }
        
        dict_sweep['parameters'] = parameters

        dict_sweep['early_terminate'] = {'type' : 'hyperband', 'min_iter' : 3, 'eta': 2}
    
    elif args.wandb == 'transformTuning':
        parameters = {  'rndRot' : {'values':[True, False]},
                        'rndHzFlip' : {'values':[True, False]},
                        'rndVertFlip' : {'values':[True, False]},
                        'colorJitter' : {'values':[True, False]},
                                   }
        dict_sweep['parameters'] = parameters
        

    project_name = "test_hyp_sweeps_AdamvsSGD20e"
    if args.sweep_id == None:
        sweep_id = wandb.sweep(dict_sweep, project = project_name)

    else:
        sweep_id = args.sweep_id

    train_func = lambda: sweep_train(args=args)
    wandb.agent(sweep_id = sweep_id, function = train_func, count = 2 ,project = project_name)


def sweep_train(args, config = None):
    
    with wandb.init(config = config):
        config = wandb.config
        print(f'Initializing model...')
        model = model_init(args)
        model.cuda()
        print('Done.')

        if args.wandb == 'transformTuning':
            train_transforms, test_transforms = get_sweep_transforms2(args, config)
            train_datasets, test_datasets = get_datasets(args=args, train_transforms = train_transforms , test_transforms = test_transforms)
            train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, model)
            metrics = set_metrics(args)
            server = Server(args, train_clients, test_clients, model, metrics)
            path = 'configs/runSingola.yaml'
            configHyp = yaml_to_dict(path)
            server.distribute_config_dict(configHyp)


        elif args.wandb == 'hypTuning':
            train_datasets, test_datasets = get_datasets(args=args)
            train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, model)
            metrics = set_metrics(args)
            server = Server(args, train_clients, test_clients, model, metrics)
            server.distribute_config_dict(config)
        
        server.train() 


def main():
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    #print(f'Initializing model...')
    #model = model_init(args)
    #model.cuda()
    #print('Done.')

    if args.wandb == 'hypTuning' or args.wandb == 'transformTuning':
        sweeping(args)
    
    elif args.wandb == 'singleRun':
        wandb.login()

        with open('configs/runSingola.yaml', 'r') as f:
            yaml_config = yaml.safe_load(f)

        wandb.init(
            project = 'singleRuns',
            config = yaml_config)
        
        config = wandb.config

        print(f'Initializing model...')
        model = model_init(args)
        model.cuda()
        print('Done.')
        print('Generate datasets...')
        train_datasets, test_datasets = get_datasets(args)
        print('Done.')

        metrics = set_metrics(args)
        train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, model)
        server = Server(args, train_clients, test_clients, model, metrics)
        
        server.distribute_config_dict(config)
        server.train()
        

    elif args.wandb == None:
        print(f'Initializing model...')
        model = model_init(args)
        model.cuda()
        print('Done.')
        print('Generate datasets...')
        train_datasets, test_datasets = get_datasets(args)
        print('Done.')

        metrics = set_metrics(args)
        train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, model)
        server = Server(args, train_clients, test_clients, model, metrics)
        path = 'configs/runSingola.yaml'
        config = yaml_to_dict(path)
        config =  {
    'optimizer':{'name': 'Adam',
                'lr':0.005,
                'weight_decay': 0.5},
    'scheduler':{'name': 'ExponentialLR',
                 'gamma':0.005}
                 }
        server.distribute_config_dict(config)
        
        server.train()


if __name__ == '__main__':
    main()
