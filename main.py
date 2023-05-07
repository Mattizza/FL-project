
import os
import json
import torch
import random
import wandb
import numpy as np

import datasets.ss_transforms as sstr
import datasets.np_transforms as nptr

from torch import nn
from torchvision.models import resnet18

from client import Client
from server import Server
from centralized import Centralized
from datasets.idda import IDDADataset
from datasets.femnist import Femnist

from collections import defaultdict
from utils.args import get_parser
from utils.stream_metrics import StreamSegMetrics, StreamClsMetrics
from models.deeplabv3 import deeplabv3_mobilenetv2


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
                clients[i].append(Client(args, ds, model, test_client=i == 1))

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

def sweeping(args):
    wandb.login()
    #!passare a file yaml
    sweep_config = {
    
    'method': 'random',    # We specify the search strategy.
    
    'metric': {            # We set the metric and the goal (minimize/maximize).
        'name': 'loss',
        'goal': 'minimize'
        },

    # Here we need to set all the parameters we want to tune. Notice that the key
    # 'parameters' is mandatory and can't be omitted.
    'parameters': {
        
        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        # Hyperparameters that need additional hyperparameters, e.g. optimizer and scheduler.
        # In this case we need to specify both the name and the settings related to each.        
        'optimizer': {
            'parameters': {
                           'name': {'values': ['Adam', 'SGD']},
                           'settings': {'parameters': {                           
                                                       'lr'      : {'max': 0.1,   
                                                                    'min': 0.01},
                                                       'momentum': {'max': 1,
                                                                    'min': 0}
                                                       }
                                        }
                           }
                      },

        'scheduler': {
            'parameters': {
                           'name'  : {'values': ['ConstantLR', 'ExponentialLR']},
                           'settings': {'parameters':{      
                                                      'gamma' : {'max': 1,
                                                                 'min': 0.1},
                                                      'factor': {'max': 1,
                                                                 'min': 0.1}}
                                        }
                            }
                      },

        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        # Hyperparameters that are related to the whole process, e.g. #layers or dropout.
        # These can be seen as global, and don't need specific hyperparameters related to
        # them.

        # We have 8 transforms. We want to create different combinations of them. In
        # order to do this we create a dictionary of parameters and for each key (i.e. 
        # each possible transform) we choose whether to use it or not. Notice we have set
        # a probability distribution over them since we don't want to have lists full of
        # them every time.
        'transforms': {
            'parameters': {
                           'tr1': {'values': ['RndRot', np.nan],
                                   'distribution': 'categorical',
                                   'probs': [0.3, 0.7]},
                           
                           'tr2': {'values': ['RndHzFlip', np.nan],
                                   'distribution': 'categorical',
                                   'probs': [0.3, 0.7]},
                           
                           'tr3': {'values': ['RndVertFlip', np.nan],
                                   'distribution': 'categorical',
                                   'probs': [0.3, 0.7]},
                           
                           'tr4': {'values': ['RndScale', np.nan],
                                   'distribution': 'categorical',
                                   'probs': [0.3, 0.7]},
                           
                           'tr5': {'values': ['RndCrop', np.nan],
                                   'distribution': 'categorical',
                                   'probs': [0.3, 0.7]},
                           
                           'tr6': {'values': ['RndResizedCrop', np.nan],
                                   'distribution': 'categorical',
                                   'probs': [0.3, 0.7]},
                           
                           'tr7': {'values': ['ClrJitter', np.nan],
                                   'distribution': 'categorical',
                                   'probs': [0.3, 0.7]},
                           
                           'tr8': {'values': ['RndScaleRndCrop', np.nan],
                                   'distribution': 'categorical',
                                   'probs': [0.3, 0.7]},
            }
         }
        }
    }
    
    sweep_id = wandb.sweep(sweep_config, project="pytorch-augmentation1-sweeps")
    #wandb.agent(sweep_id, sweep_train(args=args, model=model), count = 2)
    train_func = lambda: sweep_train(args=args)
    wandb.agent(sweep_id, train_func, count = 1)


def sweep_train(args, config = None):
    with wandb.init(config = config):
        config = wandb.config
        print(f'Initializing model...')
        model = model_init(args)
        model.cuda()
        print('Done.')
        train_transforms, test_transforms = get_sweep_transforms(args, config)
        train_datasets, test_datasets = get_datasets(args=args, train_transforms = train_transforms , test_transforms = test_transforms)
        train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, model)
        metrics = set_metrics(args)
        server = Server(args, train_clients, test_clients, model, metrics)
        opt_params = {'optimizer': {
                            'name'    : 'Adam',
                            'settings': {'lr'   : 0.01}
                            },
              'scheduler': {
                            'name'    : 'ConstantLR',
                            'settings': {'factor': 0.33}
                            }
              }
        train_clients[0].set_opt(opt_params)
        server.train()    


def main():
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    #print(f'Initializing model...')
    #model = model_init(args)
    #model.cuda()
    #print('Done.')

    if args.do_sweep == True:
        sweeping(args)

    else:
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
        server.train()


if __name__ == '__main__':
    main()
