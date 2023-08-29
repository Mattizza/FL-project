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
from utils.args import get_parser
from datasets.idda import IDDADataset
from models.deeplabv3 import deeplabv3_mobilenetv2
from utils.stream_metrics import StreamSegMetrics, StreamClsMetrics
from customWAggServer import CustomWaggServer

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
        raise NotImplementedError
    raise NotImplementedError

def getTrainTransformsFromYaml(transformConfig):
    """
        transformConfig: dict containing the configuration of the transforms.
        Example: transformConfig = {'ColorJitter': {'brightness': 0.5, 'contrast': 0.2, 'saturation': 0.6, 'hue': 0.1}, 'RandomHorizontalFlip' : {'p' : 0.4}}
    """
    transfrom_functions = {
        'ColorJitter': sstr.ColorJitter,
        'RandomResizedCrop': sstr.RandomResizedCrop,
        'RandomHorizontalFlip': sstr.RandomHorizontalFlip,
        'RandomVerticalFlip': sstr.RandomVerticalFlip,
        'RandomRotation': sstr.RandomRotation
    }
    customTransformsList = []
    for k in transformConfig.keys():
        transformClass = transfrom_functions[k]
        params = transformConfig[k]
        transform = transformClass(**params)
        customTransformsList.append(transform)
    
    base_train_transforms = [
            sstr.RandomResizedCrop((512, 928), scale=(0.5, 2.0)),
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
    return sstr.Compose(customTransformsList + base_train_transforms)

def get_transforms(args):
    if args.transformConfig != None:
        with open('transformConfigs/' + args.transformConfig, 'r') as f:
            transformConfig = yaml.safe_load(f)
            train_transforms = getTrainTransformsFromYaml(transformConfig)
            test_transforms = sstr.Compose([
                                            sstr.ToTensor(),
                                            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
    
    else:
        if args.model == 'deeplabv3_mobilenetv2':
            train_transforms = sstr.Compose([
                sstr.ColorJitter(brightness=0.5,
                                contrast=0.2,
                                saturation=0.6,
                                hue=0.1),
                sstr.RandomResizedCrop((512, 928), scale=(0.5, 2.0)),
                sstr.ToTensor(),
                sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            test_transforms = sstr.Compose([
                sstr.ToTensor(),
                sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        #elif args.model == 'cnn' or args.model == 'resnet18':
        #    train_transforms = nptr.Compose([
        #        nptr.ToTensor(),
        #        nptr.Normalize((0.5,), (0.5,)),
        #    ])
        #    test_transforms = nptr.Compose([
        #        nptr.ToTensor(),
        #        nptr.Normalize((0.5,), (0.5,)),
        #    ])
        else:
            raise NotImplementedError

    #print("Train transforms:", train_transforms)
    #print("\nTest transfroms:", test_transforms)
    return train_transforms, test_transforms


def get_datasets(args, train_transforms = None, test_transforms = None):

    train_datasets = []
    test_train_datasets = []
    for_style_extraction_dataset = []

    if train_transforms == None or test_transforms == None:
        train_transforms, test_transforms = get_transforms(args)
    
    print("Train transforms:", train_transforms)
    print("\nTest transfroms:", test_transforms)

    if args.dataset == 'idda':
        root = 'data/idda'

        with open(os.path.join(root, 'train.json'), 'r') as f:
            all_data = json.load(f)
        for client_id in all_data.keys():
            train_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], transform=train_transforms,
                                            client_name=client_id))
            
            test_train_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], transform = test_transforms,
                                            client_name=client_id))
            
            for_style_extraction_dataset.append(IDDADataset(root=root, list_samples=all_data[client_id], transform=None,
                                        client_name=client_id))
        
        with open(os.path.join(root, 'test_same_dom.txt'), 'r') as f:
            test_same_dom_data = f.read().splitlines()
            test_same_dom_dataset = IDDADataset(root=root, list_samples=test_same_dom_data, transform=test_transforms,
                                                client_name='test_same_dom')
            
        with open(os.path.join(root, 'test_diff_dom.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()
            test_diff_dom_dataset = IDDADataset(root=root, list_samples=test_diff_dom_data, transform=test_transforms,
                                                client_name='test_diff_dom')
        test_datasets = [test_same_dom_dataset, test_diff_dom_dataset]

    else:
        raise NotImplementedError

    return train_datasets, test_train_datasets, for_style_extraction_dataset, test_datasets

    #[train_dataset], [test_train_dataset], [for_style_extraction_dataset],[test_same_dom_data, test_diff__dom]


def get_datasets_DA(args, train_transforms = None, test_transforms = None):
    """
    Function to create all the needed dataset when using gta5 in a DA framework
    """

    #Note: when in gta/DA framework we should apply the train_transforms to source_dataset (gta5), and test_transforms to target_datasets (idda)
    if train_transforms == None or test_transforms == None:
        train_transforms, test_transforms = get_transforms(args)
    
    print("Train transforms:", train_transforms)
    print("\nTest transfroms:", test_transforms)

    
    idda_root = 'data/idda'

    #Create the idda_clients_datasets, these dataset are used just to extract the style of the client
    idda_clients_datasets = []
    with open(os.path.join(idda_root, 'train.json'), 'r') as f:
            all_data = json.load(f)
    for client_id in all_data.keys():
        idda_clients_datasets.append(IDDADataset(root=idda_root, list_samples=all_data[client_id], transform=None,
                                        client_name=client_id))

    #Create idda_eval dataset
    with open(os.path.join(idda_root, 'train.txt'), 'r') as f:
        idda_train_data = f.read().splitlines()
        idda_train = IDDADataset(root=idda_root, list_samples=idda_train_data, transform=test_transforms,
                                            client_name="idda_test")
    
    #Create idda_same_dom dataset
    with open(os.path.join(idda_root, 'test_same_dom.txt'), 'r') as f:
        test_same_dom_data = f.read().splitlines()
        test_same_dom_dataset = IDDADataset(root=idda_root, list_samples=test_same_dom_data, transform=test_transforms,
                                            client_name='test_same_dom')
    
    #Create idda_diff_dom dataset
    with open(os.path.join(idda_root, 'test_diff_dom.txt'), 'r') as f:
        test_diff_dom_data = f.read().splitlines()
        test_diff_dom_dataset = IDDADataset(root=idda_root, list_samples=test_diff_dom_data, transform=test_transforms,
                                            client_name='test_diff_dom')
        
    eval_and_test_datasets = [idda_train, test_same_dom_dataset, test_diff_dom_dataset]

    test_dataset_gta = GTA5(test_transform = test_transforms, client_name = 'test_gta5', target_dataset='idda', test=True)

    return train_dataset, test_dataset_gta ,idda_clients_datasets, eval_and_test_datasets


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
            'test_diff_dom': StreamSegMetrics(num_classes, 'test_diff_dom'),
            'test_gta5': StreamSegMetrics(num_classes, 'test_gta5')
        }

    elif args.model == 'resnet18' or args.model == 'cnn':
        metrics = {
            'eval_train': StreamClsMetrics(num_classes, 'eval_train'),
            'test': StreamClsMetrics(num_classes, 'test')
        }
    else:
        raise NotImplementedError
    return metrics


def gen_clients(args, train_datasets, test_train_datasets, test_datasets, for_style_extraction_dataset, model):
    clients = [[], [], []]

    for train_dataset, test_train_dataset in zip(train_datasets, test_train_datasets):
        clients[0].append(Client(args, train_dataset = train_dataset, test_dataset = test_train_dataset, model = model))
    
    for test_dataset in test_datasets:
        clients[1].append(Client(args, train_dataset=None, test_dataset = test_dataset, model = model, test_client=True))

    for se_dataset in for_style_extraction_dataset:
        clients[2].append(Client(args, train_dataset=None, test_dataset = se_dataset, model = model, test_client=True, isTarget=True))

    return clients[0], clients[1], clients[2]


def gen_clients_dom_adapt(args, idda_clients_datasets, test_datasets, model):
    clients = [[],[]] # ix=0 clients with idda partition, ix=1 clients with eval and tests partition (idda)
    
    #Creates the various clients, each one having a partition of the idda dataset
    for idda_client_dataset in idda_clients_datasets:
        clients[0].append(Client(args, train_dataset=None, test_dataset = idda_client_dataset, model = model, test_client=True, isTarget=True))

    #Creates 3 test clients: idda_test(file train.txt), idda_same_dom, idda_diff_dom
    for test_dataset in test_datasets:
        clients[1].append(Client(args, train_dataset=None, test_dataset = test_dataset, model = model, test_client=True))

    return clients



def main():
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    #get the configuration from a file given in the command line
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

    metrics = set_metrics(args)
    
    train_datasets, test_train_datasets, for_style_extraction_dataset, test_datasets = get_datasets(args)
    print('Done.')
    train_clients, test_clients, clients_for_style_extraction = gen_clients(args, train_datasets, test_train_datasets, test_datasets, for_style_extraction_dataset, model)
    
    server = CustomWaggServer(args, train_clients, test_clients, clients_for_style_extraction, model, metrics, config)
    
    #debugging
    #print('num_immagini',  train_clients[0].num_train_samples)
    ###
    
    #compute syles
    print('Extracting stlyes from clients...')
    server.load_styles() #extract_styles from clients
    print('Done.')
    
    #compute clusters
    print('\nComputing clusters...')
    server.compute_clusters() #compute clusters and return the tot num of clusters
    print('Done.')
    
    server.train()
    #server.eval_train()
    #server.test()
    if args.name_checkpoint_to_save != None:
        server.checkpoint_recap()
    if args.framework == 'federated' and args.r != None:
        server.download_mious_as_csv()


if __name__ == '__main__':
    main()
