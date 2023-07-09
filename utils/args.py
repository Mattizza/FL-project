import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--dataset', type=str, choices=['idda', 'gta5'], required=True, help='dataset name')
    parser.add_argument('--niid', action='store_true', default=False,
                        help='Run the experiment with the non-IID partition (IID by default). Only on FEMNIST dataset.')
    parser.add_argument('--model', type=str, choices=['deeplabv3_mobilenetv2', 'resnet18', 'cnn'], default = 'deeplabv3_mobilenetv2' ,help='model name')
    parser.add_argument('--num_rounds', type=int, default = 1, help='number of rounds')
    parser.add_argument('--num_epochs', type=int, required = True, help='number of local epochs')
    parser.add_argument('--clients_per_round', type=int, default=1, help='number of clients trained per round')
    parser.add_argument('--hnm', action='store_true', default=False, help='Use hard negative mining reduction or not')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--bs', type=int, default=4, help='batch size')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--m', type=float, default=0.9, help='momentum')
    parser.add_argument('--print_train_interval', type=int, default=10, help='client print train interval')
    parser.add_argument('--print_test_interval', type=int, default=10, help='client print test interval')
    parser.add_argument('--eval_interval', type=int, default=10, help='eval interval')
    parser.add_argument('--test_interval', type=int, default=10, help='test interval')

    parser.add_argument('--framework', type=str, choices = ['centralized', 'federated'], required=True, help = 'Choose a centralized or a federated framework')
    parser.add_argument('--config', type=str, default='bestHypSameDom.yaml', help = 'name of the file containing the configuration, stored in the configs folder (include extention .yaml)')

    #wandb args
    parser.add_argument('--wandb', type = str, choices=[None, 'singleRun', 'hypTuning', 'transformTuning'], default= None, help='None: deactivate wandb, singleRun: track a single run with fixed parameters, hypTuning: do hyperparam tuning')
    parser.add_argument('--sweep_id', type = str, default= None, help='pass a sweep id to log all the runs into the same sweep')
    parser.add_argument('--wb_project_name', type = str, default = 'generalProject', help='The name of the project to store the wandb runs')
    parser.add_argument('--sweep_config', type = str, default = None, help = 'name of the sweep_config file, contained in the configs folder (include extention .yaml)' )

    #saving model
    parser.add_argument('--name_checkpoint_to_save', type = str, default = None, help='chose a name for the checkpoint you want to save, omit to avoid saving the checkpoint (include extention .pth.tar)')
    parser.add_argument('--checkpoint_to_load', type = str, default=None, help='write the name of the checkpoint to be loaded, otherwise leave default')

    return parser
