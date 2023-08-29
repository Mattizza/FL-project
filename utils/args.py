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
    parser.add_argument('--transformConfig', type=str, default=None, help = 'name of the file containing the transforms instructions, stored in the transformsConfigs folder (include extention .yaml)')
    parser.add_argument('--mode', type=str, default='train', help = 'Mode can be train or test')
    
    #Task 2
    parser.add_argument('--r', type=int, default=None, help='num. of round beween each evaluation in federated framework.')
    
    #Task 3
    parser.add_argument('--t', type=int, default=1, help='num. of epoch between each evaluation on target dataset.')
    parser.add_argument('--fda', type=str, default='False', help = 'set True if you want to apply FDA.')
    parser.add_argument('--b', type=int, default=None, help='size of the window to apply FDA.')

    #Task 4
    parser.add_argument('--self_train', type=str, default='false', help = 'set true if you want to use self train loss.')
    parser.add_argument('--T', type=int, default=None, help='num. of round between each teacher model update.')
    parser.add_argument('--source_trained_ckpt', type=str, default=None, help='the name of the checkpoint of the model pretrained on gta. Include .pth.tar' )

    #Task 5
    parser.add_argument('--our_self_train', type=str, default='false', help = 'Use our method for getting pseudolabels.')
    parser.add_argument('--use_entropy', type=str, default='false', help = 'set true if you need to use entropy.')
    parser.add_argument('--custom_client_selection', type=str, default='false', help = 'set true to use custom client selection.')
    
    #Task 5 Client Selection
    parser.add_argument('--beta', type=float, default=0.5, help = 'Weight related to the entropy of the clients. It must be comprised in [0, 1]')
    parser.add_argument('--llambda', type=float, default=1.0, help = 'Lambda parameter in the sigmoid function.')

    #Task 5 Weight Aggregation
    parser.add_argument('--custom_weight_agg', type=str, default=None, help = 'Set true to use custom weight aggregation.')
    parser.add_argument('--llambda_weight_agg', type=float, default=1.0, help = 'Lambda parameter in the sigmoid function for the weight aggregation.')
    parser.add_argument('--beta_weight_agg', type=float, default=0.5, help = 'Weight related to the entropy of the clusters. It must be comprised in [0, 1]')
    #parser.add_argument('--use_clustered_weight_agg', type=str, default=None, help = 'set true if you want to use clustering in the weight aggregation.')
    parser.add_argument('--alpha_weight_agg', type=float, default=0.0, help = 'Total number of clusters.')



    #wandb args
    parser.add_argument('--wandb', type = str, choices=[None, 'singleRun', 'hypTuning', 'transformTuning'], default= None, help='None: deactivate wandb, singleRun: track a single run with fixed parameters, hypTuning: do hyperparam tuning')
    parser.add_argument('--sweep_id', type = str, default= None, help='pass a sweep id to log all the runs into the same sweep')
    parser.add_argument('--wb_project_name', type = str, default = 'generalProject', help='The name of the project to store the wandb runs')
    parser.add_argument('--sweep_config', type = str, default = None, help = 'name of the sweep_config file, contained in the configs folder (include extention .yaml)' )

    #saving checkpoints
    parser.add_argument('--name_checkpoint_to_save', type = str, default = None, help='chose a name for the checkpoint you want to save, omit to avoid saving the checkpoint (include extention .pth.tar)')
    parser.add_argument('--checkpoint_to_load', type = str, default=None, help='write the name of the checkpoint to be loaded, otherwise leave default')

    return parser
