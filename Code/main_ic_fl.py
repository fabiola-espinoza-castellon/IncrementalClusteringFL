import datetime
import time
import argparse

# local imports
from DataProcessing.labelSwapData import *
from DataProcessing.imageRotData import *
from FedAVG.federated_training import *
from DataProcessing.server import *

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('case', type=str, help="Non-IID case. Can be 'emnist', 'labelSwap' or 'imageRot'.")
parser.add_argument('nb_users', type=int, help='Number of users to be created.')
parser.add_argument('E', type=int, help='Number of local epochs.')
parser.add_argument('B', type=int, help='Local batch size.')
parser.add_argument('lr', type=float, help='Local learning rate.')
parser.add_argument('C', type=float, help='Proportion of clients sampled at each round.')
parser.add_argument('T', type=int, help='Federated rounds before clustering.')
parser.add_argument('Tf', type=int, help='Federated rounds after clustering.')
parser.add_argument('--cuda', action='store_true', help='Use option to computed in GPU. Otherwise do not mention.')

args = parser.parse_args()

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

cuda = args.cuda

def main():
    if cuda and torch.cuda.is_available():
        print('Training will be done on GPU in device {} of id {}'.format(
            torch.cuda.get_device_name(torch.cuda.current_device()), torch.cuda.current_device()))

    results = {}
    case = args.case  # emnist, labelSwap or imageRot, labelSwap_iid
    nb_users = args.nb_users

    globalPath = os.path.dirname(os.getcwd())
    dataPath = os.path.join(os.path.dirname(globalPath), "Data", "emnist-mnist.mat")

    # set parameters and hyperparameters
    global_hyperparams = {'rounds': args.T, 'C': args.C, 'criterion': nn.CrossEntropyLoss()}
    local_hyperparams = {'B': args.B, 'E': args.E, 'lr': args.lr, 'criterion': nn.CrossEntropyLoss(),
                         'optimizer': torch.optim.SGD, 'inputChannel': 1, 'outputSize': 10}
    results['cuda'] = cuda
    results['seed'] = seed
    results['local_lr'] = local_hyperparams['lr']  # We used 0.01
    results['C'] = global_hyperparams['C']
    results['B'] = local_hyperparams['B']
    results['E'] = local_hyperparams['E']
    results['rounds'] = global_hyperparams['rounds']
    results['case'] = case
    results['nb_users'] = nb_users

    if case == 'emnist':
        dataPath = os.path.join(os.path.dirname(globalPath), "Data", "emnist-byclass.mat")
        local_hyperparams['outputSize'] = 62
        data = EntireDataset(dataPath, cuda=cuda)
        data.preprocess_data(K=nb_users)
    elif case == 'labelSwap':
        data = LabelSwapData(dataPath, {0: {0: 1, 1: 0}, 1: {2: 3, 3: 2}, 2: {4: 5, 5: 4}, 3: {6: 7, 7: 6},
                                        4: {8: 9, 9: 8}}, nb_users, cuda=cuda)
        data.preprocess_data()
    elif case == 'imageRot':
        data = ImageRotData(dataPath, {0: 0, 1: 90, 2: 180, 3: 270}, nb_users, cuda=cuda)
        data.preprocess_data()

    # nets
    user_net = CNNetwork
    server_net = CNNetwork

    resultsPath = os.path.join(os.getcwd(), "Results", case, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.mkdir(resultsPath)
    os.mkdir(os.path.join(resultsPath, 'cache'))

    train = FedAVG(server_net, user_net, data, global_hyperparams, local_hyperparams, preload=False, cuda=cuda,
                   resultsPath=resultsPath)

    b = time.time()
    server_acc = train.start_fedavg_training(online_cluster=True, save_accuracies=False)
    clustering_ours = train.clustering_training(rounds=args.Tf)
    train.choose_unclustered()

    results['total_time'] = str(datetime.timedelta(seconds=time.time() - b))

    file = open(os.path.join(resultsPath, 'params.txt'), 'w')
    for key in results:
        file.write(key + ', ' + str(results[key]) + '\n')
    file.close()


if __name__ == '__main__':
    main()
