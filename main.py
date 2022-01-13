import argparse
from models import load_models, load_model_states
from dataset import load_dataset
from train import training_loop
from test import test_loop

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', help='Number of Epochs.', type=int, default=150)
    parser.add_argument('--train', help='Pass this to Train.', action='store_true')
    parser.add_argument('--num_workers',
                        help='Pass the number of workers you want for the dataloader.',
                        type = int, default = 2)
    parser.add_argument('--cpu',
                        help='Pass this to train in the CPU. Can not tell you how much I do not recommend this for training, but testing with pre-trained models is fine.',
                        action='store_true')
    args = parser.parse_args()

    if args.cpu and args.train:
        print('WARNING: Training in the CPU.')
        device = 'cpu'
    elif args.cpu and not args.train:
        print('Testing in the CPU.')
        device = 'cpu'
    elif args.cpu == False and args.train:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Training in {device}')
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Testing in {device}.')

    print('Loading the Dataset.')
    train_dataloader, test_dataloader = load_dataset(args.num_workers)

    print('Loading Models')
    cnn_3, cnn_5, cnn_7 = load_models(device)

    if args.train:
        print('Training CNN_3.')
        cnn3 = training_loop(cnn_3, args.n_epochs, 'cnn_3', train_dataloader)
        print('Training CNN_5.')
        cnn5 = training_loop(cnn_5, args.n_epochs, 'cnn_5', train_dataloader)
        print('Training CNN_7.')
        cnn7 = training_loop(cnn_7, args.n_epochs, 'cnn_7', train_dataloader)

    else:
        print('Loading Pre-Trained Model States.')
        cnn3, cnn5, cnn7 = load_model_states(cnn_3, cnn_5, cnn_7)

    print('Testing the Models.')
    test_loop(cnn3, cnn5, cnn7, test_dataloader, device)
