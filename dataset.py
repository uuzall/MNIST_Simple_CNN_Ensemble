from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def load_dataset(num_workers):
    trans = transforms.Compose([transforms.RandomRotation(20),
                                            transforms.RandomAffine(0, translate=(0.2, 0.2)),
                                           transforms.ToTensor(), ])
    train = datasets.MNIST('./', train=True, download=True, transform = trans)
    test = datasets.MNIST('./', train=False, download=False, transform =
                                     transforms.Compose([transforms.ToTensor()]))

    train_dataloader = DataLoader(train, batch_size=120, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test, batch_size=100, shuffle=False, num_workers=num_workers)
    return train_dataloader, test_dataloader

if __name__ == '__main__':
    train, test = load_dataset(4)

    print('test successful')
