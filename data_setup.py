import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

def load_cifar10(batch_size):
    """
    Parmeters:
        batch_size
    Return:
        trainloader
        testloader
        classes (Tuple) : 10 classes
    """

    # Transform Data
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Load training set, download if not downloaded
    trainset = datasets.CIFAR10(root='./data/cifar10', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    # Load test set, download if not downloaded
    testset = datasets.CIFAR10(root='./data/cifar10', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    # Classes in CIFAR10 dataset 
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

if __name__ == '__main__':
    
    # Load data with batch size of 4
    trainloader, testloader, classes =  load_cifar10(4)

    # Get next batch
    image, label = next(iter(trainloader))
    
    # Plot the image 
    plt.imshow(image[0].permute(1, 2, 0))
    plt.title(classes[label[0]])
    plt.axis(False)
    plt.show()


