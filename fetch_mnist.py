import pickle
from torchvision import datasets, transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
dataset2 = datasets.MNIST("../data", train=False, transform=transform)

X_train = dataset1.data.numpy()
y_train = dataset1.targets.numpy()

X_test = dataset2.data.numpy()
y_test = dataset2.targets.numpy()

data = X_train, X_test, y_train, y_test

with open("./data/mnist-data.pickle", "wb") as file:
    pickle.dump(data, file)
