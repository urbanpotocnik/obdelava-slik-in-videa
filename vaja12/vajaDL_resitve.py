import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

# vaje je povzeta po: https://nextjournal.com/gkoehler/pytorch-mnist

## definicija parametrov
if __name__ == "__main__":
    train_ratio = 0.85
    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    val_log_epoch_interval = 1

    batch_size_val = batch_size_train
    train_log_step_interval = 100
    val_ratio = 1 - train_ratio

    random_seed = 42
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)


def count_samples_in_class(dataset, classes):
    N = len(dataset)
    targets = [case[1] for case in dataset]
    for cl in classes:
        s = sum([1 for t in targets if t == cl])
        print(f"Number of cases in class {cl}: {s}, {s/N*100:.2f} %")


if __name__ == "__main__":
    train_and_val_dataset = torchvision.datasets.MNIST(
        "/files/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                # matrike zapisemo v obliko torch.Tensor (na nek nacin podobno kot numpy array)
                torchvision.transforms.ToTensor(),
                # vsako sliko normliziramo https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Normalize
                # 0.1307 je povprecna vrednost in 0.3081 standardna deviacije MNIST ucne mnozice
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
    test_dataset = torchvision.datasets.MNIST(
        "/files/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    N_train_and_val = len(train_and_val_dataset)

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_and_val_dataset,
        [int(N_train_and_val * train_ratio), int(N_train_and_val * val_ratio)],
        generator=torch.Generator().manual_seed(random_seed),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size_train,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
    )
    
    count_samples_in_class(train_dataset, list(range(10)))
    count_samples_in_class(val_dataset, list(range(10)))
    count_samples_in_class(test_dataset, list(range(10)))


def showMultipleImages(nRows, nCols, iImages, iLabels, iPrediction=None):
    """
    Funkcija za prikaz slicic
    """

    fig, axs = plt.subplots(nrows=nRows, ncols=nCols, squeeze=False, facecolor="white")
    count = 0
    for row in range(nRows):
        for col in range(nCols):
            title_color = "k"
            ax = axs[row, col]
            ax.imshow(iImages[count], cmap="gray")
            title = f"GT: {iLabels[count]}"
            if iPrediction is not None:
                title += f" , P: {iPrediction[count]}"
                if iLabels[count] == iPrediction[count]:
                    title_color = "g"
                else:
                    title_color = "r"
            ax.set_title(title, color=title_color)
            ax.set_xticks([])
            ax.set_yticks([])
            count += 1
    plt.tight_layout()
    return fig


## 2. NALOGA
if __name__ == "__main__":
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(f"Shape {example_data.shape}")
    example_data = np.squeeze(example_data.numpy())

    showMultipleImages(5, 6, example_data, iLabels=example_targets.numpy())


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.soft_max = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return self.soft_max(x)


if __name__ == "__main__":
    # inicializacija nevronske mreze
    network = Net()
    # optimizacijska metoda
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    # kriterijska funkcija
    loss_fcn = nn.NLLLoss(reduction="mean")

    # pripravimo sezname, kamor bomo shranjevali epoche in vrednost kriterijske funkcije pri trenutni epochi
    train_counter = []
    train_losses = []
    val_counter = []
    val_losses = []


def calculate_accuracy(gt, predicted):
    """
    Funkcija za izracun natancnosti razvrscanja
    """
    N_correct = (np.array(gt) == np.array(predicted)).sum()
    acc = N_correct / len(gt)
    return N_correct, acc


def train(loader, epoch, counter_list, losses_list, log_step=100):
    """
    Funkcija za ucenje modela

    Args:
        loader (torch.utils.data.dataloader.DataLoader): training set loader
        counter_list (list): list of epochs when loss value is logged
        losses_list (list): list of logged loss value
        epoch (int): current epoch
        log_step (int): number of steps after logging is done
    """
    for batch_idx, (data, target) in enumerate(loader):
        # postavimo gradiente na nic
        optimizer.zero_grad()
        # spustimo slike skozi nevronsko mrezo in izracunamo izhodni vektor
        output = network(data)
        # izracunamo vrednost kriterijske funkcije
        loss = loss_fcn(output, target)
        # izracunaj gradiente za vse parametre v mrezi
        loss.backward()
        # popravi vrednosti parametrov
        optimizer.step()

        # ko skozi mrezo spustimo veckratnik 'train_log_step_interval' paketov podatkov, potem izpisemo na zaslon trenutno vrednost kriterijske funkcije
        if batch_idx % log_step == 0:
            print(
                f"TRAINING:\tEpoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}\t({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
            losses_list.append(loss.item())
            counter_list.append(epoch + batch_idx / len(train_loader))

            # ce zelimo shraniti model
            # torch.save(network.state_dict(), "/results/model.pth")
            # torch.save(optimizer.state_dict(), "/results/optimizer.pth")


def inference(loader, batch_size, phase="", epoch=None):
    network.eval()
    loss_value = 0
    gt = []
    predicted = []
    images = []
    with torch.no_grad():
        for data, target in loader:
            output = network(data)
            loss_value += loss_fcn(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            predicted += pred.flatten().tolist()
            gt += target.flatten().tolist()
            if phase.lower() == "test":
                images += list(np.squeeze(data.numpy()))
    loss_value /= len(loader.dataset) / batch_size

    N_correct, acc = calculate_accuracy(gt, predicted)

    if phase.lower() == "validation":
        print(
            f"\n{phase}:\tEpoch: {epoch}: Avg. loss: {loss_value:.4f}, Accuracy: {N_correct}/{len(loader.dataset)}\t({acc*100:.4f}%)\n"
        )

    return loss_value, gt, predicted, N_correct, acc, images


if __name__ == "__main__":
    for epoch in range(n_epochs):
        # vsakih 'val_log_interval' izracunaj vrednost kriterijske funkcije na validacijskih podatkih
        if epoch % val_log_epoch_interval == 0:
            val_loss, gt, predicted, _, _, _ = inference(
                val_loader, batch_size_val, phase="VALIDATION", epoch=epoch
            )
            val_counter.append(epoch)
            val_losses.append(val_loss)

        # ucenje modela
        network.train()
        train(
            train_loader,
            epoch=epoch,
            counter_list=train_counter,
            losses_list=train_losses,
            log_step=train_log_step_interval,
        )

    # prikazi potek kriterijske funkcije
    fig = plt.figure()
    # prikazi potek kriterijske funkcije za učno množico
    plt.plot(train_counter, train_losses, color="blue")
    # prikazi potek kriterijske funkcije za validacijsko množico
    plt.scatter(val_counter, val_losses, color="red")
    plt.legend(["Train Loss", "Val Loss"], loc="upper right")
    plt.xlabel("Epochs")
    plt.ylabel("Loss function value")


if __name__ == "__main__":
    test_loss, gt_test, predicted_test, N_correct_test, acc_test, images = inference(
        test_loader, batch_size_test, phase="test"
    )
    showMultipleImages(5, 6, images, gt_test, predicted_test)
