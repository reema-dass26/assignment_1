import torch
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# for wandb users:
# from dlvc.wandb_logger import WandBLogger


class BaseTrainer(metaclass=ABCMeta):
    """
    Base class of all Trainers.
    """

    @abstractmethod
    def train(self) -> None:
        """
        Holds training logic.
        """

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float, float]:
        """
        Holds validation logic for one epoch.
        """

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float, float]:
        """
        Holds training logic for one epoch.
        """

        pass


class ImgClassificationTrainer(BaseTrainer):
    """
    Class that stores the logic reversefor training a model for image classification.
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        lr_scheduler,
        train_metric,
        val_metric,
        train_data,
        val_data,
        device,
        num_epochs: int,
        training_save_dir: Path,
        batch_size: int = 4,
        val_frequency: int = 5,
    ) -> None:
        """
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of training set
            val_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of validation set
            train_data (dlvc.datasets.cifar10.CIFAR10Dataset): Train dataset
            val_data (dlvc.datasets.cifar10.CIFAR10Dataset): Validation dataset
            device (torch.device): cuda or cpu - device used to train the network
            num_epochs (int): number of epochs to train the network
            training_save_dir (Path): the path to the folder where the best model is stored
            batch_size (int): number of samples in one batch
            val_frequency (int): how often validation is conducted during training (if it is 5 then every 5th
                                epoch we evaluate model on validation set)

        What does it do:
            - Stores given variables as instance variables for use in other class methods e.g. self.model = model.
            - Creates data loaders for the train and validation datasets
            - Optionally use weights & biases for tracking metrics and loss: initializer W&B logger

        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = training_save_dir
        self.batch_size = batch_size
        self.val_frequency = val_frequency

        # Create data loaders for training and validation datasets
        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

        self.train_loss_history = []
        self.train_accuracy_history = []
        self.train_pcacc_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []
        self.val_pcacc_history = []

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.train()  # Set model to train mode
        train_loss = 0.0
        self.train_metric.reset()

        for batch_idx, (data, target) in tqdm(enumerate(self.train_loader)):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.train_metric.update(output, target)

        mean_train_loss = train_loss / len(self.train_loader)
        mean_train_accuracy, mean_train_pcacc = self.train_metric.compute()

        self.train_loss_history.append(mean_train_loss)
        self.train_accuracy_history.append(mean_train_accuracy)
        self.train_pcacc_history.append(mean_train_pcacc)

        print(
            f"Epoch {epoch_idx + 1}/{self.num_epochs}, Train Loss: {mean_train_loss:.4f}, "
            f"Train mAcc: {mean_train_accuracy:.4f}, Train mPCAcc: {mean_train_pcacc:.4f}"
        )

        return mean_train_loss, mean_train_accuracy, mean_train_pcacc

    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        self.val_metric.reset()

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss_fn(output, target)
                val_loss += loss.item()
                self.val_metric.update(output, target)

        mean_val_loss = val_loss / len(self.val_loader)
        mean_val_accuracy, mean_val_pcacc = self.val_metric.compute()

        self.val_loss_history.append(mean_val_loss)
        self.val_accuracy_history.append(mean_val_accuracy)
        self.val_pcacc_history.append(mean_val_pcacc)

        print(
            f"Epoch {epoch_idx + 1}/{self.num_epochs}, Validation Loss: {mean_val_loss:.4f}, "
            f"Validation mAcc: {mean_val_accuracy:.4f}, Validation mPCAcc: {mean_val_pcacc:.4f}"
        )

        return mean_val_loss, mean_val_accuracy, mean_val_pcacc

    def train(self) -> None:
        print("Starting")
        for epoch in tqdm(range(self.num_epochs)):
            # TODO Need CV?
            train_loss, train_accuracy, train_pcacc = self._train_epoch(epoch)
            val_loss, val_accuracy, val_pcacc = self._val_epoch(epoch)

            if val_pcacc > self.best_val_pcacc:
                self.best_val_pcacc = val_pcacc
                torch.save(self.model.state_dict(), self.training_save_dir)
                print("Best model saved.")

            if (epoch + 1) % self.val_frequency == 0:
                self.lr_scheduler.step(
                    val_loss
                )  # Adjust learning rate if using scheduler

        print("Training finished")

    def _plot(self):
        epochs = range(1, self.num_epochs + 1)

        # Plot training and validation loss
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_loss_history, label="Train")
        plt.plot(epochs, self.val_loss_history, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()

        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracy_history, label="Train")
        plt.plot(epochs, self.val_accuracy_history, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()
