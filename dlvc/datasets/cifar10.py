import pickle
from typing import Tuple
import numpy as np
from pathlib import Path


from dlvc.datasets.dataset import Subset, ClassificationDataset


class CIFAR10Dataset(ClassificationDataset):
    """
    Custom CIFAR-10 Dataset.
    """

    def __init__(self, fdir: str, subset: Subset, transform=None):
        """
        Loads the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all images from "data_batch_5".
          - The test set contains all images from "test_batch".

        Images are loaded in the order they appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in RGB channel order.
        """
        p = Path(fdir)

        data = sorted(list(p.glob("data_*")))
        test = list(p.glob("test_*"))

        if not p.exists() or len(data) + len(test) != 6:
            raise ValueError

        training, validation = data[:-1], data[-1]

        training_dict = self._unpickle(training[0])
        training_data = training_dict[b"data"]
        training_labels = training_dict[b"labels"]

        for d in training[1:]:
            training_dict = self._unpickle(d)
            training_data = np.vstack((training_data, training_dict[b"data"]))
            training_labels += training_dict[b"labels"]

        training_data = training_data.reshape((len(training_data), 3, 32, 32))
        training_data = np.rollaxis(training_data, 1, 4)
        training_labels = np.array(training_labels)

        validation_dict = self._unpickle(validation)
        validation_data = validation_dict[b"data"]
        validation_labels = validation_dict[b"labels"]
        validation_data = validation_data.reshape(len(validation_data), 3, 32, 32)
        validation_data = np.rollaxis(validation_data, 1, 4)
        validation_labels = np.array(validation_labels)

        test_dict = self._unpickle(test[0])
        test_data = test_dict[b"data"]
        test_labels = test_dict[b"labels"]
        test_data = test_data.reshape(len(test_data), 3, 32, 32)
        test_data = np.rollaxis(test_data, 1, 4)
        test_labels = np.array(test_labels)

        if subset == Subset.TRAINING:
            self.data = training_data
            self.labels = training_labels

        if subset == Subset.TEST:
            self.data = test_data
            self.labels = test_labels

        if subset == Subset.VALIDATION:
            self.data = validation_data
            self.labels = validation_labels

        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

        self.transform = transform

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """

        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns the idx-th sample in the dataset, which is a tuple,
        consisting of the image and labels.
        Applies transforms if not None.
        Raises IndexError if the index is out of bounds.
        """
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def num_classes(self) -> int:
        """
        Returns the number of classes.
        """
        return len(self.classes)

    def _unpickle(self, file):
        """load the cifar-10 data"""
        with open(file, "rb") as fo:
            data = pickle.load(fo, encoding="bytes")
        return data
