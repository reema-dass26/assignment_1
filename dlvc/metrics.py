from abc import ABCMeta, abstractmethod
import torch
from dlvc.datasets.cifar10 import CIFAR10Dataset

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass



class Accuracy(PerformanceMeasure):
    '''
    Average classification accuracy.
    '''

    def __init__(self, classes) -> None:
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        '''
        Reset Method:
        The reset method initializes or resets the internal state of the accuracy tracker. 
        This is often used at the beginning of a new evaluation or training session to clear any previous data.
        It resets counters or variables used to track the number of correct predictions and total predictions.
        '''
        self.correct_count = 0
        self.total_count = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''
        print(prediction)
        print(target)


        try:
            self.predictions=prediction
            self.targets=target

            for pred, true in zip(prediction, target):
                if pred:
                    self.correct_count += 1
                self.total_count += 1
        except ValueError as e:
            print('The data shape or values are unsupported')
            print(e)

    def __str__(self):
        """
        Generate a string representation of the performance including overall accuracy
        and per-class accuracy.

        Args:
        accuracy (float): Overall accuracy.
        per_class_accuracy (list): List of per-class accuracies.

        Returns:
        str: String representation of the performance.
        """
        accuracy=self.accuracy()
        per_class_accuracy=self.per_class_accuracy()
        performance_str = "Overall Accuracy: {:.2f}\n".format(accuracy)
        performance_str += "Per-class Accuracy:\n"
        for class_idx, class_acc in enumerate(per_class_accuracy):
            performance_str += "  Class {}: {:.2f}\n".format(class_idx, class_acc)
        return performance_str


    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''

        if self.total_count == 0:
            return 0
        return self.correct_count / self.total_count
    

    
    def per_class_accuracy(self) -> float:
        """
        Compute and return the per-class accuracy as a list of floats between 0 and 1.
        Returns a list of length num_classes containing accuracies for each class.
        Returns 0 for classes with no data available (after resets).

        Args:
        predicted_labels (list): List of predicted labels.
        true_labels (list): List of true labels.
        num_classes (int): Number of classes in the dataset.

        Returns:
        list: List of per-class accuracies.
        """
        per_class_correct_count = [0] * self.classes
        per_class_total_count = [0] * self.classes
        
        for pred, true in zip(self.predictions, self.targets):
                per_class_total_count[true] += 1
                if pred:
                    per_class_correct_count[true] += 1
        
        per_class_accuracy = [correct / total if total != 0 else 0 for correct, total in zip(per_class_correct_count, per_class_total_count)]
        
        return per_class_accuracy

       