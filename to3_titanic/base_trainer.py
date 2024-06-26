import logging
from abc import abstractmethod

class Base_trainer:
    def __init__(self, config):
        self.config = config
        self.train_dataset = ...
        self.model = ...
        ...

    
    @abstractmethod
    def train(self):
        raise NotImplementedError
    
    @abstractmethod
    def train_one_epoch(self):
        raise NotImplementedError
    
    @abstractmethod
    def eval(self):
        raise NotImplementedError

