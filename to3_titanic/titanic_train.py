import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from abc import abstractmethod
from sklearn.utils import shuffle
from base_trainer import Base_trainer
from config import Config
from tqdm import tqdm

# Model Class
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

class DropoutNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

class MaNN(nn.Module):
    def __init__(self, input_size):
        super(MaNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128000)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

class MaNN1(nn.Module):
    def __init__(self, input_size):
        super(MaNN1, self).__init__()
        self.fc1 = nn.Linear(input_size, 128000)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128000, 2)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)

        return x


class MaNN2(nn.Module):
    def __init__(self, input_size, hidden_size=20, num_layers=1000):
        super(MaNN2, self).__init__()

        self.fc = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.relu = nn.ReLU()
        for _ in range(num_layers - 1):
            self.fc.append(nn.Linear(hidden_size, hidden_size))
        self.fc.append(nn.Linear(hidden_size, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.fc[:-1]:
            x = layer(x)
            x = self.sigmoid(x)
        x = self.fc[-1](x)
        x = self.sigmoid(x)
        return x

# Dataset Class
class TitanicDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.scaler = StandardScaler()
        self.process_data()

    def process_data(self):
        self.data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
        self.data['Age'].fillna(self.data['Age'].median(), inplace=True)
        self.data['Embarked'].fillna(self.data['Embarked'].mode()[0], inplace=True)
        self.data = pd.get_dummies(self.data, columns=['Sex', 'Embarked'], drop_first=True)
        self.data = self.data.astype('float32')
        self.data.iloc[:, 1:] = self.scaler.fit_transform(self.data.iloc[:, 1:])
        self.X = self.data.drop('Survived', axis=1).values.astype(np.float32)
        self.y = self.data['Survived'].values.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'features': torch.tensor(self.X[idx]), 'label': torch.tensor(self.y[idx])}

class TitanicDatasetTest(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.scaler = StandardScaler()
        self.process_data()

    def process_data(self):
        self.data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
        self.data['Age'].fillna(self.data['Age'].median(), inplace=True)
        self.data['Embarked'].fillna(self.data['Embarked'].mode()[0], inplace=True)
        self.data = pd.get_dummies(self.data, columns=['Sex', 'Embarked'], drop_first=True)
        self.data = self.data.astype('float32')
        self.data.iloc[:, 1:] = self.scaler.fit_transform(self.data.iloc[:, 1:])
        self.X = self.data.values.astype(np.float32)
        # self.y = self.data['Survived'].values.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'features': torch.tensor(self.X[idx])}

# Trainer Class
class TitanicTrainer(Base_trainer):
    def __init__(self, config):
        super().__init__(config)
        self.logger = config.logger
        self.train_dataset = TitanicDataset(config.train_data_path)
        self.test_dataset = TitanicDatasetTest(config.test_data_path)
        self.logger.info("Dataset size: " + str(len(self.train_dataset)))
        # self.test_dataset = TitanicDataset(config.test_data_path)
        self.model = MaNN2(input_size=self.train_dataset.X.shape[1])
        self.train_loader, self.val_loader, self.test_loader = self.create_data_loaders()

    def create_data_loaders(self):
        train_size = int(0.8 * len(self.train_dataset))
        val_size = int(0.2 * len(self.train_dataset))
        test_size = len(self.train_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            self.train_dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        return train_loader, val_loader, test_loader

    def train(self):
        criterion = nn.BCELoss()
        if type(self.model).__name__ == 'MaNN1':
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        for epoch in tqdm(range(self.config.num_epochs), ncols=100):
            self.train_one_epoch(epoch, criterion, optimizer)
            
            early_stopping = self.eval()
            if early_stopping:
                break
            else:
                continue

    def train_one_epoch(self, epoch, criterion, optimizer):
        self.model.train()
        train_loss = 0.0
        for batch in self.train_loader:
            inputs, batch_labels = batch['features'], batch['label']
            optimizer.zero_grad()
            batch_outputs = self.model(inputs)
            
            if type(self.model).__name__ == 'MaNN1':
                loss = criterion(batch_outputs, batch_labels.long().squeeze())
            else:
                loss = criterion(batch_outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(self.train_loader.dataset)
        self.logger.info(f"Epoch [{epoch + 1}/{self.config.num_epochs}], Train Loss: {train_loss:.4f}")

    def eval(self):
        self.model.eval()
        val_loss = 0.0
        outputs = []
        labels = []
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, batch_labels = batch['features'], batch['label']
                batch_outputs = self.model(inputs)
                if type(self.model).__name__ == 'MaNN1':
                    loss = nn.CrossEntropyLoss()(batch_outputs, batch_labels.long().squeeze())
                else:
                    loss = nn.BCELoss()(batch_outputs, batch_labels)
                val_loss += loss.item() * inputs.size(0)
                outputs.append(batch_outputs)
                labels.append(batch_labels)
        val_loss /= len(self.val_loader.dataset)
        self.logger.info(f"Validation Loss: {val_loss:.4f}")
        
        # Function to calculate accuracy
        def calculate_accuracy(outputs, labels, threshold=0.5):
            # Convert outputs to binary predictions
            predictions = (outputs > threshold).float()

            # Compare predictions with labels
            correct = (predictions == labels).float().sum()

            # Calculate accuracy
            accuracy = correct / labels.size(0)
            
            return accuracy.item()
        outputs = torch.cat(outputs, dim=0).squeeze()
        labels = torch.cat(labels, dim=0).squeeze()
        if type(self.model).__name__ == 'MaNN1':
            val_acc = calculate_accuracy(torch.max(outputs, dim=1)[1], labels)
        else:
            val_acc = calculate_accuracy(outputs, labels)
        self.logger.info(f"Validation Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss >= self.config.prev_val_loss+self.config.tolerance:
            self.logger.info("Validation loss increased. Early stopping.")
            return True
        else:
            self.config.prev_val_loss = min(val_loss, self.config.prev_val_loss)
            return False
    def test(self):
        test_dataloader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset))
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_dataloader:
                inputs = batch['features']
                outputs = self.model(inputs)
                
        predictions = (outputs > 0.5).int()
        import pandas as pd

        # Define the range of PassengerId
        passenger_ids = range(892, 1310)
        print(predictions.squeeze().shape)
        df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions.squeeze()})

        # Save DataFrame with no index
        df.to_csv('/mnt/disk1/AI_Lab/to3_titanic/predictions.csv', index=False)


def main(config):
    trainer = TitanicTrainer(config)
    def read_module_content():
        try:
            module_path = __import__(__name__).__file__
            with open(module_path, 'r', encoding='utf-8') as file:
                module_content = file.read()
            return module_content
        except Exception as e:
            print(f"Error occurred while reading module '{module_name}': {e}")
            return None
    
    config.logger.info('\n <<< Train Content <<<\n'+read_module_content())
    
    config.logger.info('Training start')
    trainer.train()
    # trainer.test()


    

if __name__ == '__main__':
    config = Config()
    
    main(config)