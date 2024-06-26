import torch
import torch.nn as nn
import torch.optim as optim

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = 10   # Number of input features
hidden_size = 20  # Number of features in hidden state
output_size = 1   # Number of output classes
num_layers = 2    # Number of stacked RNN layers
num_epochs = 100
learning_rate = 0.001

# Create an instance of the model
model = RNNModel(input_size, hidden_size, output_size, num_layers)

# Loss and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Dummy input data and target
# For actual use, replace with your dataset loader
x_train = torch.randn((100, 5, input_size))  # Example input tensor (batch_size, sequence_length, input_size)
y_train = torch.randn((100, output_size))    # Example output tensor (batch_size, output_size)

# Training loop
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing the model
model.eval()
with torch.no_grad():
    sample = torch.randn((1, 5, input_size))  # Example single input (batch_size=1, sequence_length=5, input_size=10)
    prediction = model(sample)
    print(f'Prediction: {prediction}')

