"""
Neural Network Model for Fertility Risk Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FertilityRiskNet(nn.Module):
    """
    Deep Neural Network for fertility risk prediction
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], num_classes=2, dropout=0.3):
        super(FertilityRiskNet, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            #layers.append(nn.BatchNorm1d(hidden_dim))
               # Use 1 group or a small number of groups
            layers.append(nn.GroupNorm(num_groups=1, num_channels=hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def get_model(input_dim, num_classes=2):
    """
    Factory function to create model
    """
    return FertilityRiskNet(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        num_classes=num_classes,
        dropout=0.3
    )

def train_one_epoch(model, trainloader, optimizer, criterion, device):
    """
    Train model for one epoch
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for X_batch, y_batch in trainloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    avg_loss = total_loss / len(trainloader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def evaluate(model, testloader, criterion, device):
    """
    Evaluate model on test data
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in testloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(testloader)
    accuracy = correct / total
    
    return avg_loss, accuracy, all_preds, all_labels