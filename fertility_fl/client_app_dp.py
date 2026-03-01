"""
Flower Client with Differential Privacy
"""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from fertility_fl.task import load_partition_data, get_model_config
from fertility_fl.model import get_model, evaluate

class FertilityClientDP(NumPyClient):
    """
    Flower client with Differential Privacy
    """
    def __init__(self, trainloader, valloader, model_config, 
                 local_epochs=3, noise_multiplier=1.0, max_grad_norm=1.0):
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # DP parameters
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        
        # Initialize model
        self.model = get_model(
            input_dim=model_config['input_dim'],
            num_classes=model_config['num_classes']
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Attach privacy engine
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.trainloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
    
    def fit(self, parameters, config):
        """Train with differential privacy"""
        self.set_parameters(parameters)
        
        # Train for multiple epochs
        for epoch in range(self.local_epochs):
            self.model.train()
            for X_batch, y_batch in self.trainloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
        
        # Get privacy spent
        epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
        
        # Compute final metrics
        train_loss, train_acc = self._evaluate_train()
        
        return self.get_parameters(), len(self.trainloader.dataset), {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "epsilon": epsilon,  # Privacy budget spent
            "delta": 1e-5
        }
    
    def evaluate(self, parameters, config):
        """Evaluate model"""
        self.set_parameters(parameters)
        val_loss, val_acc, _, _ = evaluate(
            self.model, self.valloader, self.criterion, self.device
        )
        return val_loss, len(self.valloader.dataset), {
            "val_accuracy": val_acc,
            "val_loss": val_loss
        }
    
    def _evaluate_train(self):
        """Evaluate on training data"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in self.trainloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        return total_loss / len(self.trainloader), correct / total
    
    def get_parameters(self):
        """Get model parameters"""
        return [val.cpu().numpy() for val in self.model._module.state_dict().values()]
    
    def set_parameters(self, parameters):
        """Set model parameters"""
        params_dict = zip(self.model._module.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model._module.load_state_dict(state_dict, strict=True)


def client_fn(context: Context):
    """Factory function for DP client"""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    trainloader, valloader, metadata = load_partition_data(partition_id)
    model_config = get_model_config()
    
    # Get DP parameters from config
    noise_multiplier = context.run_config.get("noise-multiplier", 1.0)
    max_grad_norm = context.run_config.get("max-grad-norm", 1.0)
    
    return FertilityClientDP(
        trainloader=trainloader,
        valloader=valloader,
        model_config=model_config,
        local_epochs=3,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm
    ).to_client()

# Create ClientApp
app = ClientApp(client_fn=client_fn)