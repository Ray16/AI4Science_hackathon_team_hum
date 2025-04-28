import os
import torch
from datetime import datetime
import helper
from helper import GraphDataset
import pickle
from lightning import GraphRegressionTask
from gnn import MPNNModel
from torch import nn as nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

torch.set_float32_matmul_precision('medium')

class TextLoggingCallback(Callback):
    """PyTorch Lightning callback to log metrics to a text file."""
    # ... (Your TextLoggingCallback remains the same) ...
    def __init__(self, log_dir: str):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.log_dir, "metrics.log")

    def log_info(self, s: str):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        with open(self.log_file_path, "a") as f:
            f.write(f"{dt_string} {s}\n")

    def on_train_start(self, trainer, pl_module):
        self.log_info("Training is starting...")

    def on_train_end(self, trainer, pl_module):
        self.log_info("Training has finished.")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            self.log_info(f"Epoch {epoch} Train Loss: {train_loss:.8f}")
            with open(self.log_file_path, "a") as f:
                f.write(f"{epoch},{train_loss:.8f},")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return None
        epoch = trainer.current_epoch
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            self.log_info(f"Epoch {epoch} Validation Loss: {val_loss:.8f}")
            with open(self.log_file_path, "a") as f:
                f.write(f"{val_loss:.8f}\n")
        else:
            with open(self.log_file_path, "a") as f:
                f.write("\n")

# load input json data into a dictionary of graph objects
graph_data_json = helper.load_data_from_file("graph_data.json")
graph_list = [graph_data_json[key] for key in graph_data_json.keys()]

# dataset splitting BEFORE creating the GraphDataset
dataset_size = len(graph_list)
train_size = int(0.9 * dataset_size)
val_size = int(0.05 * dataset_size)
test_size = dataset_size - train_size - val_size
train_graphs_list, val_graphs_list, test_graphs_list = random_split(
    graph_list, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
)

# Calculate mean and std from the training graphs_list
train_targets = []
for graph_data in train_graphs_list:
    for _, node in graph_data.nodes(data=True):
        binding_energies = node.get('binding_energies', [])
        # remove binding energy=-1 from mean and standard deviation calculation
        valid_energies = [energy for energy in binding_energies if energy != -1]
        train_targets.extend(valid_energies)

train_mean = torch.tensor(sum(train_targets) / len(train_targets), dtype=torch.float)
train_std = torch.tensor((sum([(x - train_mean)**2 for x in train_targets]) / (len(train_targets) - 1))**0.5 if len(train_targets) > 1 else 1.0, dtype=torch.float)

print(f"Training binding energy mean: {train_mean}")
print(f"Training binding energy std: {train_std}")

# save training binding energy mean and std to external pickle
with open('binding_e_mean_std_train.pkl', "wb") as file:
    pickle.dump([train_mean,train_std], file)

# create dataset
print('Creating datasets')
train_dataset = GraphDataset(train_graphs_list, target_mean=train_mean, target_std=train_std)
val_dataset = GraphDataset(val_graphs_list, target_mean=train_mean, target_std=train_std)
test_dataset = GraphDataset(test_graphs_list, target_mean=train_mean, target_std=train_std)

# Calculate the base feature dimension and total feature dimension
base_feature_dim = len(train_dataset.all_elements) + 1 + 1 + 1 + 1
total_feature_dim = base_feature_dim + len(train_dataset.all_orbitals)
print(f"Total feature dimension: {total_feature_dim}")
print(f"Base feature dimension: {base_feature_dim}")

# create dataloader
print('Creating dataloaders')
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=63)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=63)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=63)

# Instantiate the MPNN model
num_layers = 4
emb_dim = 64
edge_dim = 3

# The in_dim and edge_dim will be inferred automatically by LazyLinear
print('Initializing model')
model = MPNNModel(num_layers=num_layers, emb_dim=emb_dim, edge_dim=edge_dim, out_dim=1)

losses = [nn.MSELoss()]

# Set up logging and callbacks
logs_directory = "lightning_logs"
name = "graph_regression"
i = 0
while os.path.isdir(f"{logs_directory}/{name}/version_{i}"):
    i += 1
version_name = f"version_{i}"
os.makedirs(f"{logs_directory}/{name}/{version_name}",exist_ok=True)
log_file_path = f"{logs_directory}/{name}/{version_name}/metrics.log"
text_logger_callback = TextLoggingCallback(log_file_path)
logger = TensorBoardLogger(logs_directory, name=name, version=version_name)

# Initialize ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',       # Metric to monitor
    dirpath=f"{logs_directory}/{name}/{version_name}/checkpoints", # Path to save checkpoints
    filename='best-model-{epoch:02d}-{val_loss:.4f}', # Filename format
    save_top_k=1,             # Save only the best model
    mode='min'                # Monitor 'val_loss' and save when it decreases
)

# Initialize PyTorch Lightning Trainer
trainer = pl.Trainer(
    max_epochs=500,
    callbacks=[text_logger_callback, checkpoint_callback], # Add ModelCheckpoint to callbacks
    logger=logger
)

csv_path = f"{logs_directory}/{name}/{version_name}/test_predictions.csv"

# Instantiate the Lightning Module, passing the test_dataset and base_feature_dim
lightning_model = GraphRegressionTask(
    model,
    learning_rate=0.001,
    train_mean=train_mean,
    train_std=train_std,
    losses=losses,
    csv_path=csv_path,
    test_dataset=test_dataset,
    base_feature_dim=base_feature_dim
)

# Model training
print('Starting model training')
trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Model testing (optional: you can also test the best model specifically)
print('Starting model testing')
trainer.test(lightning_model, dataloaders=test_loader)

# You can also load the best model from the checkpoint path if you want to test it specifically later:
# best_model_path = checkpoint_callback.best_model_path
# best_model = GraphRegressionTask.load_from_checkpoint(best_model_path, model=model, learning_rate=0.001, train_mean=train_mean, train_std=train_std, losses=losses, csv_path=csv_path, test_dataset=test_dataset, base_feature_dim=base_feature_dim)
# trainer.test(best_model, dataloaders=test_loader)
