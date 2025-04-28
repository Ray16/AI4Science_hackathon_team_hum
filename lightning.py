import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error

class GraphRegressionTask(pl.LightningModule):
    def __init__(self, model, learning_rate=0.001, train_mean=0.0, train_std=1.0, losses=None, csv_path=None, test_dataset=None, base_feature_dim=None):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.train_mean = train_mean
        self.train_std = train_std
        self.losses = losses if losses is not None else [nn.MSELoss()] # Default to MSELoss if no losses provided
        self.test_predictions = []
        self.test_targets = []
        self.csv_path = csv_path
        self.test_dataset_ref = test_dataset # Store a reference to the test dataset
        self.base_feature_dim_size = base_feature_dim # Store the base feature dimension

    def forward(self, data):
        return self.model.forward(data)

    def calculate_loss(self, data):
        results = self.forward(data)
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn(results, data.y.squeeze(1))
            total_loss += loss
        return total_loss

    def training_step(self, batch, batch_idx):
        predictions = self.forward(batch)
        loss_mask = (batch.y != -1).float()  # Mask for non -1 targets

        # Find indices where the target is not -1
        valid_indices = torch.where(loss_mask == 1)[0]

        if valid_indices.numel() > 0:
            # Select predictions and targets for the valid indices
            valid_predictions = predictions[valid_indices]
            valid_targets = batch.y.squeeze(1)[valid_indices]

            # Calculate the loss only on the valid predictions and targets
            loss = F.mse_loss(valid_predictions, valid_targets)
        else:
            loss = torch.tensor(0.0, device=self.device) # If all targets are -1, loss is 0
        batch_size = batch.batch.max() + 1
        self.log("train_loss", loss.mean(), on_epoch=True, batch_size=batch_size, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        predictions = self.forward(batch)
        loss_mask = (batch.y != -1).float()  # Mask for non -1 targets

        # Find indices where the target is not -1
        valid_indices = torch.where(loss_mask == 1)[0]

        if valid_indices.numel() > 0:
            # Select predictions and targets for the valid indices
            valid_predictions = predictions[valid_indices]
            valid_targets = batch.y.squeeze(1)[valid_indices]

            # Calculate the loss only on the valid predictions and targets
            loss = F.mse_loss(valid_predictions, valid_targets)
        else:
            loss = torch.tensor(0.0, device=self.device) # If all targets are -1, loss is 0

        batch_size = batch.batch.max() + 1
        self.log("val_loss", loss.mean(), on_epoch=True, batch_size=batch_size, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        predictions = self.forward(batch)
        loss_mask = (batch.y != -1).float()  # Mask for non -1 targets

        # Find indices where the target is not -1
        valid_indices = torch.where(loss_mask == 1)[0]

        if valid_indices.numel() > 0:
            # Select predictions and targets for the valid indices
            valid_predictions = predictions[valid_indices]
            valid_targets = batch.y.squeeze(1)[valid_indices]

            # Denormalize predictions and targets *before* calculating metrics
            preds_denorm = (valid_predictions.cpu() * self.train_std + self.train_mean).numpy()
            targets_denorm = (valid_targets.cpu() * self.train_std + self.train_mean).numpy()

            # Calculate evaluation metrics
            rmse = np.sqrt(mean_squared_error(targets_denorm, preds_denorm))
            mae = mean_absolute_error(targets_denorm, preds_denorm)
            loss = F.mse_loss(valid_predictions, valid_targets) # Calculate loss here
        else:
            loss = torch.tensor(0.0, device=self.device) # If all targets are -1, loss is 0
            rmse = np.nan
            mae = np.nan
            preds_denorm = np.array([])
            targets_denorm = np.array([])

        # Handle the case where the '-1' orbital feature might be present
        if hasattr(self, 'test_dataset_ref') and self.test_dataset_ref is not None and hasattr(self, 'base_feature_dim_size') and self.base_feature_dim_size is not None:
            negative_one_orbital_index = self.test_dataset_ref.orbital_to_index.get(-1)
            if negative_one_orbital_index is not None:
                negative_one_orbital_feature_column = self.base_feature_dim_size + negative_one_orbital_index
                if negative_one_orbital_feature_column < batch.x.shape[1]:
                    negative_one_orbital_present_mask = (batch.x[:, negative_one_orbital_feature_column] == 1.0)
                    # Assign -1.0 to the denormalized predictions where the '-1' orbital is present
                    # This needs to be applied to the full batch predictions before filtering
                    full_preds_denorm_tensor = (predictions.cpu() * self.train_std + self.train_mean)
                    full_preds_denorm = full_preds_denorm_tensor.numpy()
                    full_preds_denorm[negative_one_orbital_present_mask.cpu().numpy()] = -1

                    # Now, when extending, consider the original targets for masking
                    original_targets = batch.y.squeeze(1).cpu().numpy()
                    train_std_np = np.array(self.train_std)  # Convert to NumPy
                    train_mean_np = np.array(self.train_mean) # Convert to NumPy

                    denormed_original_targets = (original_targets * train_std_np + train_mean_np).astype(np.float64)
                    valid_original_mask = (original_targets != -1)

                    # Ensure that we extend with lists, not individual values
                    molecules = batch.molecule if hasattr(batch, 'molecule') else [''] * len(valid_original_mask)
                    elements = batch.element if hasattr(batch, 'element') else [''] * len(valid_original_mask)
                    orbitals = batch.orbital if hasattr(batch, 'orbital') else [''] * len(valid_original_mask)
                    self.test_predictions.extend(list(zip(molecules, elements, orbitals, full_preds_denorm[valid_original_mask])))
                    self.test_targets.extend(list(denormed_original_targets[valid_original_mask]))
                else:
                    # If the '-1' orbital feature column is out of bounds
                    original_targets = batch.y.squeeze(1).cpu().numpy()
                    train_std_np = np.array(self.train_std)  # Convert to NumPy
                    train_mean_np = np.array(self.train_mean) # Convert to NumPy
                    denormed_original_targets = (original_targets * train_std_np + train_mean_np).astype(np.float64)  # Denormalize GT
                    valid_original_mask = (original_targets != -1)
                    full_preds_denorm_tensor = (predictions.cpu() * self.train_std + self.train_mean)
                    molecules = batch.molecule if hasattr(batch, 'molecule') else [''] * len(valid_original_mask)
                    elements = batch.element if hasattr(batch, 'element') else [''] * len(valid_original_mask)
                    orbitals = batch.orbital if hasattr(batch, 'orbital') else [''] * len(valid_original_mask)
                    self.test_predictions.extend(list(zip(molecules, elements, orbitals, full_preds_denorm_tensor.cpu().numpy()[valid_original_mask])))
                    self.test_targets.extend(list(denormed_original_targets[valid_original_mask]))
            else:
                # If '-1' orbital index is not found
                original_targets = batch.y.squeeze(1).cpu().numpy()
                train_std_np = np.array(self.train_std)  # Convert to NumPy
                train_mean_np = np.array(self.train_mean) # Convert to NumPy
                denormed_original_targets = (original_targets * train_std_np + train_mean_np).astype(np.float64)  # Denormalize GT
                valid_original_mask = (original_targets != -1)
                full_preds_denorm_tensor = (predictions.cpu() * self.train_std + self.train_mean)
                molecules = batch.molecule if hasattr(batch, 'molecule') else [''] * len(valid_original_mask)
                elements = batch.element if hasattr(batch, 'element') else [''] * len(valid_original_mask)
                orbitals = batch.orbital if hasattr(batch, 'orbital') else [''] * len(valid_original_mask)
                self.test_predictions.extend(list(zip(molecules, elements, orbitals, full_preds_denorm_tensor.cpu().numpy()[valid_original_mask])))
                self.test_targets.extend(list(denormed_original_targets[valid_original_mask]))
        else:
            # If test_dataset_ref or base_feature_dim_size is not available
            original_targets = batch.y.squeeze(1).cpu().numpy()
            train_std_np = np.array(self.train_std)  # Convert to NumPy
            train_mean_np = np.array(self.train_mean) # Convert to NumPy
            denormed_original_targets = (original_targets * train_std_np + train_mean_np).astype(np.float64)  # Denormalize GT
            valid_original_mask = (original_targets != -1)
            full_preds_denorm_tensor = (predictions.cpu() * self.train_std + self.train_mean)
            molecules = batch.molecule if hasattr(batch, 'molecule') else [''] * len(valid_original_mask)
            elements = batch.element if hasattr(batch, 'element') else [''] * len(valid_original_mask)
            orbitals = batch.orbital if hasattr(batch, 'orbital') else [''] * len(valid_original_mask)
            self.test_predictions.extend(list(zip(molecules, elements, orbitals, full_preds_denorm_tensor.cpu().numpy()[valid_original_mask])))
            self.test_targets.extend(list(denormed_original_targets[valid_original_mask]))

        batch_size = batch.batch.max() + 1
        self.log("test_loss", loss.mean(), on_epoch=True, batch_size=batch_size)
        self.log("test_rmse", rmse, on_epoch=True, batch_size=batch_size)
        self.log("test_mae", mae, on_epoch=True, batch_size=batch_size)
        return loss

    def on_test_epoch_end(self):
        """
        This hook is called at the end of the test epoch.  It's the ideal place
        to save the collected predictions and targets to a CSV file.
        """
        # Create a list of lists, where each inner list represents a row
        rows = []
        for (molecule, element, orbital, predicted_energy), ground_truth_energy in zip(self.test_predictions, self.test_targets):
            rows.append([molecule, element, orbital, predicted_energy])

        # Create the Pandas DataFrame
        results_df = pd.DataFrame(rows, columns=['Molecule', 'Element', 'Orbital', 'Binding Energy'])

        # Save to csv
        if self.csv_path:  # Only save if a path is provided
            results_df.to_csv(self.csv_path, index=False)
            print(f"Test predictions saved to {self.csv_path}")
        else:
            print("csv_path not provided.  Not saving test predictions to CSV.")

        # Clear the lists after saving
        self.test_predictions = []
        self.test_targets = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
