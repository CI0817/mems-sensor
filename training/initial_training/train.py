"""
train.py
--------
This script orchestrates the training of the inertial odometry model.
Usage: python train.py <dataset> <output_checkpoint_name> --model_type <lstm/tcn/transformer>
"""

import argparse
import math
import csv
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

# Import model and data preparation modules
from model import create_model, CustomMultiLossLayer, LSTM_PROPERTIES, TCNBLOCK_PROPERTIES
from data_preparation import (
    build_file_lists, load_and_process_data, split_sequences,
    concat_and_build_dataset, create_dataloaders, quaternion_angle_error
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train inertial odometry model.")
    parser.add_argument('dataset', choices=['oxiod', 'euroc'], help='Dataset name.')
    parser.add_argument('output', help='Output checkpoint name (without extension).')
    parser.add_argument('--model_type', choices=['lstm', 'tcn', 'transformer'], default='lstm', help='Model type.')
    return parser.parse_args()

def train_loop(model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, num_epochs: int = 100, scheduler: torch.optim.lr_scheduler._LRScheduler = None, checkpoint_path: str = 'checkpoint.pt', hyperparams: dict = None) -> dict:
    """
    Train a model using the provided training and validation data loaders.
    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        criterion (callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device): Device to run the training on (e.g., 'cpu' or 'cuda').
        num_epochs (int, optional): Number of epochs to train the model. Default is 100.
        scheduler (torch.optim.lr_scheduler._LRScheduler or torch.optim.lr_scheduler.ReduceLROnPlateau, optional): Learning rate scheduler. Default is None.
        checkpoint_path (str, optional): Path to save the model checkpoint. Default is 'checkpoint.pt'.
        hyperparams (dict, optional): Additional hyperparameters to log. Default is None.
    Returns:
        dict: A dictionary containing training and validation metrics:
            - 'epochs': List of epoch numbers.
            - 'train_loss': List of training losses for each epoch.
            - 'val_loss': List of validation losses for each epoch.
            - 'val_pos_rmse': List of validation position RMSE for each epoch.
            - 'val_quat_angle_deg': List of validation quaternion angle errors in degrees for each epoch.
    """
    # Initialize hyperparameters if not provided
    if hyperparams is None:
        hyperparams = {}
    # Generate a timestamp for the training session; use provided timestamp if available
    training_start_ts = hyperparams.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
    # Retrieve the model type from hyperparameters (default is 'lstm')
    model_type = hyperparams.get('model_type', 'lstm')
    # Create a CSV filename based on model type and timestamp
    csv_filename = f"{model_type}_{training_start_ts}.csv"
    # Get the absolute path for the CSV file
    csv_path = Path(csv_filename).resolve()

    # If the CSV file does not exist, create it and write the header row
    if not csv_path.exists():
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            # Define header with metrics and additional hyperparameters keys
            header = ['epoch', 'train_loss', 'val_loss', 'val_pos_rmse', 'val_quat_angle_deg', 'epoch_time', 'current_lr']
            header.extend(list(hyperparams.keys()))
            writer.writerow(header)

    # Initialize the best validation loss and lists to store metrics for each epoch
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    val_pos_rmse_list, val_quat_angle_deg_list = [], []

    # Loop over each epoch
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0.0  # Initialize cumulative training loss for the epoch
        
        # Iterate over training batches
        for (x_gyro, x_acc, y_delta_p, y_delta_q) in train_loader:
            # Transfer input data to the specified device
            x_gyro, x_acc = x_gyro.to(device), x_acc.to(device)
            # Transfer target data to the specified device
            y_delta_p, y_delta_q = y_delta_p.to(device), y_delta_q.to(device)
            optimizer.zero_grad()  # Clear previous gradients
            # Forward pass: get model predictions for position and quaternion
            pos_pred, quat_pred = model(x_gyro, x_acc)
            # Compute loss based on the predictions and true values
            loss = criterion([y_delta_p, y_delta_q], [pos_pred, quat_pred])
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model parameters
            epoch_loss += loss.item()  # Accumulate loss over the batch
        
        # Compute the average training loss for the epoch
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        model.eval()  # Set model to evaluation mode
        val_loss = 0.0  # Initialize cumulative validation loss
        val_pos_mse_sum = 0.0  # Sum of MSE for position predictions over validation batches
        val_quat_angle_sum = 0.0  # Sum of quaternion angle errors over validation batches
        val_batches = 0  # Counter for the number of validation batches

        # Disable gradient computation during validation for efficiency
        with torch.no_grad():
            for (x_gyro, x_acc, y_delta_p, y_delta_q) in val_loader:
                # Transfer validation data to the device
                x_gyro, x_acc = x_gyro.to(device), x_acc.to(device)
                y_delta_p, y_delta_q = y_delta_p.to(device), y_delta_q.to(device)
                # Forward pass: compute model predictions on validation data
                pos_pred, quat_pred = model(x_gyro, x_acc)
                # Compute and accumulate batch loss
                batch_loss = criterion([y_delta_p, y_delta_q], [pos_pred, quat_pred])
                val_loss += batch_loss.item()
                # Calculate mean squared error for position predictions
                batch_pos_mse = nn.functional.mse_loss(pos_pred, y_delta_p, reduction='mean')
                val_pos_mse_sum += batch_pos_mse.item()
                # Calculate quaternion angle error for the batch
                batch_angles = quaternion_angle_error(y_delta_q, quat_pred)
                val_quat_angle_sum += batch_angles.mean().item()
                val_batches += 1  # Increment batch counter

        # Compute average validation loss over all batches
        val_loss /= val_batches
        val_losses.append(val_loss)
        # Calculate root mean squared error for position predictions
        mean_pos_rmse = math.sqrt(val_pos_mse_sum / val_batches)
        val_pos_rmse_list.append(mean_pos_rmse)
        # Convert average quaternion angle error to degrees
        mean_angle_degrees = (val_quat_angle_sum / val_batches * 180.0) / math.pi
        val_quat_angle_deg_list.append(mean_angle_degrees)

        # Print metrics for the current epoch
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Pos RMSE: {mean_pos_rmse:.4f} | Angle (deg): {mean_angle_degrees:.4f}")

        # Adjust learning rate if a scheduler is provided
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)  # Use validation loss to adjust learning rate
            else:
                scheduler.step()  # Standard scheduler step

        # Retrieve current learning rate from optimizer
        current_lr = optimizer.param_groups[0]['lr']
        # Save model checkpoint if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, checkpoint_path)

        # Log epoch metrics and hyperparameters to the CSV file
        epoch_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Record current time as epoch timestamp
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            # Prepare row data with epoch metrics and current learning rate
            row = [epoch + 1, epoch_loss, val_loss, mean_pos_rmse, mean_angle_degrees, epoch_time_str, current_lr]
            # Append hyperparameter values to the row
            row.extend([hyperparams[k] for k in hyperparams])
            writer.writerow(row)

    # Return a dictionary of metrics collected during training
    return {
        'epochs': list(range(1, num_epochs + 1)),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_pos_rmse': val_pos_rmse_list,
        'val_quat_angle_deg': val_quat_angle_deg_list
    }


def plot_training_history(metrics: dict, output_prefix: str):
    """
    Plot training and validation metrics, saving the figures using the given prefix.
    
    Args:
        metrics (dict): Dictionary containing 'epochs', 'train_loss', 'val_loss',
                        'val_pos_rmse', and 'val_quat_angle_deg'.
        output_prefix (str): Prefix used for saving plot images.
    """
    epochs = metrics['epochs']
    training_losses = metrics['train_loss']
    validation_losses = metrics['val_loss']
    position_rmse = metrics['val_pos_rmse']
    angle_errors = metrics['val_quat_angle_deg']

    # Plot Training and Validation Loss
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, training_losses, label='Training Loss')
    ax.plot(epochs, validation_losses, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss over Epochs')
    ax.legend()
    ax.grid(True)
    textstr = f'Final train_loss = {training_losses[-1]:.4f}\nFinal val_loss   = {validation_losses[-1]:.4f}'
    ax.text(0.95, 0.50, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right')
    plt.savefig(f'{output_prefix}_loss.png')
    plt.show()

    # Plot Position RMSE and Angle Error
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_rmse = 'tab:blue'
    ax1.plot(epochs, position_rmse, color=color_rmse, label='Position RMSE')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Position RMSE (m)', color=color_rmse)
    ax1.tick_params(axis='y', labelcolor=color_rmse)
    ax1.grid(True)
    ax2 = ax1.twinx()
    color_angle = 'tab:red'
    ax2.plot(epochs, angle_errors, color=color_angle, label='Angle Error')
    ax2.set_ylabel('Angle Error (deg)', color=color_angle)
    ax2.tick_params(axis='y', labelcolor=color_angle)
    textstr = f'Final rmse_error  = {position_rmse[-1]:.4f}\nFinal angle_error = {angle_errors[-1]:.4f}'
    ax1.text(0.95, 0.50, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right')
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    plt.title('Position RMSE and Angle Error over Epochs')
    plt.savefig(f'{output_prefix}_error.png')
    plt.show()

def main():
    """
    Main function to train a model for MEMS AI project.
    This function performs the following steps:
    1. Parses command-line arguments.
    2. Sets up the device (CPU or GPU) and random seed.
    3. Configures hyperparameters and learning rate scheduler.
    4. Prepares the dataset by loading and processing data, and splitting it into training and validation sets.
    5. Creates data loaders for training and validation datasets.
    6. Initializes the model, loss function, optimizer, and learning rate scheduler.
    7. Trains the model using the training loop and saves the best model checkpoint.
    8. Plots the training history.
    Args:
        None
    Returns:
        None
    """
    args = parse_arguments()  # Parse command-line arguments
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device to GPU if available, otherwise use CPU
    torch.manual_seed(0)  # Set random seed for reproducibility

    # Hyperparameters and scheduler setup
    model_type = args.model_type  # Determine the model type from the parsed arguments
    window_size, stride = 200, 10  # Define the window size and stride for segmenting the data
    batch_size, num_epochs = 32, 100  # Set batch size and the number of epochs for training
    learning_rate = 1e-3  # Set the learning rate for the optimizer
    scheduler_type = 'MultiStepLR'  # Specify which scheduler to use
    # Define properties for MultiStepLR scheduler
    multiStepLR_props = {'type': 'MultiStepLR', 'milestones': [20, 50, 80], 'gamma': 0.1}
    # Define properties for ReduceLROnPlateau scheduler
    reduceLROnPlateau_props = {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.1, 'patience': 10, 'threshold': 0.0001, 'min_lr': 1e-6}
    # Choose scheduler properties based on the scheduler type
    scheduler_props = multiStepLR_props if scheduler_type == 'MultiStepLR' else reduceLROnPlateau_props
    # Get additional model-specific hyperparameters based on model type
    hyperparams_model = LSTM_PROPERTIES if model_type == 'lstm' else (TCNBLOCK_PROPERTIES if model_type == 'tcn' else {})
    # Prepare scheduler properties for logging by converting them to a list of strings
    lr_scheduler_props = [f"{k}={v}" for k, v in scheduler_props.items()]
    optimizer_name = "Adam"  # Specify the optimizer name

    # Data preparation via data_preparation module
    imu_files, gt_files = build_file_lists(args.dataset)  # Build lists of IMU and ground truth files from the dataset directory
    # Load and process data into sequences using the specified window size and stride
    sequence_data, _ = load_and_process_data(args.dataset, imu_files, gt_files, window_size, stride)
    # Split the sequence data into training and validation sets (70% training, 30% validation)
    train_sequences, val_sequences = split_sequences(sequence_data, train_ratio=0.7)
    print(f"\nTotal sequences: {len(sequence_data)}")
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}\n")
    # Print the number of windows in each training sequence for detailed logging
    for i, seq in enumerate(train_sequences):
        print(f"  Training Sequence {i} has {len(seq['x_gyro'])} windows")
    # Print the number of windows in each validation sequence for detailed logging
    for i, seq in enumerate(val_sequences):
        print(f"  Validation Sequence {i} has {len(seq['x_gyro'])} windows")

    # Concatenate sequences and build datasets for training and validation
    train_dataset = concat_and_build_dataset(train_sequences)
    val_dataset = concat_and_build_dataset(val_sequences)
    print(f"\nTotal training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}\n")

    # Log hyperparameter details to the console for verification
    print(f"window_size: {window_size}, stride: {stride}, batch_size: {batch_size}, num_epochs: {num_epochs}, learning_rate: {learning_rate}")
    if scheduler_type == 'MultiStepLR':
        print(f"lr_scheduler: {scheduler_type}, milestones: {scheduler_props['milestones']}, gamma: {scheduler_props['gamma']}")
    else:
        print(f"lr_scheduler: {scheduler_type}, mode: {scheduler_props['mode']}, factor: {scheduler_props['factor']}, patience: {scheduler_props['patience']}, threshold: {scheduler_props['threshold']}, min_lr: {scheduler_props['min_lr']}")

    # Create data loaders for training and validation datasets with the specified batch size
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size=batch_size)
    print(f"Creating model: {model_type}...")
    model = create_model(model_type=model_type)  # Initialize the model based on the specified model type
    model.to(device)  # Move the model to the configured device (CPU or GPU)

    # Initialize the loss function and optimizer
    criterion = CustomMultiLossLayer(nb_outputs=2)  # Use a custom loss layer that handles multiple outputs (position and quaternion)
    optimizer = Adam(model.parameters(), lr=learning_rate)  # Initialize the Adam optimizer with model parameters and learning rate

    # Initialize the learning rate scheduler based on the specified scheduler type
    if scheduler_props['type'] == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, milestones=scheduler_props['milestones'], gamma=scheduler_props['gamma'])
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode=scheduler_props['mode'], factor=scheduler_props['factor'],
                                      patience=scheduler_props['patience'], threshold=scheduler_props['threshold'],
                                      min_lr=scheduler_props['min_lr'])

    # Generate a timestamp and define a checkpoint path for saving the best model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = f'{args.output}_{timestamp}.pt'
    # Create a dictionary of hyperparameters to log and pass to the training loop
    hyperparams = {
        'model_type': model_type,
        'sequence_length': window_size,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'lr_scheduler': lr_scheduler_props,
        'optimizer': optimizer_name,
        'timestamp': timestamp
    }
    hyperparams.update(hyperparams_model)  # Update with model-specific hyperparameters

    # Train the model using the training loop and capture the training metrics
    metrics = train_loop(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, scheduler, checkpoint_path, hyperparams)

    # Plot the training history using the collected metrics
    plot_training_history(metrics, output_prefix=f'{args.output}_{timestamp}')


if __name__ == '__main__':
    main()
