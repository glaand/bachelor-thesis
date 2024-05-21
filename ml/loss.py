import torch
import torch.nn.functional as F

def loss_cosine_similarity(pred_error, true_error, residual, grid_size_x, grid_size_y):
    # grid-like to vector-like
    true_error = true_error.view(-1, grid_size_x*grid_size_y)
    pred_error = pred_error.view(-1, grid_size_x*grid_size_y)
    
    # Compute cosine similarity
    cosine_similarity = F.cosine_similarity(pred_error, true_error, dim=1)  # Along the batch dimension
    
    # Compute the distance from 1.0
    alignment_loss = 1.0 - cosine_similarity
    
    # Take mean over the spatial dimensions and then mean across the batch
    alignment_loss = torch.mean(alignment_loss)
    
    # Total loss
    total_loss = alignment_loss
    
    return total_loss


def loss_mse(pred_error, true_error, residual, grid_size_x, grid_size_y):
    # Compute the mean squared error
    mse = F.mse_loss(pred_error, true_error)
    
    # Total loss
    total_loss = mse
    
    return total_loss


def loss_rmse(pred_error, true_error, residual, grid_size_x, grid_size_y):
    # Compute the root mean squared error
    mse = F.mse_loss(pred_error, true_error)
    rmse = torch.sqrt(mse)
    
    # Total loss
    total_loss = rmse
    
    return total_loss
