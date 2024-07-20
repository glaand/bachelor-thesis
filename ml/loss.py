import torch
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

def loss_cosine_similarity(pred_error, true_error, residual, grid_size_x, grid_size_y):
    # grid-like to vector-like
    true_error = true_error.view(-1, grid_size_x*grid_size_y)
    pred_error = pred_error.view(-1, grid_size_x*grid_size_y)
    
    # Compute cosine similarity using PyTorch built-in function
    cosine_similarity = F.cosine_similarity(true_error, pred_error, dim=1)
    
    # Subtract the range [-1, 1] to [1, 0]
    cosine_similarity = (-1*(cosine_similarity - 1.0))/2.0
    
    # Take mean over the spatial dimensions and then mean across the batch
    alignment_loss = torch.mean(cosine_similarity)
    
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


def loss_huber(pred_error, true_error, residual, grid_size_x, grid_size_y):
    # Compute the Huber loss
    huber = F.huber_loss(pred_error, true_error)
    
    # Total loss
    total_loss = huber
    
    return total_loss

def loss_ssim(pred_error, true_error, residual, grid_size_x, grid_size_y):
    # Compute the mean structural similarity index

    # Scale pred_error to range of true_error
    total_loss = 0
    for i in range(pred_error.shape[0]):
        left = pred_error[i].cpu().detach().unsqueeze(0)
        right = true_error[i].cpu().detach().unsqueeze(0)
        print(left.shape, right.shape)
        total_loss += 1 - ssim(left, right, data_range=right.max()-right.min())

    total_loss = total_loss/pred_error.shape[0]
    
    return total_loss