import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target,ctrl_pts,control_points):

        # Compute the MSE loss between the output and target
        mse_loss_image = self.mse(output, target)

        # print(f"image:{mse_loss_image}")
        
        # compute the euclidean distance between the control points
        mse_loss_control_points = self.mse(ctrl_pts,control_points[:,:-1,:].flatten(1))

        # print(f"ctrl:{mse_loss_control_points}")
        # # print the loss of the image
        # print(f"image:{mse_loss_image}")
        # # their ratio as well
        # print(f"ratio:{mse_loss_control_points/mse_loss_image}")
        # print(f"ctrl:{mse_loss_control_points}")

        # let's combine the two losses
        return mse_loss_control_points + mse_loss_image*2
    # def vae_loss(recon_x, x, mu, log_var):
    #     # Calculate the binary cross-entropy loss
    #     MSE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
    #     #print(f'MSE', MSE)
    #     # KL divergence
    #     KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    #     #print(f'KLD', KLD)
    #     return MSE + KLD
class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.mse = nn.MSELoss()
    def forward(self, recon_x,x,mu,log_var):
        # Compute the MSE loss between the output and target
        mse_loss_image = self.mse(recon_x, x)
        # print(f"image:{mse_loss_image}")
        # KL divergence
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # print(f"KLD:{KLD}")
        return mse_loss_image + KLD

