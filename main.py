from dataloader.loader import CustomDataset
from torch.utils.data import DataLoader
from model.mapping import ConstrainedAutoEncoder,VAE
from executor.train import train
from executor.train import load_model_epoch
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from ops.signed_distance import curve_point_nurbs
from configreader.configreader import parse_config
import sys
from evaluations.evaluator import calculate_lpips,calculate_psnr,calculate_ssim
import imageio
from model.loss import CustomLoss,VAELoss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# Assuming you have a function to convert torch tensor to NumPy array
def to_numpy(tensor):
    return tensor.cpu().detach().numpy() if tensor.requires_grad else tensor.cpu().numpy()
# the file is in experiments/splinetofield/executor/train.py

def extract_boundary_points(sdf, threshold=0.0125):
    """
    Extract boundary points from the signed distance field.

    Args:
        sdf (ndarray): Signed distance field.
        threshold (float): Threshold value to determine the boundary points.

    Returns:
        tuple: Arrays of x and y coordinates of the boundary points.
    """
    boundary_points = np.argwhere(np.abs(sdf) < threshold)
    x_coords = boundary_points[:, 1]
    y_coords = boundary_points[:, 0]
    return x_coords, y_coords
def compute_area_inside_bbox_numerical(x_coords, y_coords):
    """
    Compute the area inside the bounding box defined by the boundary points using numerical integration.

    Args:
        x_coords (ndarray): Array of x coordinates of the boundary points.
        y_coords (ndarray): Array of y coordinates of the boundary points.

    Returns:
        float: Area inside the bounding box.
    """
    if len(x_coords) == 0 or len(y_coords) == 0:
        return 0.0
    
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    
    # Define the grid for numerical integration
    x_grid = np.linspace(min_x, max_x, num=len(x_coords))
    y_grid = np.linspace(min_y, max_y, num=len(y_coords))
    
    # Compute the area using the trapezoidal rule
    area = np.trapz(y_grid, x=x_grid)
    
    return area

def train_func(batch_size,num_epochs,train_data_cp,train_data_sdf,test_data_cp,test_data_sdf,save_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConstrainedAutoEncoder(train_data_sdf[0][0].shape,latent_dim=16,output_size=128).to(device)
    print(count_parameters(model))
    # model = VAE(train_data_cp.shape).to(device)
    exit(1)
    custom_dataset = CustomDataset(train_data_cp, train_data_sdf,False)
    train_loader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=True)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = VAELoss()
    
    train(train_loader, model, optimizer, criterion, device,num_epochs,save_path,test_data_cp,test_data_sdf)

def test_func_visualize(test_data_cp,test_data_sdf,save_path):
    print("Testing the model")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConstrainedAutoEncoder(test_data_cp[0][0].shape,latent_dim=16,output_size=128).to(device)
    # load model
    model = load_model_epoch(model,save_path)
    model.eval()
    model.to(device)
    test_data_cp=test_data_cp.to(device)
    test_data_sdf=test_data_sdf.to(device)
    # # evaluate on test data
    print("Evaluating on test data")
    computed_sdf,ctrl_pts = model(test_data_sdf.unsqueeze(dim=1))
    print(computed_sdf.shape)
    ############################################################################################################
    ctrl_pts_np = to_numpy(ctrl_pts)
    test_data_cp_np = to_numpy(test_data_cp)

    # ctrl_pts_np is of shape [batch_size,16]
    # reshape it to [batch_size,8,2]
    ctrl_pts_np = ctrl_pts_np.reshape(test_data_cp_np.shape[0],8,2)

    # repeat the first control point to make it 9 control points
    ctrl_pts_np = np.concatenate([ctrl_pts_np,np.expand_dims(ctrl_pts_np[:,0],axis=1)],axis=1)


    # # # evaluate on test data
    print("Evaluating on test data")


    image_data = []  # Use a local variable to collect images
    # mse between the true and predicted control points
    # use numpy to calculate the mse
    mse = np.mean(np.square(test_data_cp_np - ctrl_pts_np))
    print(f"MSE between true and predicted control points: {mse:.10f}")
    selected_indices = [i for i in range(10)]
    area_diff = []
    for i in range(len(selected_indices)):


        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        # plot the true curve

        # use the above code to plot the true curve
        x_data_i = test_data_cp_np[selected_indices[i]]
        # Generate polygon
        polygon = curve_point_nurbs(x_data_i)
        # Plot the polygon
        axs[2].add_patch(Polygon(polygon, edgecolor='red', fill=None, linewidth=2))
        axs[2].set_xlim(-1, 1)
        axs[2].set_ylim(-1, 1)
        axs[2].set_title(f'Nurbs - Sample {selected_indices[i]}')

        # plot the predicted curve
        x_data_i = ctrl_pts_np[selected_indices[i]] 
        # Generate polygon
        polygon = curve_point_nurbs(x_data_i)

        # Plot the polygon
        axs[1].add_patch(Polygon(polygon, edgecolor='red', fill=None, linewidth=2))
        axs[1].set_xlim(-1, 1)
        axs[1].set_ylim(-1, 1)
        axs[1].set_title(f'Predicted Nurbs - Sample {selected_indices[i]}')


        # plot the true signed distance field
        
        y_data_i_true = to_numpy(test_data_sdf[selected_indices[i]])
        # print(y_data_i_true.shape)
        img_true = axs[0].imshow(y_data_i_true,origin='lower')
        axs[0].set_title(f'Input True Signed Distance - Sample {selected_indices[i]}')
        axs[0].set_xlim(-1, 1)
        axs[0].set_ylim(-1, 1)
        # compute area for the true as well
        x_coords, y_coords = extract_boundary_points(y_data_i_true)
        # compute the area inside the bounding box
        area_true = compute_area_inside_bbox_numerical(x_coords, y_coords)
        # plot the predicted signed distance field

        y_data_i_pred = to_numpy(computed_sdf[i][0])

        x_coords, y_coords = extract_boundary_points(y_data_i_pred)
        # compute the area inside the bounding box
        area_pred = compute_area_inside_bbox_numerical(x_coords, y_coords)
        axs[3].set_xlim(-1, 1)
        axs[3].set_ylim(-1, 1)
        axs[3].set_title(f'Predicted Signed Distance - Sample {selected_indices[i]}')

        
        # add the area difference and append to the list
        area_diff.append(np.abs(area_true - area_pred)/area_true)

        plt.tight_layout()  
        plt.savefig(f'me592_experiments/bb_test_visualization_both{i}.png')
        image_data.append(f'me592_experiments/bb_test_visualization_both{i}.png')
        plt.close()
        
    # filenames = [f'test_results_visualization_{i}.png' for i in range(len(selected_indices))]
    # # Convert the images into a GIF
    images = [imageio.v2.imread(filename) for filename in image_data]
    imageio.mimsave('me592_experiments/vv_test_visualization_full_model.gif', images, duration=300)  # Adjust fps (frames per second) as needed
    # compute the mse area_diff
    area_diff = np.mean(area_diff)
    print(f"Mean area difference between true and predicted signed distance fields: {area_diff:.10f}")
    exit(1)
    # load the npy file and plot the curve and the signed distance field
    # predicted by model
    control_points = np.load("/work/mech-ai-scratch/samundra/experiments/me592_project/code/notebooks/move_in_control_points.npy")

    # control_points = torch.tensor(control_points,dtype=torch.float32).to(device)
    # remove the last control point to make it 8 control points
    control_points_model = control_points[:,:-1]

    control_points_model = torch.tensor(control_points_model,dtype=torch.float32).to(device)
    print(control_points_model.shape)
    computed_sdf = model.decoder(control_points_model.reshape((15, 16)))

    # convert to numpy

    computed_sdf = to_numpy(computed_sdf)
    image_data = []  # Use a local variable to collect images
    # for all the images in the batch
    for i in range(computed_sdf.shape[0]):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # plot the true curve
        x_data_i = control_points[i]
        # Generate polygon
        polygon = curve_point_nurbs(x_data_i)
        # Plot the polygon
        axs[0].add_patch(Polygon(polygon, edgecolor='red', fill=None, linewidth=2))
        axs[0].set_xlim(-1, 1)
        axs[0].set_ylim(-1, 1)
        axs[0].set_title(f'Squeezing Curve- Sample {i}')

        # plot the predicted signed distance field
        y_data_i_pred = computed_sdf[i][0]
        img_pred = axs[1].imshow(y_data_i_pred, origin='lower')
        axs[1].set_title(f'Predicted Signed Distance - Sample {i}')

        plt.tight_layout()
        image_data.append(f'me592_experiments/squeezing plot{i}.png')
        plt.savefig(f'me592_experiments/squeezing plot{i}.png')
        # plt.close()
     # filenames = [f'test_results_visualization_{i}.png' for i in range(len(selected_indices))]

    # # Convert the images into a GIF
    images = [imageio.v2.imread(filename) for filename in image_data]
    imageio.mimsave('me592_experiments/squeezing.gif', images, duration=50)  # Adjust fps (frames per second) as needed

    # plot the true curve and the predicted signed distance field



    # compute the mean ssim
    
    # print(f"Mean ssim is {ssim}")
    # # Select 10 test samples
    # selected_indices = [0, 1, 2, 3]
    # selected_test_data_cp = test_data_cp[selected_indices]
    # selected_test_data_sdf = test_data_sdf[selected_indices]

    # # Evaluate on selected test data
    # computed_sdf = model(selected_test_data_cp.to(device))
    # computed_sdf_np = to_numpy(computed_sdf)

    # Create a figure with subplots for each sample in the batch


    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # for i in range(len(selected_indices)):
    #     # Clear the previous plots
    #     for ax in axs:
    #         ax.clear()

    #     # Extract control points and reshape signed distance function
    #     x_data_i = to_numpy(selected_test_data_cp[i])
    #     y_data_i_true = to_numpy(selected_test_data_sdf[i])
    #     y_data_i_pred = computed_sdf_np[i]

    #     # Generate polygon
    #     polygon = curve_point_nurbs(x_data_i)

    #     # Plot the polygon
    #     axs[0].add_patch(Polygon(polygon, edgecolor='red', fill=None, linewidth=2))
    #     axs[0].set_xlim(-1, 1)
    #     axs[0].set_ylim(-1, 1)
    #     axs[0].set_title(f'Nurbs - Sample {selected_indices[i]}')

    #     # Plot the true signed distance field
    #     img_true = axs[1].imshow(y_data_i_true,origin='lower')
    #     axs[1].set_title(f'True Signed Distance - Sample {selected_indices[i]}')

    #     # Plot the predicted signed distance field
    #     img_pred = axs[2].imshow(y_data_i_pred[0], origin='lower')
    #     axs[2].set_title(f'Predicted Signed Distance - Sample {selected_indices[i]}')

    #     # Save the plot as an image file
    #     plt.tight_layout()
    #     filename = f'test_results_visualization_{i}.png'
    #     plt.savefig(filename)

    # # Create a list of filenames
    # # Create a list of filenames
    # filenames = [f'test_results_visualization_{i}.png' for i in range(len(selected_indices))]

    # # Convert the images into a GIF
    # images = [imageio.v2.imread(filename) for filename in filenames]
    # imageio.mimsave('test_results_visualization.gif', images, fps=1)  # Adjust fps (frames per second) as needed
    # fig, axs = plt.subplots(len(selected_indices), 3, figsize=(15, 5 * len(selected_indices)))
    
    # for i in range(len(selected_indices)):
    #     # Extract control points and reshape signed distance function
    #     x_data_i = to_numpy(selected_test_data_cp[i])
    #     y_data_i_true = to_numpy(selected_test_data_sdf[i])
    #     y_data_i_pred = computed_sdf_np[i]

    #     # Generate polygon
    #     polygon = curve_point_nurbs(x_data_i)

    #     # Plot the polygon
    #     axs[i, 0].add_patch(Polygon(polygon, edgecolor='red', fill=None, linewidth=2))
    #     axs[i, 0].set_xlim(-1, 1)
    #     axs[i, 0].set_ylim(-1, 1)
    #     axs[i, 0].set_title(f'Nurbs - Sample {selected_indices[i]}')

    #     # Plot the true signed distance field
    #     img_true = axs[i, 1].imshow(y_data_i_true,origin='lower')
    #     axs[i, 1].set_title(f'True Signed Distance - Sample {selected_indices[i]}')

    #     # Plot the predicted signed distance field
    #     img_pred = axs[i, 2].imshow(y_data_i_pred[0], origin='lower')
    #     axs[i, 2].set_title(f'Predicted Signed Distance - Sample {selected_indices[i]}')

    # Show the plot
    # plt.tight_layout()
    # plt.savefig('full_visualization.png')
    # plt.show()
    # plot the results

def test_func_compute(test_loader,save_path,test_data_cp):

    print("Testing the model")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConstrainedAutoEncoder(test_data_cp[0][0].shape, latent_dim=16, output_size=128).to(device)
    # Load model
    model = load_model_epoch(model, save_path)
    model.eval()
    model.to(device)

    # Lists to store results
    psnr_values = []
    ssim_values = []
    lpips_values = []
    mse_errors = []


    # Iterate over batches in the test loader
    for batch_idx, batch in enumerate(test_loader):
        print(f"Evaluating batch {batch_idx + 1}/{len(test_loader)}")

        print("Evaluating on test data")
        test_data_cp = batch['control_point'].to(device)
        test_data_sdf = batch['signed_distance_function'].to(device)

        # Evaluate on test data

        computed_sdf, ctrl_pts = model(test_data_sdf.unsqueeze(dim=1))

        # Convert computed SDF and control points to NumPy
        computed_sdf_np = to_numpy(computed_sdf)
        test_data_sdf_np= to_numpy(test_data_sdf)

        # Compute PSNR and SSIM for the batch
        batch_psnr = calculate_psnr(computed_sdf_np, test_data_sdf_np)
        batch_ssim = calculate_ssim(computed_sdf_np, test_data_sdf_np)
        batch_lpips = calculate_lpips(computed_sdf, test_data_sdf)

        # Compute MSE error between true and predicted control points for the batch
        batch_mse_error = np.mean(np.square(test_data_sdf_np - computed_sdf_np))

        # Append batch results to lists
        psnr_values.append(batch_psnr)
        ssim_values.append(batch_ssim)
        mse_errors.append(batch_mse_error)
        lpips_values.append(batch_lpips)

        # calculate the mse between the true and predicted control points
        # use the mse function
        # mse = torch.nn.MSELoss()
        # mse_val = mse(ctrl_pts, test_data_cp)



        # del test_data_cp
        # del test_data_sdf
        # del computed_sdf


    # Compute mean PSNR, SSIM, and MSE error across all batches
    mean_psnr = np.mean(psnr_values)
    mean_ssim = np.mean(ssim_values)
    mean_mse_error = np.mean(mse_errors)

    # Print mean values
    print(f"Mean PSNR across batches: {mean_psnr}")
    print(f"Mean SSIM across batches: {mean_ssim}")
    print(f"Mean squared error between true and predicted sdf: {mean_mse_error}")

    # Compute mean LPIPS across all batches
    mean_lpips = np.mean(lpips_values)
    print(f"Mean LPIPS across batches: {mean_lpips}")

# def test_func_visualize(test_data_cp,test_data_sdf,save_path):
#     print("Testing the model")
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model = VAE(test_data_cp.shape).to(device)
#     # load model
#     model = load_model_epoch(model,save_path)
#     model.eval()
#     model.to(device)
#     test_data_cp=test_data_cp.to(device)
#     test_data_sdf=test_data_sdf.to(device)
#     # # evaluate on test data
#     print("Evaluating on test data")
#     computed_sdf,_,_ = model(test_data_cp)
#     print(computed_sdf.shape)
#     ############################################################################################################

#     test_data_cp_np = to_numpy(test_data_cp)

#     selected_indices = [i for i in range(10)]
#     image_data = []  # Use a local variable to collect images
#     for i in range(len(selected_indices)):


#         fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#         # plot the true curve

#         # use the above code to plot the true curve
#         x_data_i = test_data_cp_np[selected_indices[i]]
#         # Generate polygon
#         polygon = curve_point_nurbs(x_data_i)
#         # Plot the polygon
#         axs[0].add_patch(Polygon(polygon, edgecolor='red', fill=None, linewidth=2))
#         axs[0].set_xlim(-1, 1)
#         axs[0].set_ylim(-1, 1)
#         axs[0].set_title(f'Nurbs - Sample {selected_indices[i]}')



#         # plot the true signed distance field

#         y_data_i_pred = to_numpy(computed_sdf[i][0])
#         print(y_data_i_pred.shape)
#         x_coords, y_coords = extract_boundary_points(y_data_i_pred)
#         img_pred = axs[1].imshow(y_data_i_pred, origin='lower')
#         axs[1].scatter(x_coords, y_coords, c='red', s=1)
#         axs[1].set_title(f'Predicted Signed Distance - Sample {selected_indices[i]}')

#         plt.tight_layout()  
#         plt.savefig(f'me592_experiments/curve_vae_visualization_both{i}.png')
#         image_data.append(f'me592_experiments/curve_vae_visualization_both{i}.png')
#         plt.close()
        
#     # filenames = [f'test_results_visualization_{i}.png' for i in range(len(selected_indices))]
#     # # Convert the images into a GIF
#     images = [imageio.v2.imread(filename) for filename in image_data]
#     imageio.mimsave('me592_experiments/curve_vae_visualization_full_model.gif', images, duration=300)  # Adjust fps (frames per second) as needed
#     exit(1)
#     # load the npy file and plot the curve and the signed distance field
#     # predicted by model
#     control_points = np.load("/work/mech-ai-scratch/samundra/experiments/me592_project/code/notebooks/move_in_control_points.npy")

#     # control_points = torch.tensor(control_points,dtype=torch.float32).to(device)
#     # remove the last control point to make it 8 control points
#     control_points_model = control_points[:,:-1]

#     control_points_model = torch.tensor(control_points_model,dtype=torch.float32).to(device)
#     print(control_points_model.shape)
#     computed_sdf = model.decoder(control_points_model.reshape((15, 16)))

#     # convert to numpy

#     computed_sdf = to_numpy(computed_sdf)
#     image_data = []  # Use a local variable to collect images
#     # for all the images in the batch
#     for i in range(computed_sdf.shape[0]):
#         fig, axs = plt.subplots(1, 2, figsize=(10, 5))

#         # plot the true curve
#         x_data_i = control_points[i]
#         # Generate polygon
#         polygon = curve_point_nurbs(x_data_i)
#         # Plot the polygon
#         axs[0].add_patch(Polygon(polygon, edgecolor='red', fill=None, linewidth=2))
#         axs[0].set_xlim(-1, 1)
#         axs[0].set_ylim(-1, 1)
#         axs[0].set_title(f'Squeezing Curve- Sample {i}')

#         # plot the predicted signed distance field
#         y_data_i_pred = computed_sdf[i][0]
#         img_pred = axs[1].imshow(y_data_i_pred, origin='lower')
#         axs[1].set_title(f'Predicted Signed Distance - Sample {i}')

#         plt.tight_layout()
#         image_data.append(f'me592_experiments/squeezing plot{i}.png')
#         plt.savefig(f'me592_experiments/squeezing plot{i}.png')
#         # plt.close()
#      # filenames = [f'test_results_visualization_{i}.png' for i in range(len(selected_indices))]

#     # # Convert the images into a GIF
#     images = [imageio.v2.imread(filename) for filename in image_data]
#     imageio.mimsave('me592_experiments/squeezing.gif', images, duration=50)  # Adjust fps (frames per second) as needed

#     # plot the true curve and the predicted signed distance field



#     # compute the mean ssim
    
#     # print(f"Mean ssim is {ssim}")
#     # # Select 10 test samples
#     # selected_indices = [0, 1, 2, 3]
#     # selected_test_data_cp = test_data_cp[selected_indices]
#     # selected_test_data_sdf = test_data_sdf[selected_indices]

#     # # Evaluate on selected test data
#     # computed_sdf = model(selected_test_data_cp.to(device))
#     # computed_sdf_np = to_numpy(computed_sdf)

#     # Create a figure with subplots for each sample in the batch


#     # fig, axs = plt.subplots(1, 3, figsize=(15, 5))

#     # for i in range(len(selected_indices)):
#     #     # Clear the previous plots
#     #     for ax in axs:
#     #         ax.clear()

#     #     # Extract control points and reshape signed distance function
#     #     x_data_i = to_numpy(selected_test_data_cp[i])
#     #     y_data_i_true = to_numpy(selected_test_data_sdf[i])
#     #     y_data_i_pred = computed_sdf_np[i]

#     #     # Generate polygon
#     #     polygon = curve_point_nurbs(x_data_i)

#     #     # Plot the polygon
#     #     axs[0].add_patch(Polygon(polygon, edgecolor='red', fill=None, linewidth=2))
#     #     axs[0].set_xlim(-1, 1)
#     #     axs[0].set_ylim(-1, 1)
#     #     axs[0].set_title(f'Nurbs - Sample {selected_indices[i]}')

#     #     # Plot the true signed distance field
#     #     img_true = axs[1].imshow(y_data_i_true,origin='lower')
#     #     axs[1].set_title(f'True Signed Distance - Sample {selected_indices[i]}')

#     #     # Plot the predicted signed distance field
#     #     img_pred = axs[2].imshow(y_data_i_pred[0], origin='lower')
#     #     axs[2].set_title(f'Predicted Signed Distance - Sample {selected_indices[i]}')

#     #     # Save the plot as an image file
#     #     plt.tight_layout()
#     #     filename = f'test_results_visualization_{i}.png'
#     #     plt.savefig(filename)

#     # # Create a list of filenames
#     # # Create a list of filenames
#     # filenames = [f'test_results_visualization_{i}.png' for i in range(len(selected_indices))]

#     # # Convert the images into a GIF
#     # images = [imageio.v2.imread(filename) for filename in filenames]
#     # imageio.mimsave('test_results_visualization.gif', images, fps=1)  # Adjust fps (frames per second) as needed
#     # fig, axs = plt.subplots(len(selected_indices), 3, figsize=(15, 5 * len(selected_indices)))
    
#     # for i in range(len(selected_indices)):
#     #     # Extract control points and reshape signed distance function
#     #     x_data_i = to_numpy(selected_test_data_cp[i])
#     #     y_data_i_true = to_numpy(selected_test_data_sdf[i])
#     #     y_data_i_pred = computed_sdf_np[i]

#     #     # Generate polygon
#     #     polygon = curve_point_nurbs(x_data_i)

#     #     # Plot the polygon
#     #     axs[i, 0].add_patch(Polygon(polygon, edgecolor='red', fill=None, linewidth=2))
#     #     axs[i, 0].set_xlim(-1, 1)
#     #     axs[i, 0].set_ylim(-1, 1)
#     #     axs[i, 0].set_title(f'Nurbs - Sample {selected_indices[i]}')

#     #     # Plot the true signed distance field
#     #     img_true = axs[i, 1].imshow(y_data_i_true,origin='lower')
#     #     axs[i, 1].set_title(f'True Signed Distance - Sample {selected_indices[i]}')

#     #     # Plot the predicted signed distance field
#     #     img_pred = axs[i, 2].imshow(y_data_i_pred[0], origin='lower')
#     #     axs[i, 2].set_title(f'Predicted Signed Distance - Sample {selected_indices[i]}')

#     # Show the plot
#     # plt.tight_layout()
#     # plt.savefig('full_visualization.png')
#     # plt.show()
#     # plot the results

# def test_func_compute(test_loader,save_path,test_data_cp):

#     print("Testing the model")
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model = ConstrainedAutoEncoder(test_data_cp[0][0].shape, latent_dim=16, output_size=128).to(device)
#     # Load model
#     model = load_model_epoch(model, save_path)
#     model.eval()
#     model.to(device)

#     # Lists to store results
#     psnr_values = []
#     ssim_values = []
#     lpips_values = []
#     mse_errors = []


#     # Iterate over batches in the test loader
#     for batch_idx, batch in enumerate(test_loader):
#         print(f"Evaluating batch {batch_idx + 1}/{len(test_loader)}")

#         print("Evaluating on test data")
#         test_data_cp = batch['control_point'].to(device)
#         test_data_sdf = batch['signed_distance_function'].to(device)

#         # Evaluate on test data

#         computed_sdf, ctrl_pts = model(test_data_sdf.unsqueeze(dim=1))

#         # Convert computed SDF and control points to NumPy
#         computed_sdf_np = to_numpy(computed_sdf)
#         test_data_sdf_np= to_numpy(test_data_sdf)

#         # Compute PSNR and SSIM for the batch
#         batch_psnr = calculate_psnr(computed_sdf_np, test_data_sdf_np)
#         batch_ssim = calculate_ssim(computed_sdf_np, test_data_sdf_np)
#         batch_lpips = calculate_lpips(computed_sdf, test_data_sdf)

#         # Compute MSE error between true and predicted control points for the batch
#         batch_mse_error = np.mean(np.square(test_data_sdf_np - computed_sdf_np))

#         # Append batch results to lists
#         psnr_values.append(batch_psnr)
#         ssim_values.append(batch_ssim)
#         mse_errors.append(batch_mse_error)
#         lpips_values.append(batch_lpips)

#         # calculate the mse between the true and predicted control points
#         # use the mse function
#         # mse = torch.nn.MSELoss()
#         # mse_val = mse(ctrl_pts, test_data_cp)



#         # del test_data_cp
#         # del test_data_sdf
#         # del computed_sdf


#     # Compute mean PSNR, SSIM, and MSE error across all batches
#     mean_psnr = np.mean(psnr_values)
#     mean_ssim = np.mean(ssim_values)
#     mean_mse_error = np.mean(mse_errors)

#     # Print mean values
#     print(f"Mean PSNR across batches: {mean_psnr}")
#     print(f"Mean SSIM across batches: {mean_ssim}")
#     print(f"Mean squared error between true and predicted sdf: {mean_mse_error}")

#     # Compute mean LPIPS across all batches
#     mean_lpips = np.mean(lpips_values)
#     print(f"Mean LPIPS across batches: {mean_lpips}")




if __name__ == '__main__':
    config_file_path = sys.argv[1]
    print(config_file_path)
    mode, batch_size, learning_rate, epochs, root_directory, save_path = parse_config(config_file_path)
    
    if mode == 'train':
        train_data_cp, test_data_cp, train_data_sdf, test_data_sdf = CustomDataset.return_test_train(root_directory)
        print("Using 10K images for training and 2K images for testing")
        train_func(batch_size,epochs,train_data_cp,train_data_sdf,test_data_cp,test_data_sdf,save_path)

        # create a gif  first  50 cp and 50 sdf to visualize the data 
        # test_func_visualize(test_data_cp[:50],test_data_sdf[:50],save_path)



    elif mode == 'test':
        batch_size = 64
        test_data_cp, test_data_sdf = CustomDataset.return_test_only(root_directory)
        custom_dataset = CustomDataset(test_data_cp, test_data_sdf,False)
        test_loader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=False)
        print("Using Remaining 2K images")
        # test_func_compute(test_loader,save_path,test_data_cp)
        test_func_visualize(test_data_cp,test_data_sdf,save_path)
    else:
        print("Mode not supported")

