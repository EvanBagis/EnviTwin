import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import folium
from scipy.spatial import KDTree, cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import getpass
import tarfile
import os
import gc
from tqdm import tqdm
import json
from dateutil.tz import tz
from datetime import datetime, timedelta, timezone
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

import segmentation_models_pytorch as smp

import tifffile

from xgboost import XGBClassifier

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.classification import Dice

from geopy.distance import geodesic

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import os
project_root = '/home/envitwin/Desktop/venvs/EnviTwin/Operational_LULC/'


def extract_lon_lat_from_tiff(tiff_path):
    # Open the GeoTIFF file
    with rasterio.open(tiff_path) as dataset:
        # Read the altitude (elevation) data
        elevation_data = dataset.read(1)  # First band
        
        # Get the transform to compute coordinates
        transform = dataset.transform
        
        # Get the bounds of the image
        bounds = dataset.bounds
        print(f"Bounds of the TIFF: {bounds}")
        
        # Get the resolution (pixel size)
        resolution = (transform[0], transform[4])
        print(f"Resolution of the TIFF: {resolution}")
        
        # Generate latitude and longitude arrays
        rows, cols = elevation_data.shape
        latitudes = np.linspace(bounds.top, bounds.bottom, rows)
        longitudes = np.linspace(bounds.left, bounds.right, cols)
        
        # Create a meshgrid of latitudes and longitudes
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
        
        return lon_grid, lat_grid
        
def find_closest_coordinates_and_check(points_list, meshgrid_lat, meshgrid_lon, max_distance_km=0.001):
    """
    Find closest coordinates and check if the match is valid by comparing the distance to the closest point.

    Args:
        points_list (list of tuples): List of lat, lon coordinates to compare. [(lat1, lon1), (lat2, lon2), ...]
        meshgrid_lat (2D array): 2D array of latitudes from the meshgrid.
        meshgrid_lon (2D array): 2D array of longitudes from the meshgrid.
        max_distance_km (float): Maximum acceptable distance (in km) for a match to be considered valid.

    Returns:
        list of tuples: Closest coordinates from the meshgrid for each point in the list.
    """
    # Flatten the meshgrid and prepare KDTree
    mesh_points = np.column_stack((meshgrid_lat.ravel(), meshgrid_lon.ravel()))
    tree = KDTree(mesh_points)

    closest_points = []
    valid_flags = []  # Flags to track valid matches
    mismatches = []  # Track points that were incorrectly matched

    for point in points_list:
        lat, lon = point
        
        # Find the index of the nearest meshgrid point using KDTree
        _, idx = tree.query(point, eps=1e-8)
        closest_lat = mesh_points[idx][0]
        closest_lon = mesh_points[idx][1]
        
        # Calculate the initial distance between the query point and the matched point
        initial_distance = np.sqrt((lat - closest_lat)**2 + (lon - closest_lon)**2)

        # If the initial distance is within the acceptable range, accept the KDTree match
        if initial_distance <= max_distance_km:
            closest_points.append((closest_lat, closest_lon))

    return closest_points

def match_coords(path, label_file, lat, lon):
    label_coords = pd.read_csv(path + label_file)
    if 'Unnamed: 2' in label_coords.columns:
        label_coords = label_coords.drop('Unnamed: 2', axis=1)

    label_list = []
    for i in range(len(label_coords)):
        label_list.append(tuple(label_coords.iloc[i,:]))
    
    closest = find_closest_coordinates_and_check(label_list, lat, lon)
        
    label_list = pd.DataFrame(label_list, columns=['lat', 'lon'])
    closest = pd.DataFrame(closest, columns=['lat', 'lon'])
    return closest

def coordinates_to_mesh_indices(coords_list, meshgrid_lat, meshgrid_lon):
    """
    Convert a list of coordinates into indices from lat-lon meshes.

    Args:
        coords_list (list of tuples): List of (latitude, longitude) coordinates.
        meshgrid_lat (2D array): 2D array of latitude values.
        meshgrid_lon (2D array): 2D array of longitude values.

    Returns:
        list of tuples: List of (row_index, col_index) for each coordinate.
    """
    label_list = []
    for i in range(len(coords_list)):
        #print(i,label_coords.iloc[i,:])
        label_list.append(tuple(coords_list.iloc[i,:]))
    
    # Flatten the meshgrid and prepare KDTree
    mesh_points = np.column_stack((meshgrid_lat.ravel(), meshgrid_lon.ravel()))
    tree = cKDTree(mesh_points)

    # Query the KDTree for the closest point for each coordinate
    indices = []; dist = []
    for coord in label_list:
        d, idx = tree.query(coord)  # Get the flattened index
        # Convert the flat index to 2D indices
        row, col = np.unravel_index(idx, meshgrid_lat.shape)
        indices.append((row, col)); dist.append(d)

    return indices, dist

def replace_nan_inf_with_mean(arr):
    """
    Replace NaN and Inf values in a NumPy array with the mean of the valid values.

    Args:
        arr (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Array with NaN and Inf values replaced by the mean.
    """
    # Copy the array to avoid modifying the original
    arr_clean = np.copy(arr)
    
    # Create a mask for valid values (not NaN and not Inf)
    valid_mask = np.isfinite(arr_clean)
    
    # Calculate the mean of valid values
    valid_mean = arr_clean[valid_mask].mean()
    
    # Replace NaN and Inf values with the mean
    arr_clean[~valid_mask] = valid_mean
    
    return arr_clean

class UNet(nn.Module):
    def __init__(self, in_channels, out_classes):
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU6(),
                nn.Dropout(0.1),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU6(),
                nn.Dropout(0.1)
            )

        def up_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.ReLU6(),
                #nn.Dropout(0.2)
            )
        multiplier = 20
        self.encoder1 = conv_block(in_channels, multiplier)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = conv_block(multiplier, 2*multiplier)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.middle = conv_block(2*multiplier, 4*multiplier)
        
        self.up2 = up_conv(4*multiplier, 2*multiplier)
        self.decoder2 = conv_block(4*multiplier, 2*multiplier)
        self.up1 = up_conv(2*multiplier, multiplier)
        self.decoder1 = conv_block(2*multiplier, multiplier)

        self.final_conv = nn.Conv2d(multiplier, out_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        middle = self.middle(self.pool2(enc2))

        dec2 = self.up2(middle)
        diffY2 = enc2.size()[2] - dec2.size()[2]
        diffX2 = enc2.size()[3] - dec2.size()[3]
        dec2 = nn.functional.pad(dec2, (diffX2 // 2, diffX2 - diffX2 // 2, diffY2 // 2, diffY2 - diffY2 // 2))
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.up1(dec2)
        diffY1 = enc1.size()[2] - dec1.size()[2]
        diffX1 = enc1.size()[3] - dec1.size()[3]
        dec1 = nn.functional.pad(dec1, (diffX1 // 2, diffX1 - diffX1 // 2, diffY1 // 2, diffY1 - diffY1 // 2))
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.final_conv(dec1)

from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class EarlyStopping:
    """Stops training when the validation loss increases and saves the best predictions & true labels."""
    def __init__(self, patience=30, save_path="best_model.pth", pred_save_path=None, reference_tif=None):
        """
        Args:
            patience (int): Number of epochs to wait before stopping after an increase.
            save_path (str): Path to save the best model.
            pred_save_path (str): Path to save the best predictions as a TIFF file.
            reference_tif (str): Path to a reference TIFF file to use for georeferencing the prediction.
        """
        self.patience = patience
        self.best_loss = np.inf  # Best validation loss starts as infinity
        self.counter = 0
        self.save_path = save_path
        self.pred_save_path = pred_save_path
        self.reference_tif = reference_tif
        self.best_predictions = None  # Store best predictions
        self.true_labels = None  # Store true labels

    def __call__(self, val_loss, model, predictions, true_labels, full_predictions=None):
        """Check if training should stop based on validation loss."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)  # Save best model
            if self.pred_save_path and self.reference_tif and full_predictions is not None:
                save_prediction_as_tiff(full_predictions, self.reference_tif, self.pred_save_path)
            self.best_predictions = predictions.detach().cpu().numpy()  # Store best predictions
            self.true_labels = true_labels.detach().cpu().numpy()  # Store true labels
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"⏹️ Early stopping triggered after {self.patience} epochs of increasing loss!")
                return True, self.best_predictions, self.true_labels  # Stop training
        return False, None, None  # Continue training


def scale_array(arr, method="standard"):
    """
    Scales a 2D NumPy array using Min-Max scaling or Standardization (Z-score normalization).
    
    Parameters:
        arr (np.ndarray): Input 2D NumPy array.
        method (str): Scaling method. Either 'minmax' (default) or 'standard'.
    
    Returns:
        np.ndarray: Scaled 2D array.
    """
    
    if method == "minmax":
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val == min_val:
            return np.zeros_like(arr)  # Avoid division by zero
        return (arr - min_val) / (max_val - min_val)
    
    elif method == "standard":
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return np.zeros_like(arr)  # Avoid division by zero
        return (arr - mean) / std

def plot_LULC(preds, fold):
    colors = ['green', 'brown', 'darkgreen', 'purple', 'blue', 'orange', 'red', 'black', 'lightgray']  # Add colors for all unique integers
    cmap = ListedColormap(colors)# 'lightgray','red',
    
    # Define bounds and normalization (to map integers to colors correctly)
    bounds = np.arange(len(colors) + 1)  # One more than the number of colors
    norm = BoundaryNorm(bounds, cmap.N)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(preds.cpu().numpy(), cmap=cmap, norm=norm)
    
    cbar = plt.colorbar(ticks=np.arange(len(colors)))
    cbar.ax.set_yticklabels(['green_urban_space', 'agriculture',#,
                  'forest', 'open_spaces','water','cement_roofs', 'roof_tiles', 'industrial',
                             'roads'])
    plt.title("Land Use Visualization")
    
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.savefig(project_root + '/plots/' + f'LU_pred_unet_{fold}.png', dpi=1500)
    #plt.show()

import rasterio
from rasterio.transform import Affine

def save_prediction_as_tiff(pred_array, reference_tif, out_path, dtype='uint8'):
    import numpy as np
    import torch
    import rasterio

    # Convert torch tensor to numpy if needed
    if isinstance(pred_array, torch.Tensor):
        pred_array = pred_array.detach().cpu().numpy()

    with rasterio.open(reference_tif) as ref:
        transform = ref.transform
        crs = ref.crs
        height, width = ref.height, ref.width

    if pred_array.shape != (height, width):
        raise ValueError(
            f"Shape mismatch: prediction {pred_array.shape} vs reference ({height}, {width})"
        )

    with rasterio.open(
        out_path,
        'w',
        driver='GTiff',
        height=pred_array.shape[0],
        width=pred_array.shape[1],
        count=1,  # single-band
        dtype=dtype,
        crs=crs,
        transform=transform,
        #compress='lzw'
    ) as dst:
        dst.write(pred_array.astype(dtype), 1)

    print(f"✅ Saved prediction to {out_path}")

s2_path = '/home/envitwin/Desktop/venvs/EnviTwin/Operational_LULC/data/S2/2025-09-19/3115394842281513757/'
s1_path = '/home/envitwin/Desktop/venvs/EnviTwin/Operational_LULC/data/S1/2025-09-20/3115394842281513757/'

# extract lat lon
lat, lon = extract_lon_lat_from_tiff(s2_path + 'B01.tif')

label_files = ['green_urban_space.txt', 'agri2.txt',#,'roads.txt',
              'forest.txt', 'open_spaces.txt','Water_500.txt', 'cement_roofs.txt', 
               'roof_tiles.txt', 'Industrial.txt', 'roads_g.txt']#'roof_tiles.txt',
#label_files = ['cement_roofs.txt','roof_tiles.txt', 'industrial.txt']
    
# match the coordintates
labels_matched = {}
for file in label_files:
    labels_matched[file] = match_coords(project_root + '/labels/', file, lon, lat)

# Get indices
labels_ind = {}
for label in labels_matched:
    indices, _ = coordinates_to_mesh_indices(labels_matched[label], lon, lat)
    labels_ind[label] = indices
#print(labels_ind)

# construct X data from the sentinel bands
s2_bands = ["B01","B02","B03","B04","B05","B06", "B07","B08","B09","B11","B12"]
s1_bands = ["VH", "VV"]

bands_list = []
bands_dict = {}

# Load S2 bands
for band in s2_bands:
    with rasterio.open(s2_path + band + '.tif') as dataset:
        temp = dataset.read(1)
        transform = dataset.transform
        crs = dataset.crs
        bounds = dataset.bounds
        bands_list.append(scale_array(temp))
        bands_dict[band] = temp

# Load S1 bands
for band in s1_bands:
    with rasterio.open(s1_path + band + '.tif') as dataset:
        temp = dataset.read(1)
        bands_list.append(scale_array(temp))
        bands_dict[band] = temp

#print(np.stack(bands_list, axis=-1).shape)

# NDVI = b8-b4/(b8+b4)
ndvi = (bands_dict["B08"] - bands_dict["B04"])/(bands_dict["B08"]+bands_dict["B04"]+1e-8)
bands_list.append(scale_array(replace_nan_inf_with_mean(ndvi)))

# NDBI = b11-b8/(b11+b8)
ndbi = (bands_dict["B11"] - bands_dict["B08"])/(bands_dict["B11"]+bands_dict["B08"]+1e-8)
bands_list.append(scale_array(replace_nan_inf_with_mean(ndbi)))

# base soil index = (b11+b4)-(b8+b2)/((b11+b4)+(b8+b2))
bsi = ((bands_dict["B11"] + bands_dict["B04"])-(bands_dict["B08"] + bands_dict["B02"]))/((bands_dict["B11"]+bands_dict["B04"])+(bands_dict["B08"] + bands_dict["B02"])+1e-8)
bands_list.append(scale_array(replace_nan_inf_with_mean(bsi)))

#green chlorophyll index = (b8/b3)-1
gci = (bands_dict["B08"]/bands_dict["B03"]+1e-8) -1
bands_list.append(scale_array(replace_nan_inf_with_mean(gci)))

# moisture stress index = b11/b8
msi = bands_dict["B11"]/(bands_dict["B08"]+1e-8)
bands_list.append(scale_array(replace_nan_inf_with_mean(msi)))

ndwi = (bands_dict["B03"] - bands_dict["B08"])/(bands_dict["B03"]+bands_dict["B08"]+1e-8)
bands_list.append(scale_array(replace_nan_inf_with_mean(ndwi)))

bu = ndbi - ndvi
bands_list.append(scale_array(replace_nan_inf_with_mean(bu)))

# modified normalized difference water index
ndwi2 = (bands_dict["B03"] - bands_dict["B11"])/(bands_dict["B03"]+bands_dict["B11"]+1e-8)
bands_list.append(scale_array(replace_nan_inf_with_mean(ndwi2)))

# NDVIre
NDVIre = (bands_dict["B05"] - bands_dict["B04"])/(bands_dict["B05"]+bands_dict["B04"]+1e-8)
bands_list.append(scale_array(replace_nan_inf_with_mean(NDVIre)))

# normalized difference tillage index NDTI
NDTI = (bands_dict["B11"] - bands_dict["B12"])/(bands_dict["B11"]+bands_dict["B12"]+1e-8)
bands_list.append(scale_array(replace_nan_inf_with_mean(NDTI)))

# building height
with rasterio.open(project_root + '/auxiliary/building_height_resampled_nearest.tif') as dataset:
    bh_data = dataset.read(1)
    bands_list.append(scale_array(bh_data))

stack_bands = np.stack(bands_list, axis=-1)
print(stack_bands.shape)

x_np = stack_bands

y_np = np.ones((x_np.shape[0], x_np.shape[1]))*999
for i, file in enumerate(label_files):
    temp = pd.DataFrame(labels_ind[file], columns=['rows', 'cols'])    
    y_np[temp.loc[:,'rows'], temp.loc[:,'cols']] = i

out_classes = 9

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mask = (y_np != 999)#.to(device)
valid_indices = np.argwhere(mask).T
#criterion = nn.CrossEntropyLoss()

alpha=torch.tensor([0.1, 0.05, 0.1, 0.2, 0.01, 0.1, 0.1, 0.05, 0.1])
criterion = FocalLoss(gamma=2, alpha=alpha) # g=0.001, F1=0.81

K = 10
kf_y = {f"fold = {k}":[] for k in range(K)}
kf_pred = {f"fold = {k}":[] for k in range(K)}
epochs = 1001

kf = KFold(n_splits=K, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(valid_indices.T)):

    torch.cuda.empty_cache()
    gc.collect()
    
    # Delete any remaining references
    for obj in list(globals().keys()):
        if isinstance(globals()[obj], torch.Tensor):
            #print(globals()[obj])
            del globals()[obj]
    
    torch.cuda.empty_cache()
    gc.collect()

    x = torch.tensor(x_np, dtype=torch.float32, requires_grad=True).permute(2,0,1).unsqueeze(0)
    y = torch.tensor(y_np, dtype=torch.long)
    
    height, width = y.shape
    #pad_h = (32 - height % 32) if height % 32 != 0 else 0
    #pad_w = (32 - width % 32) if width % 32 != 0 else 0
    #x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
    #y = F.pad(y, (0, pad_w, 0, pad_h), mode='constant', value=999)
    
    mask = (y != 999)#.to(device)
    valid_indices = np.argwhere(mask).T 
    in_channels = x.shape[1]

    x = x.to(device); y = y.to(device)
    mask = mask.to(device)
    #valid_indices = torch.argwhere(mask).T 
    
    early_stopping = EarlyStopping(patience=5,
                                   save_path=project_root + '/models/' + f'best_model_fold_{fold}.pth',
                                   pred_save_path=project_root + '/preds/' + f'best_prediction_fold_{fold}.tif',
                                   reference_tif=s2_path + 'B01.tif')
    print(f"\n🚀 Fold {fold+1}/{K}")
    model = UNet(in_channels, out_classes).to(device)
    #model = smp.Segformer(               # MAnet, dpn68
    #    encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #    encoder_weights="swsl",     # use `imagenet` pre-trained weights for encoder initialization
    #    in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #   classes=out_classes,  
    #   decoder_use_batchnorm=False,# model output channels (number of classes in your dataset)
    #    aux_params={"classes":9, "pooling":"avg","dropout":0.9}
    #).to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("total params = ", pytorch_total_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # Map back to 2D coordinates
    train_coords = valid_indices[train_idx]  # (num_train, 2) -> (row, col)
    val_coords = valid_indices[val_idx]  # (num_val, 2) -> (row, col)
    # Reconstruct masks
    train_mask = np.zeros(y.shape, dtype=bool)
    val_mask = np.zeros(y.shape, dtype=bool)

    train_mask[train_coords[:, 0], train_coords[:, 1]] = True
    val_mask[val_coords[:, 0], val_coords[:, 1]] = True

    train_mask = torch.tensor(train_mask).to(device)
    val_mask = torch.tensor(val_mask)#.to(device)
    # Training loop
    
    outputs = torch.zeros(y.shape, device=device)
    for epoch in range(epochs):
        
        model.train()
        optimizer.zero_grad()
    
        # Forward pass
        outputs = model(x)
        #outputs = outputs[0]
        #print(outputs)
        loss = criterion(outputs[:,:,train_mask], y[train_mask])  # Shape: (1834, 1672)
        loss.backward(retain_graph=True)
        optimizer.step()
    
        #print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
        #print(torch.argmax(outputs, dim=1).squeeze(0).shape)
        #final_pred = torch.argmax(outputs, dim=1).squeeze(0).detach()
    
        if epoch%10 == 0:

            model.eval()
            with torch.no_grad():
                y_val = y[val_mask]  # Corresponding labels for validation pixels
                outputs_val = model(x).detach().cpu()
                #outputs_val = outputs_val[0].detach().cpu()
                val_pred2 = torch.argmax(outputs_val, dim=1).squeeze(0).cpu()  # Get predictions for the validation set
                val_pred = val_pred2[val_mask].cpu()
                val_loss = criterion(outputs_val[:,:,val_mask], y.cpu()[val_mask])
                print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss}")
                # Calculate confusion matrix for validation
                cm = confusion_matrix(y_val.cpu().numpy(), val_pred.cpu().numpy())
                print(f"Confusion Matrix (Epoch {epoch + 1}): \n{cm}")
                stop, predictions, true_labels = early_stopping(val_loss.item(), model, val_pred, y_val, full_predictions=val_pred2)
                plot_LULC(val_pred2, fold)
                
                del outputs_val, val_pred, val_pred2, val_loss
                torch.cuda.empty_cache()
                gc.collect()
                if stop:
                    print(classification_report(true_labels, predictions))
                    kf_y[f"fold = {fold}"] = true_labels
                    kf_pred[f"fold = {fold}"] = predictions
                    break  # Stop training
            
        del loss, outputs
        torch.cuda.empty_cache()
        gc.collect()
        
    del model, train_mask, val_mask, optimizer
    torch.cuda.empty_cache()
    gc.collect()

final_y = np.concatenate([kf_y[i] for i in kf_y.keys()])
final_p = np.concatenate([kf_pred[i] for i in kf_pred.keys()])
cm = confusion_matrix(final_y, final_p)

sns.heatmap(cm, annot=True); plt.show()

print(f"Confusion Matrix (Epoch {epoch + 1}): \n{cm}")
print(classification_report(final_y, final_p))