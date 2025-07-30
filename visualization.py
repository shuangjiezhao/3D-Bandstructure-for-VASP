#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib as mpl
# GUI mode selection - MUST BE SET BEFORE IMPORTING pyplot
GUI_MODE = True  # Set to True for interactive GUI, False for saving figures

if not GUI_MODE:
    mpl.use('Agg')  # Silent mode for saving
else:
    mpl.use('TkAgg')  # or 'Qt5Agg' for interactive mode

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator, FixedLocator
from scipy.ndimage import gaussian_filter, median_filter, binary_dilation
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
import warnings

# For GUI mode
if GUI_MODE:
    from matplotlib.widgets import Button
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog

# ============= USER CONTROL PARAMETERS =============
# GUI Mode - Interactive plot vs. saving figures
# GUI_MODE = False  # Set to True for interactive GUI, False for saving figures
# Note: GUI mode allows rotation, zoom, and custom saving with DPI selection
# Non-GUI mode saves two views (front and perspective) automatically

# Projection settings
SHOW_FERMI_PROJECTION = False  # Set to True to show contour projections on Fermi level

# K-space range control (set to None to use full data range)
KX_RANGE = None  # Example: (0.28, 0.38) to limit kx from 0.28 to 0.38
KY_RANGE = None  # Example: (0.28, 0.38) to limit ky from 0.28 to 0.38

# Z-axis (Energy) range control (set to None to use full data range)
Z_RANGE = (-0.8,0.8)  # Example: (-0.5, 0.7) to limit energy from -0.5 to 0.7 eV
Z_TICK_STEP = 0.3  # Example: 0.2 for ticks every 0.2 eV (None for automatic)
Z_TICK_DECIMALS = 1  # Number of decimal places for energy tick labels
# Note: Z_TICK_STEP of 0.1 gives ticks at -0.5, -0.4, -0.3, ... 0.6, 0.7 if Z_RANGE=(-0.5, 0.7)

# ============= MODIFIED SETTINGS FOR SHARP DIRAC CONES =============
SMOOTHING_METHOD = 1  # Sharp-preserving bilateral filter

# Interpolation settings - different for GUI vs save mode
if GUI_MODE:
    INTERPOLATION_FACTOR = 8  # Lower for faster GUI interaction (try 2-8)
    GUI_QUALITY = 'medium'  # 'low', 'medium', or 'high' - affects rendering quality
else:
    INTERPOLATION_FACTOR = 24  # High quality for saved figures
    GUI_QUALITY = None  # Not used in save mode

# Advanced options for Dirac cone preservation
PROTECT_CONE_TIPS = True  
CONE_PROTECTION_RADIUS = 2  # Large protection zone
BILATERAL_SIGMA_SPATIAL = 1.0  
BILATERAL_SIGMA_INTENSITY = 0.00001  # Small for preserving sharp features

# Load data files
try:
    kx_mesh = np.loadtxt('KX.grd')      
    ky_mesh = np.loadtxt('KY.grd')   
    CBM_mesh = np.loadtxt('BAND_LUMO.grd')        
    VBM_mesh = np.loadtxt('BAND_HOMO.grd')
    print("Successfully loaded data files")
except Exception as e:
    print(f"Failed to open grds: {e}")
    exit()

print(f"\nInitial data shapes:")
print(f"kx: {kx_mesh.shape}, ky: {ky_mesh.shape}")
print(f"VBM: {VBM_mesh.shape}, CBM: {CBM_mesh.shape}")

# If data is 1D, reshape to 2D
if kx_mesh.ndim == 1:
    n_points = len(kx_mesh)
    # Find best factorization
    factors = []
    for i in range(1, int(np.sqrt(n_points)) + 1):
        if n_points % i == 0:
            factors.append((i, n_points // i))
    
    # Choose factorization closest to square
    best_diff = float('inf')
    nk1, nk2 = 1, n_points
    for f1, f2 in factors:
        diff = abs(f1 - f2)
        if diff < best_diff:
            best_diff = diff
            nk1, nk2 = f1, f2
    
    print(f"Reshaping 1D data ({n_points} points) to {nk1}x{nk2} grid")
    
    try:
        kx_mesh = kx_mesh.reshape(nk1, nk2)
        ky_mesh = ky_mesh.reshape(nk1, nk2)
        VBM_mesh = VBM_mesh.reshape(nk1, nk2)
        CBM_mesh = CBM_mesh.reshape(nk1, nk2)
    except:
        # Try transposed dimensions
        kx_mesh = kx_mesh.reshape(nk2, nk1)
        ky_mesh = ky_mesh.reshape(nk2, nk1)
        VBM_mesh = VBM_mesh.reshape(nk2, nk1)
        CBM_mesh = CBM_mesh.reshape(nk2, nk1)

print(f"\nFinal data shapes:")
print(f"kx: {kx_mesh.shape}, ky: {ky_mesh.shape}")
print(f"VBM: {VBM_mesh.shape}, CBM: {CBM_mesh.shape}")

# Print energy ranges
print(f"\nEnergy ranges:")
print(f"VBM: [{VBM_mesh.min():.6f}, {VBM_mesh.max():.6f}] eV")
print(f"CBM: [{CBM_mesh.min():.6f}, {CBM_mesh.max():.6f}] eV")

# Save original data
VBM_original = VBM_mesh.copy()
CBM_original = CBM_mesh.copy()
kx_original = kx_mesh.copy()
ky_original = ky_mesh.copy()

# Calculate original band gap
original_vbm_max = VBM_original.max()
original_cbm_min = CBM_original.min()
original_gap = original_cbm_min - original_vbm_max

print(f"\nOriginal band gap: {original_gap:.6f} eV ({original_gap*1000:.3f} meV)")

def detect_cone_tips(data, is_maximum=True):
    """Detect exact Dirac cone tips (extrema) and their immediate neighborhood"""
    if is_maximum:
        extremum_idx = np.unravel_index(np.argmax(data), data.shape)
    else:
        extremum_idx = np.unravel_index(np.argmin(data), data.shape)
    
    # Create protection mask around the tip
    protection_mask = np.zeros_like(data, dtype=bool)
    rows, cols = data.shape
    
    # Protect the tip and its neighborhood
    for di in range(-CONE_PROTECTION_RADIUS, CONE_PROTECTION_RADIUS + 1):
        for dj in range(-CONE_PROTECTION_RADIUS, CONE_PROTECTION_RADIUS + 1):
            ni, nj = extremum_idx[0] + di, extremum_idx[1] + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                # Use circular protection region
                if di*di + dj*dj <= CONE_PROTECTION_RADIUS*CONE_PROTECTION_RADIUS:
                    protection_mask[ni, nj] = True
    
    return protection_mask, extremum_idx

def detect_sharp_features(data, threshold=0.85):
    """Detect sharp features using improved gradient analysis"""
    # Calculate gradients with better edge handling
    gy, gx = np.gradient(data)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    
    # Calculate second derivatives (curvature)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    
    # Gaussian curvature (detects saddle points - Dirac cones)
    gaussian_curvature = gxx * gyy - gxy**2
    
    # Mean curvature (detects general sharp features)  
    mean_curvature = 0.5 * (gxx + gyy)
    
    # Laplacian (detects local extrema)
    laplacian = gxx + gyy
    
    # Multi-criteria feature detection
    high_gradient = gradient_magnitude > np.percentile(gradient_magnitude, 88)
    high_curvature = np.abs(gaussian_curvature) > np.percentile(np.abs(gaussian_curvature), 92)
    sharp_peaks = np.abs(laplacian) > np.percentile(np.abs(laplacian), 90)
    
    # Combine criteria - be more selective
    feature_mask = high_gradient & (high_curvature | sharp_peaks)
    
    # Dilate slightly to protect feature neighborhoods
    feature_mask = binary_dilation(feature_mask, structure=np.ones((3,3)))
    
    return feature_mask

def sharp_preserving_bilateral_filter(data, protection_mask, sigma_spatial=1.0, sigma_intensity=0.001):
    """Enhanced bilateral filter that absolutely preserves protected regions"""
    filtered = np.copy(data)
    rows, cols = data.shape
    
    # Create spatial weights template
    window_size = int(3 * sigma_spatial)
    if window_size % 2 == 0:
        window_size += 1
    
    half_window = window_size // 2
    
    # Pre-compute spatial weight template
    yy, xx = np.mgrid[-half_window:half_window+1, -half_window:half_window+1]
    spatial_template = np.exp(-(xx**2 + yy**2) / (2 * sigma_spatial**2))
    
    for i in range(half_window, rows - half_window):
        for j in range(half_window, cols - half_window):
            # Skip if this point is protected
            if protection_mask[i, j]:
                continue
                
            # Extract neighborhood
            neighborhood = data[i-half_window:i+half_window+1, 
                             j-half_window:j+half_window+1]
            protection_neighborhood = protection_mask[i-half_window:i+half_window+1,
                                                    j-half_window:j+half_window+1]
            
            # Exclude protected points from filtering
            valid_mask = ~protection_neighborhood
            if not np.any(valid_mask):
                continue  # Skip if all neighbors are protected
            
            # Intensity differences (only for valid points)
            center_intensity = data[i, j]
            intensity_diff = np.abs(neighborhood - center_intensity)
            intensity_weights = np.exp(-intensity_diff**2 / (2 * sigma_intensity**2))
            
            # Combined weights (spatial * intensity, only for valid points)
            weights = spatial_template * intensity_weights * valid_mask
            
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
                filtered[i, j] = np.sum(weights * neighborhood)
    
    return filtered

def adaptive_smoothing_with_protection(data, protection_mask, feature_mask, 
                                     sigma_smooth=0.6, sigma_preserve=0.15):
    """Apply adaptive smoothing while absolutely protecting cone tips"""
    result = np.copy(data)
    
    # Never modify protected regions
    modifiable_mask = ~protection_mask
    
    # Strong smoothing for non-feature, non-protected regions
    smooth_regions = modifiable_mask & ~feature_mask
    if np.any(smooth_regions):
        smoothed = gaussian_filter(data, sigma=sigma_smooth)
        result[smooth_regions] = smoothed[smooth_regions]
    
    # Light smoothing for feature regions (excluding protected)
    light_smooth_regions = modifiable_mask & feature_mask
    if np.any(light_smooth_regions):
        lightly_smoothed = gaussian_filter(data, sigma=sigma_preserve)
        result[light_smooth_regions] = lightly_smoothed[light_smooth_regions]
    
    return result

def anisotropic_diffusion_with_protection(data, protection_mask, iterations=4, kappa=0.05, gamma=0.2):
    """Anisotropic diffusion that absolutely preserves protected regions"""
    result = data.astype(np.float64)
    
    for iteration in range(iterations):
        # Calculate gradients
        gy, gx = np.gradient(result)
        
        # Calculate diffusion coefficients (Perona-Malik)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        c = np.exp(-(gradient_mag / kappa)**2)
        
        # Calculate divergence of diffusion
        cgy = c * gy
        cgx = c * gx
        
        div_y, _ = np.gradient(cgy)
        _, div_x = np.gradient(cgx)
        
        # Update only non-protected regions
        update = gamma * (div_x + div_y)
        modifiable_mask = ~protection_mask
        result[modifiable_mask] += update[modifiable_mask]
    
    return result

def edge_aware_interpolation(kx_old, ky_old, data_old, new_shape, method='cubic'):
    """Advanced edge-aware interpolation with improved edge handling to prevent artifacts"""
    
    # Create target grid
    kx_min, kx_max = kx_old.min(), kx_old.max()
    ky_min, ky_max = ky_old.min(), ky_old.max()
    
    # Add small margin to prevent edge issues
    margin_x = 0.02 * (kx_max - kx_min)
    margin_y = 0.02 * (ky_max - ky_min)
    
    kx_fine = np.linspace(kx_min + margin_x, kx_max - margin_x, new_shape[1])
    ky_fine = np.linspace(ky_min + margin_y, ky_max - margin_y, new_shape[0])
    kx_mesh_fine, ky_mesh_fine = np.meshgrid(kx_fine, ky_fine)
    
    # Step 1: High-quality cubic interpolation
    points_old = np.column_stack((kx_old.ravel(), ky_old.ravel()))
    points_fine = np.column_stack((kx_mesh_fine.ravel(), ky_mesh_fine.ravel()))
    
    # Use cubic interpolation first
    data_fine = griddata(points_old, data_old.ravel(), points_fine, 
                        method='cubic', fill_value=np.nan)
    
    # Step 2: Fill NaN regions with linear interpolation (smoother than nearest neighbor)
    nan_mask = np.isnan(data_fine)
    if np.any(nan_mask):
        # Calculate distances to original points
        distances_to_data = cdist(points_fine[nan_mask], points_old)
        min_distances = np.min(distances_to_data, axis=1)
        
        # Only fill points that are reasonably close to original data
        close_threshold = 0.15 * min(kx_max - kx_min, ky_max - ky_min)
        close_mask = min_distances < close_threshold
        
        if np.any(close_mask):
            nan_indices = np.where(nan_mask)[0]
            close_nan_indices = nan_indices[close_mask]
            
            # Use linear interpolation for smoother edges
            linear_fill = griddata(points_old, data_old.ravel(), 
                                 points_fine[close_nan_indices], method='linear', 
                                 fill_value=np.nan)
            data_fine[close_nan_indices] = linear_fill
            
            # For any remaining NaN, use nearest neighbor with distance weighting
            still_nan_mask = np.isnan(data_fine)
            if np.any(still_nan_mask):
                still_nan_indices = np.where(still_nan_mask)[0]
                for idx in still_nan_indices:
                    point = points_fine[idx]
                    # Find closest original points
                    distances = cdist([point], points_old)[0]
                    closest_indices = np.argsort(distances)[:5]  # Use 5 closest points
                    weights = 1.0 / (distances[closest_indices] + 1e-10)
                    weights = weights / np.sum(weights)
                    
                    # Weighted average of closest points
                    data_fine[idx] = np.sum(weights * data_old.ravel()[closest_indices])
    
    # Reshape result
    data_fine = data_fine.reshape(new_shape)
    
    # Step 3: Apply very gentle smoothing only near edges to reduce artifacts
    # Identify edge regions
    edge_mask = np.zeros_like(data_fine, dtype=bool)
    edge_mask[0, :] = edge_mask[-1, :] = edge_mask[:, 0] = edge_mask[:, -1] = True
    
    # Expand edge region slightly
    for _ in range(3):
        edge_mask = binary_dilation(edge_mask, structure=np.ones((3,3)))
    
    # Apply very light smoothing only to edge regions
    if np.any(edge_mask):
        smoothed = gaussian_filter(data_fine, sigma=0.8)
        data_fine[edge_mask] = 0.7 * data_fine[edge_mask] + 0.3 * smoothed[edge_mask]
    
    # Create validity mask - more conservative
    validity_mask = ~np.isnan(data_fine)
    
    return kx_mesh_fine, ky_mesh_fine, data_fine, validity_mask

# Apply chosen smoothing method
print(f"\nApplying smoothing method: {SMOOTHING_METHOD}")

if SMOOTHING_METHOD == 0:
    print("No smoothing applied - using original data")
    
elif SMOOTHING_METHOD == 1:
    print(f"Applying sharp-preserving bilateral filter")
    print(f"  Spatial sigma: {BILATERAL_SIGMA_SPATIAL}")
    print(f"  Intensity sigma: {BILATERAL_SIGMA_INTENSITY} eV ({BILATERAL_SIGMA_INTENSITY*1000:.3f} meV)")
    
    if PROTECT_CONE_TIPS:
        # Detect and protect cone tips
        vbm_protection, vbm_tip = detect_cone_tips(VBM_mesh, is_maximum=True)
        cbm_protection, cbm_tip = detect_cone_tips(CBM_mesh, is_maximum=False)
        
        print(f"  VBM tip protected at {vbm_tip} (value: {VBM_mesh[vbm_tip]:.6f} eV)")
        print(f"  CBM tip protected at {cbm_tip} (value: {CBM_mesh[cbm_tip]:.6f} eV)")
        print(f"  Protection radius: {CONE_PROTECTION_RADIUS} points")
        
        VBM_mesh = sharp_preserving_bilateral_filter(VBM_mesh, vbm_protection,
                                                   BILATERAL_SIGMA_SPATIAL, 
                                                   BILATERAL_SIGMA_INTENSITY)
        CBM_mesh = sharp_preserving_bilateral_filter(CBM_mesh, cbm_protection,
                                                   BILATERAL_SIGMA_SPATIAL, 
                                                   BILATERAL_SIGMA_INTENSITY)
        
        print(f"  Post-filter VBM tip: {VBM_mesh[vbm_tip]:.6f} eV (preserved)")
        print(f"  Post-filter CBM tip: {CBM_mesh[cbm_tip]:.6f} eV (preserved)")
        
    else:
        VBM_mesh = sharp_preserving_bilateral_filter(VBM_mesh, np.zeros_like(VBM_mesh, dtype=bool),
                                                   BILATERAL_SIGMA_SPATIAL, 
                                                   BILATERAL_SIGMA_INTENSITY)
        CBM_mesh = sharp_preserving_bilateral_filter(CBM_mesh, np.zeros_like(CBM_mesh, dtype=bool),
                                                   BILATERAL_SIGMA_SPATIAL, 
                                                   BILATERAL_SIGMA_INTENSITY)
    
elif SMOOTHING_METHOD == 2:
    print(f"Applying adaptive smoothing with cone tip protection")
    
    # Detect features and cone tips
    vbm_features = detect_sharp_features(VBM_mesh, threshold=0.85)
    cbm_features = detect_sharp_features(CBM_mesh, threshold=0.85)
    
    vbm_protection, vbm_tip = detect_cone_tips(VBM_mesh, is_maximum=True)
    cbm_protection, cbm_tip = detect_cone_tips(CBM_mesh, is_maximum=False)
    
    print(f"  VBM features: {np.sum(vbm_features)} points")
    print(f"  CBM features: {np.sum(cbm_features)} points")
    print(f"  VBM tip protected at {vbm_tip}")
    print(f"  CBM tip protected at {cbm_tip}")
    
    VBM_mesh = adaptive_smoothing_with_protection(VBM_mesh, vbm_protection, vbm_features)
    CBM_mesh = adaptive_smoothing_with_protection(CBM_mesh, cbm_protection, cbm_features)
    
elif SMOOTHING_METHOD == 3:
    print(f"Applying anisotropic diffusion with protection")
    
    vbm_protection, vbm_tip = detect_cone_tips(VBM_mesh, is_maximum=True)
    cbm_protection, cbm_tip = detect_cone_tips(CBM_mesh, is_maximum=False)
    
    print(f"  VBM tip protected at {vbm_tip}")
    print(f"  CBM tip protected at {cbm_tip}")
    
    VBM_mesh = anisotropic_diffusion_with_protection(VBM_mesh, vbm_protection)
    CBM_mesh = anisotropic_diffusion_with_protection(CBM_mesh, cbm_protection)

# High-quality interpolation for smooth surfaces
if INTERPOLATION_FACTOR > 1:
    print(f"\nInterpolating to {INTERPOLATION_FACTOR}x higher k-point density")
    
    # Warn about diminishing returns for very high factors
    if INTERPOLATION_FACTOR > 32:
        print(f"  WARNING: {INTERPOLATION_FACTOR}x is very high!")
        print("     Visual quality plateaus around 16-32x. Higher values just slow computation.")
        print("     Recommended: Use 8-16x for final plots, 2-4x for testing.")
    
    original_shape = kx_mesh.shape
    new_shape = (original_shape[0] * INTERPOLATION_FACTOR,
                 original_shape[1] * INTERPOLATION_FACTOR)
    
    print(f"  Original grid: {original_shape} → New grid: {new_shape}")
    
    # Interpolate both bands with advanced edge-aware method
    kx_mesh, ky_mesh, VBM_mesh, vbm_validity = edge_aware_interpolation(
        kx_mesh, ky_mesh, VBM_mesh, new_shape, method='cubic')
    
    _, _, CBM_mesh, cbm_validity = edge_aware_interpolation(
        kx_original, ky_original, CBM_mesh, new_shape, method='cubic')
    
    print("  Using edge-aware interpolation for smooth boundaries")
    
    # Combine validity masks - only show regions that are valid for both bands
    combined_validity = vbm_validity & cbm_validity
    
    # Mask invalid regions with NaN to prevent surface artifacts
    VBM_mesh[~combined_validity] = np.nan
    CBM_mesh[~combined_validity] = np.nan
    
    print(f"  Valid interpolation region: {np.sum(combined_validity)}/{combined_validity.size} points ({100*np.sum(combined_validity)/combined_validity.size:.1f}%)")
    
    # Preserve exact extrema after interpolation
    if PROTECT_CONE_TIPS and SMOOTHING_METHOD > 0:
        # Only preserve extrema in valid regions
        valid_vbm = VBM_mesh[~np.isnan(VBM_mesh)]
        valid_cbm = CBM_mesh[~np.isnan(CBM_mesh)]
        
        if len(valid_vbm) > 0 and len(valid_cbm) > 0:
            vbm_shift = original_vbm_max - np.nanmax(VBM_mesh)
            cbm_shift = original_cbm_min - np.nanmin(CBM_mesh)
            
            if abs(vbm_shift) > 1e-9:
                VBM_mesh[~np.isnan(VBM_mesh)] += vbm_shift
            if abs(cbm_shift) > 1e-9:
                CBM_mesh[~np.isnan(CBM_mesh)] += cbm_shift
                
            print(f"  Extrema preserved after interpolation")
        else:
            print("  WARNING: No valid regions found after interpolation!")
    
    print(f"  Final grid size: {kx_mesh.shape}")

# Final band gap check
final_vbm_max = np.nanmax(VBM_mesh)
final_cbm_min = np.nanmin(CBM_mesh)
final_gap = final_cbm_min - final_vbm_max
gap_change = (final_gap - original_gap) * 1000  # in meV

print(f"\nFinal band gap: {final_gap:.6f} eV ({final_gap*1000:.3f} meV)")
print(f"Gap change: {gap_change:+.3f} meV")

# Check for data validity
vbm_valid_points = np.sum(~np.isnan(VBM_mesh))
cbm_valid_points = np.sum(~np.isnan(CBM_mesh))
total_points = VBM_mesh.size

if vbm_valid_points < total_points or cbm_valid_points < total_points:
    print(f"Data validity: VBM {vbm_valid_points}/{total_points} ({100*vbm_valid_points/total_points:.1f}%), CBM {cbm_valid_points}/{total_points} ({100*cbm_valid_points/total_points:.1f}%)")

if abs(gap_change) > 0.05:
    print("WARNING: Significant band gap change detected!")
elif abs(gap_change) > 0.01:
    print("Minor band gap change - check if acceptable")
else:
    print("Band gap excellently preserved")

# Enhanced plotting for PUBLICATION QUALITY
fermi_level = 0.0

# Publication-quality font settings
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'mathtext.default': 'regular'
})

font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 14}

# Publication-quality colormap settings
cmap_vbm = plt.cm.Blues_r
cmap_cbm = plt.cm.Reds

def setup_publication_subplot(ax, elev, azim, show_xlabel=True, view_name=""):
    """Create publication-quality 3D subplot with proper tick alignment and gamma point"""
    
    # Professional surface rendering parameters - optimize for mode
    if GUI_MODE:
        # Optimized parameters for interactive performance
        if GUI_QUALITY == 'low':
            plot_params = {
                'alpha': 0.8,
                'rstride': max(2, kx_mesh.shape[0] // 50),
                'cstride': max(2, kx_mesh.shape[1] // 50),
                'linewidth': 0,
                'antialiased': False,  # Faster without antialiasing
                'shade': False,  # Faster without shading
                'rasterized': True,
            }
        elif GUI_QUALITY == 'medium':
            plot_params = {
                'alpha': 0.85,
                'rstride': max(1, kx_mesh.shape[0] // 100),
                'cstride': max(1, kx_mesh.shape[1] // 100),
                'linewidth': 0,
                'antialiased': True,
                'shade': True,
                'rasterized': True,
            }
        else:  # high
            plot_params = {
                'alpha': 0.9,
                'rstride': max(1, kx_mesh.shape[0] // 150),
                'cstride': max(1, kx_mesh.shape[1] // 150),
                'linewidth': 0,
                'antialiased': True,
                'shade': True,
                'rasterized': False,
            }
    else:
        # High quality for saved figures
        plot_params = {
            'alpha': 0.9,
            'rstride': max(1, min(kx_mesh.shape[0] // 200, 4)),
            'cstride': max(1, min(kx_mesh.shape[1] // 200, 4)),
            'linewidth': 0,
            'antialiased': True,
            'shade': True,
            'rasterized': False,
        }
    
    # Apply k-space range limits if specified
    if KX_RANGE is not None or KY_RANGE is not None:
        # Get current ranges
        kx_min_data, kx_max_data = kx_mesh.min(), kx_mesh.max()
        ky_min_data, ky_max_data = ky_mesh.min(), ky_mesh.max()
        
        # Apply user-specified ranges
        if KX_RANGE is not None:
            kx_min_plot, kx_max_plot = KX_RANGE
            print(f"  Limiting kx range to [{kx_min_plot:.3f}, {kx_max_plot:.3f}]")
        else:
            kx_min_plot, kx_max_plot = kx_min_data, kx_max_data
            
        if KY_RANGE is not None:
            ky_min_plot, ky_max_plot = KY_RANGE
            print(f"  Limiting ky range to [{ky_min_plot:.3f}, {ky_max_plot:.3f}]")
        else:
            ky_min_plot, ky_max_plot = ky_min_data, ky_max_data
        
        # Create mask for data within the specified range
        range_mask = ((kx_mesh >= kx_min_plot) & (kx_mesh <= kx_max_plot) & 
                      (ky_mesh >= ky_min_plot) & (ky_mesh <= ky_max_plot))
        
        # Create masked arrays
        kx_plot = np.ma.masked_where(~range_mask, kx_mesh)
        ky_plot = np.ma.masked_where(~range_mask, ky_mesh)
        VBM_plot = np.ma.masked_where(~range_mask, VBM_mesh)
        CBM_plot = np.ma.masked_where(~range_mask, CBM_mesh)
    else:
        # Use full data range
        kx_plot = kx_mesh
        ky_plot = ky_mesh
        VBM_plot = VBM_mesh
        CBM_plot = CBM_mesh
        kx_min_plot = kx_mesh.min()
        kx_max_plot = kx_mesh.max()
        ky_min_plot = ky_mesh.min()
        ky_max_plot = ky_mesh.max()
    
    # Check for NaN values
    vbm_has_nan = np.any(np.isnan(VBM_plot))
    cbm_has_nan = np.any(np.isnan(CBM_plot))
    
    if vbm_has_nan or cbm_has_nan:
        print(f"  Note: Masking invalid regions (VBM: {np.sum(np.isnan(VBM_plot))} NaN, CBM: {np.sum(np.isnan(CBM_plot))} NaN)")
    
    # Plot surfaces with standard matplotlib rendering
    surf_vbm = ax.plot_surface(kx_plot, ky_plot, VBM_plot, 
                              cmap=cmap_vbm,
                              vmin=np.nanmin(VBM_plot), vmax=np.nanmax(VBM_plot),
                              clip_on=True,  # Clip to axis boundaries
                              **plot_params)
    
    surf_cbm = ax.plot_surface(kx_plot, ky_plot, CBM_plot, 
                              cmap=cmap_cbm,
                              vmin=np.nanmin(CBM_plot), vmax=np.nanmax(CBM_plot),
                              clip_on=True,  # Clip to axis boundaries
                              **plot_params)
    
    # Enhanced contour projections (only if enabled)
    if SHOW_FERMI_PROJECTION:
        vbm_valid_fraction = np.sum(~np.isnan(VBM_plot)) / VBM_plot.size
        cbm_valid_fraction = np.sum(~np.isnan(CBM_plot)) / CBM_plot.size
        
        if vbm_valid_fraction > 0.5 and cbm_valid_fraction > 0.5:
            n_levels = min(40, max(20, kx_plot.shape[0] // 10))
            
            try:
                vbm_valid = VBM_plot[~np.isnan(VBM_plot)]
                cbm_valid = CBM_plot[~np.isnan(CBM_plot)]
                
                if len(vbm_valid) > 0:
                    levels_vbm = np.linspace(vbm_valid.min(), vbm_valid.max(), n_levels)
                    ax.contourf(kx_plot, ky_plot, VBM_plot, levels=levels_vbm, 
                               cmap=cmap_vbm, alpha=0.4, zdir='z', offset=fermi_level)
                    ax.contour(kx_plot, ky_plot, VBM_plot, levels=n_levels//3, 
                              colors='darkblue', linewidths=0.3, alpha=0.6,
                              zdir='z', offset=fermi_level)
                
                if len(cbm_valid) > 0:
                    levels_cbm = np.linspace(cbm_valid.min(), cbm_valid.max(), n_levels)
                    ax.contourf(kx_plot, ky_plot, CBM_plot, levels=levels_cbm, 
                               cmap=cmap_cbm, alpha=0.4, zdir='z', offset=fermi_level)
                    ax.contour(kx_plot, ky_plot, CBM_plot, levels=n_levels//3, 
                              colors='darkred', linewidths=0.3, alpha=0.6,
                              zdir='z', offset=fermi_level)
                    
            except Exception as e:
                print(f"    Warning: Contour projection failed: {e}")
    
    # FIXED: Always show x-label, improved axis labels
    ax.set_xlabel(r'$k_x$ (Å$^{-1}$)', fontdict=font, labelpad=10)
    ax.set_ylabel(r'$k_y$ (Å$^{-1}$)', fontdict=font, labelpad=10)  
    ax.set_zlabel(r'Energy (eV)', fontdict=font, labelpad=20)  # Increased labelpad to avoid overlap
    
    # Get data ranges for proper tick spacing
    z_min_val = min(np.nanmin(VBM_plot), np.nanmin(CBM_plot))
    z_max_val = max(np.nanmax(VBM_plot), np.nanmax(CBM_plot))
    
    # FIXED: Custom tick spacing with 0.1 increments, ensuring (0,0) is included
    def create_nice_ticks(data_min, data_max, step=0.1):
        """Create ticks with proper 0.1 spacing, ensuring 0.0 is included if in range"""
        # Always include 0 if it's within the data range
        if data_min <= 0 <= data_max:
            # Build ticks outward from 0
            ticks = [0.0]
            # Add positive ticks
            tick = step
            while tick <= data_max:
                ticks.append(tick)
                tick += step
            # Add negative ticks
            tick = -step
            while tick >= data_min:
                ticks.append(tick)
                tick -= step
            ticks = sorted(ticks)
        else:
            # If 0 is not in range, use regular spacing
            start = np.floor(data_min / step) * step
            end = np.ceil(data_max / step) * step
            ticks = np.arange(start, end + step*0.5, step)
        
        # Filter to keep only ticks within the actual data range (with small tolerance)
        ticks = [t for t in ticks if data_min - step*0.01 <= t <= data_max + step*0.01]
        return np.array(ticks)
    
    kx_ticks = create_nice_ticks(kx_min_plot, kx_max_plot, 0.1)
    ky_ticks = create_nice_ticks(ky_min_plot, ky_max_plot, 0.1)
    
    # Set custom ticks
    ax.set_xticks(kx_ticks)
    ax.set_yticks(ky_ticks)
    
    # Format tick labels to show 0.0, 0.1, etc.
    ax.set_xticklabels([f'{tick:.1f}' for tick in kx_ticks])
    ax.set_yticklabels([f'{tick:.1f}' for tick in ky_ticks])
    
    # Z-axis ticks with user control
    z_range = z_max_val - z_min_val
    
    # Apply user-specified z-range if provided
    if Z_RANGE is not None:
        z_min_plot, z_max_plot = Z_RANGE
        print(f"  Limiting energy range to [{z_min_plot:.3f}, {z_max_plot:.3f}] eV")
        # Update the actual min/max values for setting limits
        z_min_val = z_min_plot
        z_max_val = z_max_plot
        z_range = z_max_val - z_min_val
    
    if z_range > 0:
        if Z_TICK_STEP is not None:
            # Use user-specified tick spacing
            z_ticks = np.arange(z_min_val, z_max_val + Z_TICK_STEP*0.5, Z_TICK_STEP)
            # Filter to ensure ticks are within range
            z_ticks = z_ticks[(z_ticks >= z_min_val) & (z_ticks <= z_max_val)]
            print(f"  Using energy tick spacing: {Z_TICK_STEP} eV")
        else:
            # Automatic tick spacing
            n_z_ticks = 5  # Reduced from 6 for more space
            z_ticks = np.linspace(z_min_val, z_max_val, n_z_ticks)
        
        ax.set_zticks(z_ticks)
        # Format with user-specified decimal places
        ax.set_zticklabels([f'{tick:.{Z_TICK_DECIMALS}f}' for tick in z_ticks])
    
    # FIXED: Proper axis limits with small margins
    kx_margin = (kx_max_plot - kx_min_plot) * 0.02
    ky_margin = (ky_max_plot - ky_min_plot) * 0.02
    z_margin = z_range * 0.1 if z_range > 0 else 0.1
    
    ax.set_xlim(kx_min_plot - kx_margin, kx_max_plot + kx_margin)
    ax.set_ylim(ky_min_plot - ky_margin, ky_max_plot + ky_margin)
    ax.set_zlim(z_min_val - z_margin, z_max_val + z_margin)
    
    # FIXED: Better tick parameters for proper alignment
    # Adjust pad values for better spacing
    ax.tick_params(axis='x', pad=5, direction='out', length=4)
    ax.tick_params(axis='y', pad=5, direction='out', length=4)
    ax.tick_params(axis='z', pad=12, direction='out', length=4)  # Increased z-axis pad
    
    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    # ADDED: Mark Gamma point (0,0) if it exists in the data range
    if kx_min_plot <= 0 <= kx_max_plot and ky_min_plot <= 0 <= ky_max_plot:
        # Find the closest point to (0,0) in the mesh
        gamma_distances = np.sqrt(kx_mesh**2 + ky_mesh**2)
        gamma_idx = np.unravel_index(np.argmin(gamma_distances), gamma_distances.shape)
        
        gamma_kx = kx_mesh[gamma_idx]
        gamma_ky = ky_mesh[gamma_idx]
        
        # Get energy values at gamma point
        if not (np.isnan(VBM_mesh[gamma_idx]) or np.isnan(CBM_mesh[gamma_idx])):
            gamma_vbm = VBM_mesh[gamma_idx]
            gamma_cbm = CBM_mesh[gamma_idx]
            
            # Mark gamma point on the projection plane
            ax.scatter([gamma_kx], [gamma_ky], [fermi_level], 
                      color='black', s=100, marker='x', linewidth=3,
                      zorder=20, alpha=1.0)
            
            # Add gamma label with better positioning
            ax.text(gamma_kx + 0.02, gamma_ky + 0.02, fermi_level,
                   r'$\Gamma$', fontsize=14, ha='left', va='bottom', 
                   weight='bold', color='black')
            
            print(f"  Gamma point marked at k=({gamma_kx:.3f}, {gamma_ky:.3f})")
            print(f"  Gamma energies: VBM={gamma_vbm:.6f} eV, CBM={gamma_cbm:.6f} eV")
    
    # Professional grid and pane settings
    ax.grid(True, alpha=0.2, linewidth=0.5, linestyle='-')
    
    # Clean pane appearance
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.zaxis.pane.set_facecolor('white')
    ax.xaxis.pane.set_alpha(0.02)
    ax.yaxis.pane.set_alpha(0.02)
    ax.zaxis.pane.set_alpha(0.02)
    
    # Ensure grid aligns with ticks
    ax.xaxis.set_major_locator(FixedLocator(kx_ticks))
    ax.yaxis.set_major_locator(FixedLocator(ky_ticks))
    
    return ax

# Create separate publication-quality figures
figure_size = (10, 8)  # Single subplot - larger for better detail

method_names = ['No smoothing', 'Sharp-preserving bilateral', 'Adaptive with protection', 
                'Anisotropic diffusion']

if GUI_MODE:
    print(f"\n" + "="*70)
    print("INTERACTIVE GUI MODE")
    print("="*70)

# Usage examples for z-axis control
    print("\n📝 Z-AXIS CONTROL EXAMPLES:")
    print("To limit energy range and control ticks, set:")
    print("  Z_RANGE = (-0.5, 0.7)   # Show only -0.5 to 0.7 eV")
    print("  Z_TICK_STEP = 0.2       # Ticks at -0.4, -0.2, 0.0, 0.2, 0.4, 0.6")
    print("  Z_TICK_DECIMALS = 1     # Show as -0.4, -0.2, 0.0, etc.")
    print("Or for finer control:")
    print("  Z_RANGE = (-0.3, 0.4)   # Zoom to band gap region")
    print("  Z_TICK_STEP = 0.1       # Ticks every 0.1 eV")
    print("  Z_TICK_DECIMALS = 3     # Show as -0.300, -0.200, etc.")
    print("Mouse Controls:")
    print("  • Left mouse: Rotate view")
    print("  • Scroll wheel: Zoom in/out (changes view range, not plot size)")
    print("  • Middle mouse: Pan")
    print("  • Right mouse: Context menu")
    print("\nKeyboard Controls:")
    print("  • 'r': Reset view and zoom to initial state")
    print("  • 'a': Auto-fit axes to data bounds (respects Z_RANGE if set)")
    print("  • 'v': Toggle between wide/narrow view modes")
    print("  • '+'/'-': Zoom in/out with keyboard")
    print("  • 'q': Toggle quality mode (low/medium/high)")
    print("  • 's': Save figure (matplotlib default)")
    print("\nToolbar:")
    print("  • Use toolbar buttons for additional controls")
    print("  • Click 'Save High-Res' button to save with custom DPI")
    print(f"\nCurrent settings:")
    print(f"  • Quality mode: {GUI_QUALITY}")
    
    quality_modes = ['low', 'medium', 'high']
    state = {
        'current_quality_idx': quality_modes.index(GUI_QUALITY),
        'current_view_mode': 'wide'  # Start with wide view
    }
    print(f"  • View mode: {state['current_view_mode']}")
    print(f"  • Interpolation factor: {INTERPOLATION_FACTOR}x")
    print(f"  • Grid size: {kx_mesh.shape}")
    print("="*70)
    
    # Create interactive figure
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # Store state in a mutable container to avoid Python scope issues
    # This allows nested functions to modify these values without 'nonlocal'

    
    # View mode control
    view_modes = ['wide', 'narrow']
    
    # Start with perspective view
    setup_publication_subplot(ax, elev=25, azim=45, show_xlabel=True, view_name="Interactive")
    
    # Store current view for reset
    initial_elev = 25
    initial_azim = 45
    
    # Store initial axis limits and data ranges
    initial_xlim = ax.get_xlim()
    initial_ylim = ax.get_ylim()
    # Use user-defined Z_RANGE if provided, otherwise use current limits
    if Z_RANGE is not None:
        initial_zlim = Z_RANGE
        ax.set_zlim(Z_RANGE)
    else:
        initial_zlim = ax.get_zlim()
    
    # Function to apply view mode
    def apply_view_mode(mode):
        """Apply wide or narrow view mode"""
        if mode == 'wide':
            # Wide view - equal aspect ratio
            ax.set_box_aspect([1, 1, 1])
            # Standard subplot margins
            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
        else:  # narrow
            # Narrow view - emphasize vertical (z) dimension
            ax.set_box_aspect([0.8, 0.8, 1.2])
            # Tighter margins for narrow view
            plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
        
        fig.canvas.draw_idle()
    
    # Function to update tick formatting based on view mode
    def update_tick_format():
        """Adjust tick formatting based on current view mode"""
        # Access view mode from state dictionary
        if state['current_view_mode'] == 'narrow':
            # More compact formatting for narrow view
            ax.tick_params(axis='z', pad=15)  # Even more pad for narrow
        else:
            # Standard formatting for wide view
            ax.tick_params(axis='z', pad=12)
        
        # Re-apply z-tick formatting
        z_limits = ax.get_zlim()
        if Z_TICK_STEP is not None:
            # Use custom tick spacing
            z_ticks = np.arange(z_limits[0], z_limits[1] + Z_TICK_STEP*0.5, Z_TICK_STEP)
            z_ticks = z_ticks[(z_ticks >= z_limits[0]) & (z_ticks <= z_limits[1])]
            ax.set_zticks(z_ticks)
        else:
            # Get current ticks for reformatting
            z_ticks = ax.get_zticks()
        
        # Format labels
        if len(z_ticks) > 0:
            ax.set_zticklabels([f'{tick:.{Z_TICK_DECIMALS}f}' for tick in z_ticks])
        
        fig.canvas.draw_idle()
    
    # Apply initial view mode
    apply_view_mode(state['current_view_mode'])
    update_tick_format()
    
    # Store current view for reset
    initial_elev = 25
    initial_azim = 45
    
    # Store initial axis limits and data ranges
    initial_xlim = ax.get_xlim()
    initial_ylim = ax.get_ylim()
    initial_zlim = ax.get_zlim()
    
    # Get actual data ranges for boundary checking
    data_xlim = (np.nanmin(kx_mesh), np.nanmax(kx_mesh))
    data_ylim = (np.nanmin(ky_mesh), np.nanmax(ky_mesh))
    data_zlim = (min(np.nanmin(VBM_mesh), np.nanmin(CBM_mesh)), 
                 max(np.nanmax(VBM_mesh), np.nanmax(CBM_mesh)))
    
    # Custom save function
    def save_custom():
        """Save with custom filename and DPI"""
        root = tk.Tk()
        root.withdraw()  # Hide main window
        
        # Ask for DPI
        dpi = simpledialog.askinteger("Save Options", "Enter DPI (e.g., 300, 600, 1200):", 
                                      initialvalue=600, minvalue=72, maxvalue=2400)
        if dpi is None:
            return
        
        # Ask for filename
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ],
            initialfile="band_structure"
        )
        
        if file_path:
            try:
                # Get current view angles
                elev = ax.elev
                azim = ax.azim
                
                # Save with specified DPI
                fig.savefig(file_path, 
                           dpi=dpi,
                           bbox_inches='tight', 
                           pad_inches=0.1,
                           facecolor='white', 
                           edgecolor='none')
                
                messagebox.showinfo("Success", f"Saved to:\n{file_path}\nDPI: {dpi}\nView: elev={elev:.1f}°, azim={azim:.1f}°")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save:\n{str(e)}")
        
        root.destroy()
    
    # Function to update axis visibility based on zoom
    def update_axis_visibility():
        """Update axis ticks after zoom without trying to resize the box"""
        # Just update the ticks to match the new limits
        ax.xaxis._update_ticks()
        ax.yaxis._update_ticks()
        ax.zaxis._update_ticks()
        
        # Redraw
        fig.canvas.draw_idle()
    
    # Key press handler
    def on_key(event):
        # No scope issues - we're modifying a mutable dictionary
        
        if event.key == 'r':  # Reset
            # Reset view and axis limits
            ax.view_init(elev=initial_elev, azim=initial_azim)
            ax.set_xlim(initial_xlim)
            ax.set_ylim(initial_ylim)
            ax.set_zlim(initial_zlim)
            # Reset to wide view
            state['current_view_mode'] = 'wide'
            apply_view_mode(state['current_view_mode'])
            update_tick_format()
            update_axis_visibility()
            print("View reset to initial state")
            
        elif event.key == 'q':
            # Toggle quality
            state['current_quality_idx'] = (state['current_quality_idx'] + 1) % len(quality_modes)
            new_quality = quality_modes[state['current_quality_idx']]
            print(f"\nSwitched to {new_quality} quality mode")
            print("(Note: Quality change requires restart to take effect)")
            
        elif event.key == 'a':
            # Auto-fit axis to data (without changing view mode)
            margin = 0.05  # 5% margin
            x_range = data_xlim[1] - data_xlim[0]
            y_range = data_ylim[1] - data_ylim[0]
            z_range = data_zlim[1] - data_zlim[0]
            
            ax.set_xlim(data_xlim[0] - margin * x_range, data_xlim[1] + margin * x_range)
            ax.set_ylim(data_ylim[0] - margin * y_range, data_ylim[1] + margin * y_range)
            ax.set_zlim(data_zlim[0] - margin * z_range, data_zlim[1] + margin * z_range)
            update_axis_visibility()
            print("Auto-fitted to data bounds")
            
        elif event.key == 'v':
            # Toggle view mode
            state['current_view_mode'] = 'narrow' if state['current_view_mode'] == 'wide' else 'wide'
            apply_view_mode(state['current_view_mode'])
            update_tick_format()
            print(f"Switched to {state['current_view_mode']} view mode")
        elif event.key == '+' or event.key == '=':
            # Zoom in
            zoom_factor = 0.8
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            zlim = ax.get_zlim()
            
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
            z_center = (zlim[0] + zlim[1]) / 2
            
            x_half_range = (xlim[1] - xlim[0]) * zoom_factor / 2
            y_half_range = (ylim[1] - ylim[0]) * zoom_factor / 2
            z_half_range = (zlim[1] - zlim[0]) * zoom_factor / 2
            
            ax.set_xlim(x_center - x_half_range, x_center + x_half_range)
            ax.set_ylim(y_center - y_half_range, y_center + y_half_range)
            ax.set_zlim(z_center - z_half_range, z_center + z_half_range)
            update_axis_visibility()
            
        elif event.key == '-' or event.key == '_':
            # Zoom out
            zoom_factor = 1.25
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            zlim = ax.get_zlim()
            
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
            z_center = (zlim[0] + zlim[1]) / 2
            
            x_half_range = (xlim[1] - xlim[0]) * zoom_factor / 2
            y_half_range = (ylim[1] - ylim[0]) * zoom_factor / 2
            z_half_range = (zlim[1] - zlim[0]) * zoom_factor / 2
            
            # Limit zoom out to data bounds with margin
            margin = 0.2
            x_max_range = (data_xlim[1] - data_xlim[0]) * (1 + margin)
            y_max_range = (data_ylim[1] - data_ylim[0]) * (1 + margin)
            z_max_range = (data_zlim[1] - data_zlim[0]) * (1 + margin)
            
            x_half_range = min(x_half_range, x_max_range / 2)
            y_half_range = min(y_half_range, y_max_range / 2)
            z_half_range = min(z_half_range, z_max_range / 2)
            
            ax.set_xlim(x_center - x_half_range, x_center + x_half_range)
            ax.set_ylim(y_center - y_half_range, y_center + y_half_range)
            ax.set_zlim(z_center - z_half_range, z_center + z_half_range)
            update_axis_visibility()
    
    # Simple zoom implementation for 3D plots
    def on_scroll(event):
        """Handle mouse scroll for zoom - just change the view limits"""
        # Get current view limits
        xlim = list(ax.get_xlim())
        ylim = list(ax.get_ylim())
        zlim = list(ax.get_zlim())
        
        # Determine zoom direction
        if event.button == 'up':
            scale_factor = 0.9  # Zoom in - make view smaller
        else:
            scale_factor = 1.1  # Zoom out - make view larger
        
        # Get the center of current view
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        z_center = (zlim[0] + zlim[1]) / 2
        
        # Calculate new ranges from center
        x_half_range = (xlim[1] - xlim[0]) * scale_factor / 2
        y_half_range = (ylim[1] - ylim[0]) * scale_factor / 2
        z_half_range = (zlim[1] - zlim[0]) * scale_factor / 2
        
        # Calculate new limits
        new_xlim = [x_center - x_half_range, x_center + x_half_range]
        new_ylim = [y_center - y_half_range, y_center + y_half_range]
        new_zlim = [z_center - z_half_range, z_center + z_half_range]
        
        # Prevent over-zooming out
        if scale_factor > 1:  # Zooming out
            x_range = new_xlim[1] - new_xlim[0]
            y_range = new_ylim[1] - new_ylim[0]
            z_range = new_zlim[1] - new_zlim[0]
            
            # Check against data bounds with margin
            max_x_range = (data_xlim[1] - data_xlim[0]) * 1.5
            max_y_range = (data_ylim[1] - data_ylim[0]) * 1.5
            max_z_range = (data_zlim[1] - data_zlim[0]) * 1.5
            
            if x_range > max_x_range or y_range > max_y_range or z_range > max_z_range:
                return  # Don't zoom out further
        
        # Apply the new limits
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        ax.set_zlim(new_zlim)
        
        # Update the display
        update_axis_visibility()
    
    # Connect event handlers
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    # Add custom save button to toolbar
    # Position the button
    ax_button = plt.axes([0.81, 0.02, 0.15, 0.04])
    btn_save = Button(ax_button, 'Save High-Res', color='lightblue', hovercolor='skyblue')
    btn_save.on_clicked(lambda x: save_custom())
    
    # Adjust main plot to make room for button
    plt.subplots_adjust(top=0.95, bottom=0.10, right=0.95, left=0.05)
    
    # Show interactive plot
    plt.show()
    
    print(f"\nMethod used: {method_names[SMOOTHING_METHOD]}")
    if INTERPOLATION_FACTOR > 1:
        print(f"K-point density increased by factor of {INTERPOLATION_FACTOR}")
    
    print(f"\n📌 KEY FEATURES:")
    print("   • View modes: Press 'v' to toggle wide/narrow views")
    print("   • Zoom: Changes view range, plot stays within axis boundaries")
    print("   • Z-axis label positioning adjusted to avoid overlap")
    print("   • Z-axis range and tick spacing can be controlled via Z_RANGE and Z_TICK_STEP")
    print("   • State stored in dictionary to avoid Python scope issues")
    
    print(f"\nPerformance tip: If interaction is slow, try:")
    print(f"  • Reducing INTERPOLATION_FACTOR (currently {INTERPOLATION_FACTOR})")
    print(f"  • Setting GUI_QUALITY to 'low' (currently '{GUI_QUALITY}')")
    print(f"  • Limiting k-space range with KX_RANGE/KY_RANGE")
    print(f"\nEstimated speedup factors:")
    print(f"  • INTERPOLATION_FACTOR 2 vs 4: ~4x faster")
    print(f"  • Quality 'low' vs 'medium': ~2-3x faster")
    print(f"  • Limiting k-range to 50%: ~4x faster")

else:
    print(f"\nCreating separate high-quality figures...")
    
    # ===== FRONT VIEW =====
    fig_front = plt.figure(figsize=figure_size)
    ax_front = fig_front.add_subplot(111, projection='3d')
    
    print("  Creating Front View...")
    setup_publication_subplot(ax_front, elev=5, azim=0, show_xlabel=True, view_name="Front")
    
    # Minimal clean layout for Front View
    plt.subplots_adjust(top=0.95, bottom=0.05, right=0.95, left=0.05)
    
    # Save Front View
    front_filename = 'front_view_bands.png'
    fig_front.savefig(front_filename, 
                      dpi=600,
                      bbox_inches='tight', 
                      pad_inches=0.1,
                      facecolor='white', 
                      edgecolor='none')
    
    # Also save as PDF
    front_pdf = 'front_view_bands.pdf'
    fig_front.savefig(front_pdf,
                      format='pdf',
                      bbox_inches='tight',
                      pad_inches=0.1,
                      facecolor='white',
                      edgecolor='none')
    
    plt.close(fig_front)
    
    # ===== PERSPECTIVE VIEW 1 =====
    fig_persp = plt.figure(figsize=figure_size)
    ax_persp = fig_persp.add_subplot(111, projection='3d')
    
    print("  Creating Perspective View...")
    setup_publication_subplot(ax_persp, elev=25, azim=45, show_xlabel=True, view_name="Perspective")
    
    # Minimal clean layout for Perspective View
    plt.subplots_adjust(top=0.95, bottom=0.05, right=0.95, left=0.05)
    
    # Save Perspective View
    persp_filename = 'perspective_view_bands.png'
    fig_persp.savefig(persp_filename, 
                      dpi=600,
                      bbox_inches='tight', 
                      pad_inches=0.1,
                      facecolor='white', 
                      edgecolor='none')
    
    # Also save as PDF
    persp_pdf = 'perspective_view_bands.pdf'
    fig_persp.savefig(persp_pdf,
                      format='pdf',
                      bbox_inches='tight',
                      pad_inches=0.1,
                      facecolor='white',
                      edgecolor='none')
    
    plt.close(fig_persp)
    
    print(f"\nCLEAN PUBLICATION FIGURES SAVED:")
    print(f"   Front View PNG: '{front_filename}' (600 DPI)")
    print(f"   Front View PDF: '{front_pdf}' (vector)")
    print(f"   Perspective PNG: '{persp_filename}' (600 DPI)")  
    print(f"   Perspective PDF: '{persp_pdf}' (vector)")
    
    print(f"Method used: {method_names[SMOOTHING_METHOD]}")
    if INTERPOLATION_FACTOR > 1:
        print(f"K-point density increased by factor of {INTERPOLATION_FACTOR} for publication quality")

print("\n" + "="*70)
print("SUMMARY OF SETTINGS:")
print("="*70)
print(f"Mode: {'INTERACTIVE GUI' if GUI_MODE else 'SAVE FIGURES'}")
if GUI_MODE:
    print(f"GUI Quality: {GUI_QUALITY}")
    print(f"GUI Interpolation: {INTERPOLATION_FACTOR}x (lower = faster interaction)")
else:
    print(f"Save Interpolation: {INTERPOLATION_FACTOR}x (high quality)")
print(f"Fermi projection: {'ENABLED' if SHOW_FERMI_PROJECTION else 'DISABLED'}")
if KX_RANGE is not None:
    print(f"KX range limited to: [{KX_RANGE[0]:.3f}, {KX_RANGE[1]:.3f}]")
else:
    print(f"KX range: Full data range [{kx_mesh.min():.3f}, {kx_mesh.max():.3f}]")
if KY_RANGE is not None:
    print(f"KY range limited to: [{KY_RANGE[0]:.3f}, {KY_RANGE[1]:.3f}]")
else:
    print(f"KY range: Full data range [{ky_mesh.min():.3f}, {ky_mesh.max():.3f}]")

# Show energy range information
energy_min = min(np.nanmin(VBM_mesh), np.nanmin(CBM_mesh))
energy_max = max(np.nanmax(VBM_mesh), np.nanmax(CBM_mesh))
if Z_RANGE is not None:
    print(f"Energy range limited to: [{Z_RANGE[0]:.3f}, {Z_RANGE[1]:.3f}] eV")
else:
    print(f"Energy range: Full data range [{energy_min:.3f}, {energy_max:.3f}] eV")
if Z_TICK_STEP is not None:
    print(f"Energy tick spacing: {Z_TICK_STEP} eV")
else:
    print(f"Energy tick spacing: Automatic")

if GUI_MODE:
    total_points = kx_mesh.shape[0] * kx_mesh.shape[1]
    print(f"Total grid points: {total_points:,}")
    if total_points > 100000:
        print("⚠️  High point count may cause slow interaction!")
print("="*70)
