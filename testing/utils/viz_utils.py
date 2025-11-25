import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def visualize_image(currFrame, currPoints, prevPoints, status):
    flowVis = cv2.cvtColor(currFrame, cv2.COLOR_GRAY2BGR)
    for i in range(len(currPoints)):
        if status[i] == 1:
            p1 = tuple(map(int, prevPoints[i]))
            p2 = tuple(map(int, currPoints[i]))
            cv2.arrowedLine(flowVis, p1, p2, (0, 0, 255), 2)
    cv2.imshow("OF_raw", flowVis)
    # cv2.waitKey(delay)

def visualize_gt_flow(flow_gt, event_frame, win_name="GT Flow", apply_mask=True):
    """
    Visualizza un optical flow ground-truth (2,H,W) usando HSV:
    - Hue = direzione
    - Value = magnitudo
    - Saturation = piena
    """
    if flow_gt is None:
        blank = np.zeros((event_frame.shape[0], event_frame.shape[1], 3), dtype=np.uint8)
        cv2.imshow(win_name, blank)
        cv2.waitKey(1)
        return

    u = flow_gt[0]
    v = flow_gt[1]
    H, W = u.shape

    # Magnitude e angolo direzione
    mag = np.sqrt(u * u + v * v)
    ang = np.arctan2(v, u)  # [-pi, pi]

    # OpenCV HSV:
    # Hue ∈ [0,180]  (!= 360)
    hue = ((ang + np.pi) / (2 * np.pi)) * 180.0   # mappa [-pi,pi] → [0,180]

    # Magnitudo normalizzata per Value ∈ [0,255]
    if mag.max() > 0:
        val = (mag / mag.max()) * 255.0
    else:
        val = mag * 0

    # Saturation fissa al massimo
    sat = np.ones_like(val) * 255.0

    # Stack HSV
    hsv = np.stack([hue, sat, val], axis=-1).astype(np.uint8)

    # Convert HSV → BGR con OpenCV
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Maschera eventi:
    # pixel = 128 => no-event
    if apply_mask == True:
        mask = (event_frame != 128)
    else:
        mask = np.ones_like(event_frame)
        mask = True

    # punti senza eventi → nero
    out = np.zeros_like(bgr)
    out[mask] = bgr[mask]

    cv2.imshow(win_name, out)

def overlay_gt_flow_on_events(flow_gt, event_frame, alpha=0.5, win_name="GT Flow Overlay"):
    """
    Sovrappone GT optical flow all'event_frame con blending alpha.
    """
    if flow_gt is None:
        black = np.zeros((event_frame.shape[0], event_frame.shape[1], 3), dtype=np.uint8)
        cv2.imshow(win_name, black)
        return

    # Visualize only flow first
    H, W = event_frame.shape
    temp = np.zeros((H, W, 3), dtype=np.uint8)
    visualize_gt_flow(flow_gt, event_frame, win_name="__temp__")  # internal display
    temp = cv2.imread("__temp__.png") if os.path.exists("__temp__.png") else None

    # fallback se non salviamo file
    if temp is None:
        return

    # Convert event_frame to grayscale 3-channels
    event_gray = cv2.cvtColor(event_frame, cv2.COLOR_GRAY2BGR)

    blended = cv2.addWeighted(event_gray, 1 - alpha, temp, alpha, 0)

    cv2.imshow(win_name, blended)


def visualize_event_frame(event_frame, use_cv2=True, wait=0, window_name="Event Frame"):
    """
    Visualize a single event frame (CV_8UC1-style grayscale).

    Args:
        event_frame: torch.Tensor | np.ndarray, shape (H, W)
        use_cv2 (bool): if True, display with OpenCV (fast); else use matplotlib
        wait (int): delay in ms for cv2.waitKey()
        window_name (str): optional window title
    """

    # Convert torch → numpy if needed
    if torch.is_tensor(event_frame):
        img = event_frame.detach().cpu().numpy()
    elif isinstance(event_frame, np.ndarray):
        img = event_frame
    else:
        raise TypeError("Input must be a torch.Tensor or np.ndarray")

    # Ensure 2D grayscale
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")

    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if use_cv2:
        # OpenCV expects BGR but for grayscale we just pass single channel
        cv2.imshow(window_name, img)
        key = cv2.waitKey(wait)
        if key == 27:  # Esc to close
            cv2.destroyWindow(window_name)
    else:
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.title(window_name)
        plt.show()

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

# visualize events
def visualize_sparse_voxel(voxel):
    bins = voxel.shape[0]

    plt.figure(figsize=(20, 20))
    for i in range(bins):
        plt.subplot(1, bins, i+1)
        plt.spy(abs(voxel[i]))
    plt.show()


def visualize_N_voxels(voxels_in, EPS=1e-3, idx_plot_vox=[0]):
    # Custom red-white-blue colormap
    colors = ['blue', 'white', 'red']
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list("custom_rwb", colors)

    device = voxels_in.device

    if device != 'cpu':
        voxels_in = voxels_in.detach().cpu()

    voxels = torch.clone(voxels_in)
    N = voxels.shape[0]
    assert N > sorted(idx_plot_vox)[-1]

    if len(idx_plot_vox) > 7:
       N = 7
       idx_plot_vox = idx_plot_vox[:N]
       print(f"Only plotting first {N} voxels")

    voxels = voxels[idx_plot_vox]
    bins = voxels.shape[1]

    # Zero out small values
    voxels[(voxels < EPS) & (voxels > 0)] = 0
    voxels[(voxels > -EPS) & (voxels < 0)] = 0

    # Find min and max for scaling
    vmin = voxels.min().item()
    vmax = voxels.max().item()

    # Avoid division by zero
    if vmax != 0:
        voxels[voxels > 0] = voxels[voxels > 0] / vmax
    if vmin != 0:
        voxels[voxels < 0] = voxels[voxels < 0] / abs(vmin)

    fig = plt.figure(figsize=(20, 20))
    for i in range(N*bins):
        ax = fig.add_subplot(N, bins, i+1)
        ax.imshow(voxels[i % N][i % bins], cmap=cmap, vmin=-1, vmax=1)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # Restore to original device
    if device != 'cpu':
        voxels = voxels.to(device)



def visualize_voxel(*voxel_in, EPS=1e-3, save=False, folder="results/voxels", index=None):
    # cmaps
    colors = ['red', 'white', 'blue']
    cmap = plt.cm.colors.ListedColormap(colors)
    norm = plt.Normalize(vmin=-1, vmax=1)

    plt.figure(figsize=(20, 20))
    for i, vox in enumerate(voxel_in):

        if vox.device != 'cpu':
            vox = vox.detach().cpu()
        
        voxel = torch.clone(vox)
        bins = vox.shape[0]

        voxel[torch.bitwise_and(voxel<EPS, voxel>0)] = 0
        voxel[torch.bitwise_and(voxel>-EPS, voxel<0)] = 0

        voxel[voxel<0] = -1
        voxel[voxel>0] = 1
        for j in range(bins):
            plt.subplot(len(voxel_in), bins, i*bins + j + 1)
            plt.imshow(voxel[j], cmap=cmap, norm=norm)
            ax = plt.gca()
            ax.grid(False)
    if save:
        os.makedirs(folder, exist_ok=True)
        if index is not None:
            string = "voxel_" + str(index)
        else:
            string = datetime.today().strftime('%Y-%m-%d_%H:%M:%S.%f')
        plt.axis('off')
        plt.savefig(f'{folder}/{string}.png', bbox_inches='tight', transparent=True, pad_inches=0)
    else:    
        plt.show()
    plt.close()