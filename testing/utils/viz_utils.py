import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt



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