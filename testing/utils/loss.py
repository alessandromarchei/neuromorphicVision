import numpy as np


def compute_AEE(
    estimated_flow,      # Nx2xHxW or 2xHxW
    gt_flow,             # 2xHxW
    dt_input_ms,         # duration of current window (ms)
    dt_gt_ms,            # duration of GT (ms)
    invalid_gt_value=0.0,
    eps=1e-6             # epsilon to avoid division by zero
):
    """
    Compute AEE (Average Endpoint Error) and REE (Relative Endpoint Error)
    for a single event window. Works with MVSEC slicing, fixed, or adaptive.

    Returns:
        AEE (float), REE (float), outlier_percentage (float), valid_pixel_count
    """

    if len(estimated_flow.shape) == 4:
        estimated_flow = estimated_flow[0]

    # Scale GT to match dt of input
    scaling = dt_input_ms / float(dt_gt_ms + eps)
    gt_scaled = gt_flow * scaling

    # Compute per-pixel endpoint error
    error = np.sqrt(
        (estimated_flow[0] - gt_scaled[0])**2 +
        (estimated_flow[1] - gt_scaled[1])**2
    )

    # Validity masks
    mask_event = (estimated_flow[0] != 0) | (estimated_flow[1] != 0)
    mask_gt = ~((gt_flow[0] == invalid_gt_value) & (gt_flow[1] == invalid_gt_value))
    mask = mask_event & mask_gt

    if mask.sum() == 0:
        print("no valid mask applied, all zeroes")
        return np.nan, np.nan, np.nan, 0

    error_valid = error[mask]

    # AEE
    AEE = error_valid.mean()

    # Ground-truth magnitude
    mag_gt = np.sqrt(gt_scaled[0]**2 + gt_scaled[1]**2)
    mag_gt_valid = mag_gt[mask]

    # Relative error (REE), as mean over valid pixels containing vectors
    REE = (error_valid / (mag_gt_valid + eps)).mean()

    # Outlier percentage
    outliers = (error_valid > 3.0) & (error_valid > 0.05 * mag_gt_valid)
    outlier_percentage = outliers.sum() / float(len(error_valid) + eps)

    return AEE, outlier_percentage, mask.sum(), REE
