import numpy as np

def compute_AEE(
    estimated_flow,      # Nx2xHxW oppure 2xHxW
    gt_flow,             # 2xHxW
    dt_input_ms,         # durata finestra corrente (ms)
    dt_gt_ms,            # durata GT (ms)
    invalid_gt_value=0.0
):
    """
    Calcola l'AEE (Average Endpoint Error) per una singola finestra eventi.
    Funziona per MVSEC default, fixed slicing e adaptive slicing.

    PARAMETRI:
    estimated_flow: optical flow stimato (2xHxW)
    gt_flow: optical flow GT (2xHxW)
    sparse_mask: pixel mask with detected features only (HxW) (0 for the pixel != feature)
    dt_input_ms: durata finestra corrente (ms)
    dt_gt_ms: durata della finestra GT (ms)
    invalid_gt_value: valore usato per indicare GT mancante (es: 0)

    RITORNA:
    AEE (float), outlier_percentage (float), numero_pixel_validi
    """

    # assicurati che siano 2×H×W
    if len(estimated_flow.shape) == 4:
        estimated_flow = estimated_flow[0]

        #here apply the feature sparsity mask to the GT flow since we only have FLOW at specific corners (FAST corner detector)



    # scaling temporale
    # GT è per un dt diverso → adattiamo il GT al tuo dt
    scaling = dt_input_ms / float(dt_gt_ms + 1e-9)
    gt_scaled = gt_flow * scaling

    # ERROR per pixel
    error = np.sqrt(
        (estimated_flow[0] - gt_scaled[0])**2 +
        (estimated_flow[1] - gt_scaled[1])**2
    )

    # maschera: pixels with detected keypoints 
    mask_event = (estimated_flow != 0)    #1 only where = detected feature

    # maschera: pixel with GT != 0 (eliminates those "black areas")
    mask_gt = ~(
        (gt_flow[0] == invalid_gt_value) &
        (gt_flow[1] == invalid_gt_value)
    )

    mask = mask_event & mask_gt

    if mask.sum() == 0:
        print("no valid mask applied, all zeroes")
        return np.nan, np.nan, 0

    error_valid = error[mask]

    # AEE
    AEE = error_valid.mean()

    # Magnitudo flusso GT (per outlier ratio)
    mag_gt = np.sqrt(gt_scaled[0]**2 + gt_scaled[1]**2)
    mag_gt_valid = mag_gt[mask]

    # outliers = (error > 3 px) e (error > 0.05 * |flow|)
    outliers = (error_valid > 3.0) & (error_valid > 0.05 * mag_gt_valid)
    outlier_percentage = outliers.sum() / float(len(error_valid) + 1e-9)

    return AEE, outlier_percentage, mask.sum()
