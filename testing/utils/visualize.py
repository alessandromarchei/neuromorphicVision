import cv2
import numpy as np
import os
import time


def visualize_image(currFrame, currPoints, prevPoints, status, delay):
    flowVis = cv2.cvtColor(currFrame, cv2.COLOR_GRAY2BGR)
    for i in range(len(currPoints)):
        if status[i] == 1:
            p1 = tuple(map(int, prevPoints[i]))
            p2 = tuple(map(int, currPoints[i]))
            cv2.arrowedLine(flowVis, p1, p2, (0, 0, 255), 2)
    cv2.imshow("OF_raw", flowVis)
    cv2.waitKey(delay)

def visualize_gt_flow(flow_gt, event_frame, win_name="GT Flow"):
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
    mask = (event_frame != 128)

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
