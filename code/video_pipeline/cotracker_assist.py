"""Local CoTracker3 inference for baseball-center annotation proposals.

Predictions are candidates only. The model is a general point tracker and is
not trained specifically on baseballs; every generated point must be reviewed.
"""
from __future__ import annotations

import os
from pathlib import Path

# Permit unsupported MPS ops to fall back to CPU; this is harmless on CPU-only
# setups and lets us try Apple's backend when the installed runtime exposes it.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

FPS_LABEL_SCALE = 60.0  # The existing label schema uses 60-Hz frame indices.
_MODEL = None
_DEVICE = None


def _get_model():
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL, _DEVICE

    import torch
    torch.hub.set_dir(str(Path(__file__).resolve().parent / '.cotracker_cache'))

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Official Meta Hub entrypoint downloads the public CoTracker3 checkpoint
    # on first use and caches it locally for subsequent runs.
    model = torch.hub.load(
        "facebookresearch/co-tracker",
        "cotracker3_offline",
        pretrained=True,
        trust_repo=True,
    )
    model = model.to(device).eval()
    _MODEL, _DEVICE = model, device
    return model, device


def track_points(video_path, start_time, end_time, keyframes, sample_id):
    """Track annotated center points across a selected pitch window.

    keyframes: sequence of dicts with `time_seconds`, `x`, `y`, `frame_index`.
    Returns: dict mapping `<sample_id>:<frame_index>` to candidate records.
    """
    import cv2
    import numpy as np
    import torch

    if len(keyframes) < 2:
        raise ValueError("至少需要两个已保存的可见关键帧")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("无法打开本地视频")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or FPS_LABEL_SCALE
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(start_time)) * 1000.0)

    frames, times, frame_ids = [], [], []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        # OpenCV reports the decoded frame's presentation time after read.
        time_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if time_seconds > float(end_time) + 0.5 / source_fps:
            break
        if time_seconds < float(start_time) - 0.5 / source_fps:
            continue
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        times.append(float(time_seconds))
        frame_ids.append(int(round(time_seconds * FPS_LABEL_SCALE)))
    cap.release()

    if len(frames) < 3:
        raise RuntimeError("投球时间窗内读取到的帧数不足")

    # Remove frame-index collisions caused by rounding 59.6-fps media to the
    # existing 60-Hz annotation schema. Keep one decoded frame per key.
    unique_frames, unique_times, unique_ids = [], [], []
    for image, timestamp, frame_id in zip(frames, times, frame_ids):
        if unique_ids and frame_id == unique_ids[-1]:
            unique_frames[-1], unique_times[-1] = image, timestamp
        else:
            unique_frames.append(image)
            unique_times.append(timestamp)
            unique_ids.append(frame_id)
    frames, times, frame_ids = unique_frames, unique_times, unique_ids

    # CoTracker operates on RGB video tensors shaped (B,T,C,H,W), in 0..255.
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].float()
    query_rows = []
    for point in sorted(keyframes, key=lambda q: q["time_seconds"]):
        local_frame = min(
            range(len(times)), key=lambda i: abs(times[i] - float(point["time_seconds"]))
        )
        query_rows.append([float(local_frame), float(point["x"]), float(point["y"])])
    queries = torch.tensor(query_rows, dtype=torch.float32)[None]

    model, device = _get_model()
    video = video.to(device)
    queries = queries.to(device)
    with torch.inference_mode():
        tracks, visibility = model(video, queries=queries, backward_tracking=True)

    tracks = tracks[0].detach().float().cpu().numpy()  # T,N,2
    visibility = visibility[0].detach().float().cpu().numpy()  # T,N
    if visibility.ndim == 3:
        visibility = visibility[..., 0]

    results = {}
    query_frames = [int(round(q[0])) for q in query_rows]
    for t, frame_id in enumerate(frame_ids):
        # Combine visible tracks, favoring the query nearest in time. The spread
        # between independently initialized tracks is retained for review.
        active = [j for j in range(len(query_rows)) if visibility[t, j] > 0.5]
        xy = None
        spread = None
        if active:
            weights = np.array(
                [1.0 / (1.0 + abs(t - query_frames[j])) for j in active],
                dtype=np.float32,
            )
            points = tracks[t, active]
            xy = np.average(points, axis=0, weights=weights)
            spread = float(np.max(np.linalg.norm(points - xy[None, :], axis=1)))

        results[f"{sample_id}:{frame_id}"] = {
            "sample_id": sample_id,
            "frame_index": frame_id,
            "time_seconds": round(times[t], 6),
            "visible": bool(xy is not None),
            "x": round(float(xy[0]), 2) if xy is not None else None,
            "y": round(float(xy[1]), 2) if xy is not None else None,
            "visibility": round(float(np.max(visibility[t])), 4),
            "track_disagreement_px": round(spread, 2) if spread is not None else None,
            "review_required": True,
            "label_source": "cotracker3",
        }

    metadata = {
        "device": str(device),
        "fps": round(float(source_fps), 4),
        "frames_processed": len(frames),
        "keyframes_used": len(query_rows),
    }
    return results, metadata
