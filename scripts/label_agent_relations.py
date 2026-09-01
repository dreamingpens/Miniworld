#!/usr/bin/env python3
# pyright: basic, reportMissingImports=false

"""Label conservative multi-agent relations and render review clips.

The script consumes the frame-aligned ``*_actions.pt`` and per-agent RGB videos
written by ``scripts.generate_videos``. Labels are computed for every unordered
agent pair, then short side-by-side clips are produced for each stable positive
interval.

Example:
    python -m scripts.label_agent_relations \
        --input-prefix ./out/multi/multi \
        --output-dir ./out/multi/relation_review
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


Pair = tuple[int, int]
Interval = tuple[int, int]


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap radians to [-pi, pi)."""

    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _forward_vectors(headings: np.ndarray) -> np.ndarray:
    """Convert MiniWorld headings to unit XZ vectors."""

    return np.stack([np.cos(headings), -np.sin(headings)], axis=-1)


def _horizontal_fov_rad(vertical_fov_deg: float, aspect_ratio: float) -> float:
    vertical_fov = math.radians(vertical_fov_deg)
    return 2.0 * math.atan(math.tan(vertical_fov * 0.5) * aspect_ratio)


def _frustum_triangle(
    position: np.ndarray,
    heading: float,
    half_horizontal_fov: float,
    max_range: float,
) -> np.ndarray:
    """Return a conservative 2D camera-frustum triangle in world XZ coordinates."""

    ray_angles = np.array(
        [heading - half_horizontal_fov, heading + half_horizontal_fov],
        dtype=np.float32,
    )
    rays = np.stack([np.cos(ray_angles), -np.sin(ray_angles)], axis=-1)
    return np.asarray(
        [position, position + max_range * rays[0], position + max_range * rays[1]],
        dtype=np.float32,
    )


def _convex_iou(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    intersection_area, _ = cv2.intersectConvexConvex(poly_a, poly_b)
    area_a = abs(float(cv2.contourArea(poly_a)))
    area_b = abs(float(cv2.contourArea(poly_b)))
    union_area = area_a + area_b - float(intersection_area)
    if union_area <= 0.0:
        return 0.0
    return float(intersection_area) / union_area


def _stable_intervals(mask: np.ndarray, min_frames: int) -> tuple[np.ndarray, list[Interval]]:
    """Keep only true runs at least ``min_frames`` long."""

    stable = np.zeros_like(mask, dtype=bool)
    intervals: list[Interval] = []
    start = None

    for frame_idx, active in enumerate(np.concatenate([mask, np.array([False])])):
        if active and start is None:
            start = frame_idx
        elif not active and start is not None:
            end = frame_idx - 1
            if end - start + 1 >= min_frames:
                stable[start : end + 1] = True
                intervals.append((start, end))
            start = None

    return stable, intervals


def compute_relation_labels(
    positions: np.ndarray,
    headings: np.ndarray,
    *,
    face_max_angle_deg: float,
    face_min_distance: float,
    face_max_distance: float,
    shared_max_heading_deg: float,
    shared_min_distance: float,
    shared_max_distance: float,
    shared_max_lateral_offset: float,
    shared_max_forward_offset: float,
    shared_min_frustum_iou: float,
    vertical_fov_deg: float,
    camera_aspect_ratio: float,
    frustum_range: float,
    min_event_frames: int,
) -> tuple[dict, list[dict]]:
    """Compute metrics, raw labels, stable labels, and event intervals."""

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"expected positions shaped (T,N,3), got {positions.shape}")
    if headings.shape != positions.shape[:2]:
        raise ValueError(
            f"headings shape {headings.shape} does not match positions {positions.shape[:2]}"
        )

    frame_count, num_agents = headings.shape
    pair_indices: list[Pair] = list(itertools.combinations(range(num_agents), 2))
    pair_count = len(pair_indices)
    positions_xz = positions[:, :, (0, 2)]
    forward = _forward_vectors(headings)

    distance = np.zeros((frame_count, pair_count), dtype=np.float32)
    facing_error = np.zeros((frame_count, pair_count, 2), dtype=np.float32)
    heading_error = np.zeros((frame_count, pair_count), dtype=np.float32)
    lateral_offset = np.zeros((frame_count, pair_count), dtype=np.float32)
    forward_offset = np.zeros((frame_count, pair_count), dtype=np.float32)
    frustum_iou = np.zeros((frame_count, pair_count), dtype=np.float32)

    half_horizontal_fov = _horizontal_fov_rad(
        vertical_fov_deg, camera_aspect_ratio
    ) * 0.5

    for pair_idx, (agent_i, agent_j) in enumerate(pair_indices):
        relative = positions_xz[:, agent_j] - positions_xz[:, agent_i]
        pair_distance = np.linalg.norm(relative, axis=-1)
        direction_ij = relative / np.maximum(pair_distance[:, None], 1e-8)

        dot_i = np.sum(forward[:, agent_i] * direction_ij, axis=-1)
        dot_j = np.sum(forward[:, agent_j] * -direction_ij, axis=-1)
        facing_error[:, pair_idx, 0] = np.degrees(
            np.arccos(np.clip(dot_i, -1.0, 1.0))
        )
        facing_error[:, pair_idx, 1] = np.degrees(
            np.arccos(np.clip(dot_j, -1.0, 1.0))
        )
        distance[:, pair_idx] = pair_distance

        pair_heading_error = np.abs(
            np.degrees(_wrap_angle(headings[:, agent_i] - headings[:, agent_j]))
        )
        heading_error[:, pair_idx] = pair_heading_error

        average_forward = forward[:, agent_i] + forward[:, agent_j]
        average_norm = np.linalg.norm(average_forward, axis=-1, keepdims=True)
        average_forward = average_forward / np.maximum(average_norm, 1e-8)
        opposite = average_norm[:, 0] < 1e-6
        average_forward[opposite] = forward[opposite, agent_i]
        average_right = np.stack(
            [-average_forward[:, 1], average_forward[:, 0]], axis=-1
        )
        lateral_offset[:, pair_idx] = np.abs(
            np.sum(relative * average_right, axis=-1)
        )
        forward_offset[:, pair_idx] = np.abs(
            np.sum(relative * average_forward, axis=-1)
        )

        for frame_idx in range(frame_count):
            frustum_i = _frustum_triangle(
                positions_xz[frame_idx, agent_i],
                float(headings[frame_idx, agent_i]),
                half_horizontal_fov,
                frustum_range,
            )
            frustum_j = _frustum_triangle(
                positions_xz[frame_idx, agent_j],
                float(headings[frame_idx, agent_j]),
                half_horizontal_fov,
                frustum_range,
            )
            frustum_iou[frame_idx, pair_idx] = _convex_iou(frustum_i, frustum_j)

    face_to_face_raw = (
        (facing_error[:, :, 0] <= face_max_angle_deg)
        & (facing_error[:, :, 1] <= face_max_angle_deg)
        & (distance >= face_min_distance)
        & (distance <= face_max_distance)
    )
    shared_view_raw = (
        (heading_error <= shared_max_heading_deg)
        & (distance >= shared_min_distance)
        & (distance <= shared_max_distance)
        & (lateral_offset <= shared_max_lateral_offset)
        & (forward_offset <= shared_max_forward_offset)
        & (frustum_iou >= shared_min_frustum_iou)
    )

    face_to_face = np.zeros_like(face_to_face_raw)
    shared_view = np.zeros_like(shared_view_raw)
    events: list[dict] = []

    for pair_idx, pair in enumerate(pair_indices):
        stable, intervals = _stable_intervals(
            face_to_face_raw[:, pair_idx], min_event_frames
        )
        face_to_face[:, pair_idx] = stable
        for start, end in intervals:
            events.append(
                {
                    "label": "face_to_face",
                    "pair_index": pair_idx,
                    "agents": list(pair),
                    "start_frame": start,
                    "end_frame": end,
                    "duration_frames": end - start + 1,
                }
            )

        stable, intervals = _stable_intervals(
            shared_view_raw[:, pair_idx], min_event_frames
        )
        shared_view[:, pair_idx] = stable
        for start, end in intervals:
            events.append(
                {
                    "label": "shared_view",
                    "pair_index": pair_idx,
                    "agents": list(pair),
                    "start_frame": start,
                    "end_frame": end,
                    "duration_frames": end - start + 1,
                }
            )

    events.sort(key=lambda event: (event["label"], event["start_frame"], event["agents"]))
    labels = {
        "pair_indices": np.asarray(pair_indices, dtype=np.int64),
        "distance": distance,
        "facing_error_deg": facing_error,
        "heading_error_deg": heading_error,
        "lateral_offset": lateral_offset,
        "forward_offset": forward_offset,
        "frustum_iou": frustum_iou,
        "face_to_face_raw": face_to_face_raw,
        "shared_view_raw": shared_view_raw,
        "face_to_face": face_to_face,
        "shared_view": shared_view,
    }
    return labels, events


def _collect_review_frames(
    events: Sequence[dict], context_frames: int, frame_count: int
) -> dict[int, set[int]]:
    requested: dict[int, set[int]] = {}
    for event in events:
        clip_start = max(0, event["start_frame"] - context_frames)
        clip_end = min(frame_count - 1, event["end_frame"] + context_frames)
        event["clip_start_frame"] = clip_start
        event["clip_end_frame"] = clip_end
        for agent_idx in event["agents"]:
            requested.setdefault(agent_idx, set()).update(range(clip_start, clip_end + 1))
    return requested


def _load_requested_video_frames(
    input_prefix: Path,
    requested: dict[int, set[int]],
    expected_frame_count: int,
) -> tuple[dict[int, dict[int, np.ndarray]], float, float]:
    cached: dict[int, dict[int, np.ndarray]] = {}
    detected_fps = None
    detected_aspect = None

    for agent_idx, frame_indices in requested.items():
        video_path = Path(f"{input_prefix}_agent_{agent_idx}_rgb.mp4")
        if not video_path.exists():
            raise FileNotFoundError(f"missing agent video: {video_path}")

        reader = imageio.get_reader(video_path)
        metadata = reader.get_meta_data()
        if detected_fps is None:
            detected_fps = float(metadata.get("fps", 15.0))
        size = metadata.get("size") or metadata.get("source_size")
        if size and detected_aspect is None:
            detected_aspect = float(size[0]) / float(size[1])

        agent_frames: dict[int, np.ndarray] = {}
        last_frame_idx = -1
        try:
            for frame_idx, frame in enumerate(reader):
                last_frame_idx = frame_idx
                if frame_idx in frame_indices:
                    agent_frames[frame_idx] = np.asarray(frame)
                if frame_idx >= expected_frame_count - 1 and len(agent_frames) == len(frame_indices):
                    break
        finally:
            reader.close()

        missing = sorted(frame_indices.difference(agent_frames))
        if missing:
            raise ValueError(
                f"{video_path} is missing requested frames {missing[:5]}; "
                f"last decoded frame was {last_frame_idx}"
            )
        cached[agent_idx] = agent_frames

    return cached, detected_fps or 15.0, detected_aspect or 1.0


def _pad_for_h264(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    padded_height = int(math.ceil(height / 16.0) * 16)
    padded_width = int(math.ceil(width / 16.0) * 16)
    if (padded_height, padded_width) == (height, width):
        return frame
    return cv2.copyMakeBorder(
        frame,
        0,
        padded_height - height,
        0,
        padded_width - width,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def _annotate_pair_frame(
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    event: dict,
    frame_idx: int,
    labels: dict,
) -> np.ndarray:
    agent_i, agent_j = event["agents"]
    pair_idx = event["pair_index"]
    height = max(frame_i.shape[0], frame_j.shape[0])
    width = max(frame_i.shape[1], frame_j.shape[1])

    if frame_i.shape[:2] != (height, width):
        frame_i = cv2.resize(frame_i, (width, height), interpolation=cv2.INTER_AREA)
    if frame_j.shape[:2] != (height, width):
        frame_j = cv2.resize(frame_j, (width, height), interpolation=cv2.INTER_AREA)

    positive = event["start_frame"] <= frame_idx <= event["end_frame"]
    border_color = (40, 220, 40) if positive else (130, 130, 130)
    cv2.rectangle(frame_i, (1, 1), (width - 2, height - 2), border_color, 3)
    cv2.rectangle(frame_j, (1, 1), (width - 2, height - 2), border_color, 3)

    header_height = 64
    canvas = np.zeros((height + header_height, width * 2, 3), dtype=np.uint8)
    canvas[header_height:, :width] = frame_i
    canvas[header_height:, width:] = frame_j
    cv2.line(canvas, (width, header_height), (width, height + header_height), (255, 255, 255), 1)

    state = "POSITIVE" if positive else "context"
    title = (
        f"{event['label']}  agents {agent_i}-{agent_j}  "
        f"frame {frame_idx}  {state}"
    )
    cv2.putText(
        canvas,
        title,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        border_color,
        1,
        cv2.LINE_AA,
    )

    distance = labels["distance"][frame_idx, pair_idx]
    if event["label"] == "face_to_face":
        errors = labels["facing_error_deg"][frame_idx, pair_idx]
        metrics = (
            f"distance={distance:.2f}  facing errors=({errors[0]:.1f}, {errors[1]:.1f}) deg"
        )
    else:
        metrics = (
            f"distance={distance:.2f}  heading={labels['heading_error_deg'][frame_idx, pair_idx]:.1f} deg  "
            f"lateral={labels['lateral_offset'][frame_idx, pair_idx]:.2f}  "
            f"forward={labels['forward_offset'][frame_idx, pair_idx]:.2f}  "
            f"frustum IoU={labels['frustum_iou'][frame_idx, pair_idx]:.2f}"
        )
    cv2.putText(
        canvas,
        metrics,
        (8, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"agent {agent_i}",
        (8, header_height + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"agent {agent_j}",
        (width + 8, header_height + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return _pad_for_h264(canvas)


def _write_review_clips(
    output_dir: Path,
    events: Sequence[dict],
    cached_frames: dict[int, dict[int, np.ndarray]],
    labels: dict,
    fps: float,
) -> None:
    counters = {"face_to_face": 0, "shared_view": 0}

    for event in events:
        label = event["label"]
        event_idx = counters[label]
        counters[label] += 1
        agent_i, agent_j = event["agents"]
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"event_{event_idx:03d}_agents_{agent_i}_{agent_j}_"
            f"frames_{event['start_frame']:04d}-{event['end_frame']:04d}.mp4"
        )
        clip_path = label_dir / filename
        event["clip_path"] = str(clip_path)

        with imageio.get_writer(
            clip_path,
            fps=fps,
            codec="libx264",
            format="FFMPEG",
            pixelformat="yuv420p",
            bitrate="8M",
        ) as writer:
            for frame_idx in range(
                event["clip_start_frame"], event["clip_end_frame"] + 1
            ):
                annotated = _annotate_pair_frame(
                    cached_frames[agent_i][frame_idx].copy(),
                    cached_frames[agent_j][frame_idx].copy(),
                    event,
                    frame_idx,
                    labels,
                )
                writer.append_data(annotated)


def _to_torch_payload(labels: dict, thresholds: dict, events: Sequence[dict]) -> dict:
    payload = {
        "schema_version": 1,
        "thresholds": thresholds,
        "events": list(events),
    }
    for key, value in labels.items():
        payload[key] = torch.from_numpy(value)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create conservative multi-agent relation labels and review clips."
    )
    parser.add_argument(
        "--input-prefix",
        type=Path,
        required=True,
        help="rollout prefix, e.g. out/multi/multi",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: <input-prefix>_relation_review",
    )
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--min-event-frames", type=int, default=3)

    parser.add_argument("--face-max-angle-deg", type=float, default=10.0)
    parser.add_argument("--face-min-distance", type=float, default=0.8)
    parser.add_argument("--face-max-distance", type=float, default=6.0)

    parser.add_argument("--shared-max-heading-deg", type=float, default=5.0)
    parser.add_argument("--shared-min-distance", type=float, default=0.8)
    parser.add_argument("--shared-max-distance", type=float, default=2.5)
    parser.add_argument("--shared-max-lateral-offset", type=float, default=1.5)
    parser.add_argument("--shared-max-forward-offset", type=float, default=2.0)
    parser.add_argument("--shared-min-frustum-iou", type=float, default=0.55)

    parser.add_argument("--vertical-fov-deg", type=float, default=60.0)
    parser.add_argument("--camera-aspect-ratio", type=float, default=None)
    parser.add_argument("--frustum-range", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_prefix = args.input_prefix
    output_dir = args.output_dir or Path(f"{input_prefix}_relation_review")
    output_dir.mkdir(parents=True, exist_ok=True)

    actions_path = Path(f"{input_prefix}_actions.pt")
    if not actions_path.exists():
        raise FileNotFoundError(f"missing trajectory metadata: {actions_path}")
    try:
        metadata = torch.load(actions_path, map_location="cpu", weights_only=True)
    except TypeError:
        metadata = torch.load(actions_path, map_location="cpu")

    if "multi_agent_pos" not in metadata or "multi_agent_dir" not in metadata:
        raise KeyError(
            f"{actions_path} must contain multi_agent_pos and multi_agent_dir"
        )
    positions = metadata["multi_agent_pos"].cpu().numpy().astype(np.float32)
    headings = metadata["multi_agent_dir"].cpu().numpy().astype(np.float32)
    frame_count, num_agents = headings.shape

    first_video = Path(f"{input_prefix}_agent_0_rgb.mp4")
    if not first_video.exists():
        raise FileNotFoundError(f"missing agent video: {first_video}")
    reader = imageio.get_reader(first_video)
    try:
        video_metadata = reader.get_meta_data()
    finally:
        reader.close()
    video_size = video_metadata.get("size") or video_metadata.get("source_size")
    detected_aspect = (
        float(video_size[0]) / float(video_size[1]) if video_size else 1.0
    )
    camera_aspect_ratio = args.camera_aspect_ratio or detected_aspect

    thresholds = {
        "face_max_angle_deg": args.face_max_angle_deg,
        "face_min_distance": args.face_min_distance,
        "face_max_distance": args.face_max_distance,
        "shared_max_heading_deg": args.shared_max_heading_deg,
        "shared_min_distance": args.shared_min_distance,
        "shared_max_distance": args.shared_max_distance,
        "shared_max_lateral_offset": args.shared_max_lateral_offset,
        "shared_max_forward_offset": args.shared_max_forward_offset,
        "shared_min_frustum_iou": args.shared_min_frustum_iou,
        "vertical_fov_deg": args.vertical_fov_deg,
        "camera_aspect_ratio": camera_aspect_ratio,
        "frustum_range": args.frustum_range,
        "min_event_frames": args.min_event_frames,
    }

    labels, events = compute_relation_labels(
        positions,
        headings,
        face_max_angle_deg=args.face_max_angle_deg,
        face_min_distance=args.face_min_distance,
        face_max_distance=args.face_max_distance,
        shared_max_heading_deg=args.shared_max_heading_deg,
        shared_min_distance=args.shared_min_distance,
        shared_max_distance=args.shared_max_distance,
        shared_max_lateral_offset=args.shared_max_lateral_offset,
        shared_max_forward_offset=args.shared_max_forward_offset,
        shared_min_frustum_iou=args.shared_min_frustum_iou,
        vertical_fov_deg=args.vertical_fov_deg,
        camera_aspect_ratio=camera_aspect_ratio,
        frustum_range=args.frustum_range,
        min_event_frames=args.min_event_frames,
    )

    requested = _collect_review_frames(events, args.context_frames, frame_count)
    if events:
        cached_frames, fps, _ = _load_requested_video_frames(
            input_prefix, requested, frame_count
        )
        _write_review_clips(output_dir, events, cached_frames, labels, fps)
    else:
        fps = float(video_metadata.get("fps", 15.0))

    relations_path = output_dir / "relations.pt"
    torch.save(_to_torch_payload(labels, thresholds, events), relations_path)

    counts = {
        label: sum(event["label"] == label for event in events)
        for label in ("face_to_face", "shared_view")
    }
    summary = {
        "input_prefix": str(input_prefix),
        "actions_path": str(actions_path),
        "frame_count": frame_count,
        "num_agents": num_agents,
        "fps": fps,
        "thresholds": thresholds,
        "event_counts": counts,
        "events": events,
        "relations_path": str(relations_path),
        "limitations": [
            "The current rollout has no per-agent object-ID masks, so line-of-sight and exact shared-object visibility are not verified.",
            "shared_view uses conservative pose, offset, and finite-frustum overlap constraints as a proxy.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote labels: {relations_path}")
    print(f"Wrote summary: {summary_path}")
    print(
        "Events: "
        + ", ".join(f"{label}={count}" for label, count in counts.items())
    )
    for event in events:
        print(
            f"- {event['label']} agents={event['agents']} "
            f"frames={event['start_frame']}-{event['end_frame']} "
            f"clip={event.get('clip_path')}"
        )


if __name__ == "__main__":
    main()
