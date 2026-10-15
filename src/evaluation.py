import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from diffusers.utils import export_to_video, load_video
from PIL import Image, ImageDraw
from torchvision.models.video import R3D_18_Weights, r3d_18
from transformers import CLIPModel, CLIPProcessor, pipeline


def calculate_iou(pred_masks: np.ndarray, gt_masks: np.ndarray) -> float:
    intersection = np.logical_and(pred_masks, gt_masks)
    union = np.logical_or(pred_masks, gt_masks)
    return float(np.sum(intersection) / (np.sum(union) + 1e-5))


def calculate_ap50(pred_masks: np.ndarray, gt_masks: np.ndarray) -> float:
    intersection = np.logical_and(pred_masks, gt_masks)
    union = np.logical_or(pred_masks, gt_masks)
    iou_score = np.sum(intersection, axis=(-1, -2)) / (np.sum(union, axis=(-1, -2)) + 1e-5)
    non_zero_mask = np.sum(gt_masks, axis=(-1, -2)) > 0
    return float(np.mean(iou_score[non_zero_mask] >= 0.5))


def calculate_intersection_over_min(pred_masks: np.ndarray, gt_masks: np.ndarray) -> float:
    intersection = np.logical_and(pred_masks, gt_masks)
    min_area = np.minimum(np.sum(pred_masks), np.sum(gt_masks))
    return float(np.sum(intersection) / (min_area + 1e-5))


def calculate_ap50_over_min(pred_masks: np.ndarray, gt_masks: np.ndarray) -> float:
    intersection = np.logical_and(pred_masks, gt_masks)
    min_area = np.minimum(np.sum(pred_masks, axis=(-1, -2)), np.sum(gt_masks, axis=(-1, -2)))
    intersection_over_min = np.sum(intersection, axis=(-1, -2)) / (min_area + 1e-5)
    non_zero_mask = np.sum(gt_masks, axis=(-1, -2)) > 0
    return float(np.mean(intersection_over_min[non_zero_mask] >= 0.5))


def calculate_centroid_distance(pred_masks: np.ndarray, gt_masks: np.ndarray, normalized: bool = False) -> float:
    def get_centroid(mask):
        x_sum = np.sum(mask, axis=-1)
        y_sum = np.sum(mask, axis=-2)
        x_center = (np.argmax(x_sum, axis=-1) + np.argmax(np.cumsum(x_sum, axis=-1), axis=-1)) / 2
        y_center = (np.argmax(y_sum, axis=-1) + np.argmax(np.cumsum(y_sum, axis=-1), axis=-1)) / 2
        return np.stack([x_center, y_center], axis=1)

    dist = float(np.linalg.norm(get_centroid(pred_masks) - get_centroid(gt_masks), ord=2, axis=-1).mean())
    if normalized and pred_masks.shape[-2:]:
        h, w = pred_masks.shape[-2:]
        frame_size = float(np.linalg.norm([h, w]))
        dist /= frame_size
    return dist


def load_masks_from_video(video_path: str, target_height: int, target_width: int, num_frames: int | None = None) -> np.ndarray:
    frames = load_video(video_path)
    if num_frames and len(frames) != num_frames:
        indices = np.rint(np.linspace(0, len(frames) - 1, num_frames)).astype(np.int64)
        frames = [frames[i] for i in indices]
    masks = [(np.array(f.resize((target_width, target_height), Image.NEAREST).convert("L")) > 127).astype(np.uint8) for f in frames]
    return np.stack(masks)


@torch.no_grad()
def calculate_clip_similarity(frames: list[Image.Image], prompt: str, clip_model: CLIPModel, clip_processor: CLIPProcessor) -> float:
    image_inputs = clip_processor(images=frames, return_tensors="pt", padding=True).to(clip_model.device)
    text_inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True).to(clip_model.device)

    image_features = clip_model.get_image_features(**image_inputs)
    text_features = clip_model.get_text_features(**text_inputs)

    image_features = image_features / image_features.norm(dim=1, keepdim=True).to(torch.float32)
    text_features = text_features / text_features.norm(dim=1, keepdim=True).to(torch.float32)

    similarities = (image_features * text_features).sum(dim=1)
    return float(similarities.mean().cpu())


def load_clip_model(model_name: str = "openai/clip-vit-base-patch32", device: str | None = None) -> tuple[CLIPModel, CLIPProcessor]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def detect_owlvit(
    frames: list[Image.Image],
    subject: str,
    detector_model: str = "google/owlvit-large-patch14",
    device: str | None = None,
) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    detector = pipeline("zero-shot-object-detection", model=detector_model, device=device)
    tokenizer = detector.tokenizer
    max_len = getattr(detector.model.config, "max_position_embeddings", 16)
    subject_safe = tokenizer.decode(
        tokenizer(subject, truncation=True, max_length=max_len, add_special_tokens=True)["input_ids"],
        skip_special_tokens=True,
    )

    all_masks = []
    for frame in frames:
        w, h = frame.size
        mask = np.zeros((h, w), dtype=np.uint8)
        predictions = detector(frame, candidate_labels=[subject_safe])
        if predictions:
            try:
                b = predictions[0]["box"]
                x1, y1 = int(np.clip(b["xmin"], 0, w - 1)), int(np.clip(b["ymin"], 0, h - 1))
                x2, y2 = int(np.clip(b["xmax"], 0, w)), int(np.clip(b["ymax"], 0, h))
                mask[y1:y2, x1:x2] = 1
            except (KeyError, IndexError):
                pass
        all_masks.append(mask)

    return np.array(all_masks)


def preprocess_videos_for_r3d(video_frames: list[list[Image.Image]], num_frames: int = 16, device: str = "cuda") -> torch.Tensor:
    batch_tensors = []
    for frames in video_frames:
        indices = np.linspace(0, len(frames) - 1, min(num_frames, len(frames))).astype(int)
        sampled = [frames[i].convert("RGB").resize((112, 112)) for i in indices]

        video = torch.stack([torch.from_numpy(np.array(f)).permute(2, 0, 1) for f in sampled])
        video = video.float() / 255.0

        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1)
        video = (video - mean) / std

        batch_tensors.append(video.permute(1, 0, 2, 3))

    return torch.stack(batch_tensors).to(device)


@torch.no_grad()
def extract_features_batch(videos: torch.Tensor, model: nn.Module) -> torch.Tensor:
    features = model(videos)
    if features.dim() > 2:
        features = features.view(features.size(0), -1)
    return features


def matrix_sqrt_eig(matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = torch.clamp(eigenvalues, min=0)
    eigenvalues = eigenvalues + eps
    return eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T


def calculate_frechet_distance_torch(mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor, eps: float = 1e-4) -> float:
    diff = mu1 - mu2
    mean_distance = float(diff @ diff)

    if sigma1.shape[0] == 1 or torch.allclose(sigma1, torch.zeros_like(sigma1)) or torch.allclose(sigma2, torch.zeros_like(sigma2)):
        return mean_distance

    sigma1_stable = sigma1 + eps * torch.eye(sigma1.size(0), device=sigma1.device)
    sigma2_stable = sigma2 + eps * torch.eye(sigma2.size(0), device=sigma2.device)

    sigma1_sqrt = matrix_sqrt_eig(sigma1_stable, eps)
    product = sigma1_sqrt @ sigma2_stable @ sigma1_sqrt
    product_sqrt = matrix_sqrt_eig(product, eps)

    trace_term = torch.trace(sigma1_stable + sigma2_stable - 2 * product_sqrt)

    distance = mean_distance + float(trace_term)
    return max(0.0, distance)


def calculate_fvd(
    generated_videos: list[list[Image.Image]],
    reference_videos: list[list[Image.Image]],
    batch_size: int = 8,
    device: str | None = None,
) -> float:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    model.fc = nn.Identity()
    model = model.to(device).eval()

    def extract_all_features(videos: list[list[Image.Image]]) -> torch.Tensor:
        all_features = []
        for i in range(0, len(videos), batch_size):
            batch = videos[i : i + batch_size]
            batch_tensor = preprocess_videos_for_r3d(batch, device=device)
            features = extract_features_batch(batch_tensor, model)
            all_features.append(features.cpu())
        return torch.cat(all_features, dim=0)

    gen_features = extract_all_features(generated_videos)
    ref_features = extract_all_features(reference_videos)

    mu_gen = gen_features.mean(dim=0)
    mu_ref = ref_features.mean(dim=0)

    if gen_features.size(0) == 1 or ref_features.size(0) == 1:
        diff = mu_gen - mu_ref
        return float(diff @ diff)

    gen_centered = gen_features - mu_gen
    ref_centered = ref_features - mu_ref

    sigma_gen = (gen_centered.T @ gen_centered) / (gen_features.size(0) - 1)
    sigma_ref = (ref_centered.T @ ref_centered) / (ref_features.size(0) - 1)

    return calculate_frechet_distance_torch(mu_gen, sigma_gen, mu_ref, sigma_ref)


def load_reference_videos(reference_dir: str) -> list[list[Image.Image]]:
    ref_path = Path(reference_dir)
    video_files = sorted(ref_path.glob("*.mp4"))
    return [load_video(str(vf)) for vf in video_files]


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == size:
        return mask
    resized = Image.fromarray((mask * 255).astype(np.uint8)).resize((size[1], size[0]), Image.NEAREST)
    return (np.array(resized) > 0).astype(np.uint8)


def create_debug_frame(
    frame: Image.Image,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray | None = None,
    bbox: tuple | None = None,
) -> Image.Image:
    vis = np.array(frame.convert("RGB"))
    h, w = vis.shape[:2]

    if gt_mask is not None:
        gt_mask = resize_mask(gt_mask, (h, w))
        overlay = np.zeros_like(vis)
        overlay[gt_mask.astype(bool)] = [0, 255, 0]
        vis = (vis * 0.55 + overlay * 0.45).astype(np.uint8) if gt_mask.any() else vis

    if pred_mask.any():
        overlay = np.zeros_like(vis)
        overlay[pred_mask.astype(bool)] = [255, 0, 0]
        vis = (vis * 0.55 + overlay * 0.45).astype(np.uint8)

    vis_img = Image.fromarray(vis)
    if bbox:
        ImageDraw.Draw(vis_img).rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=(255, 0, 0), width=2)

    return vis_img


def create_debug_video(
    frames: list[Image.Image],
    pred_masks: np.ndarray,
    gt_masks: np.ndarray | None = None,
    output_path: str | None = None,
    fps: int = 8,
) -> list[Image.Image]:
    debug_frames = []
    for idx, frame in enumerate(frames):
        pred = pred_masks[idx] if idx < len(pred_masks) else np.zeros((frame.size[1], frame.size[0]), dtype=np.uint8)
        gt = gt_masks[idx] if gt_masks is not None and idx < len(gt_masks) else None
        debug_frames.append(create_debug_frame(frame, pred, gt))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        export_to_video(debug_frames, output_path, fps=fps)

    return debug_frames


def save_results_to_json(results: dict, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
