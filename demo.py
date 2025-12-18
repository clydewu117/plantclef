"""
Demo script for PlantCLEF classification and retrieval models.

This script demonstrates how to:
1. Load a trained model checkpoint
2. Run inference on sample images
3. Display predictions

Usage:
    python demo.py --task classification --model dinov2 --image test_examples/sample1.jpg
    python demo.py --task retrieval --model convnext --query test_examples/sample1.jpg
"""

import os
import argparse
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import pandas as pd

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def load_dinov2_model(checkpoint_path, num_classes=None):
    """
    Load DINOv2 model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        num_classes: Number of classes (only for classification)

    Returns:
        backbone, head: Loaded model components
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Load DINOv2 backbone
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    backbone = backbone.to(DEVICE)
    backbone.load_state_dict(checkpoint['backbone'])

    # Load head
    embed_dim = 768  # DINOv2 ViT-B/14 embedding dimension
    if num_classes:  # Classification head
        head = nn.Linear(embed_dim, num_classes).to(DEVICE)
    else:  # Retrieval projection head (single linear layer to 512-dim)
        head = nn.Linear(embed_dim, 512).to(DEVICE)

    head.load_state_dict(checkpoint['head'])

    return backbone, head, checkpoint.get('label_map', None)


def load_convnext_model(checkpoint_path, num_classes=None):
    """
    Load ConvNeXt model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        num_classes: Number of classes (only for classification)

    Returns:
        backbone, head: Loaded model components
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Load ConvNeXt backbone
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(*list(model.children())[:-1])
    backbone = backbone.to(DEVICE)
    backbone.load_state_dict(checkpoint['backbone'])

    # Load head
    in_dim = 768  # ConvNeXt Tiny feature dimension
    if num_classes:  # Classification head
        head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, num_classes)
        ).to(DEVICE)
    else:  # Retrieval projection head (3 layers: pool, flatten, linear)
        head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(in_dim, 512)
        ).to(DEVICE)

    head.load_state_dict(checkpoint['head'])

    return backbone, head, checkpoint.get('label_map', None)


def get_transforms(model_type):
    """
    Get image transformations based on model type.

    Args:
        model_type: 'dinov2' or 'convnext'

    Returns:
        transform: torchvision transforms
    """
    if model_type == 'dinov2':
        # DINOv2 standard transforms
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:  # convnext
        # ConvNeXt ImageNet transforms
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        mean = weights.transforms().mean
        std = weights.transforms().std
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return transform


@torch.no_grad()
def classify_image(image_path, model_type='dinov2', checkpoint_path=None):
    """
    Classify a single image using trained model.

    Args:
        image_path: Path to input image
        model_type: 'dinov2' or 'convnext'
        checkpoint_path: Path to checkpoint (if None, uses default)

    Returns:
        predictions: Dictionary with top-5 predictions
    """
    # Default checkpoint paths
    if checkpoint_path is None:
        checkpoint_path = f"checkpoints/{model_type}_class_best.pth"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please download from: https://drive.google.com/drive/folders/14c_88401aQ4go1w_g9b9W1LIK-qRnaGY"
        )

    # Load checkpoint to get number of classes
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    label_map = checkpoint['label_map']
    num_classes = len(label_map)

    # Create reverse mapping (label -> species_id)
    id_to_species = {v: k for k, v in label_map.items()}

    # Load model
    if model_type == 'dinov2':
        backbone, head, _ = load_dinov2_model(checkpoint_path, num_classes)
    else:
        backbone, head, _ = load_convnext_model(checkpoint_path, num_classes)

    backbone.eval()
    head.eval()

    # Load and preprocess image
    transform = get_transforms(model_type)
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Forward pass
    features = backbone(img_tensor)
    logits = head(features)

    # Get top-5 predictions
    probs = torch.softmax(logits, dim=1)
    top5_probs, top5_indices = torch.topk(probs, k=5, dim=1)

    # Format results
    predictions = []
    for prob, idx in zip(top5_probs[0].cpu().numpy(), top5_indices[0].cpu().numpy()):
        species_id = id_to_species[idx]
        predictions.append({
            'species_id': species_id,
            'confidence': float(prob),
            'label': int(idx)
        })

    return predictions


@torch.no_grad()
def extract_features(image_path, model_type='dinov2', checkpoint_path=None):
    """
    Extract features from an image for retrieval.

    Args:
        image_path: Path to input image
        model_type: 'dinov2' or 'convnext'
        checkpoint_path: Path to checkpoint (if None, uses default)

    Returns:
        features: Normalized feature vector (512-dim)
    """
    # Default checkpoint paths
    if checkpoint_path is None:
        checkpoint_path = f"checkpoints/{model_type}_retrieval_best.pth"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please download from: https://drive.google.com/drive/folders/14c_88401aQ4go1w_g9b9W1LIK-qRnaGY"
        )

    # Load model
    if model_type == 'dinov2':
        backbone, head, _ = load_dinov2_model(checkpoint_path, num_classes=None)
    else:
        backbone, head, _ = load_convnext_model(checkpoint_path, num_classes=None)

    backbone.eval()
    head.eval()

    # Load and preprocess image
    transform = get_transforms(model_type)
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Forward pass
    features = backbone(img_tensor)
    embeddings = head(features)

    # L2 normalize
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser(description='PlantCLEF Demo')
    parser.add_argument('--task', type=str, choices=['classification', 'retrieval'],
                       default='classification', help='Task type')
    parser.add_argument('--model', type=str, choices=['dinov2', 'convnext'],
                       default='dinov2', help='Model type')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint (optional)')

    args = parser.parse_args()

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return

    print(f"\n{'='*60}")
    print(f"PlantCLEF Demo - {args.task.upper()}")
    print(f"Model: {args.model.upper()}")
    print(f"Image: {args.image}")
    print(f"{'='*60}\n")

    if args.task == 'classification':
        # Run classification
        predictions = classify_image(args.image, args.model, args.checkpoint)

        print("Top-5 Predictions:")
        print(f"{'Rank':<6} {'Species ID':<15} {'Confidence':<12}")
        print("-" * 35)
        for i, pred in enumerate(predictions, 1):
            print(f"{i:<6} {pred['species_id']:<15} {pred['confidence']*100:>6.2f}%")

    else:  # retrieval
        # Extract features
        features = extract_features(args.image, args.model, args.checkpoint)

        print(f"Feature Extraction Complete!")
        print(f"Feature dimension: {features.shape[0]}")
        print(f"Feature norm: {(features**2).sum()**0.5:.4f} (should be ~1.0)")
        print(f"\nFirst 10 feature values:")
        print(features[:10])
        print("\nThese features can be used for similarity search in a gallery set.")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
