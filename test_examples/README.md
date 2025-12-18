# Test Examples

This directory contains sample plant images for testing and demonstration purposes.

## Usage

Place a few sample plant images in this directory to test the models. The images should be:
- Format: JPEG or PNG
- Content: Plant species images (similar to PlantCLEF dataset)

## Example Directory Structure

```
test_examples/
├── README.md (this file)
├── sample1.jpg
├── sample2.jpg
├── sample3.jpg
└── ...
```

## Running Demo with Test Examples

### Classification Demo

Test a single image classification:
```bash
python demo.py --task classification --model dinov2 --image test_examples/sample1.jpg
```

Or with ConvNeXt:
```bash
python demo.py --task classification --model convnext --image test_examples/sample1.jpg
```

### Retrieval Demo

Extract features for retrieval:
```bash
python demo.py --task retrieval --model dinov2 --image test_examples/sample1.jpg
```

## Sample Images

You can use images from:
1. PlantCLEF dataset test set
2. Your own plant photos
3. Downloaded plant images from the internet

Make sure the images are clear and contain visible plant features (leaves, flowers, etc.).
