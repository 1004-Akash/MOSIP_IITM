# Handwritten Hindi Support

## Overview
The OCR system now supports handwritten Hindi text using optimized Tesseract OCR configurations and preprocessing.

## How to Use

### API Request with Handwritten Hindi

```bash
curl -X POST "http://localhost:8000/api/v1/ocr/extract" \
  -F "file=@hindi_handwritten_certificate.pdf" \
  -F "document_type=CERTIFICATE" \
  -F "language=hin" \
  -F "handwritten=true"
```

### Parameters

- **`language`**: Set to `hin` for Hindi, or `eng+hin` for mixed English/Hindi
- **`handwritten`**: Set to `true` for handwritten text (this enables special preprocessing and PSM modes)

## Features

### 1. **Optimized Preprocessing for Handwritten Text**
- Adaptive thresholding (better for variable handwriting quality)
- Gentle Gaussian blur (reduces noise without losing detail)
- Grayscale conversion optimized for handwriting recognition

### 2. **Multiple PSM Mode Testing**
When `handwritten=true`, the system tries multiple Page Segmentation Modes:
- **PSM 11**: Sparse text (best for handwritten forms with fields)
- **PSM 7**: Single text line (for continuous handwritten text)
- **PSM 13**: Raw line (treats image as single text line)
- **PSM 6**: Uniform block (fallback)

The system automatically selects the PSM mode that extracts the most text.

### 3. **Automatic Tesseract Selection**
For handwritten text or Hindi language, Tesseract is automatically used as the primary OCR engine (TrOCR doesn't support handwritten Hindi).

## Tips for Better Results

1. **Use high-resolution images**: Clearer images = better recognition
2. **Good lighting**: Ensure the document is well-lit without shadows
3. **Minimal skew**: Keep documents as straight as possible
4. **Clean background**: White or light backgrounds work best

## For Your Hindi Certificate

Based on your certificate example, use:
- `language=hin` or `language=eng+hin` (since it has both)
- `handwritten=true` (because it contains handwritten fields)

Example:
```bash
curl -X POST "http://localhost:8000/api/v1/ocr/extract" \
  -F "file=@hindi.pdf" \
  -F "document_type=CERTIFICATE" \
  -F "language=eng+hin" \
  -F "handwritten=true"
```

This will:
1. Use Tesseract with both English and Hindi language models
2. Apply handwritten-optimized preprocessing
3. Try multiple PSM modes to find the best result
4. Extract both printed Hindi text and handwritten English fields

