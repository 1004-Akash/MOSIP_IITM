# Performance Optimizations

## Speed Improvements Implemented

### 1. **Default Fast Mode** ⚡
- **Tesseract is now the primary OCR engine by default** (faster than TrOCR)
- TrOCR is used as a fallback when needed for quality
- Can be controlled via `USE_TESSERACT_FIRST` environment variable (default: `true`)

### 2. **Single Page Processing by Default** 📄
- Only processes the first page of PDFs by default (much faster)
- To process all pages, set `all_pages=true` in the API request
- Significantly reduces processing time for multi-page documents

### 3. **Optional Expensive Operations** 🎯
- **Quality Metrics** (`include_quality`) - OFF by default
  - Blur detection, brightness, contrast analysis
  - Only calculated when `include_quality=true`
  
- **Confidence Zones** (`include_zones`) - OFF by default
  - Word-level bounding boxes and confidence scores
  - Only calculated when `include_zones=true`

### 4. **Optimized Image Processing** 🖼️
- PDF pages limited by default (only first page)
- Efficient image loading and conversion
- No unnecessary preprocessing overhead

## API Usage

### Fast Mode (Default - Recommended)
```bash
curl -X POST "http://localhost:8000/api/v1/ocr/extract" \
  -F "file=@document.pdf" \
  -F "document_type=CERTIFICATE"
```

This uses:
- ✅ Tesseract (fast)
- ✅ First page only
- ✅ No quality metrics
- ✅ No confidence zones

### Full Quality Mode (Slower)
```bash
curl -X POST "http://localhost:8000/api/v1/ocr/extract" \
  -F "file=@document.pdf" \
  -F "document_type=CERTIFICATE" \
  -F "all_pages=true" \
  -F "include_quality=true" \
  -F "include_zones=true" \
  -F "fast_mode=false"
```

This uses:
- ⚠️ TrOCR (slower but more accurate)
- ⚠️ All pages
- ⚠️ Quality metrics
- ⚠️ Confidence zones

## Performance Comparison

| Mode | Avg Processing Time | Use Case |
|------|---------------------|----------|
| **Fast (Default)** | ~1-3 seconds | Production, real-time processing |
| **Balanced** | ~3-5 seconds | Standard extraction with quality |
| **Full Quality** | ~5-15 seconds | Maximum accuracy required |

## Environment Variables

```bash
# Use Tesseract first (faster) - DEFAULT
USE_TESSERACT_FIRST=true

# Use TrOCR first (slower, more accurate)
USE_TESSERACT_FIRST=false

# Use TrOCR Large model (slowest, most accurate)
USE_TROCR_LARGE=true
```

## Tips for Maximum Speed

1. **Use default settings** - Already optimized for speed
2. **Process single pages** - Don't set `all_pages=true` unless needed
3. **Skip quality metrics** - Only enable if you need blur/brightness analysis
4. **Skip confidence zones** - Only enable if you need word-level details
5. **Use Tesseract** - Already the default, faster than TrOCR

## Tips for Maximum Quality

1. **Set `fast_mode=false`** - Uses TrOCR primary
2. **Set `USE_TROCR_LARGE=true`** - Uses larger TrOCR model
3. **Enable `include_quality=true`** - Get image quality scores
4. **Enable `include_zones=true`** - Get word-level confidence

