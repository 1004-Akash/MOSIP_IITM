# Hindi Language Support

## Overview
The OCR system now supports Hindi language documents using Tesseract OCR.

## How to Use

### API Request with Hindi Language

```bash
curl -X POST "http://localhost:8000/api/v1/ocr/extract" \
  -F "file=@hindi_document.pdf" \
  -F "document_type=CERTIFICATE" \
  -F "language=hin"
```

### Language Options

- **`eng`** - English (default)
- **`hin`** - Hindi
- **`eng+hin`** - English + Hindi (multilingual documents)

## Installation Requirements

To use Hindi language support, you need to install the Hindi language data for Tesseract:

### Windows
1. Download Hindi language data: `hin.traineddata`
2. Place it in: `C:\Program Files\Tesseract-OCR\tessdata\`
3. Or download from: https://github.com/tesseract-ocr/tessdata

### Linux/Mac
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr-hin

# macOS
brew install tesseract-lang
```

### Verify Installation
```python
import pytesseract
print(pytesseract.get_languages())  # Should include 'hin'
```

## How It Works

1. **Tesseract Primary**: For non-English languages (like Hindi), Tesseract is used as the primary OCR engine since TrOCR doesn't support Hindi.

2. **Automatic Detection**: When you specify `language=hin`, the system automatically uses Tesseract with Hindi language model.

3. **Multilingual Support**: Use `language=eng+hin` for documents containing both English and Hindi text.

## Notes

- **TrOCR Limitation**: TrOCR (Microsoft's transformer-based OCR) only supports English. For Hindi, the system automatically uses Tesseract.

- **Performance**: Tesseract is fast and reliable for Hindi text recognition.

- **Field Extraction**: The current field extraction regex patterns are optimized for English. For Hindi documents, you'll get the raw text, but structured field extraction may need Hindi-specific patterns.

