# Install Hindi Language Support for Tesseract

## ⚠️ IMPORTANT: Hindi Language Data Required

Your system currently does NOT have Hindi language data installed. This is why Hindi text extraction is not working.

## Current Status

Available languages: `['eng', 'osd']`  
Hindi available: ❌ **NO**

## Installation Instructions

### Windows

1. **Download Hindi language data:**
   - Go to: https://github.com/tesseract-ocr/tessdata
   - Download: `hin.traineddata`

2. **Install:**
   - Place the file in: `C:\Program Files\Tesseract-OCR\tessdata\`
   - Or if Tesseract is installed elsewhere, find the `tessdata` folder and place it there

3. **Verify:**
   ```python
   import pytesseract
   print(pytesseract.get_languages())  # Should now include 'hin'
   ```

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr-hin
```

### Linux (Other distributions)

1. Download `hin.traineddata` from https://github.com/tesseract-ocr/tessdata
2. Place it in `/usr/share/tesseract-ocr/tessdata/` or `/usr/local/share/tessdata/`

### macOS

```bash
brew install tesseract-lang
```

Or manually:
1. Download `hin.traineddata` from https://github.com/tesseract-ocr/tessdata
2. Place it in `/usr/local/share/tessdata/` or `$(brew --prefix)/share/tessdata/`

## Verify Installation

After installing, verify with:

```python
import pytesseract
langs = pytesseract.get_languages()
print("Available languages:", langs)
if 'hin' in langs:
    print("✓ Hindi is now available!")
else:
    print("✗ Hindi is still not available. Check installation path.")
```

## Restart Server

After installing Hindi language data, **restart your FastAPI server** for changes to take effect.

## Test

Once installed, test with:

```bash
curl -X POST "http://localhost:8000/api/v1/ocr/extract" \
  -F "file=@hindi.pdf" \
  -F "document_type=CERTIFICATE" \
  -F "language=hin"
```

The server logs will now show:
- `✓ Using Tesseract language(s): hin`
- `✓ Detected X Hindi (Devanagari) characters in extracted text`

## Quick Fix Script (Windows)

If you have Python, you can use this to download and install:

```python
import urllib.request
import os
import shutil

# Download Hindi language data
url = "https://github.com/tesseract-ocr/tessdata/raw/main/hin.traineddata"
tessdata_path = r"C:\Program Files\Tesseract-OCR\tessdata\hin.traineddata"

print("Downloading Hindi language data...")
urllib.request.urlretrieve(url, tessdata_path)
print(f"✓ Installed to: {tessdata_path}")

# Verify
import pytesseract
langs = pytesseract.get_languages()
if 'hin' in langs:
    print("✓ Hindi is now available!")
else:
    print("✗ Installation may have failed. Check path manually.")
```

