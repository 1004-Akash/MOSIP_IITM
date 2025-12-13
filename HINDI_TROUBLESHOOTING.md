# Hindi OCR Troubleshooting Guide

## Problem: Extracted text is not in Hindi

If you're getting garbled text or English-only results for Hindi documents, follow these steps:

### 1. Verify Hindi Language Data is Installed

**Check available languages:**
```python
import pytesseract
print(pytesseract.get_languages())
```

You should see `'hin'` in the list. If not, install it:

**Windows:**
1. Download `hin.traineddata` from: https://github.com/tesseract-ocr/tessdata
2. Place it in: `C:\Program Files\Tesseract-OCR\tessdata\`
3. Restart your application

**Linux:**
```bash
sudo apt-get install tesseract-ocr-hin
```

**macOS:**
```bash
brew install tesseract-lang
```

### 2. Use Correct Language Parameter

Make sure you're passing the correct language:

```bash
# For Hindi only
-F "language=hin"

# For Hindi + English (multilingual)
-F "language=eng+hin"
```

### 3. Check Server Logs

The server logs will show:
- Available languages detected
- Language being used
- Any warnings about missing language data

Look for messages like:
- `✓ Using Tesseract language(s): hin`
- `CRITICAL: Language(s) ['hin'] not available in Tesseract!`

### 4. Test Language Detection

Create a test script:

```python
import pytesseract
from PIL import Image

# Check languages
print("Available languages:", pytesseract.get_languages())

# Test with a Hindi image
image = Image.open("hindi_test.png")
text = pytesseract.image_to_string(image, lang='hin')
print("Extracted text:", text)
```

### 5. Common Issues

**Issue: Language falls back to English**
- **Cause**: Hindi language data not installed
- **Fix**: Install `hin.traineddata` (see step 1)

**Issue: Garbled text output**
- **Cause**: Wrong preprocessing or PSM mode
- **Fix**: The system tries multiple modes automatically, but you can also try:
  - `handwritten=true` if document has handwritten parts
  - Different PSM modes (system tries automatically)

**Issue: No text extracted**
- **Cause**: Image quality or language data issue
- **Fix**: 
  - Check image resolution (should be at least 300 DPI)
  - Verify Hindi language data is installed
  - Check server logs for errors

### 6. Force Hindi Language

If the system is falling back to English, you can verify the language is being used by checking the logs. The system will:
1. Check if 'hin' is available
2. Log a warning if it's not
3. Try to use 'hin' anyway (may fail gracefully)

### 7. Verify Installation

Run this in Python:
```python
import pytesseract
langs = pytesseract.get_languages()
if 'hin' in langs:
    print("✓ Hindi language data is installed")
else:
    print("✗ Hindi language data is NOT installed")
    print(f"Available languages: {langs}")
```

### 8. API Request Example

```bash
curl -X POST "http://localhost:8000/api/v1/ocr/extract" \
  -F "file=@hindi_document.pdf" \
  -F "document_type=CERTIFICATE" \
  -F "language=hin" \
  -F "handwritten=false"
```

Check the response and server logs to see what language was actually used.

