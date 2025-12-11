# Quick Start Guide

## Installation Steps

1. **Install Python 3.8+** (if not already installed)

2. **Install Tesseract OCR** (optional, for fallback):

   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

3. **Create virtual environment**:

   ```bash
   python -m venv venv
   ```

4. **Activate virtual environment**:

   - Windows: `venv\Scripts\activate`
   - Linux/macOS: `source venv/bin/activate`

5. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

6. **Start the server**:

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

7. **Access the application**:
   - Web UI: http://localhost:8000/ui
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## First Test

1. Open http://localhost:8000/ui in your browser
2. Upload a sample document (ID card, form, or certificate)
3. Select document type and click "Extract Text"
4. Review the extracted fields and confidence scores

## API Testing with curl

### Extract Text:

```bash
curl -X POST "http://localhost:8000/api/v1/ocr/extract" \
  -F "file=@your_document.jpg" \
  -F "document_type=ID_CARD"
```

### Verify Data:

```bash
curl -X POST "http://localhost:8000/api/v1/verify" \
  -F "file=@your_document.jpg" \
  -F 'form_data={"name": "John Doe", "dob": "1990-01-15"}'
```

## Troubleshooting

- **Model download slow**: TrOCR models are downloaded on first use (~500MB). Ensure stable internet connection.
- **CUDA errors**: If you have GPU issues, the system will automatically fall back to CPU.
- **Memory errors**: For large images, the system automatically resizes. If issues persist, reduce image size manually.
- **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`

## Next Steps

- See `README.md` for detailed documentation
- See `example_usage.py` for Python API examples
- Check logs in `logs/app.log` for debugging
