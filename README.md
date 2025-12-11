# OCR Text Extraction and Verification System

An end-to-end system for Optical Character Recognition (OCR) on identity and enrollment documents with text extraction and verification capabilities.

## Features

- **OCR Extraction API**: Extract structured data from scanned documents (ID cards, forms, certificates)
- **Data Verification API**: Verify user-filled form data against extracted document data
- Support for multiple document types and layouts
- Fuzzy matching for data verification
- Confidence scores for extracted fields
- Optional web UI for document upload and verification

## Tech Stack

- **Backend**: FastAPI (Python)
- **OCR Engine**: TrOCR (Transformer-based OCR)
- **Image Processing**: OpenCV, Pillow, scikit-image
- **Text Matching**: TheFuzz (fuzzy string matching)
- **PDF Support**: PyMuPDF (no external dependencies required)

## Installation

### Prerequisites

1. **Python 3.8 or higher** - The only requirement!
   - No external binaries needed (PyMuPDF handles PDFs natively)
   - All dependencies install via pip

### Setup

1. Clone the repository and navigate to the project directory.

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. The TrOCR model will be downloaded automatically on first use (requires internet connection).

## Running the APIs

### Start the FastAPI server:

**Windows:**
```bash
start_server.bat
```

**Linux/macOS:**
```bash
chmod +x start_server.sh
./start_server.sh
```

**Or manually:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

- **API**: http://localhost:8000
- **Interactive API Docs**: http://localhost:8000/docs
- **Web UI**: http://localhost:8000/ui

## API Endpoints

### 1. OCR Extraction API

**Endpoint**: `POST /api/v1/ocr/extract`

**Request**:

- Method: POST
- Content-Type: multipart/form-data
- Body:
  - `file`: Image file (jpg, png, pdf) or PDF file
  - `document_type`: String (e.g., 'ID_CARD', 'FORM', 'CERTIFICATE')

**Example using curl**:

```bash
curl -X POST "http://localhost:8000/api/v1/ocr/extract" \
  -F "file=@sample_id_card.jpg" \
  -F "document_type=ID_CARD"
```

**Response**:

```json
{
  "document_type": "ID_CARD",
  "raw_text": "Full extracted text...",
  "structured_data": {
    "name": "John Doe",
    "dob": "1990-01-15",
    "id_number": "ABC123456",
    "address": "123 Main St, City, State"
  },
  "confidence_scores": {
    "name": 0.95,
    "dob": 0.92,
    "id_number": 0.98,
    "address": 0.88
  },
  "overall_confidence": 0.93
}
```

### 2. Data Verification API

**Endpoint**: `POST /api/v1/verify`

**Request**:

- Method: POST
- Content-Type: multipart/form-data
- Body:
  - `file`: Original scanned document (image/PDF)
  - `form_data`: JSON string with user-filled form fields

**Example using curl**:

```bash
curl -X POST "http://localhost:8000/api/v1/verify" \
  -F "file=@sample_id_card.jpg" \
  -F 'form_data={"name": "John Doe", "dob": "1990-01-15", "id_number": "ABC123456"}'
```

**Response**:

```json
{
  "overall_verification_score": 0.96,
  "field_results": [
    {
      "field": "name",
      "document_value": "John Doe",
      "form_value": "John Doe",
      "status": "match",
      "similarity_score": 1.0
    },
    {
      "field": "dob",
      "document_value": "1990-01-15",
      "form_value": "1990-01-15",
      "status": "match",
      "similarity_score": 1.0
    },
    {
      "field": "id_number",
      "document_value": "ABC123456",
      "form_value": "ABC12345",
      "status": "mismatch",
      "similarity_score": 0.89
    }
  ],
  "mismatched_fields": ["id_number"],
  "missing_fields": []
}
```

## Project Structure

```
.
├── main.py                 # FastAPI application and routes
├── app/
│   ├── __init__.py
│   ├── ocr_engine.py      # OCR extraction using TrOCR
│   ├── preprocessor.py    # Image pre-processing
│   ├── field_extractor.py # Field extraction logic
│   ├── verifier.py        # Data verification logic
│   └── models.py          # Pydantic models
├── templates/             # HTML templates for web UI
│   └── index.html
├── uploads/               # Temporary storage for uploaded files
├── logs/                  # Application logs
├── requirements.txt
└── README.md
```

## Model Choices and Improvements

### OCR Model: TrOCR

- **Why TrOCR**: State-of-the-art transformer-based OCR that handles noisy, skewed, and low-resolution images better than traditional OCR engines.
- **Limitations**: Requires more computational resources and may be slower than Tesseract for simple documents.

### Future Improvements:

1. **Fine-tuning**: Fine-tune TrOCR on domain-specific documents (ID cards, forms) for better accuracy
2. **Layout Detection**: Implement advanced layout analysis (e.g., LayoutLM) for better field localization
3. **Multi-language Support**: Add support for Hindi and other languages by using multilingual OCR models
4. **Confidence Calibration**: Improve confidence score calculation using calibrated models
5. **Template Learning**: Automatic template learning from labeled documents
6. **GPU Acceleration**: Optimize for GPU inference for faster processing

## Testing

Test the APIs using the provided examples or the interactive docs at `/docs`. Sample test files should be placed in a `test_samples/` directory.

## License

This project is open-source and available under the MIT License.
