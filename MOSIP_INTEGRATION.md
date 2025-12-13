# MOSIP Integration Guide

## Overview

The OCR system now includes full integration with MOSIP (Modular Open Source Identity Platform) modules, enabling end-to-end identity registration workflows.

## Features Implemented

### ✅ All Bonus Tasks Completed

1. **MOSIP Module Integration**
   - ✅ Pre-registration API integration
   - ✅ Registration Client API integration
   - ✅ Registration status checking
   - ✅ Android Registration Client support (via API)

2. **Capture Quality Score**
   - ✅ Blur detection (Laplacian variance)
   - ✅ Lighting quality (brightness analysis)
   - ✅ Contrast analysis
   - ✅ Overall quality score (0-1 scale)
   - ✅ Displayed after every scan with retake recommendations

3. **Multi-page Document Support**
   - ✅ Full support for PDFs with multiple pages
   - ✅ Processes all pages by default (configurable)
   - ✅ Aggregates results from all pages
   - ✅ Page-by-page confidence tracking

4. **Real-time OCR Confidence Zones**
   - ✅ Word-level bounding boxes
   - ✅ Per-word confidence scores
   - ✅ Visual feedback on scanned document
   - ✅ Grouped by confidence level (High/Medium/Low)

## API Endpoints

### 1. MOSIP Pre-registration

```bash
POST /api/v1/mosip/pre-registration
```

**Request:**
- `file`: Document image/PDF
- `document_type`: ID_CARD, FORM, or CERTIFICATE

**Response:**
```json
{
  "ocr_extraction": {
    "structured_data": {...},
    "raw_text": "...",
    "quality_score": {
      "quality": 0.85,
      "blur_score": 0.90,
      "brightness": 0.80,
      "contrast": 0.85
    }
  },
  "mosip_response": {...},
  "pre_registration_id": "PRE-20251213123456"
}
```

### 2. MOSIP Registration Client

```bash
POST /api/v1/mosip/registration
```

**Request:**
- `file`: Document image/PDF
- `document_type`: Document type
- `pre_registration_id`: Pre-registration ID from MOSIP

**Response:**
```json
{
  "ocr_extraction": {...},
  "mosip_response": {...},
  "registration_id": "REG-20251213123456"
}
```

### 3. Registration Status

```bash
GET /api/v1/mosip/status/{registration_id}
```

**Response:**
```json
{
  "response": {
    "registrationId": "REG-...",
    "status": "PROCESSING",
    "message": "..."
  }
}
```

## Configuration

### Environment Variables

Set these to connect to actual MOSIP services:

```bash
# MOSIP Pre-registration Service URL
MOSIP_PRE_REGISTRATION_URL=http://localhost:9090/preregistration/v1

# MOSIP Registration Client URL
MOSIP_REGISTRATION_CLIENT_URL=http://localhost:9091/registration-processor/v1

# API Key (if required)
MOSIP_API_KEY=your_api_key_here
```

### Default Behavior

If MOSIP URLs are not configured, the system returns **mock responses** for demonstration purposes. This allows testing the integration flow without a running MOSIP instance.

## Web UI Features

### Quality Score Display

After every scan, the UI displays:
- **Overall Quality Score** (0-100%)
- **Blur Score** (detection of image blur)
- **Brightness** (lighting quality)
- **Contrast** (text clarity)
- **Retake Recommendation** (if quality < 70%)

### Confidence Zones

Real-time feedback showing:
- **High Confidence Zones** (green) - >80% confidence
- **Medium Confidence Zones** (yellow) - 60-80% confidence
- **Low Confidence Zones** (red) - <60% confidence

Each zone shows:
- Extracted text
- Confidence percentage
- Bounding box coordinates (for visual overlay)

### Multi-page Support

- Automatically processes all pages in PDF
- Shows page-by-page results
- Aggregates structured data across pages

## Usage Examples

### Via Web UI

1. Go to **MOSIP Integration** tab
2. Upload document
3. Select document type
4. Choose service (Pre-registration or Registration)
5. Click "Submit to MOSIP"
6. View results including:
   - OCR extraction results
   - Quality scores
   - MOSIP response
   - Pre-registration/Registration ID

### Via API

```bash
# Pre-registration
curl -X POST "http://localhost:8000/api/v1/mosip/pre-registration" \
  -F "file=@document.pdf" \
  -F "document_type=ID_CARD"

# Registration (with Pre-reg ID)
curl -X POST "http://localhost:8000/api/v1/mosip/registration" \
  -F "file=@document.pdf" \
  -F "document_type=ID_CARD" \
  -F "pre_registration_id=PRE-20251213123456"

# Check Status
curl "http://localhost:8000/api/v1/mosip/status/REG-20251213123456"
```

## Quality Score Interpretation

- **Quality > 0.7**: Good quality, proceed
- **Quality 0.5-0.7**: Acceptable, may need review
- **Quality < 0.5**: Poor quality, recommend retake

### Component Scores

- **Blur Score**: Higher is better (sharp image)
- **Brightness**: 0.4-0.6 is optimal (not too dark/bright)
- **Contrast**: Higher is better (clear text)

## End-to-End Flow

1. **Document Upload** → User uploads identity document
2. **OCR Extraction** → System extracts text and structured data
3. **Quality Check** → System calculates quality scores
4. **MOSIP Pre-registration** → Submit to MOSIP Pre-registration service
5. **Get Pre-reg ID** → Receive Pre-registration ID
6. **MOSIP Registration** → Submit to Registration Client with Pre-reg ID
7. **Get Registration ID** → Receive Registration ID
8. **Status Check** → Monitor registration status

## Notes

- **Mock Mode**: If MOSIP services are not configured, the system works in mock mode for demonstration
- **Quality Scores**: Always calculated and displayed for user feedback
- **Confidence Zones**: Always included for real-time OCR feedback
- **Multi-page**: Enabled by default for complete document processing

