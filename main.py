"""
Main FastAPI application for OCR Text Extraction and Verification System
"""
import os
# Force transformers to use PyTorch, not TensorFlow (set before any imports)
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TORCH'] = '1'

import logging
import shutil
import io
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

from app.ocr_engine import TROCREngine
from app.preprocessor import ImagePreprocessor
from app.field_extractor import FieldExtractor
from app.verifier import DataVerifier
from app.models import OCRExtractionResponse, VerificationResponse, VerificationFieldResult
from app.mosip_integration import get_mosip_integration

# Create necessary directories first
os.makedirs('uploads', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OCR Text Extraction and Verification API",
    description="End-to-end OCR system for document text extraction and verification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web UI
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components (lazy loading)
ocr_engine: Optional[TROCREngine] = None
preprocessor = ImagePreprocessor()
field_extractor = FieldExtractor()
verifier = DataVerifier()

# Setup templates for UI
templates = Jinja2Templates(directory="templates")


def get_ocr_engine():
    """Lazy initialization of OCR engine - Uses Tesseract by default (fast and reliable)"""
    global ocr_engine
    if ocr_engine is None:
        # Default to Tesseract for speed and reliability, allow override via env var
        env_prefer = os.getenv("USE_TESSERACT_FIRST", "true").lower() == "true"
        prefer_tesseract = env_prefer  # Default: Tesseract first (fast and reliable)
        use_large_model = os.getenv("USE_TROCR_LARGE", "false").lower() == "true"
        if prefer_tesseract:
            logger.info("Initializing OCR engine: Tesseract primary (faster), TrOCR fallback")
        else:
            logger.info("Initializing OCR engine: TrOCR primary (default), Tesseract fallback")
        if use_large_model:
            logger.info("Using TrOCR LARGE model (slower, higher quality)")
        ocr_engine = TROCREngine(prefer_tesseract=prefer_tesseract, use_large_model=use_large_model)
    return ocr_engine


def get_confidence_zones(image: Image.Image, lang: str = 'eng') -> list:
    """
    Get OCR confidence zones using Tesseract data (word boxes with confidence)
    
    Args:
        image: PIL Image
        lang: Language code for OCR (default: 'eng')
    """
    zones = []
    try:
        import pytesseract
        # Use the specified language for confidence zones
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
        n = len(data['text'])
        for i in range(n):
            txt = data['text'][i]
            conf = float(data['conf'][i])
            if txt.strip() and conf > 0:
                zones.append({
                    "text": txt,
                    "confidence": conf / 100.0,
                    "x": int(data['left'][i]),
                    "y": int(data['top'][i]),
                    "width": int(data['width'][i]),
                    "height": int(data['height'][i]),
                })
    except Exception as e:
        logger.warning(f"Could not compute confidence zones: {e}")
    return zones


def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    """Save uploaded file to destination"""
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return destination
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")


def load_images_from_file(file_path: Path, max_pages: int = None) -> list:
    """Load images from file (supports multi-page PDF)
    
    Args:
        file_path: Path to file
        max_pages: Maximum pages to load (None = all, 1 = first page only for speed)
    """
    file_ext = file_path.suffix.lower()
    
    if file_ext == '.pdf':
        if not PDF_SUPPORT:
            error_msg = (
                "PDF support is not available. Please install PyMuPDF: pip install PyMuPDF"
            )
            logger.error("PyMuPDF not available for PDF conversion")
            raise HTTPException(status_code=400, detail=error_msg)
        
        logger.info("Converting PDF to images using PyMuPDF...")
        images = []
        try:
            pdf_document = fitz.open(str(file_path))
            total_pages = len(pdf_document)
            if total_pages == 0:
                pdf_document.close()
                raise HTTPException(status_code=400, detail="PDF file is empty")
            
            # Limit pages for speed (process first page only by default)
            pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
            
            # Higher DPI for better OCR accuracy, especially for Hindi
            zoom = 400 / 72  # 400 DPI (increased from 300 for better quality)
            mat = fitz.Matrix(zoom, zoom)
            for page_num in range(pages_to_process):
                page = pdf_document[page_num]
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                images.append(image)
            pdf_document.close()
            logger.info(f"PDF converted to {len(images)} page(s) (of {total_pages} total)")
            return images
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error converting PDF: {e}")
            raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")
    else:
        # Load image directly
        try:
            image = Image.open(file_path)
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            return [image]
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise HTTPException(status_code=400, detail=f"Error loading image: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting OCR API server...")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Root endpoint - redirects to UI"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/ui", response_class=HTMLResponse)
async def ui(request: Request):
    """Web UI endpoint"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/v1/ocr/extract", response_model=OCRExtractionResponse)
async def extract_ocr(
    file: UploadFile = File(..., description="Image or PDF file"),
    document_type: str = Form(..., description="Document type: ID_CARD, FORM, CERTIFICATE"),
    language: str = Form("eng", description="Language code: 'eng' (English), 'hin' (Hindi), 'eng+hin' (multilingual)"),
    handwritten: str = Form("false", description="Set to 'true' if document contains handwritten text"),
    all_pages: str = Form("false", description="Process all PDF pages (default: false, faster)"),
    include_quality: str = Form("true", description="Include quality metrics (default: true for MOSIP integration)"),
    include_zones: str = Form("true", description="Include confidence zones (default: true for real-time feedback)")
):
    """
    OCR Extraction API
    
    Extracts text and structured fields from uploaded document
    """
    try:
        # Validate document type
        valid_types = ['ID_CARD', 'FORM', 'CERTIFICATE']
        if document_type.upper() not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid document_type. Must be one of: {', '.join(valid_types)}"
            )
        
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        if PDF_SUPPORT:
            valid_extensions.append('.pdf')
        if file_ext not in valid_extensions:
            if file_ext == '.pdf' and not PDF_SUPPORT:
                raise HTTPException(
                    status_code=400,
                    detail="PDF support is not available. Please install PyMuPDF: pip install PyMuPDF"
                )
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Supported: {', '.join(valid_extensions)}"
            )
        
        logger.info(f"Processing file: {file.filename}, type: {document_type}")
        
        # Save uploaded file
        file_path = Path("uploads") / file.filename
        save_upload_file(file, file_path)
        
        try:
            # Load images (process only first page by default for speed)
            process_all = all_pages.lower() == "true"
            images = load_images_from_file(file_path, max_pages=None if process_all else 1)
            
            all_raw = []
            page_confidences = []
            structured_aggregate = {}
            confidence_aggregate = {}
            field_details_all = []
            
            # Always calculate quality and zones for better user feedback (especially for Hindi)
            quality_score = {}
            confidence_zones = []
            if images:
                quality_score = preprocessor.quality_metrics(images[0])
                # Use the same language for confidence zones as OCR
                confidence_zones = get_confidence_zones(images[0], lang=language)
            
            # Use Tesseract by default (fast and reliable)
            engine = get_ocr_engine()
            
            for idx, image in enumerate(images):
                # Prepare image (TrOCR needs RGB)
                if image.mode != "RGB":
                    processed_image = image.convert("RGB")
                else:
                    processed_image = image
                
                is_handwritten = handwritten.lower() == "true"
                logger.info(f"Running OCR on page {idx+1}/{len(images)} with language: {language}, handwritten: {is_handwritten}...")
                raw_text, ocr_confidence = engine.extract_text(processed_image, lang=language, handwritten=is_handwritten)
                all_raw.append(raw_text or "")
                page_confidences.append(ocr_confidence or 0.0)
                
                # Extract structured fields per page
                field_results = field_extractor.extract_fields(
                    raw_text or "",
                    document_type,
                    ocr_confidence or 0.8
                )
                # Merge: keep first non-empty for each field
                for k, v in field_results.items():
                    if k not in structured_aggregate or not structured_aggregate.get(k):
                        structured_aggregate[k] = v[0]
                        confidence_aggregate[k] = v[1]
                field_details_all.extend([
                    {"field": k, "value": v[0], "confidence": v[1]}
                    for k, v in field_results.items()
                ])
            
            # Clean up raw text - remove garbage OCR artifacts
            def clean_raw_text(text: str) -> str:
                """Remove garbage OCR artifacts and normalize text"""
                if not text:
                    return ""
                lines = text.split('\n')
                cleaned_lines = []
                for line in lines:
                    line = line.strip()
                    # Skip lines that are mostly garbage (single characters, random symbols)
                    if len(line) > 0:
                        # Skip lines that are mostly non-alphanumeric
                        alnum_ratio = sum(1 for c in line if c.isalnum() or c.isspace()) / len(line) if line else 0
                        if alnum_ratio > 0.3:  # At least 30% alphanumeric
                            # Skip very short lines with mostly symbols
                            if len(line) > 2 or (len(line) <= 2 and line.isalnum()):
                                cleaned_lines.append(line)
                return '\n'.join(cleaned_lines)
            
            raw_text_combined = "\n\n".join([clean_raw_text(text) for text in all_raw])
            # Overall confidence: mean of page confidences or mean of field scores
            if confidence_aggregate:
                overall_confidence = sum(confidence_aggregate.values()) / len(confidence_aggregate)
            elif page_confidences:
                overall_confidence = sum(page_confidences) / len(page_confidences)
            else:
                overall_confidence = 0.0
            
            if not raw_text:
                logger.warning("No text extracted from document")
                raw_text = ""
                ocr_confidence = 0.0
            
            response = OCRExtractionResponse(
                document_type=document_type.upper(),
                raw_text=raw_text_combined,
                structured_data=structured_aggregate,
                confidence_scores=confidence_aggregate,
                overall_confidence=overall_confidence,
                field_details=field_details_all,
                quality_score=quality_score,
                confidence_zones=confidence_zones
            )
            
            logger.info(f"Extraction completed. Overall confidence: {overall_confidence:.2f}")
            return response
        
        finally:
            # Clean up uploaded file
            if file_path.exists():
                file_path.unlink()
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in OCR extraction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/v1/mosip/pre-registration")
async def mosip_pre_registration(
    file: UploadFile = File(..., description="Document image or PDF"),
    document_type: str = Form(..., description="Document type: ID_CARD, FORM, CERTIFICATE")
):
    """
    MOSIP Pre-registration Integration
    Extracts data from document and submits to MOSIP Pre-registration service
    """
    try:
        # Save file temporarily
        file_path = Path("uploads") / file.filename
        save_upload_file(file, file_path)
        
        # Extract OCR data with quality and zones enabled
        images = load_images_from_file(file_path, max_pages=None)
        if not images:
            raise HTTPException(status_code=400, detail="Could not load images from file")
        
        engine = get_ocr_engine()
        all_raw = []
        structured_aggregate = {}
        quality_score = preprocessor.quality_metrics(images[0]) if images else {}
        
        for image in images:
            if image.mode != "RGB":
                processed_image = image.convert("RGB")
            else:
                processed_image = image
            raw_text, ocr_confidence = engine.extract_text(processed_image, lang='eng', handwritten=False)
            all_raw.append(raw_text or "")
            
            field_results = field_extractor.extract_fields(
                raw_text or "",
                document_type,
                ocr_confidence or 0.8
            )
            for k, v in field_results.items():
                if k not in structured_aggregate or not structured_aggregate.get(k):
                    structured_aggregate[k] = v[0]
        
        # Get document images as bytes for MOSIP
        image_bytes_list = []
        for img in images:
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            image_bytes_list.append(img_bytes.getvalue())
        
        # Submit to MOSIP
        mosip = get_mosip_integration()
        result = mosip.submit_pre_registration(
            extracted_data=structured_aggregate,
            document_images=image_bytes_list,
            quality_scores=quality_score
        )
        
        return {
            "ocr_extraction": {
                "structured_data": structured_aggregate,
                "raw_text": "\n\n".join(all_raw),
                "quality_score": quality_score
            },
            "mosip_response": result,
            "pre_registration_id": result.get("response", {}).get("preRegistrationId") if isinstance(result, dict) and "response" in result else None
        }
    
    except Exception as e:
        logger.error(f"Error in MOSIP Pre-registration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"MOSIP integration error: {str(e)}")
    finally:
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()


@app.post("/api/v1/mosip/registration")
async def mosip_registration(
    pre_registration_id: str = Form(..., description="Pre-registration ID from MOSIP"),
    file: UploadFile = File(..., description="Document image or PDF"),
    document_type: str = Form(..., description="Document type")
):
    """
    MOSIP Registration Client Integration
    Submits registration data to MOSIP Registration Client
    """
    try:
        # Save file temporarily
        file_path = Path("uploads") / file.filename
        save_upload_file(file, file_path)
        
        # Extract OCR data
        images = load_images_from_file(file_path, max_pages=None)
        if not images:
            raise HTTPException(status_code=400, detail="Could not load images from file")
        
        engine = get_ocr_engine()
        structured_aggregate = {}
        
        for image in images:
            if image.mode != "RGB":
                processed_image = image.convert("RGB")
            else:
                processed_image = image
            raw_text, ocr_confidence = engine.extract_text(processed_image, lang='eng', handwritten=False)
            
            field_results = field_extractor.extract_fields(
                raw_text or "",
                document_type,
                ocr_confidence or 0.8
            )
            for k, v in field_results.items():
                if k not in structured_aggregate or not structured_aggregate.get(k):
                    structured_aggregate[k] = v[0]
        
        # Submit to MOSIP Registration Client
        mosip = get_mosip_integration()
        result = mosip.submit_registration(
            pre_registration_id=pre_registration_id,
            extracted_data=structured_aggregate
        )
        
        return {
            "ocr_extraction": {
                "structured_data": structured_aggregate
            },
            "mosip_response": result,
            "registration_id": result.get("response", {}).get("registrationId") if isinstance(result, dict) and "response" in result else None
        }
    
    except Exception as e:
        logger.error(f"Error in MOSIP Registration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"MOSIP integration error: {str(e)}")
    finally:
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()


@app.get("/api/v1/mosip/status/{registration_id}")
async def mosip_registration_status(registration_id: str):
    """
    Get MOSIP Registration Status
    """
    try:
        mosip = get_mosip_integration()
        result = mosip.get_registration_status(registration_id)
        return result
    except Exception as e:
        logger.error(f"Error getting MOSIP status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/v1/verify", response_model=VerificationResponse)
async def verify_data(
    file: UploadFile = File(..., description="Original scanned document"),
    form_data: str = Form(..., description="JSON string with user-filled form data")
):
    """
    Data Verification API
    
    Verifies user-filled form data against extracted document data
    """
    try:
        import json
        
        # Parse form data JSON
        try:
            form_data_dict = json.loads(form_data)
            if not isinstance(form_data_dict, dict):
                raise ValueError("form_data must be a JSON object")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in form_data: {str(e)}")
        
        logger.info(f"Verifying form data with {len(form_data_dict)} fields")
        
        # Save uploaded file
        file_path = Path("uploads") / file.filename
        save_upload_file(file, file_path)
        
        try:
            # Load images (multi-page support)
            images = load_images_from_file(file_path)
            engine = get_ocr_engine()
            
            all_raw = []
            page_confidences = []
            structured_aggregate = {}
            
            for image in images:
                if image.mode != "RGB":
                    processed_image = image.convert("RGB")
                else:
                    processed_image = image
                
                logger.info("Extracting document data for verification...")
                # Try to detect document type from form data or use default
                # Check if form data has certificate-specific fields
                if any(field in form_data_dict for field in ['course', 'certificate_number', 'date']):
                    document_type = 'CERTIFICATE'
                elif any(field in form_data_dict for field in ['dob', 'id_number']):
                    document_type = 'ID_CARD'
                else:
                    document_type = 'FORM'  # Default for forms
                
                logger.info(f"Using document type: {document_type} for verification")
                
                raw_text, ocr_confidence = engine.extract_text(processed_image)
                all_raw.append(raw_text or "")
                page_confidences.append(ocr_confidence or 0.0)
                
                field_results = field_extractor.extract_fields(
                    raw_text or "",
                    document_type,
                    ocr_confidence or 0.8
                )
                
                for k, v in field_results.items():
                    if k not in structured_aggregate or not structured_aggregate.get(k):
                        structured_aggregate[k] = v[0]
            
            document_data = structured_aggregate
            
            # Verify form data against document data
            logger.info("Verifying form data...")
            verification_result = verifier.verify(document_data, form_data_dict)
            
            # Format response
            field_results_list = [
                VerificationFieldResult(
                    field=r['field'],
                    document_value=r['document_value'],
                    form_value=r['form_value'],
                    status=r['status'],
                    similarity_score=r['similarity_score']
                )
                for r in verification_result['field_results']
            ]
            
            response = VerificationResponse(
                overall_verification_score=verification_result['overall_verification_score'],
                field_results=field_results_list,
                mismatched_fields=verification_result['mismatched_fields'],
                missing_fields=verification_result['missing_fields']
            )
            
            logger.info(f"Verification completed. Score: {response.overall_verification_score:.2f}")
            return response
        
        finally:
            # Clean up uploaded file
            if file_path.exists():
                file_path.unlink()
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in verification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "OCR API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

