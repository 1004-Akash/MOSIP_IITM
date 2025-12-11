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
    """Lazy initialization of OCR engine"""
    global ocr_engine
    if ocr_engine is None:
        logger.info("Initializing OCR engine (prefer fast Tesseract first)...")
        # Prefer Tesseract for speed; TrOCR remains available as fallback for quality
        ocr_engine = TROCREngine(prefer_tesseract=True, use_large_model=False)
    return ocr_engine


def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    """Save uploaded file to destination"""
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return destination
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")


def load_image_from_file(file_path: Path) -> Image.Image:
    """Load image from file (supports PDF conversion using PyMuPDF)"""
    file_ext = file_path.suffix.lower()
    
    if file_ext == '.pdf':
        if not PDF_SUPPORT:
            error_msg = (
                "PDF support is not available. Please install PyMuPDF: pip install PyMuPDF"
            )
            logger.error("PyMuPDF not available for PDF conversion")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Convert PDF to image using PyMuPDF (fitz)
        logger.info("Converting PDF to image using PyMuPDF...")
        try:
            pdf_document = fitz.open(str(file_path))
            if len(pdf_document) == 0:
                raise HTTPException(status_code=400, detail="PDF file is empty")
            
            # Get first page
            page = pdf_document[0]
            
            # Convert to image with high DPI for better OCR quality
            # Use 300 DPI for good balance between quality and performance
            zoom = 300 / 72  # 300 DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to PIL Image (more efficient format)
            img_data = pix.tobytes("png")  # PNG is more efficient than PPM
            image = Image.open(io.BytesIO(img_data))
            
            # Clean up pixmap
            pix = None
            
            pdf_document.close()
            logger.info("PDF converted to image successfully")
            return image
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error converting PDF: {e}")
            raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")
    else:
        # Load image directly
        try:
            image = Image.open(file_path)
            # Convert to RGB if necessary (some formats like RGBA need conversion)
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            return image
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
    document_type: str = Form(..., description="Document type: ID_CARD, FORM, CERTIFICATE")
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
            # Load image
            image = load_image_from_file(file_path)
            
            # TrOCR works best with original RGB images - minimal preprocessing
            logger.info("Preparing image for OCR...")
            # Ensure RGB mode (TrOCR requires RGB)
            if image.mode != "RGB":
                processed_image = image.convert("RGB")
            else:
                processed_image = image
            
            # TrOCR's processor handles normalization internally
            # Aggressive preprocessing (grayscale, denoising) actually hurts TrOCR performance
            
            # Extract text using OCR
            logger.info("Running OCR...")
            engine = get_ocr_engine()
            raw_text, ocr_confidence = engine.extract_text(processed_image)
            
            if not raw_text:
                logger.warning("No text extracted from document")
                raw_text = ""
                ocr_confidence = 0.0
            
            # Extract structured fields
            logger.info("Extracting structured fields...")
            field_results = field_extractor.extract_fields(
                raw_text,
                document_type,
                ocr_confidence or 0.8
            )
            
            # Format response
            structured_data = {k: v[0] for k, v in field_results.items()}
            confidence_scores = {k: v[1] for k, v in field_results.items()}
            
            # Calculate overall confidence
            if confidence_scores:
                overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)
            else:
                overall_confidence = ocr_confidence or 0.0
            
            # Create field details
            field_details = [
                {"field": k, "value": v[0], "confidence": v[1]}
                for k, v in field_results.items()
            ]
            
            response = OCRExtractionResponse(
                document_type=document_type.upper(),
                raw_text=raw_text,
                structured_data=structured_data,
                confidence_scores=confidence_scores,
                overall_confidence=overall_confidence,
                field_details=field_details
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
            # Load image
            image = load_image_from_file(file_path)
            
            # TrOCR works best with original RGB images
            if image.mode != "RGB":
                processed_image = image.convert("RGB")
            else:
                processed_image = image
            
            # Extract text and fields (reuse OCR)
            logger.info("Extracting document data for verification...")
            engine = get_ocr_engine()
            raw_text, ocr_confidence = engine.extract_text(processed_image)
            
            # Determine document type (try to infer or use generic)
            document_type = 'ID_CARD'  # Default, could be enhanced with detection
            
            # Extract structured fields from document
            field_results = field_extractor.extract_fields(
                raw_text,
                document_type,
                ocr_confidence or 0.8
            )
            
            document_data = {k: v[0] for k, v in field_results.items()}
            
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

