"""
OCR Engine module using TrOCR (Transformer-based OCR)
"""
import os
# Force transformers to use PyTorch, not TensorFlow
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TORCH'] = '1'

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import logging
import numpy as np
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class TROCREngine:
    """TrOCR-based OCR engine with Tesseract fallback"""
    
    def __init__(self, model_name: str = "microsoft/trocr-base-printed", prefer_tesseract: bool = False, use_large_model: bool = False):
        """
        Initialize TrOCR model
        
        Args:
            model_name: HuggingFace model identifier
                - "microsoft/trocr-base-printed": For printed text (default)
                - "microsoft/trocr-base-handwritten": For handwritten text
                - "microsoft/trocr-large-printed": Larger model, more accurate (better for certificates)
        """
        # For certificates, consider using large model for better accuracy
        if use_large_model:
            model_name = "microsoft/trocr-large-printed"
            logger.info("Using TrOCR LARGE model for better accuracy (slower but more accurate)")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prefer_tesseract = prefer_tesseract
        
        # Check if Tesseract is available
        self.tesseract_available = False
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info("Tesseract OCR is available")
        except:
            logger.warning("Tesseract OCR not available - install pytesseract and tesseract-ocr")
        
        if not prefer_tesseract:
            logger.info(f"Loading TrOCR model: {model_name} on {self.device}")
            try:
                self.processor = TrOCRProcessor.from_pretrained(model_name)
                self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                logger.info("TrOCR model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading TrOCR model: {e}")
                if self.tesseract_available:
                    logger.info("Falling back to Tesseract-only mode")
                    self.prefer_tesseract = True
                else:
                    raise
        else:
            logger.info("Using Tesseract as primary OCR engine")
            self.processor = None
            self.model = None
    
    def extract_text(self, image: Image.Image, return_confidence: bool = True, lang: str = 'eng', handwritten: bool = False) -> Tuple[str, Optional[float]]:
        """
        Extract text from image
        
        Args:
            image: PIL Image
            return_confidence: Whether to return confidence score
            lang: Language code (e.g., 'eng', 'hin', 'eng+hin')
            handwritten: Whether text is handwritten
        
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        # Save original image for fallback
        original_image = image.copy() if hasattr(image, 'copy') else image
        
        # If Tesseract is preferred or TrOCR not available, use Tesseract
        # For non-English languages (like Hindi) or handwritten text, prefer Tesseract
        if self.prefer_tesseract or self.model is None or lang != 'eng' or handwritten:
            if self.tesseract_available:
                if lang != 'eng':
                    logger.info(f"Using Tesseract as primary OCR for language: {lang} (TrOCR doesn't support non-English well)")
                elif handwritten:
                    logger.info("Using Tesseract as primary OCR for handwritten text")
                else:
                    logger.info("Using Tesseract as primary OCR")
                return self.extract_text_tesseract(original_image, lang=lang, handwritten=handwritten)
            else:
                if lang != 'eng':
                    raise ValueError(f"Tesseract not available but required for language: {lang}")
                raise ValueError("Tesseract not available and TrOCR not loaded")
        
        try:
            # Ensure image is RGB (TrOCR requires RGB)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Enhance image quality for TrOCR (subtle enhancement, not aggressive)
            # TrOCR works better with slightly enhanced contrast
            try:
                from PIL import ImageEnhance
                # Slight contrast enhancement (1.1 = 10% increase)
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.1)
                # Slight sharpness enhancement
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.05)
                logger.info("Applied subtle image enhancement for TrOCR")
            except Exception as e:
                logger.warning(f"Image enhancement failed: {e}, using original")
            
            # TrOCR works best with images around 384px height (its training size)
            # For certificates, we need to process in a way that works with TrOCR's design
            width, height = image.size
            
            # Strategy: For tall certificates, we'll process in horizontal strips
            # TrOCR is designed for single-line text, so we need to adapt
            optimal_height = 384  # TrOCR's optimal input height
            max_width = 1600  # Reasonable max width
            
            # If image is too tall, we'll need to process it differently
            # For now, resize to optimal height while maintaining aspect ratio
            if height > optimal_height * 2:
                # Very tall image - resize but keep reasonable size
                scale = (optimal_height * 1.5) / height  # Use 1.5x optimal for better quality
                new_width = int(width * scale)
                new_height = int(height * scale)
                if new_width > max_width:
                    # If still too wide, scale down width
                    scale = max_width / new_width
                    new_width = max_width
                    new_height = int(new_height * scale)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized tall image from {width}x{height} to {new_width}x{new_height} for TrOCR")
            elif width > max_width:
                # Too wide, scale down
                scale = max_width / width
                new_width = max_width
                new_height = int(height * scale)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized wide image from {width}x{height} to {new_width}x{new_height} for TrOCR")
            elif height < 64:
                # Too small, upscale
                scale = 128 / height
                new_width = int(width * scale)
                new_height = 128
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Upscaled small image from {width}x{height} to {new_width}x{new_height} for TrOCR")
            else:
                logger.info(f"Image size {width}x{height} is acceptable for TrOCR")
            
            # TrOCR processor handles normalization internally
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate text with optimized parameters for better extraction
            with torch.no_grad():
                # Get tokenizer IDs safely
                pad_token_id = self.processor.tokenizer.pad_token_id
                eos_token_id = self.processor.tokenizer.eos_token_id
                if pad_token_id is None:
                    pad_token_id = self.processor.tokenizer.unk_token_id
                
                # TrOCR generation parameters optimized for certificates
                # Use longer sequences and better beam search
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=1024,  # Increased for longer text
                    num_beams=10,  # More beams for better quality
                    early_stopping=False,  # Don't stop early
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    no_repeat_ngram_size=3,  # Prevent repetition
                    length_penalty=2.0,  # Encourage longer sequences
                    do_sample=False,
                    repetition_penalty=1.2
                )
                
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Try with even longer max_length if result seems short
                if len(generated_text.strip()) < 100:
                    logger.info("Trying with longer max_length for better extraction...")
                    generated_ids = self.model.generate(
                        pixel_values,
                        max_length=2048,  # Even longer
                        num_beams=10,
                        early_stopping=False,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        no_repeat_ngram_size=3,
                        length_penalty=2.0,
                        do_sample=False,
                        repetition_penalty=1.2
                    )
                    generated_text2 = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    if len(generated_text2) > len(generated_text):
                        generated_text = generated_text2
                        logger.info(f"Longer generation extracted {len(generated_text)} chars")
                
                logger.info(f"TrOCR generated {len(generated_ids[0])} tokens, text length: {len(generated_text)}")
            
            # Calculate confidence based on text quality
            confidence = None
            if return_confidence:
                try:
                    text_length = len(generated_text.strip())
                    if text_length > 0:
                        # Better confidence calculation
                        # Base confidence increases with text length (up to a point)
                        length_factor = min(text_length / 200.0, 1.0)  # Normalize to 200 chars
                        
                        # Check for common certificate words (indicates good extraction)
                        certificate_keywords = ['certificate', 'awarded', 'completed', 'course', 'name', 'date', 'issued']
                        keyword_count = sum(1 for keyword in certificate_keywords if keyword.lower() in generated_text.lower())
                        keyword_factor = min(keyword_count / len(certificate_keywords), 1.0)
                        
                        # Combine factors
                        confidence = 0.6 + (length_factor * 0.2) + (keyword_factor * 0.2)
                        confidence = min(0.95, max(0.3, confidence))
                    else:
                        confidence = 0.2  # Very low for empty text
                except Exception as e:
                    logger.warning(f"Could not calculate confidence: {e}")
                    confidence = 0.7  # Default confidence
            
            logger.info(f"Extracted text (length: {len(generated_text)}, confidence: {confidence:.2f})")
            
            # If very little text extracted OR hallucination detected, try region-based extraction
            if len(generated_text.strip()) < 50 or any(word in generated_text.lower() for word in ['thank', 'thanks', 'item']):
                logger.warning(f"TrOCR issue detected: '{generated_text[:100]}' - trying region-based extraction...")
                try:
                    region_text = self.extract_text_regions(original_image)
                    if len(region_text.strip()) > len(generated_text.strip()) and len(region_text.strip()) > 20:
                        logger.info(f"Region-based extraction got {len(region_text)} chars vs {len(generated_text)} chars - using region-based")
                        return region_text, confidence or 0.8
                    elif len(generated_text.strip()) < 10:
                        # If TrOCR gave almost nothing, use region-based even if it's similar
                        logger.info("TrOCR gave almost no text, using region-based result")
                        return region_text if region_text.strip() else generated_text, confidence or 0.8
                except Exception as region_error:
                    logger.warning(f"Region-based extraction failed: {region_error}")
            
            # If still minimal text OR hallucination, try Tesseract fallback immediately
            hallucination_detected = any(word in generated_text.lower() for word in ['thank', 'thanks', 'item', 'hello'])
            if len(generated_text.strip()) < 30 or hallucination_detected:
                if hallucination_detected:
                    logger.error(f"TrOCR HALLUCINATION DETECTED: '{generated_text}' - immediately using Tesseract")
                else:
                    logger.info("TrOCR extracted minimal text, trying Tesseract fallback...")
                try:
                    # Use original image for Tesseract (works better with full resolution)
                    tesseract_text, tesseract_conf = self.extract_text_tesseract(original_image, lang=lang, handwritten=handwritten)
                    if len(tesseract_text.strip()) > len(generated_text.strip()) or hallucination_detected:
                        logger.info(f"Tesseract extracted {len(tesseract_text)} chars vs TrOCR's {len(generated_text)} chars - using Tesseract")
                        return tesseract_text, tesseract_conf
                    else:
                        logger.info(f"Tesseract extracted {len(tesseract_text)} chars, TrOCR extracted {len(generated_text)} chars - using TrOCR")
                except Exception as fallback_error:
                    logger.warning(f"Tesseract fallback failed: {fallback_error}, using TrOCR result")
            
            return generated_text, confidence
        
        except Exception as e:
            logger.error(f"Error in TrOCR extraction: {e}")
            # Try fallback to Tesseract if available
            try:
                logger.info("TrOCR failed, attempting Tesseract fallback...")
                return self.extract_text_tesseract(image, lang=lang, handwritten=handwritten)
            except Exception as fallback_error:
                logger.error(f"Both TrOCR and Tesseract failed: {fallback_error}")
                raise
    
    def extract_text_tesseract(self, image: Image.Image, lang: str = 'eng', handwritten: bool = False) -> Tuple[str, Optional[float]]:
        """Fallback OCR using Tesseract - supports multiple languages and handwritten text
        
        Args:
            image: PIL Image
            lang: Language code (e.g., 'eng', 'hin', 'eng+hin' for multilingual)
            handwritten: If True, uses PSM modes optimized for handwritten text
        """
        try:
            import pytesseract
            
            # RUNTIME CHECK: Verify language availability
            original_lang = lang  # Keep original for logging
            try:
                available_langs = pytesseract.get_languages()
                logger.info(f"Available Tesseract languages: {available_langs}")
                requested_langs = lang.split('+')
                missing_langs = [l for l in requested_langs if l not in available_langs]
                
                # CRITICAL CHECK: If Hindi is requested, verify it exists
                if 'hin' in requested_langs:
                    if 'hin' not in available_langs:
                        logger.error("=" * 70)
                        logger.error("CRITICAL ERROR: Hindi language data (hin) is NOT installed!")
                        logger.error("=" * 70)
                        logger.error("Runtime check failed: 'hin' not found in pytesseract.get_languages()")
                        logger.error(f"Available languages: {available_langs}")
                        logger.error("")
                        logger.error("IMMEDIATE ACTION REQUIRED:")
                        logger.error("  1. Download: https://github.com/tesseract-ocr/tessdata/raw/main/hin.traineddata")
                        logger.error("  2. Place in: C:\\Program Files\\Tesseract-OCR\\tessdata\\hin.traineddata")
                        logger.error("  3. Verify: Run 'python -c \"import pytesseract; print(\\\"hin\\\" in pytesseract.get_languages())\"'")
                        logger.error("  4. Restart the FastAPI server")
                        logger.error("")
                        logger.error("Note: You may need Administrator rights to copy the file.")
                        logger.error("=" * 70)
                        # DO NOT fallback to English - raise error instead
                        raise ValueError(
                            "Hindi language data (hin.traineddata) is not installed. "
                            "Download from https://github.com/tesseract-ocr/tessdata/raw/main/hin.traineddata "
                            "and place in C:\\Program Files\\Tesseract-OCR\\tessdata\\"
                        )
                    else:
                        logger.info("✓ Runtime check passed: Hindi language data is available")
                
                if missing_langs:
                    logger.error(f"CRITICAL: Language(s) {missing_langs} not available in Tesseract!")
                    logger.error(f"Available languages: {available_langs}")
                    # For other missing languages (not Hindi), fallback to eng
                    if 'hin' not in missing_langs:
                        logger.warning(f"Language(s) {missing_langs} not available. Falling back to 'eng'.")
                        lang = 'eng'
                else:
                    logger.info(f"✓ Using Tesseract language(s): {lang}")
            except ValueError:
                # Re-raise ValueError (Hindi missing) - don't catch it
                raise
            except Exception as e:
                logger.error(f"Could not check available languages: {e}")
                # If Hindi was requested, still check it exists
                if 'hin' in lang:
                    try:
                        available_langs = pytesseract.get_languages()
                        if 'hin' not in available_langs:
                            raise ValueError(
                                "Hindi language data (hin.traineddata) is not installed. "
                                "Download from https://github.com/tesseract-ocr/tessdata/raw/main/hin.traineddata "
                                "and place in C:\\Program Files\\Tesseract-OCR\\tessdata\\"
                            )
                    except:
                        pass
                # Don't change lang - use what was requested
                logger.warning(f"Proceeding with requested language: {lang} (may fail if not installed)")
            
            # Preprocess image for better OCR results (especially for handwritten text and Hindi)
            processed_image = self._preprocess_for_tesseract(image, handwritten, lang)
            
            # Choose PSM mode based on content type
            # For Hindi, use specific PSM modes that work better with Devanagari script
            if 'hin' in lang.lower():
                configs = [
                    r'--oem 3 --psm 6',   # Uniform block (BEST for Hindi documents)
                    r'--oem 3 --psm 11',  # Sparse text (for mixed layout with Hindi)
                    r'--oem 3 --psm 4',   # Single column (for form-like Hindi text)
                    r'--oem 3 --psm 3',   # Fully automatic (fallback)
                    r'--oem 3 --psm 1',   # Automatic page segmentation with OSD
                ]
                logger.info("Using Hindi-optimized PSM modes for Devanagari script")
            elif handwritten:
                # Try multiple PSM modes for handwritten text - optimized for forms
                configs = [
                    r'--oem 3 --psm 11',  # Sparse text (BEST for handwritten forms with fields)
                    r'--oem 3 --psm 6',   # Uniform block (good for structured forms)
                    r'--oem 3 --psm 4',   # Single column (for form-like layouts)
                    r'--oem 3 --psm 7',   # Single line (for continuous handwritten text)
                    r'--oem 3 --psm 13',  # Raw line (treat image as single text line)
                ]
                logger.info("Using handwritten text optimized PSM modes")
            else:
                # Try multiple PSM modes for printed text
                configs = [
                    r'--oem 3 --psm 6',   # Uniform block (best for certificates)
                    r'--oem 3 --psm 11',  # Sparse text
                    r'--oem 3 --psm 3',   # Fully automatic (no PSM)
                ]
            
            best_text = ""
            best_config = configs[0]
            
            for config in configs:
                try:
                    # Explicitly pass lang parameter to ensure it's used
                    text = pytesseract.image_to_string(processed_image, lang=lang, config=config)
                    logger.debug(f"Config {config} with lang '{lang}' extracted {len(text.strip())} characters")
                    if len(text.strip()) > len(best_text.strip()):
                        best_text = text
                        best_config = config
                        logger.info(f"Best so far: Config {config} with lang '{lang}' extracted {len(text.strip())} characters")
                except Exception as e:
                    logger.warning(f"Config {config} with lang '{lang}' failed: {e}")
                    # For Hindi, DO NOT fallback to English - this would give wrong results
                    # Instead, log the error and continue trying other configs
                    if 'hin' in lang.lower():
                        logger.error(f"Hindi OCR failed with config {config}: {e}")
                        if 'language' in str(e).lower() or 'lang' in str(e).lower():
                            logger.error("CRITICAL: Hindi language data is NOT installed!")
                            logger.error("Run: python install_hindi_lang.py (as Administrator)")
                    continue
            
            # If still poor results, try with original image (less preprocessing)
            # Also try with different preprocessing strategies
            if len(best_text.strip()) < 100:
                logger.info("Trying with original image and alternative preprocessing...")
                
                # Try original image
                for config in [r'--oem 3 --psm 6', r'--oem 3 --psm 11', r'--oem 3 --psm 3']:
                    try:
                        text = pytesseract.image_to_string(image, lang=lang, config=config)
                        logger.debug(f"Original image with lang '{lang}' and {config} extracted {len(text.strip())} chars")
                        if len(text.strip()) > len(best_text.strip()):
                            best_text = text
                            best_config = config
                            logger.info(f"Original image with lang '{lang}' and {config} gave better result: {len(text.strip())} chars")
                    except Exception as e:
                        logger.debug(f"Original image with {config} failed: {e}")
                        continue
                
                # For Hindi, try multiple preprocessing strategies
                if 'hin' in lang.lower():
                    logger.info("Trying alternative preprocessing strategies for Hindi...")
                    try:
                        import cv2
                        import numpy as np
                        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        
                        # Strategy 1: Minimal preprocessing (just contrast)
                        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                        enhanced = clahe.apply(gray)
                        simple_processed = Image.fromarray(enhanced).convert('RGB')
                        
                        # Strategy 2: Original image (no preprocessing - sometimes better for Hindi)
                        original_rgb = image.convert('RGB')
                        
                        # Strategy 3: Grayscale only
                        gray_only = Image.fromarray(gray).convert('RGB')
                        
                        strategies = [
                            ("minimal_contrast", simple_processed),
                            ("original", original_rgb),
                            ("grayscale", gray_only)
                        ]
                        
                        for strategy_name, processed_img in strategies:
                            for config in [r'--oem 3 --psm 6', r'--oem 3 --psm 11', r'--oem 3 --psm 4']:
                                try:
                                    text = pytesseract.image_to_string(processed_img, lang=lang, config=config)
                                    # Check for Hindi characters
                                    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
                                    if len(text.strip()) > len(best_text.strip()) or (hindi_chars > 0 and sum(1 for c in best_text if '\u0900' <= c <= '\u097F') == 0):
                                        best_text = text
                                        best_config = config
                                        logger.info(f"{strategy_name} with {config} gave better result: {len(text.strip())} chars, {hindi_chars} Hindi chars")
                                except Exception as e:
                                    logger.debug(f"{strategy_name} with {config} failed: {e}")
                                    continue
                    except Exception as e:
                        logger.warning(f"Alternative preprocessing strategies failed: {e}")
            
            # Log final result with language info and Hindi character detection
            logger.info(f"Tesseract FINAL RESULT: {len(best_text)} characters using language '{lang}' with config {best_config}")
            if 'hin' in lang.lower():
                # Check if text contains actual Hindi characters (Devanagari Unicode range: U+0900 to U+097F)
                hindi_chars = sum(1 for c in best_text if '\u0900' <= c <= '\u097F')
                if hindi_chars == 0:
                    logger.error("=" * 70)
                    logger.error("CRITICAL ERROR: No Hindi (Devanagari) characters detected!")
                    logger.error("=" * 70)
                    logger.error(f"Language requested: {lang}")
                    logger.error(f"Text extracted: {len(best_text)} characters")
                    logger.error(f"Sample: {best_text[:100] if best_text else 'EMPTY'}")
                    logger.error("")
                    logger.error("This means Hindi language data is NOT properly installed.")
                    logger.error("")
                    logger.error("IMMEDIATE ACTION REQUIRED:")
                    logger.error("  1. Run as Administrator: python install_hindi_lang.py")
                    logger.error("  2. Restart the server")
                    logger.error("  3. Try again")
                    logger.error("")
                    logger.error("Manual installation:")
                    logger.error("  Download: https://github.com/tesseract-ocr/tessdata/raw/main/hin.traineddata")
                    logger.error("  Place in: C:\\Program Files\\Tesseract-OCR\\tessdata\\")
                    logger.error("  (You may need Administrator rights to copy the file)")
                    logger.error("=" * 70)
                else:
                    logger.info(f"SUCCESS: Detected {hindi_chars} Hindi (Devanagari) characters!")
                    logger.info(f"Hindi OCR is working correctly with language '{lang}'")
                if len(best_text.strip()) < 50:
                    logger.warning(f"Very little text extracted ({len(best_text)} chars) - may need better image quality")
            
            return best_text.strip(), 0.75  # Default confidence for Tesseract
        except ImportError:
            logger.error("Tesseract (pytesseract) not available as fallback. Install: pip install pytesseract")
            return "", 0.0
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return "", 0.0
    
    def _preprocess_for_tesseract(self, image: Image.Image, handwritten: bool = False, lang: str = 'eng') -> Image.Image:
        """Preprocess image for better Tesseract OCR results, especially for Hindi and handwritten text"""
        try:
            import cv2
            import numpy as np
            
            # Upscale image if too small (better for Hindi recognition)
            width, height = image.size
            min_dimension = min(width, height)
            if min_dimension < 1000:
                scale = 1500 / min_dimension
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Upscaled image from {width}x{height} to {new_width}x{new_height} for better OCR")
            
            # Convert PIL to OpenCV
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # For Hindi text, use specific preprocessing optimized for Devanagari script
            if 'hin' in lang.lower():
                # CRITICAL: Hindi/Devanagari needs very careful preprocessing
                # Enhance contrast significantly (Devanagari has complex character shapes)
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))  # Higher clip limit
                gray = clahe.apply(gray)
                
                # Apply gentle denoising (preserve character details)
                gray = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)  # Reduced strength
                
                # Use adaptive thresholding with larger block for Hindi
                # Devanagari characters are more complex, need larger context
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 25, 12  # Larger block size for Hindi
                )
                
                # Morphological operations to connect broken Devanagari characters
                kernel = np.ones((2, 2), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                
                # Additional dilation for better character recognition
                kernel_dilate = np.ones((1, 1), np.uint8)
                binary = cv2.dilate(binary, kernel_dilate, iterations=1)
                
            elif handwritten:
                # For handwritten text, use specialized preprocessing
                # First, enhance contrast to make handwriting clearer
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                
                # Apply gentle Gaussian blur to reduce noise (very gentle for handwritten)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                
                # Use adaptive thresholding with larger block size for handwritten text
                # This helps with variable handwriting quality
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 21, 10  # Larger block size for handwritten
                )
                
                # Morphological operations to connect broken characters
                kernel = np.ones((2, 2), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            else:
                # For printed English text, use standard preprocessing
                # Apply slight denoising
                denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
                
                # Enhance contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(denoised)
                
                # Apply thresholding for better contrast
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to RGB for Tesseract
            processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            
            # Convert back to PIL
            return Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}, using original image")
            return image
    
    def extract_text_regions(self, image: Image.Image) -> str:
        """
        Extract text from certificate by processing in horizontal regions
        TrOCR works best with single-line text, so we split tall certificates
        """
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            width, height = image.size
            logger.info(f"Processing certificate in regions: {width}x{height}")
            
            # Split into horizontal strips
            strip_height = 384  # TrOCR's optimal height
            num_strips = max(1, height // strip_height)
            
            all_text = []
            for i in range(num_strips):
                top = i * strip_height
                bottom = min((i + 1) * strip_height, height)
                strip = image.crop((0, top, width, bottom))
                
                # Process strip with TrOCR
                pixel_values = self.processor(images=strip, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        pixel_values,
                        max_length=512,
                        num_beams=5,
                        early_stopping=False,
                        pad_token_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.unk_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        no_repeat_ngram_size=2,
                        length_penalty=1.5
                    )
                    text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    if text.strip():
                        all_text.append(text)
            
            return "\n".join(all_text)
        except Exception as e:
            logger.error(f"Region-based extraction failed: {e}")
            return ""
