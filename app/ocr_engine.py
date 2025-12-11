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
    
    def extract_text(self, image: Image.Image, return_confidence: bool = True) -> Tuple[str, Optional[float]]:
        """
        Extract text from image
        
        Args:
            image: PIL Image
            return_confidence: Whether to return confidence score
        
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        # Save original image for fallback
        original_image = image.copy() if hasattr(image, 'copy') else image
        
        # If Tesseract is preferred or TrOCR not available, use Tesseract
        if self.prefer_tesseract or self.model is None:
            if self.tesseract_available:
                logger.info("Using Tesseract as primary OCR")
                return self.extract_text_tesseract(original_image)
            else:
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
                    max_length=512,  # Start with 512, can be increased if needed
                    num_beams=10,  # Increased beams for better quality (was 5)
                    early_stopping=True,  # Stop at EOS token
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                    length_penalty=2.0,  # Encourage longer sequences
                    do_sample=False,  # Deterministic beam search
                    repetition_penalty=1.2  # Discourage repetition
                )
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Check if TrOCR is hallucinating (generating text not in image)
                # Common hallucinations: "THANK", "THANKS", "ITEM", single words
                hallucination_words = ['thank', 'thanks', 'item', 'hello', 'test']
                is_hallucination = (
                    len(generated_text.strip().split()) <= 2 and 
                    any(word in generated_text.lower() for word in hallucination_words)
                )
                
                if is_hallucination:
                    logger.warning(f"TrOCR appears to be hallucinating: '{generated_text}' - skipping to region-based extraction")
                    generated_text = ""  # Force region-based extraction
                
                # If text is very short or hallucination, try with longer max_length
                if (len(generated_text.strip()) < 50 and len(generated_ids[0]) < 100) or is_hallucination:
                    logger.info("TrOCR extracted short text, trying with longer max_length...")
                    generated_ids = self.model.generate(
                        pixel_values,
                        max_length=1024,  # Try longer
                        num_beams=10,
                        early_stopping=False,  # Don't stop early
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
                    tesseract_text, tesseract_conf = self.extract_text_tesseract(original_image)
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
                return self.extract_text_tesseract(image)
            except Exception as fallback_error:
                logger.error(f"Both TrOCR and Tesseract failed: {fallback_error}")
                raise
    
    def extract_text_tesseract(self, image: Image.Image) -> Tuple[str, Optional[float]]:
        """Fallback OCR using Tesseract - often more reliable for certificates"""
        try:
            import pytesseract
            # Tesseract works better with specific config for certificates
            # Use page segmentation mode 6 (uniform block of text) or 11 (sparse text)
            custom_config = r'--oem 3 --psm 6'  # OEM 3 = LSTM, PSM 6 = uniform block
            text = pytesseract.image_to_string(image, lang='eng', config=custom_config)
            
            # If that doesn't work well, try PSM 11 (sparse text)
            if len(text.strip()) < 50:
                logger.info("Trying alternative Tesseract config (PSM 11)...")
                custom_config = r'--oem 3 --psm 11'
                text2 = pytesseract.image_to_string(image, lang='eng', config=custom_config)
                if len(text2.strip()) > len(text.strip()):
                    text = text2
            
            logger.info(f"Tesseract extracted {len(text)} characters")
            return text.strip(), 0.75  # Default confidence for Tesseract
        except ImportError:
            logger.error("Tesseract (pytesseract) not available as fallback. Install: pip install pytesseract")
            return "", 0.0
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return "", 0.0
    
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
            
            # Split into horizontal strips (TrOCR works best with ~384px height)
            region_height = 384
            overlap = 50  # Overlap to avoid cutting text
            regions = []
            
            y = 0
            while y < height:
                # Crop region
                region = image.crop((0, max(0, y - overlap), width, min(height, y + region_height)))
                
                # Resize to optimal TrOCR size if needed
                r_width, r_height = region.size
                if r_height < 64:
                    # Too small, skip
                    y += region_height - overlap
                    continue
                
                # Process this region with TrOCR
                try:
                    pixel_values = self.processor(images=region, return_tensors="pt").pixel_values
                    pixel_values = pixel_values.to(self.device)
                    
                    pad_token_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.unk_token_id
                    eos_token_id = self.processor.tokenizer.eos_token_id
                    
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            pixel_values,
                            max_length=256,  # Shorter for each region
                            num_beams=5,
                            early_stopping=True,
                            pad_token_id=pad_token_id,
                            eos_token_id=eos_token_id,
                            no_repeat_ngram_size=2,
                            length_penalty=1.5  # Encourage longer text
                        )
                        region_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        
                        # Skip hallucinations in regions too
                        if region_text.strip() and not any(word in region_text.lower() for word in ['thank', 'thanks', 'item']):
                            regions.append(region_text.strip())
                        elif region_text.strip():
                            logger.warning(f"Skipping hallucinated region text: '{region_text}'")
                except Exception as e:
                    logger.warning(f"Error processing region at y={y}: {e}")
                
                y += region_height - overlap
            
            # Combine all regions
            combined_text = "\n".join(regions)
            logger.info(f"Region-based extraction got {len(combined_text)} chars from {len(regions)} regions")
            return combined_text
            
        except Exception as e:
            logger.error(f"Region-based extraction failed: {e}")
            return ""
    
    def extract_with_regions(self, image: Image.Image, regions: list) -> Dict[str, Tuple[str, float]]:
        """
        Extract text from specific regions of image
        
        Args:
            image: PIL Image
            regions: List of dicts with 'name', 'x', 'y', 'width', 'height'
        
        Returns:
            Dict mapping region names to (text, confidence) tuples
        """
        results = {}
        for region in regions:
            try:
                # Crop region
                x, y, w, h = region['x'], region['y'], region['width'], region['height']
                cropped = image.crop((x, y, x + w, y + h))
                
                # Extract text
                text, confidence = self.extract_text(cropped)
                results[region['name']] = (text.strip(), confidence or 0.5)
            except Exception as e:
                logger.error(f"Error extracting from region {region.get('name')}: {e}")
                results[region['name']] = ("", 0.0)
        
        return results

