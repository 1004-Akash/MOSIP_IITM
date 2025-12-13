"""
Image pre-processing module for OCR enhancement
Handles skew correction, noise reduction, contrast enhancement, etc.
"""
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Pre-processes images to improve OCR accuracy"""
    
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """Load image from file path"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        return img
    
    @staticmethod
    def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format"""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image to PIL format"""
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    
    @staticmethod
    def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        """Remove noise from image"""
        # Apply bilateral filter to reduce noise while keeping edges sharp
        denoised = cv2.bilateralFilter(image, 5, 50, 50)
        # Additional denoising if needed
        denoised = cv2.fastNlMeansDenoising(denoised, None, 10, 7, 21)
        return denoised
    
    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(image.shape) == 2:
            return clahe.apply(image)
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def correct_skew(image: np.ndarray) -> np.ndarray:
        """Correct skew/rotation in document image"""
        gray = ImagePreprocessor.convert_to_grayscale(image) if len(image.shape) == 3 else image
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is None or len(lines) == 0:
            logger.warning("No lines detected for skew correction")
            return image
        
        # Calculate angles
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            # Only consider angles close to 0, 90, -90
            if abs(angle) < 45:
                angles.append(angle)
        
        if not angles:
            return image
        
        # Get median angle for robustness
        median_angle = np.median(angles)
        
        # Only correct if angle is significant
        if abs(median_angle) < 0.5:
            return image
        
        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        corrected = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                   flags=cv2.INTER_CUBIC, 
                                   borderMode=cv2.BORDER_REPLICATE)
        
        logger.info(f"Corrected skew by {median_angle:.2f} degrees")
        return corrected
    
    @staticmethod
    def resize_if_needed(image: np.ndarray, min_dimension: int = 512) -> np.ndarray:
        """Resize image if too small (maintains aspect ratio)"""
        h, w = image.shape[:2]
        min_size = min(h, w)
        
        if min_size < min_dimension:
            scale = min_dimension / min_size
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            logger.info(f"Resized image from ({w}, {h}) to ({new_w}, {new_h})")
        
        return image
    
    @staticmethod
    def binarize(image: np.ndarray, method: str = "adaptive") -> np.ndarray:
        """Binarize image (convert to black and white)"""
        gray = ImagePreprocessor.convert_to_grayscale(image) if len(image.shape) == 3 else image
        
        if method == "adaptive":
            # Adaptive thresholding works better for varying illumination
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        else:
            # Otsu's thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
    
    def preprocess(self, image: Image.Image, apply_skew_correction: bool = True,
                   apply_denoising: bool = True, enhance: bool = True) -> Image.Image:
        """
        Main preprocessing pipeline
        
        Args:
            image: PIL Image
            apply_skew_correction: Whether to correct skew
            apply_denoising: Whether to apply denoising
            enhance: Whether to enhance contrast
        
        Returns:
            Preprocessed PIL Image
        """
        try:
            # Convert PIL to OpenCV
            cv_image = self.pil_to_cv2(image)
            
            # Resize if needed
            cv_image = self.resize_if_needed(cv_image)
            
            # Correct skew
            if apply_skew_correction:
                cv_image = self.correct_skew(cv_image)
            
            # Convert to grayscale for processing
            gray = self.convert_to_grayscale(cv_image)
            
            # Denoise
            if apply_denoising:
                gray = self.denoise(gray)
            
            # Enhance contrast
            if enhance:
                gray = self.enhance_contrast(gray)
            
            # Convert back to PIL
            return self.cv2_to_pil(gray)
        
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            # Return original image if preprocessing fails
            return image

    @staticmethod
    def quality_metrics(image: Image.Image) -> Dict[str, float]:
        """
        Compute basic quality metrics: blur, brightness, contrast
        Returns a quality_score (0-1) and components.
        """
        try:
            cv_img = ImagePreprocessor.pil_to_cv2(image)
            gray = ImagePreprocessor.convert_to_grayscale(cv_img)
            # Blur: variance of Laplacian
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Normalize blur score (heuristic)
            blur_score = min(1.0, lap_var / 300.0)
            # Brightness and contrast
            brightness = float(np.mean(gray) / 255.0)
            contrast = float(np.std(gray) / 128.0)
            contrast = min(1.0, contrast)
            # Aggregate quality (simple average)
            quality = (blur_score + brightness + contrast) / 3.0
            return {
                "quality": round(quality, 3),
                "blur_score": round(blur_score, 3),
                "brightness": round(brightness, 3),
                "contrast": round(contrast, 3),
                "laplacian_variance": float(round(lap_var, 2)),
            }
        except Exception as e:
            logger.error(f"Error computing quality metrics: {e}")
            return {
                "quality": 0.0,
                "blur_score": 0.0,
                "brightness": 0.0,
                "contrast": 0.0,
                "laplacian_variance": 0.0,
            }

