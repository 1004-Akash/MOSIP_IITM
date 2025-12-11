"""
Quick system test to verify all components are working
"""
import sys
import traceback

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        print("  ✓ Transformers")
        
        import fitz
        print(f"  ✓ PyMuPDF {fitz.version[0]}")
        
        from fastapi import FastAPI
        print("  ✓ FastAPI")
        
        from PIL import Image
        print("  ✓ Pillow")
        
        import cv2
        print(f"  ✓ OpenCV {cv2.__version__}")
        
        from thefuzz import fuzz
        print("  ✓ TheFuzz")
        
        from app.ocr_engine import TROCREngine
        from app.preprocessor import ImagePreprocessor
        from app.field_extractor import FieldExtractor
        from app.verifier import DataVerifier
        print("  ✓ All app modules")
        
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        traceback.print_exc()
        return False

def test_components():
    """Test component initialization"""
    print("\nTesting component initialization...")
    try:
        from app.preprocessor import ImagePreprocessor
        from app.field_extractor import FieldExtractor
        from app.verifier import DataVerifier
        
        preprocessor = ImagePreprocessor()
        print("  ✓ ImagePreprocessor initialized")
        
        field_extractor = FieldExtractor()
        print("  ✓ FieldExtractor initialized")
        
        verifier = DataVerifier()
        print("  ✓ DataVerifier initialized")
        
        # Note: OCR engine initialization is expensive, so we skip it here
        print("  ⚠ TROCREngine will initialize on first use (may take time)")
        
        return True
    except Exception as e:
        print(f"  ✗ Component error: {e}")
        traceback.print_exc()
        return False

def test_pdf_support():
    """Test PDF support"""
    print("\nTesting PDF support...")
    try:
        import fitz
        print(f"  ✓ PyMuPDF available (version {fitz.version[0]})")
        return True
    except ImportError:
        print("  ✗ PyMuPDF not available - PDF support disabled")
        return False

def main():
    print("=" * 60)
    print("OCR System Health Check")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Components", test_components()))
    results.append(("PDF Support", test_pdf_support()))
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed! System is ready.")
        return 0
    else:
        print("✗ Some tests failed. Please check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

