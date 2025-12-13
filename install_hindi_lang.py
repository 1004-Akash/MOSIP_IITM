"""
Script to automatically download and install Hindi language data for Tesseract
"""
import os
import sys
import urllib.request
import shutil
from pathlib import Path

def find_tesseract_tessdata():
    """Find Tesseract tessdata directory"""
    import pytesseract
    
    # Try to get Tesseract path
    try:
        tesseract_cmd = pytesseract.pytesseract.tesseract_cmd
        if tesseract_cmd:
            # Extract directory from tesseract_cmd
            tesseract_dir = Path(tesseract_cmd).parent.parent
            tessdata = tesseract_dir / "tessdata"
            if tessdata.exists():
                return str(tessdata)
    except:
        pass
    
    # Common Windows paths
    windows_paths = [
        r"C:\Program Files\Tesseract-OCR\tessdata",
        r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
    ]
    
    for path in windows_paths:
        if os.path.exists(path):
            return path
    
    # Try to find from pytesseract
    try:
        import pytesseract
        # Get tesseract info
        info = pytesseract.get_tesseract_version()
        # Common locations
        common_paths = [
            "/usr/share/tesseract-ocr/tessdata",
            "/usr/local/share/tessdata",
            "/opt/homebrew/share/tessdata",  # macOS
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
    except:
        pass
    
    return None


def download_hindi_language():
    """Download and install Hindi language data"""
    print("=" * 60)
    print("Installing Hindi Language Data for Tesseract")
    print("=" * 60)
    
    # Find tessdata directory
    tessdata_dir = find_tesseract_tessdata()
    
    if not tessdata_dir:
        print("❌ ERROR: Could not find Tesseract tessdata directory!")
        print("\nPlease install Tesseract OCR first:")
        print("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("  Linux:   sudo apt-get install tesseract-ocr")
        print("  macOS:   brew install tesseract")
        return False
    
    print(f"[OK] Found tessdata directory: {tessdata_dir}")
    
    # Download Hindi language data
    hin_url = "https://github.com/tesseract-ocr/tessdata/raw/main/hin.traineddata"
    output_path = os.path.join(tessdata_dir, "hin.traineddata")
    
    print(f"\nDownloading Hindi language data...")
    print(f"   URL: {hin_url}")
    print(f"   Destination: {output_path}")
    
    try:
        urllib.request.urlretrieve(hin_url, output_path)
        print(f"[OK] Successfully downloaded to: {output_path}")
        
        # Verify installation
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"[OK] File size: {file_size / (1024*1024):.2f} MB")
            print("\n" + "=" * 60)
            print("[SUCCESS] Hindi language data installed successfully!")
            print("=" * 60)
            print("\nPlease restart your FastAPI server for changes to take effect.")
            return True
        else:
            print("❌ ERROR: File was not saved correctly")
            return False
    
    except PermissionError as e:
        print(f"[ERROR] Permission denied. You need administrator rights.")
        print(f"\nPlease run as Administrator:")
        print(f"  1. Right-click Command Prompt/PowerShell")
        print(f"  2. Select 'Run as Administrator'")
        print(f"  3. Navigate to this directory")
        print(f"  4. Run: python install_hindi_lang.py")
        print(f"\nOr manually:")
        print(f"  1. Download from: {hin_url}")
        print(f"  2. Place in: {tessdata_dir}")
        print(f"  3. You may need to run as Administrator to copy the file")
        return False
    except Exception as e:
        print(f"[ERROR] Error downloading Hindi language data: {e}")
        print("\nManual installation:")
        print(f"1. Download from: {hin_url}")
        print(f"2. Place in: {tessdata_dir}")
        return False


if __name__ == "__main__":
    success = download_hindi_language()
    sys.exit(0 if success else 1)

