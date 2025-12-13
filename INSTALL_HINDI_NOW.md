# CRITICAL: Install Hindi Language Data NOW

## Quick Installation (Run as Administrator)

1. **Open PowerShell or Command Prompt as Administrator**
   - Right-click → "Run as Administrator"

2. **Navigate to project directory**
   ```powershell
   cd C:\Users\AKASH.E\IIT
   ```

3. **Run the installer**
   ```powershell
   python install_hindi_lang.py
   ```

4. **If permission denied, manually download:**
   - Download: https://github.com/tesseract-ocr/tessdata/raw/main/hin.traineddata
   - Copy to: `C:\Program Files\Tesseract-OCR\tessdata\hin.traineddata`
   - You may need Administrator rights to copy the file

5. **Restart the FastAPI server**

## Verify Installation

```powershell
python -c "import pytesseract; langs = pytesseract.get_languages(); print('Hindi available:', 'hin' in langs)"
```

Should output: `Hindi available: True`

## Why This is Critical

- Hindi text extraction **WILL NOT WORK** without `hin.traineddata`
- The system will extract garbled text or nothing
- This determines whether you win the competition!

