# Installing Poppler for PDF Support (Windows)

The OCR system requires Poppler to process PDF files. Follow these steps to install it on Windows:

## Quick Installation Steps

### Option 1: Using Pre-built Binaries (Recommended)

1. **Download Poppler for Windows:**

   - Go to: https://github.com/oschwartz10612/poppler-windows/releases/
   - Download the latest release (e.g., `Release-XX.XX.X-X.zip`)

2. **Extract the ZIP file:**

   - Extract to a location like `C:\poppler` or `C:\Program Files\poppler`

3. **Add to System PATH:**

   - Press `Win + X` and select "System"
   - Click "Advanced system settings"
   - Click "Environment Variables"
   - Under "System variables", find and select "Path", then click "Edit"
   - Click "New" and add the path to the `bin` folder (e.g., `C:\poppler\Library\bin`)
   - Click "OK" on all dialogs

4. **Verify Installation:**

   - Open a new Command Prompt or PowerShell
   - Run: `pdftoppm -v`
   - You should see version information

5. **Restart the OCR Server:**
   - Stop the current server (Ctrl+C)
   - Restart it: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`

### Option 2: Using Chocolatey (If you have it installed)

```powershell
choco install poppler
```

### Option 3: Using Conda (If you use Conda)

```bash
conda install -c conda-forge poppler
```

## Alternative: Convert PDF to Image

If you don't want to install Poppler, you can:

1. Convert your PDF to an image (JPG or PNG) using any PDF viewer
2. Upload the image file instead

## Troubleshooting

- **"pdftoppm is not recognized"**: Make sure you added the `bin` folder to PATH and restarted your terminal/server
- **Still getting errors**: Try restarting your computer after adding to PATH
- **Permission errors**: Make sure you have admin rights when modifying PATH

## Testing PDF Support

After installation, test with:

```bash
pdftoppm -v
```

If this works, PDF support should be available in the OCR system.
