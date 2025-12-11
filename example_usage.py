"""
Example usage script for OCR API
Demonstrates how to use the OCR Extraction and Verification APIs
"""
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_ocr_extraction(image_path: str, document_type: str = "ID_CARD"):
    """
    Test OCR Extraction API
    
    Args:
        image_path: Path to image file
        document_type: Type of document (ID_CARD, FORM, CERTIFICATE)
    """
    print(f"\n{'='*60}")
    print(f"Testing OCR Extraction API")
    print(f"{'='*60}")
    
    url = f"{BASE_URL}/api/v1/ocr/extract"
    
    with open(image_path, 'rb') as f:
        files = {'file': (image_path, f, 'image/jpeg')}
        data = {'document_type': document_type}
        
        print(f"Uploading {image_path}...")
        response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Extraction successful!")
        print(f"Overall Confidence: {result['overall_confidence']:.2%}")
        print(f"\nStructured Data:")
        for field, value in result['structured_data'].items():
            confidence = result['confidence_scores'].get(field, 0)
            print(f"  {field}: {value} (confidence: {confidence:.2%})")
        print(f"\nRaw Text (first 200 chars):")
        print(result['raw_text'][:200] + "...")
        return result
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None


def test_verification(image_path: str, form_data: dict):
    """
    Test Data Verification API
    
    Args:
        image_path: Path to original document image
        form_data: Dictionary with form fields to verify
    """
    print(f"\n{'='*60}")
    print(f"Testing Data Verification API")
    print(f"{'='*60}")
    
    url = f"{BASE_URL}/api/v1/verify"
    
    with open(image_path, 'rb') as f:
        files = {'file': (image_path, f, 'image/jpeg')}
        data = {'form_data': json.dumps(form_data)}
        
        print(f"Uploading {image_path}...")
        print(f"Form data: {form_data}")
        response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Verification successful!")
        print(f"Overall Verification Score: {result['overall_verification_score']:.2%}")
        print(f"\nField Results:")
        for field_result in result['field_results']:
            status_emoji = {
                'match': '✅',
                'mismatch': '❌',
                'unsure': '⚠️'
            }
            emoji = status_emoji.get(field_result['status'], '❓')
            print(f"  {emoji} {field_result['field']}: {field_result['status']}")
            print(f"     Document: {field_result['document_value']}")
            print(f"     Form: {field_result['form_value']}")
            print(f"     Similarity: {field_result['similarity_score']:.2%}")
        
        if result['mismatched_fields']:
            print(f"\n⚠️ Mismatched Fields: {', '.join(result['mismatched_fields'])}")
        if result['missing_fields']:
            print(f"⚠️ Missing Fields: {', '.join(result['missing_fields'])}")
        
        return result
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None


if __name__ == "__main__":
    # Example usage
    print("OCR API Example Usage")
    print("Make sure the API server is running: uvicorn main:app --reload")
    
    # Example 1: Extract text from an ID card
    # Uncomment and modify the path to test with your own image
    # test_ocr_extraction("test_samples/sample_id_card.jpg", "ID_CARD")
    
    # Example 2: Verify form data against document
    # form_data = {
    #     "name": "John Doe",
    #     "dob": "1990-01-15",
    #     "id_number": "ABC123456"
    # }
    # test_verification("test_samples/sample_id_card.jpg", form_data)
    
    print("\n" + "="*60)
    print("To test with your own images:")
    print("1. Place your test images in a 'test_samples' directory")
    print("2. Uncomment and modify the example calls above")
    print("3. Run: python example_usage.py")
    print("="*60)

