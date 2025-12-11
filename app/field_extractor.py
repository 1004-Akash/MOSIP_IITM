"""
Field extraction module - extracts structured fields from OCR text
Uses heuristics, regex patterns, and layout analysis
"""
import re
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class FieldExtractor:
    """Extracts structured fields from OCR text based on document type"""
    
    # Common regex patterns
    PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        'ssn': r'\d{3}-\d{2}-\d{4}',
        'date': r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
        'zipcode': r'\b\d{5}(-\d{4})?\b',
        'id_number': r'\b[A-Z0-9]{6,12}\b',  # Generic ID pattern
    }
    
    def __init__(self):
        """Initialize field extractor"""
        self.field_keywords = {
            'name': ['name', 'full name', 'fullname', 'applicant name'],
            'dob': ['date of birth', 'dob', 'birth date', 'born', 'birthday'],
            'id_number': ['id', 'identification', 'id number', 'id no', 'number'],
            'address': ['address', 'street', 'residence', 'location'],
            'phone': ['phone', 'telephone', 'mobile', 'contact'],
            'email': ['email', 'e-mail', 'mail'],
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters except basic punctuation
        text = re.sub(r'[^\w\s\.,\-:/]', '', text)
        return text.strip()
    
    def find_field_by_keyword(self, text: str, field_name: str) -> Optional[str]:
        """Find field value by searching for keywords"""
        keywords = self.field_keywords.get(field_name.lower(), [])
        text_lower = text.lower()
        
        for keyword in keywords:
            # Search for keyword followed by colon, equals, or newline
            pattern = rf'{re.escape(keyword)}[:\s]+([^\n\r]+)'
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Clean up common OCR artifacts
                value = re.sub(r'^[:\-=]+\s*', '', value)
                return self.normalize_text(value)
        
        return None
    
    def extract_date(self, text: str) -> Optional[str]:
        """Extract and normalize date - handles multiple formats"""
        # Try dateutil parser first (handles "February 24, 2024" format)
        try:
            from dateutil import parser
            # Find date-like strings
            date_patterns = [
                r'([A-Za-z]+\s+\d{1,2},\s+\d{4})',  # "February 24, 2024"
                r'([A-Za-z]+day,\s+[A-Za-z]+\s+\d{1,2},\s+\d{4})',  # "Saturday, February 24, 2024"
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # "02/24/2024" or "24-02-2024"
                r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',  # "2024-02-24"
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    try:
                        parsed_date = parser.parse(matches[0])
                        return parsed_date.strftime("%Y-%m-%d")
                    except:
                        continue
        except ImportError:
            pass  # Fall back to regex if dateutil not available
        
        # Fallback to regex patterns
        date_patterns = [
            r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b',
            r'\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b',  # YYYY-MM-DD
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            if matches:
                try:
                    # Try to parse and normalize the first date found
                    match = matches[0]
                    if len(match[2]) == 2:  # 2-digit year
                        year = '20' + match[2] if int(match[2]) < 50 else '19' + match[2]
                    else:
                        year = match[2]
                    
                    # Format as YYYY-MM-DD
                    date_str = f"{year}-{match[1].zfill(2)}-{match[0].zfill(2)}"
                    # Validate date
                    datetime.strptime(date_str, "%Y-%m-%d")
                    return date_str
                except:
                    continue
        
        return None
    
    def extract_id_card_fields(self, text: str, ocr_confidence: float) -> Dict[str, Tuple[str, float]]:
        """
        Extract fields from ID card document
        
        Returns:
            Dict mapping field names to (value, confidence) tuples
        """
        results = {}
        normalized_text = self.normalize_text(text)
        
        # Extract name (typically first few words, or after "NAME" keyword)
        name = self.find_field_by_keyword(text, 'name')
        if not name:
            # Try to extract first line or first few capitalized words
            lines = text.split('\n')
            for line in lines[:3]:
                words = line.split()
                if len(words) >= 2 and any(w[0].isupper() for w in words if w):
                    name = ' '.join(words[:3])  # First 2-3 words
                    break
        
        results['name'] = (name or "", ocr_confidence * 0.9 if name else 0.0)
        
        # Extract DOB
        dob = self.find_field_by_keyword(text, 'dob')
        if not dob:
            dob = self.extract_date(text)
        results['dob'] = (dob or "", ocr_confidence * 0.95 if dob else 0.0)
        
        # Extract ID number
        id_number = self.find_field_by_keyword(text, 'id_number')
        if not id_number:
            # Look for alphanumeric sequences that look like IDs
            id_pattern = r'\b[A-Z]{2,3}\d{6,9}\b|\b\d{6,12}\b'
            matches = re.findall(id_pattern, text)
            if matches:
                id_number = matches[0]  # Take first match
        results['id_number'] = (id_number or "", ocr_confidence * 0.92 if id_number else 0.0)
        
        # Extract address (typically multiple lines)
        address = self.find_field_by_keyword(text, 'address')
        if not address:
            # Look for lines containing address keywords
            address_keywords = ['street', 'st', 'avenue', 'ave', 'road', 'rd', 'city', 'state']
            lines = text.split('\n')
            address_lines = []
            for line in lines:
                if any(keyword in line.lower() for keyword in address_keywords):
                    address_lines.append(line.strip())
                elif address_lines:  # Continue collecting if we started
                    address_lines.append(line.strip())
                    if len(address_lines) >= 3:
                        break
            if address_lines:
                address = ', '.join(address_lines[:3])
        
        results['address'] = (self.normalize_text(address) if address else "", 
                            ocr_confidence * 0.85 if address else 0.0)
        
        # Extract phone
        phone_match = re.search(self.PATTERNS['phone'], text)
        phone = phone_match.group(0) if phone_match else None
        results['phone'] = (phone or "", ocr_confidence * 0.9 if phone else 0.0)
        
        # Extract email
        email_match = re.search(self.PATTERNS['email'], text)
        email = email_match.group(0) if email_match else None
        results['email'] = (email or "", ocr_confidence * 0.95 if email else 0.0)
        
        return results
    
    def extract_form_fields(self, text: str, ocr_confidence: float) -> Dict[str, Tuple[str, float]]:
        """
        Extract fields from form document
        
        Returns:
            Dict mapping field names to (value, confidence) tuples
        """
        results = {}
        normalized_text = self.normalize_text(text)
        
        # Forms typically have labeled fields, so keyword search works well
        for field_name in ['name', 'dob', 'address', 'phone', 'email', 'id_number']:
            value = None
            
            if field_name == 'dob':
                value = self.find_field_by_keyword(text, 'dob')
                if not value:
                    value = self.extract_date(text)
            else:
                value = self.find_field_by_keyword(text, field_name)
            
            if value:
                results[field_name] = (value, ocr_confidence * 0.9)
            else:
                results[field_name] = ("", 0.0)
        
        return results
    
    def extract_certificate_fields(self, text: str, ocr_confidence: float) -> Dict[str, Tuple[str, float]]:
        """
        Extract fields from certificate document
        
        Returns:
            Dict mapping field names to (value, confidence) tuples
        """
        results = {}
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        # Extract name - look for patterns like "awarded to NAME" or "NAME" after "awarded to"
        name = None
        name_patterns = [
            r'awarded to\s+([A-Z][A-Z\s]+?)(?:\s+for|$|\n)',
            r'awarded to\s+([A-Z][A-Z\s]{2,})',
            r'to\s+([A-Z][A-Z\s]{2,}?)(?:\s+for|$)',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                name = match.group(1).strip()
                # Clean up common OCR artifacts
                name = re.sub(r'\s+', ' ', name)
                if len(name) > 2:  # Valid name should be longer
                    break
        
        # If pattern matching failed, look for all-caps name lines (common in certificates)
        if not name:
            for i, line in enumerate(lines):
                # Look for lines that are mostly uppercase and 2-4 words
                words = line.split()
                if (len(words) >= 2 and len(words) <= 4 and 
                    all(w.isupper() or w.replace("'", "").isupper() for w in words if w) and
                    len(line) > 5 and len(line) < 50):
                    # Check if it's not a common phrase
                    if not any(phrase in line.upper() for phrase in ['THANK YOU', 'CERTIFICATE', 'AWARDED', 'COMPLETING']):
                        name = line.strip()
                        break
        
        results['name'] = (name or "", ocr_confidence * 0.9 if name else 0.0)
        
        # Extract date - look for "on [DATE]" or "Issued on: [DATE]" patterns
        date = None
        date_patterns = [
            r'Issued on:\s*([A-Za-z]+day,\s*[A-Za-z]+\s+\d{1,2},\s*\d{4})',
            r'on\s+([A-Za-z]+\s+\d{1,2},\s*\d{4})',
            r'on\s+(\w+\s+\d{1,2},\s*\d{4})',
            r'(\w+\s+\d{1,2},\s*\d{4})',  # Fallback: any date pattern
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take the first valid date match
                date_str = matches[0]
                # Try to normalize to YYYY-MM-DD format
                try:
                    from dateutil import parser
                    parsed_date = parser.parse(date_str)
                    date = parsed_date.strftime("%Y-%m-%d")
                except:
                    # If parsing fails, use the extracted string
                    date = date_str
                break
        
        # If date patterns didn't work, try the generic extract_date
        if not date:
            date = self.extract_date(text)
        
        results['date'] = (date or "", ocr_confidence * 0.9 if date else 0.0)
        
        # Extract course name - look for "completing the course [COURSE NAME]"
        course_name = None
        course_patterns = [
            r'completing the course\s+([^\n]+?)(?:\s+on|$|Issued)',
            r'completing the course\s+([A-Z][^\n]+?)(?:\s+on|$|Issued)',
            r'course\s+([A-Z][^\n]+?)(?:\s+on|$|Issued)',
        ]
        for pattern in course_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                course_name = match.group(1).strip()
                # Clean up - remove extra whitespace
                course_name = re.sub(r'\s+', ' ', course_name)
                if len(course_name) > 3:  # Valid course name
                    break
        
        if course_name:
            results['course'] = (course_name, ocr_confidence * 0.85)
        else:
            results['course'] = ("", 0.0)
        
        # Extract certificate number - look for patterns like "Certificate No:", "ID:", etc.
        cert_number = None
        cert_patterns = [
            r'certificate\s*(?:no|number|#)[:\s]+([A-Z0-9\-]+)',
            r'cert\s*(?:no|number|#)[:\s]+([A-Z0-9\-]+)',
            r'id[:\s]+([A-Z0-9\-]{6,})',
            r'number[:\s]+([A-Z0-9\-]{6,})',
        ]
        for pattern in cert_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                cert_number = match.group(1).strip()
                break
        
        results['certificate_number'] = (cert_number or "", ocr_confidence * 0.85 if cert_number else 0.0)
        
        return results
    
    def extract_fields(self, text: str, document_type: str, ocr_confidence: float) -> Dict[str, Tuple[str, float]]:
        """
        Main method to extract fields based on document type
        
        Args:
            text: Raw OCR text
            document_type: Type of document (ID_CARD, FORM, CERTIFICATE)
            ocr_confidence: Overall OCR confidence
        
        Returns:
            Dict mapping field names to (value, confidence) tuples
        """
        document_type = document_type.upper()
        
        if document_type == 'ID_CARD':
            return self.extract_id_card_fields(text, ocr_confidence)
        elif document_type == 'FORM':
            return self.extract_form_fields(text, ocr_confidence)
        elif document_type == 'CERTIFICATE':
            return self.extract_certificate_fields(text, ocr_confidence)
        else:
            # Generic extraction for unknown types
            logger.warning(f"Unknown document type: {document_type}, using generic extraction")
            return self.extract_form_fields(text, ocr_confidence)

