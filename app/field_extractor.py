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
            'name': ['name', 'full name', 'fullname', 'applicant name', 'first name', 'last name'],
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
        """Extract and normalize date - comprehensive patterns"""
        # Normalize common ordinal suffixes (1st, 2nd, 3rd, 4th -> 1,2,3,4)
        text_norm = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', text, flags=re.IGNORECASE)
        month_names = "(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)"
        
        patterns = [
            # Saturday, July 28, 2025
            rf'([A-Za-z]+day,\s+{month_names}\s+\d{{1,2}},\s+\d{{4}})',
            # July 28, 2025
            rf'({month_names}\s+\d{{1,2}},\s+\d{{4}})',
            # 28 July 2025
            rf'(\d{{1,2}}\s+{month_names}\s+\d{{4}})',
            # 2025-07-28
            r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
            # 28/07/2025 or 28-07-25 or 27-09-2000
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        ]
        try:
            from dateutil import parser
        except ImportError:
            parser = None
        
        for pattern in patterns:
            matches = re.findall(pattern, text_norm, flags=re.IGNORECASE)
            if matches:
                # Each match can be tuple when using month_names; flatten
                candidate = matches[0]
                if isinstance(candidate, tuple):
                    candidate = candidate[0]
                candidate = candidate.strip()
                # Try parser if available
                if parser:
                    try:
                        dt = parser.parse(candidate, fuzzy=True, dayfirst=False)
                        return dt.strftime("%Y-%m-%d")
                    except Exception:
                        # Try dayfirst too
                        try:
                            dt = parser.parse(candidate, fuzzy=True, dayfirst=True)
                            return dt.strftime("%Y-%m-%d")
                        except Exception:
                            pass
                # Fallback: try to parse manually for DD-MM-YYYY or DD/MM/YYYY
                date_match = re.match(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', candidate)
                if date_match:
                    d, m, y = date_match.groups()
                    y = int(y)
                    if y < 100:
                        y += 2000 if y < 50 else 1900
                    try:
                        return f"{y}-{int(m):02d}-{int(d):02d}"
                    except:
                        pass
                return candidate
        
        return None
    
    def extract_id_card_fields(self, text: str, ocr_confidence: float) -> Dict[str, Tuple[str, float]]:
        """
        Extract fields from ID card document
        
        Returns:
            Dict mapping field names to (value, confidence) tuples
        """
        results = {}
        normalized_text = self.normalize_text(text)
        
        # ID cards typically have structured layouts
        # Extract name (usually prominent)
        name = self.find_field_by_keyword(text, 'name')
        if not name:
            # Try to find capitalized name patterns
            name_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
            matches = re.findall(name_pattern, text)
            if matches:
                # Take the longest match (likely the full name)
                name = max(matches, key=len)
        
        # Extract DOB
        dob = self.find_field_by_keyword(text, 'dob')
        if not dob:
            dob = self.extract_date(text)
        
        # Extract ID number
        id_number = self.find_field_by_keyword(text, 'id_number')
        if not id_number:
            id_match = re.search(self.PATTERNS['id_number'], text)
            if id_match:
                id_number = id_match.group(0)
        
        # Extract address
        address = self.find_field_by_keyword(text, 'address')
        
        # Extract phone
        phone = self.find_field_by_keyword(text, 'phone')
        if not phone:
            phone_match = re.search(self.PATTERNS['phone'], text)
            if phone_match:
                phone = phone_match.group(0)
        
        # Extract email
        email = self.find_field_by_keyword(text, 'email')
        if not email:
            email_match = re.search(self.PATTERNS['email'], text, re.IGNORECASE)
            if email_match:
                email = email_match.group(0)
        
        
        return results
    
    def extract_form_fields(self, text: str, ocr_confidence: float) -> Dict[str, Tuple[str, float]]:
        """
        Extract fields from form document (including handwritten forms)
        
        Returns:
            Dict mapping field names to (value, confidence) tuples
        """
        results = {}
        normalized_text = self.normalize_text(text)
        
        # For handwritten forms, OCR might have errors, so we need more flexible extraction
        # Try multiple strategies for each field
        
        # Extract name - look for common patterns
        name = None
        # Try keyword-based extraction first
        name = self.find_field_by_keyword(text, 'name')
        if not name:
            # For handwritten forms, name might be on a line with "first name", "last name", etc.
            name_patterns = [
                r'(?:first\s+name|name)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'(?:full\s+name)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                r'^([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',  # Line starting with capitalized words
            ]
            for pattern in name_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    name = match.group(1).strip()
                    # Clean up common OCR errors
                    name = re.sub(r'[^\w\s]', '', name)
                    if len(name.split()) >= 2:  # At least first and last name
                        break
        
        # Extract DOB
        dob = self.find_field_by_keyword(text, 'dob')
        if not dob:
            dob = self.extract_date(text)
        
        # Extract phone - use regex pattern
        phone = None
        phone_match = re.search(self.PATTERNS['phone'], text)
        if phone_match:
            phone = phone_match.group(0).strip()
        if not phone:
            phone = self.find_field_by_keyword(text, 'phone')
        
        # Extract email - use regex pattern
        email = None
        email_match = re.search(self.PATTERNS['email'], text, re.IGNORECASE)
        if email_match:
            email = email_match.group(0).strip()
        if not email:
            email = self.find_field_by_keyword(text, 'email')
        
        # Extract address - look for multi-line addresses
        address = self.find_field_by_keyword(text, 'address')
        if not address:
            # Try to find address patterns (street, city, state, zip)
            address_patterns = [
                r'(?:address|street)[:\s]+([^\n]+(?:\n[^\n]+){0,3})',  # Address with multiple lines
                r'(road|street|avenue|lane|drive)[\s#\d,]+[^\n]+',  # Street address pattern
            ]
            for pattern in address_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    address = match.group(0).strip() if match.groups() == () else match.group(1).strip()
                    address = re.sub(r'\s+', ' ', address)  # Normalize whitespace
                    break
        
        # Extract ID number
        id_number = self.find_field_by_keyword(text, 'id_number')
        if not id_number:
            id_match = re.search(self.PATTERNS['id_number'], text)
            if id_match:
                id_number = id_match.group(0).strip()
        
        # Assign results with confidence scores
        results['name'] = (name or "", ocr_confidence * 0.9 if name else 0.0)
        results['dob'] = (dob or "", ocr_confidence * 0.9 if dob else 0.0)
        results['address'] = (address or "", ocr_confidence * 0.85 if address else 0.0)
        results['phone'] = (phone or "", ocr_confidence * 0.9 if phone else 0.0)
        results['email'] = (email or "", ocr_confidence * 0.95 if email else 0.0)  # Email regex is very reliable
        results['id_number'] = (id_number or "", ocr_confidence * 0.9 if id_number else 0.0)
        
        return results
    
    def extract_certificate_fields(self, text: str, ocr_confidence: float) -> Dict[str, Tuple[str, float]]:
        """
        Extract fields from certificate document
        
        Returns:
            Dict mapping field names to (value, confidence) tuples
        """
        results = {}
        normalized_text = self.normalize_text(text)
        
        # Certificates typically have: name, date, course, certificate number
        
        # Extract name - look for "awarded to" or similar phrases
        # CRITICAL: Stop at "for" to avoid including course name
        name = None
        
        # Strategy 1: Find "awarded to" and extract name until "for"
        awarded_match = re.search(r'awarded\s+to[:\s]+(.+?)(?:\s+for\s+successfully|\s+for\s+completing|\s+for\s+|\s+on\s+|\n|$)', text, re.IGNORECASE | re.DOTALL)
        if awarded_match:
            name_candidate = awarded_match.group(1).strip()
            
            # Split by newline and take first line (name is usually on one line)
            name_lines = name_candidate.split('\n')
            name_candidate = name_lines[0].strip()
            
            # CRITICAL: If name contains "for", split at "for" and take only the part before it
            if ' for ' in name_candidate.lower():
                name_candidate = name_candidate.split(' for ')[0].strip()
            
            # Clean up: remove any trailing "for", "on", course-related words, etc.
            name_candidate = re.sub(r'\s+(for|on|has|is|was|successfully|completing|the|course|database|management|system|part)\s*$', '', name_candidate, flags=re.IGNORECASE)
            
            # Also remove from the beginning if somehow it got there
            name_candidate = re.sub(r'^(for|on|has|is|was|successfully|completing|the|course|database|management|system|part)\s+', '', name_candidate, flags=re.IGNORECASE)
            
            # Validate: should be 2-4 words, all capitalized or mixed case
            words = name_candidate.split()
            if 2 <= len(words) <= 4:
                # Check if it looks like a name (not course-related)
                course_keywords = ['course', 'system', 'management', 'database', 'part', 'certificate', 'completion', 'successfully', 'completing']
                if not any(kw in name_candidate.lower() for kw in course_keywords):
                    # Check if words are reasonable length
                    if all(2 <= len(w) <= 20 for w in words):
                        name = name_candidate
        
        # Strategy 2: Find standalone name lines (all caps, between "awarded to" and "for")
        if not name:
            lines = text.split('\n')
            found_awarded = False
            for i, line in enumerate(lines):
                line = line.strip()
                if 'awarded to' in line.lower():
                    found_awarded = True
                    # Check next few lines for name
                    for j in range(i+1, min(i+5, len(lines))):
                        next_line = lines[j].strip()
                        # Stop if we hit "for" - that's course section
                        if 'for' in next_line.lower() and ('successfully' in next_line.lower() or 'completing' in next_line.lower()):
                            break
                        # Look for all-caps name (2-4 words, reasonable length)
                        if next_line.isupper() and 5 < len(next_line) < 50:
                            words = next_line.split()
                            if 2 <= len(words) <= 4:
                                # Filter out common phrases
                                if next_line.lower() not in ['certificate', 'appreciation', 'thank you', 'congratulations', 'completion', 'course completion certificate']:
                                    name = next_line
                                    break
                    if name:
                        break
        
        # Strategy 3: Handle case where text starts directly with name (no "awarded to")
        # Pattern: "AKASH ELAYARAJA for successfully completing..."
        if not name:
            # Look for pattern: [NAME] for successfully/completing
            direct_name_match = re.search(r'^([A-Z][A-Z\s]{4,30}?)(?:\s+for\s+successfully|\s+for\s+completing|\s+for\s+the\s+course)', text, re.IGNORECASE | re.MULTILINE)
            if direct_name_match:
                name_candidate = direct_name_match.group(1).strip()
                words = name_candidate.split()
                # Should be 2-4 words, all caps or mixed case
                if 2 <= len(words) <= 4:
                    # Filter out course-related words
                    course_keywords = ['course', 'system', 'management', 'database', 'part', 'certificate', 'completion']
                    if not any(kw in name_candidate.lower() for kw in course_keywords):
                        name = name_candidate
        
        # Extract date
        date = self.extract_date(text)
        
        # Extract course - look for "course" keyword or similar
        # CRITICAL: Find the actual course name, not "COMPLETION CERTIFICATE"
        course = None
        
        # Strategy 1: Find "completing the course [COURSE NAME]"
        course_match = re.search(r'completing\s+the\s+course[:\s]+(.+?)(?:\s+on\s+|\s+issued\s+|\n|$)', text, re.IGNORECASE | re.DOTALL)
        if course_match:
            course_candidate = course_match.group(1).strip()
            # Split by newline and take first line or lines until date
            course_lines = course_candidate.split('\n')
            course_candidate = course_lines[0].strip()
            
            # CRITICAL: If it contains "on" followed by date, split there
            if re.search(r'\s+on\s+\w+day', course_candidate, re.IGNORECASE):
                course_candidate = re.split(r'\s+on\s+', course_candidate, flags=re.IGNORECASE)[0].strip()
            
            # Clean up: remove trailing "on", "issued", etc.
            course_candidate = re.sub(r'\s+(on|issued|certificate|has|is|was)\s*$', '', course_candidate, flags=re.IGNORECASE)
            
            # Filter out common certificate header phrases
            if course_candidate.lower() not in ['completion certificate', 'certificate', 'appreciation', 'recognition', 'completion']:
                # Check if it contains course-related keywords
                course_keywords = ['database', 'management', 'system', 'cloud', 'professional', 'certified', 'training', 'part']
                if any(kw in course_candidate.lower() for kw in course_keywords) or len(course_candidate.split()) >= 3:
                    # Make sure it's not just "COMPLETION CERTIFICATE"
                    if 'completion' not in course_candidate.lower() or len(course_candidate.split()) > 2:
                        course = course_candidate
        
        # Strategy 1b: If name was extracted incorrectly (contains course text), try to fix it
        if name and any(kw in name.lower() for kw in ['course', 'system', 'management', 'database', 'part', 'completing', 'successfully', 'for']):
            # Name was extracted incorrectly - try to split it
            name_parts = name.split()
            # Find where course-related words start (look for "for" first, then other keywords)
            split_index = -1
            for i, word in enumerate(name_parts):
                if word.lower() in ['for', 'course', 'completing', 'successfully']:
                    split_index = i
                    break
                elif word.lower() in ['database', 'management', 'system'] and i > 2:  # Only if not at start
                    split_index = i
                    break
            
            if split_index > 0:
                # Extract name (everything before the split)
                name = ' '.join(name_parts[:split_index]).strip()
                # Extract course (everything after "for" or course keyword)
                if not course and split_index < len(name_parts):
                    course_candidate = ' '.join(name_parts[split_index:])
                    # Clean up course candidate - remove "for successfully completing the course"
                    course_candidate = re.sub(r'^(for\s+)?(successfully\s+)?(completing\s+)?(the\s+)?(course\s*:?\s*)', '', course_candidate, flags=re.IGNORECASE)
                    if len(course_candidate.split()) >= 2:
                        course = course_candidate.strip()
        
        # Strategy 2: Look for lines with "Part" followed by number (e.g., "Part - 2")
        if not course:
            lines = text.split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                # Look for lines with "Part" and number
                part_match = re.search(r'Part\s+[-\d]+', line, re.IGNORECASE)
                if part_match:
                    # Check if previous line(s) contain course name
                    if i > 0:
                        # Check previous 2 lines for course name
                        for j in range(max(0, i-2), i):
                            prev_line = lines[j].strip()
                            # Skip if it's the name or a header
                            if prev_line.lower() not in ['certificate', 'completion certificate', 'awarded to'] and len(prev_line) > 5:
                                # Combine if it looks like course name
                                if len(prev_line.split()) >= 2:
                                    course = f"{prev_line} {line}"
                                    break
                    if not course:
                        # Just use the line with Part if it's substantial
                        if len(line.split()) >= 2:
                            course = line
                    if course:
                        break
        
        # Strategy 3: Find multi-word lines that contain course keywords
        if not course:
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                # Look for lines with 3+ words containing course keywords
                words = line.split()
                if len(words) >= 3:
                    course_keywords = ['database', 'management', 'system', 'cloud', 'course', 'training']
                    if any(kw in line.lower() for kw in course_keywords):
                        # Filter out headers and common phrases
                        if line.lower() not in ['completion certificate', 'course completion certificate', 'certificate of completion']:
                            # Make sure it's not all caps (likely a header)
                            if not line.isupper() or len(words) > 3:
                                course = line
                                break
        
        # Extract certificate number
        cert_number = None
        cert_patterns = [
            r'certificate\s+number[:\s]+([A-Z0-9\-]+)',
            r'certificate\s+no[:\s]+([A-Z0-9\-]+)',
            r'cert\s+#[:\s]+([A-Z0-9\-]+)',
            r'certificate\s+id[:\s]+([A-Z0-9\-]+)',
            r'cert\s+id[:\s]+([A-Z0-9\-]+)',
            # Look for alphanumeric codes that might be certificate numbers
            r'\b([A-Z]{2,}\d{4,}[A-Z0-9]*)\b',  # Pattern like "INF123456" or "CERT2024001"
        ]
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take the first match that looks like a certificate number
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    match = match.strip()
                    # Filter: should be at least 6 characters, mix of letters and numbers
                    if len(match) >= 6 and any(c.isdigit() for c in match) and any(c.isalpha() for c in match):
                        cert_number = match
                        break
                if cert_number:
                    break
        
        # Assign results
        results['name'] = (name or "", ocr_confidence * 0.9 if name else 0.0)
        results['date'] = (date or "", ocr_confidence * 0.9 if date else 0.0)
        results['course'] = (course or "", ocr_confidence * 0.85 if course else 0.0)
        results['certificate_number'] = (cert_number or "", ocr_confidence * 0.9 if cert_number else 0.0)
        
        return results
    
    def extract_fields(self, text: str, document_type: str, ocr_confidence: float = 0.8) -> Dict[str, Tuple[str, float]]:
        """
        Main extraction method - routes to appropriate extractor based on document type
        
        Args:
            text: Raw OCR text
            document_type: Type of document (ID_CARD, FORM, CERTIFICATE)
            ocr_confidence: Overall OCR confidence score
        
        Returns:
            Dict mapping field names to (value, confidence) tuples
        """
        doc_type = document_type.upper()
        
        if doc_type == 'ID_CARD':
            return self.extract_id_card_fields(text, ocr_confidence)
        elif doc_type == 'FORM':
            return self.extract_form_fields(text, ocr_confidence)
        elif doc_type == 'CERTIFICATE':
            return self.extract_certificate_fields(text, ocr_confidence)
        else:
            logger.warning(f"Unknown document type: {document_type}, using form extractor")
            return self.extract_form_fields(text, ocr_confidence)
