"""
Data verification module - compares extracted document data with user-filled form data
Uses fuzzy matching for tolerant comparison
"""
import re
import logging
from typing import Dict, List, Tuple
from thefuzz import fuzz, process

logger = logging.getLogger(__name__)


class DataVerifier:
    """Verifies form data against extracted document data"""
    
    # Thresholds for matching
    MATCH_THRESHOLD = 0.85  # Above this = match
    MISMATCH_THRESHOLD = 0.70  # Below this = mismatch
    # Between = unsure
    
    def __init__(self, match_threshold: float = 0.85, mismatch_threshold: float = 0.70):
        """
        Initialize verifier
        
        Args:
            match_threshold: Similarity score above which fields are considered matching
            mismatch_threshold: Similarity score below which fields are considered mismatched
        """
        self.match_threshold = match_threshold
        self.mismatch_threshold = mismatch_threshold
    
    def normalize_value(self, value: str) -> str:
        """
        Normalize value for comparison
        
        Args:
            value: String value to normalize
        
        Returns:
            Normalized string
        """
        if not value:
            return ""
        
        # Convert to lowercase
        normalized = value.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove common punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Remove leading/trailing whitespace
        normalized = normalized.strip()
        
        return normalized
    
    def normalize_date(self, value: str) -> str:
        """Normalize date string to YYYY-MM-DD format"""
        if not value:
            return ""
        
        # Try using dateutil parser first (more robust)
        try:
            from dateutil import parser
            dt = parser.parse(value, fuzzy=True, dayfirst=False)
            return dt.strftime("%Y-%m-%d")
        except:
            pass
        
        # Fallback to regex patterns
        date_patterns = [
            r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY-MM-DD or YYYY/MM/DD
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',  # MM-DD-YYYY or DD-MM-YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, value)
            if match:
                parts = match.groups()
                if len(parts[2]) == 4:  # DD-MM-YYYY format
                    return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
                elif len(parts[0]) == 4:  # YYYY-MM-DD format
                    return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
                else:  # Assume MM-DD-YY or DD-MM-YY
                    year = parts[2]
                    if len(year) == 2:
                        year = '20' + year if int(year) < 50 else '19' + year
                    # Try both MM-DD and DD-MM
                    try:
                        from datetime import datetime
                        # Try MM-DD-YY first
                        dt = datetime.strptime(f"{parts[0]}/{parts[1]}/{year}", "%m/%d/%Y")
                        return f"{year}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
                    except:
                        # Try DD-MM-YY
                        try:
                            dt = datetime.strptime(f"{parts[1]}/{parts[0]}/{year}", "%m/%d/%Y")
                            return f"{year}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
                        except:
                            pass
        
        return self.normalize_value(value)
    
    def calculate_similarity(self, value1: str, value2: str, field_type: str = "text") -> float:
        """
        Calculate similarity between two values
        
        Args:
            value1: First value (document value)
            value2: Second value (form value)
            field_type: Type of field (text, date, numeric, etc.)
        
        Returns:
            Similarity score between 0 and 1
        """
        if not value1 and not value2:
            return 1.0  # Both empty = match
        
        if not value1 or not value2:
            return 0.0  # One empty, one not = mismatch
        
        # Normalize based on field type
        if field_type in ['date', 'dob']:
            norm1 = self.normalize_date(value1)
            norm2 = self.normalize_date(value2)
        elif field_type in ['numeric', 'id_number', 'certificate_number']:
            # Remove non-numeric characters for ID numbers
            norm1 = re.sub(r'\D', '', value1)
            norm2 = re.sub(r'\D', '', value2)
        else:
            norm1 = self.normalize_value(value1)
            norm2 = self.normalize_value(value2)
        
        # Use multiple fuzzy matching algorithms
        # Ratio: simple similarity
        ratio = fuzz.ratio(norm1, norm2) / 100.0
        
        # Partial ratio: handles partial matches (important for names with titles/suffixes)
        partial_ratio = fuzz.partial_ratio(norm1, norm2) / 100.0
        
        # Token sort ratio: handles word order differences (e.g., "John Doe" vs "Doe, John")
        token_sort_ratio = fuzz.token_sort_ratio(norm1, norm2) / 100.0
        
        # Token set ratio: handles duplicate words and extra words
        token_set_ratio = fuzz.token_set_ratio(norm1, norm2) / 100.0
        
        # Use the best match from all algorithms
        # For exact matches, use ratio
        if ratio == 1.0:
            similarity = 1.0
        # For numeric/ID fields, use ratio and partial ratio
        elif field_type in ['numeric', 'id_number', 'certificate_number']:
            similarity = max(ratio, partial_ratio)
        # For dates, use ratio (should be exact after normalization)
        elif field_type in ['date', 'dob']:
            similarity = ratio if ratio > 0.8 else max(ratio, partial_ratio)
        # For text fields (names, courses, addresses), use best of all
        else:
            similarity = max(ratio, partial_ratio, token_sort_ratio, token_set_ratio)
        
        # Boost similarity if one string is contained in the other (after normalization)
        if norm1 in norm2 or norm2 in norm1:
            similarity = max(similarity, 0.9)
        
        return min(1.0, similarity)  # Cap at 1.0
    
    def determine_status(self, similarity: float) -> str:
        """
        Determine match status based on similarity score
        
        Args:
            similarity: Similarity score (0-1)
        
        Returns:
            Status: "match", "mismatch", or "unsure"
        """
        if similarity >= self.match_threshold:
            return "match"
        elif similarity < self.mismatch_threshold:
            return "mismatch"
        else:
            return "unsure"
    
    def get_field_type(self, field_name: str) -> str:
        """Determine field type from field name"""
        field_name_lower = field_name.lower()
        
        if any(term in field_name_lower for term in ['date', 'dob', 'birth']):
            return 'date'
        elif any(term in field_name_lower for term in ['certificate_number', 'cert_number', 'cert_no', 'certificate_no', 'cert_id']):
            return 'certificate_number'
        elif any(term in field_name_lower for term in ['id', 'id_number', 'id_no']):
            return 'id_number'
        elif any(term in field_name_lower for term in ['phone', 'tel', 'mobile']):
            return 'numeric'
        else:
            return 'text'
    
    def verify(self, document_data: Dict[str, str], form_data: Dict[str, str]) -> Dict:
        """
        Verify form data against document data
        
        Args:
            document_data: Extracted fields from document
            form_data: User-filled form data
        
        Returns:
            Verification results dictionary
        """
        field_results = []
        matched_fields = []
        mismatched_fields = []
        unsure_fields = []
        missing_fields = []
        
        # Normalize field names for matching (handle variations like "date" vs "dob", "course" vs "course_name")
        field_mapping = {
            'date': ['date', 'dob', 'birth_date', 'date_of_birth'],
            'name': ['name', 'full_name', 'fullname', 'applicant_name'],
            'course': ['course', 'course_name', 'course_title'],
            'certificate_number': ['certificate_number', 'cert_number', 'cert_no', 'certificate_no', 'cert_id'],
            'id_number': ['id_number', 'id', 'identification_number', 'id_no'],
            'address': ['address', 'street_address', 'residence'],
            'phone': ['phone', 'telephone', 'mobile', 'contact_number'],
            'email': ['email', 'e-mail', 'mail']
        }
        
        # Create reverse mapping for quick lookup
        reverse_mapping = {}
        for canonical, variants in field_mapping.items():
            for variant in variants:
                reverse_mapping[variant.lower()] = canonical
        
        # Check all form fields
        for field, form_value in form_data.items():
            # Try to find matching document field (handle field name variations)
            field_lower = field.lower()
            canonical_field = reverse_mapping.get(field_lower, field_lower)
            
            # Try multiple field name variations
            possible_fields = [canonical_field, field, field_lower]
            document_value = None
            matched_field = None
            
            for possible_field in possible_fields:
                if possible_field in document_data:
                    document_value = document_data[possible_field]
                    matched_field = possible_field
                    break
            
            # If still not found, try case-insensitive search
            if not document_value:
                for doc_field, doc_value in document_data.items():
                    if doc_field.lower() == field_lower or doc_field.lower() == canonical_field:
                        document_value = doc_value
                        matched_field = doc_field
                        break
            
            if not document_value or not document_value.strip():
                missing_fields.append(field)
                field_results.append({
                    'field': field,
                    'document_value': "",
                    'form_value': form_value,
                    'status': 'unsure',
                    'similarity_score': 0.0
                })
                continue
            
            # Calculate similarity
            field_type = self.get_field_type(field)
            similarity = self.calculate_similarity(document_value, form_value, field_type)
            
            # Determine status
            status = self.determine_status(similarity)
            
            logger.debug(f"Verifying field '{field}': document='{document_value}' vs form='{form_value}' -> similarity={similarity:.3f}, status={status}")
            
            field_results.append({
                'field': field,
                'document_value': document_value,
                'form_value': form_value,
                'status': status,
                'similarity_score': similarity
            })
            
            if status == "match":
                matched_fields.append(field)
            elif status == "mismatch":
                mismatched_fields.append(field)
            else:
                unsure_fields.append(field)
        
        # Check for extra fields in document that weren't in form
        extra_fields = set(document_data.keys()) - set(form_data.keys())
        
        # Calculate overall verification score
        if not field_results:
            overall_score = 0.0
        else:
            # Weighted average: matches count more, mismatches count less
            scores = []
            for result in field_results:
                if result['status'] == 'match':
                    scores.append(result['similarity_score'])
                elif result['status'] == 'mismatch':
                    scores.append(result['similarity_score'] * 0.5)  # Penalize mismatches
                else:
                    scores.append(result['similarity_score'] * 0.7)  # Partial credit for unsure
            
            overall_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            'overall_verification_score': overall_score,
            'field_results': field_results,
            'matched_fields': matched_fields,
            'mismatched_fields': mismatched_fields,
            'unsure_fields': unsure_fields,
            'missing_fields': missing_fields,
            'extra_fields': list(extra_fields)
        }
