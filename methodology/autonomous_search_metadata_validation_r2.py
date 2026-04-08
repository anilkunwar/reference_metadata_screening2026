# app.py - BibTeX Hallucination-Free Reference Validator
# Complete expanded version with full error handling and no redactions

import streamlit as st
import bibtexparser
import requests
import re
import json
import time
import hashlib
import logging
from typing import Optional, Dict, List, Tuple, Union, Any
from bibtexparser.bparser import BibTexParser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.customization import getnames, convert_to_unicode
from urllib.parse import quote
import warnings
warnings.filterwarnings("ignore")

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Optional LLM Imports with Graceful Fallback ====================
TRANSFORMERS_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    logger.info(f"PyTorch available: {TORCH_AVAILABLE}, CUDA: {torch.cuda.is_available()}")
except ImportError:
    logger.warning("PyTorch not installed - LLM features will be disabled")
    torch = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers library available")
except ImportError:
    logger.warning("Transformers library not installed - LLM features will be disabled")
    AutoTokenizer = None
    AutoModelForCausalLM = None
    pipeline = None
    GenerationConfig = None

# ==================== Streamlit Page Configuration ====================
st.set_page_config(
    page_title="BibTeX Hallucination Checker",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/bibtex-validator',
        'Report a bug': 'https://github.com/your-repo/bibtex-validator/issues',
        'About': "# BibTeX Hallucination Validator\nValidates references against Crossref/OpenAlex APIs"
    }
)

# ==================== Global Constants ====================
CROSSREF_API_URL = "https://api.crossref.org/works/"
OPENALEX_API_URL = "https://api.openalex.org/works/https://doi.org/"
CROSSREF_SEARCH_URL = "https://api.crossref.org/works"
USER_AGENT = "BibTeX-Validator/1.0 (mailto:validator@example.com)"
DEFAULT_TIMEOUT = 10
MAX_RETRIES = 3
RETRY_DELAY = 1.0
SIMILARITY_THRESHOLD = 0.85
TITLE_SEARCH_MAX_RESULTS = 3

# Valid BibTeX entry types
VALID_ENTRY_TYPES = {
    "article", "book", "booklet", "conference", "inbook", "incollection",
    "inproceedings", "manual", "mastersthesis", "misc", "phdthesis",
    "proceedings", "techreport", "unpublished", "online", "dataset",
    "software", "preprint", "report", "thesis", "periodical", "supplement"
}

# Fields to validate and compare
FIELDS_TO_VALIDATE = ["title", "author", "journal", "booktitle", "year", 
                      "volume", "number", "pages", "doi", "publisher", "note"]

# Fields that are required for a valid entry
REQUIRED_FIELDS = {
    "article": ["author", "title", "journal", "year"],
    "inproceedings": ["author", "title", "booktitle", "year"],
    "book": ["author", "title", "publisher", "year"],
    "misc": ["author", "title", "year"]
}

# ==================== Helper Functions - String Processing ====================

def safe_string_slice(value: Any, max_length: int = 50) -> str:
    """
    Safely slice any value to string with max length, handling None and non-string types.
    
    Args:
        value: Any value that might be None, string, list, or other type
        max_length: Maximum length for the returned string
        
    Returns:
        Safely truncated string representation
    """
    if value is None:
        return "Untitled"
    
    # Convert to string if not already
    if not isinstance(value, str):
        if isinstance(value, list):
            # Handle list of authors or other fields
            value = " and ".join(str(v) for v in value if v is not None)
        else:
            value = str(value)
    
    # Clean and truncate
    cleaned = value.strip()
    if len(cleaned) > max_length:
        return cleaned[:max_length] + "..."
    return cleaned if cleaned else "Untitled"


def normalize_text(text: Optional[str]) -> str:
    """
    Normalize text for comparison: lowercase, remove extra whitespace, strip punctuation.
    
    Args:
        text: Input text string or None
        
    Returns:
        Normalized string for comparison
    """
    if text is None:
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower().strip()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common punctuation that shouldn't affect matching
    text = re.sub(r'[^\w\s\-\.]', '', text)
    
    return text.strip()


def clean_doi(doi: Optional[str]) -> Optional[str]:
    """
    Clean and normalize DOI string by removing URL prefixes and whitespace.
    
    Args:
        doi: Raw DOI string that might contain URL prefixes
        
    Returns:
        Cleaned DOI string or None if invalid
    """
    if doi is None or not isinstance(doi, str):
        return None
    
    # Remove common DOI URL prefixes
    patterns_to_remove = [
        r'^https?://doi\.org/',
        r'^https?://dx\.doi\.org/',
        r'^doi:\s*',
        r'^DOI:\s*',
        r'^http://dx\.doi\.org/',
        r'^www\.doi\.org/',
    ]
    
    cleaned = doi.strip()
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Validate DOI format (basic check)
    if cleaned and '/' in cleaned and len(cleaned) > 5:
        return cleaned.strip()
    
    return None


def format_authors_for_bibtex(authors: Union[str, List[str], None]) -> str:
    """
    Format authors list into proper BibTeX "and"-separated format.
    
    Args:
        authors: Can be string, list of strings, or None
        
    Returns:
        Properly formatted author string for BibTeX
    """
    if authors is None:
        return ""
    
    # Handle string input (already formatted or single author)
    if isinstance(authors, str):
        authors = authors.strip()
        if not authors:
            return ""
        # Check if already in "and" format
        if " and " in authors.lower():
            return authors
        # Check if in "Last, First; Last, First" format (convert to "and")
        if ";" in authors:
            return authors.replace(";", " and")
        return authors
    
    # Handle list input
    if isinstance(authors, list):
        # Filter out None/empty values and convert to strings
        valid_authors = [str(a).strip() for a in authors if a and str(a).strip()]
        if not valid_authors:
            return ""
        return " and ".join(valid_authors)
    
    # Fallback for any other type
    return str(authors).strip()


def parse_authors_string(authors_str: Optional[str]) -> List[str]:
    """
    Parse BibTeX author string into list of individual authors.
    
    Args:
        authors_str: BibTeX formatted author string (e.g., "Smith, John and Doe, Jane")
        
    Returns:
        List of individual author names
    """
    if not authors_str or not isinstance(authors_str, str):
        return []
    
    # Split by " and " (case-insensitive)
    authors = re.split(r'\s+and\s+', authors_str.strip(), flags=re.IGNORECASE)
    
    # Clean each author name
    cleaned_authors = []
    for author in authors:
        author = author.strip()
        if author:
            # Handle "Last, First" format - keep as is for BibTeX compatibility
            cleaned_authors.append(author)
    
    return cleaned_authors


def calculate_string_similarity(str1: Optional[str], str2: Optional[str]) -> float:
    """
    Calculate similarity ratio between two strings using token-based Jaccard similarity.
    
    Args:
        str1: First string for comparison
        str2: Second string for comparison
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if str1 is None or str2 is None:
        return 0.0
    
    # Normalize both strings
    norm1 = normalize_text(str1)
    norm2 = normalize_text(str2)
    
    # Exact match
    if norm1 == norm2:
        return 1.0
    
    # Empty string check
    if not norm1 or not norm2:
        return 0.0
    
    # Tokenize into words
    tokens1 = set(re.findall(r'\b[a-z0-9]+\b', norm1))
    tokens2 = set(re.findall(r'\b[a-z0-9]+\b', norm2))
    
    if not tokens1 or not tokens2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)


def extract_year_from_date(date_value: Any) -> Optional[str]:
    """
    Extract year from various date formats returned by APIs.
    
    Args:
        date_value: Date value from API (could be dict, list, string, or int)
        
    Returns:
        Year as string or None if not extractable
    """
    if date_value is None:
        return None
    
    # Handle integer year directly
    if isinstance(date_value, int):
        return str(date_value) if 1000 <= date_value <= 2100 else None
    
    # Handle string year
    if isinstance(date_value, str):
        date_value = date_value.strip()
        # Try to extract 4-digit year
        match = re.search(r'\b(1[0-9]{3}|2[0-9]{3})\b', date_value)
        if match:
            return match.group(1)
        return None
    
    # Handle Crossref/OpenAlex date-parts format: {"date-parts": [[2024, 3, 15]]}
    if isinstance(date_value, dict):
        date_parts = date_value.get("date-parts")
        if date_parts and isinstance(date_parts, list) and len(date_parts) > 0:
            first_part = date_parts[0]
            if isinstance(first_part, list) and len(first_part) > 0:
                year = first_part[0]
                if isinstance(year, int) and 1000 <= year <= 2100:
                    return str(year)
    
    # Handle list format
    if isinstance(date_value, list) and len(date_value) > 0:
        first_item = date_value[0]
        if isinstance(first_item, (int, str)):
            return extract_year_from_date(first_item)
    
    return None


# ==================== API Interaction Functions ====================

def make_api_request(url: str, headers: Dict[str, str], timeout: int, 
                    params: Optional[Dict] = None, max_retries: int = MAX_RETRIES) -> Optional[Dict]:
    """
    Make HTTP request with retry logic and error handling.
    
    Args:
        url: API endpoint URL
        headers: HTTP headers to include
        timeout: Request timeout in seconds
        params: Optional query parameters
        max_retries: Maximum number of retry attempts
        
    Returns:
        JSON response as dict or None if request fails
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url, 
                headers=headers, 
                params=params, 
                timeout=timeout,
                allow_redirects=True
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After', RETRY_DELAY)
                try:
                    wait_time = float(retry_after)
                except (ValueError, TypeError):
                    wait_time = RETRY_DELAY * (attempt + 1)
                logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
                continue
            
            # Handle other HTTP errors
            if response.status_code >= 400:
                logger.warning(f"HTTP {response.status_code} from {url}")
                if response.status_code >= 500:
                    # Server error, might be temporary
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                return None
            
            # Parse JSON response
            return response.json()
            
        except requests.exceptions.Timeout:
            last_exception = f"Timeout after {timeout}s"
            logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries}): {url}")
        except requests.exceptions.ConnectionError:
            last_exception = "Connection error"
            logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}): {url}")
        except requests.exceptions.RequestException as e:
            last_exception = str(e)
            logger.warning(f"Request exception (attempt {attempt + 1}/{max_retries}): {e}")
        except json.JSONDecodeError as e:
            last_exception = f"JSON decode error: {e}"
            logger.warning(f"Failed to parse JSON response: {e}")
            return None
        
        # Wait before retry (exponential backoff)
        if attempt < max_retries - 1:
            wait_time = RETRY_DELAY * (2 ** attempt)
            time.sleep(wait_time)
    
    logger.error(f"All {max_retries} attempts failed for {url}. Last error: {last_exception}")
    return None


def fetch_crossref_metadata(doi: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[Dict[str, Any]]:
    """
    Fetch publication metadata from Crossref API using DOI.
    
    Args:
        doi: Digital Object Identifier to lookup
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with standardized metadata fields or None if lookup fails
    """
    clean_doi_val = clean_doi(doi)
    if not clean_doi_val:
        logger.debug(f"Invalid DOI format: {doi}")
        return None
    
    url = f"{CROSSREF_API_URL}{quote(clean_doi_val)}"
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json"
    }
    
    logger.info(f"Querying Crossref for DOI: {clean_doi_val}")
    data = make_api_request(url, headers, timeout)
    
    if not data or "message" not in data:
        return None
    
    message = data["message"]
    
    # Extract and standardize metadata fields
    result = {
        "source": "crossref",
        "doi": message.get("DOI"),
        "title": None,
        "authors": [],
        "journal": None,
        "year": None,
        "volume": None,
        "issue": None,
        "pages": None,
        "publisher": None,
        "type": message.get("type"),
        "url": message.get("URL"),
        "issn": None,
        "isbn": None
    }
    
    # Extract title (Crossref returns list)
    titles = message.get("title")
    if titles and isinstance(titles, list) and len(titles) > 0:
        result["title"] = titles[0]
    
    # Extract authors
    authors_data = message.get("author", [])
    if authors_data and isinstance(authors_data, list):
        for author in authors_data:
            if isinstance(author, dict):
                given = author.get("given", "") or ""
                family = author.get("family", "") or ""
                # Handle cases where name is in "name" field instead
                if not given and not family:
                    name = author.get("name", "")
                    if name:
                        result["authors"].append(name)
                        continue
                # Combine given and family names
                full_name = f"{given} {family}".strip()
                if full_name:
                    result["authors"].append(full_name)
    
    # Extract journal/container title
    container_titles = message.get("container-title", [])
    short_titles = message.get("short-container-title", [])
    
    if container_titles and isinstance(container_titles, list) and len(container_titles) > 0:
        result["journal"] = container_titles[0]
    elif short_titles and isinstance(short_titles, list) and len(short_titles) > 0:
        result["journal"] = short_titles[0]
    
    # Extract year from various date fields
    for date_field in ["published-print", "published-online", "created", "deposited"]:
        date_data = message.get(date_field)
        if date_data:
            year = extract_year_from_date(date_data)
            if year:
                result["year"] = year
                break
    
    # Extract volume, issue, pages
    result["volume"] = message.get("volume")
    result["issue"] = message.get("issue")
    result["pages"] = message.get("page")
    
    # Extract publisher
    result["publisher"] = message.get("publisher")
    
    # Extract ISSN/ISBN
    issn = message.get("ISSN", [])
    isbn = message.get("ISBN", [])
    if issn and isinstance(issn, list) and len(issn) > 0:
        result["issn"] = issn[0]
    if isbn and isinstance(isbn, list) and len(isbn) > 0:
        result["isbn"] = isbn[0]
    
    return result


def fetch_openalex_metadata(doi: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[Dict[str, Any]]:
    """
    Fetch publication metadata from OpenAlex API as fallback when Crossref fails.
    
    Args:
        doi: Digital Object Identifier to lookup
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with standardized metadata fields or None if lookup fails
    """
    clean_doi_val = clean_doi(doi)
    if not clean_doi_val:
        return None
    
    url = f"{OPENALEX_API_URL}{quote(clean_doi_val)}"
    headers = {"User-Agent": USER_AGENT}
    
    logger.info(f"Querying OpenAlex for DOI: {clean_doi_val}")
    data = make_api_request(url, headers, timeout)
    
    if not data or data.get("id") is None:
        return None
    
    # Extract and standardize metadata fields
    result = {
        "source": "openalex",
        "doi": data.get("doi"),
        "title": data.get("title"),
        "authors": [],
        "journal": None,
        "year": data.get("publication_year"),
        "volume": None,
        "issue": None,
        "pages": None,
        "publisher": None,
        "type": data.get("type"),
        "url": data.get("doi"),
        "issn": None,
        "isbn": None
    }
    
    # Extract authors from authorships
    authorships = data.get("authorships", [])
    if authorships and isinstance(authorships, list):
        for authorship in authorships:
            if isinstance(authorship, dict):
                author_data = authorship.get("author", {})
                if isinstance(author_data, dict):
                    display_name = author_data.get("display_name")
                    if display_name:
                        result["authors"].append(display_name)
    
    # Extract journal/source information
    primary_location = data.get("primary_location", {})
    if primary_location and isinstance(primary_location, dict):
        source_info = primary_location.get("source", {})
        if source_info and isinstance(source_info, dict):
            result["journal"] = source_info.get("display_name")
            # Extract ISSN if available
            issns = source_info.get("issn_l") or source_info.get("issn")
            if issns:
                result["issn"] = issns if isinstance(issns, str) else issns[0] if isinstance(issns, list) else None
    
    # Extract volume/issue/pages from biblio field
    biblio = data.get("biblio", {})
    if biblio and isinstance(biblio, dict):
        result["volume"] = biblio.get("volume")
        result["issue"] = biblio.get("issue")
        # OpenAlex has first_page and last_page
        first_page = biblio.get("first_page")
        last_page = biblio.get("last_page")
        if first_page and last_page:
            result["pages"] = f"{first_page}--{last_page}"
        elif first_page:
            result["pages"] = str(first_page)
    
    return result


def search_crossref_by_title(title: str, max_results: int = TITLE_SEARCH_MAX_RESULTS, 
                            timeout: int = DEFAULT_TIMEOUT) -> List[Dict[str, Any]]:
    """
    Search Crossref API by title as fallback when DOI lookup fails.
    
    Args:
        title: Publication title to search for
        max_results: Maximum number of results to return
        timeout: Request timeout in seconds
        
    Returns:
        List of matching publications with metadata and match scores
    """
    if not title or not isinstance(title, str):
        return []
    
    url = CROSSREF_SEARCH_URL
    params = {
        "query.title": title,
        "rows": max_results,
        "select": "DOI,title,author,container-title,published-print,volume,page,type,score"
    }
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    
    logger.info(f"Searching Crossref by title: {safe_string_slice(title, 60)}")
    data = make_api_request(url, headers, timeout, params)
    
    if not data or "message" not in data:
        return []
    
    items = data["message"].get("items", [])
    results = []
    
    for item in items:
        if not isinstance(item, dict):
            continue
            
        # Extract basic metadata
        entry = {
            "doi": item.get("DOI"),
            "title": None,
            "authors": [],
            "journal": None,
            "year": None,
            "volume": item.get("volume"),
            "pages": item.get("page"),
            "type": item.get("type"),
            "match_score": item.get("score", 0.0)
        }
        
        # Extract title
        titles = item.get("title", [])
        if titles and isinstance(titles, list) and len(titles) > 0:
            entry["title"] = titles[0]
        
        # Extract authors
        authors_data = item.get("author", [])
        if authors_data and isinstance(authors_data, list):
            for author in authors_data:
                if isinstance(author, dict):
                    given = author.get("given", "") or ""
                    family = author.get("family", "") or ""
                    full_name = f"{given} {family}".strip()
                    if full_name:
                        entry["authors"].append(full_name)
        
        # Extract journal
        containers = item.get("container-title", [])
        if containers and isinstance(containers, list) and len(containers) > 0:
            entry["journal"] = containers[0]
        
        # Extract year
        published = item.get("published-print", {})
        if published and isinstance(published, dict):
            year = extract_year_from_date(published)
            if year:
                entry["year"] = year
        
        # Only include if we have at least a DOI or title
        if entry["doi"] or entry["title"]:
            results.append(entry)
    
    # Sort by match score descending
    results.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    
    return results


# ==================== Metadata Comparison Functions ====================

def compare_metadata_fields(original: Dict[str, Any], verified: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Compare original BibTeX entry fields with verified API metadata.
    
    Args:
        original: Original BibTeX entry dictionary
        verified: Verified metadata from external API
        
    Returns:
        Dictionary mapping field names to comparison results
    """
    comparison_results = {}
    
    for field in FIELDS_TO_VALIDATE:
        orig_value = original.get(field)
        verify_value = verified.get(field)
        
        # Handle authors field specially (can be string or list)
        if field == "author" or field == "authors":
            orig_authors = parse_authors_string(format_authors_for_bibtex(orig_value))
            verify_authors = verified.get("authors", []) if isinstance(verified.get("authors"), list) else parse_authors_string(verified.get("authors"))
            
            # Compare author sets (order-independent)
            orig_set = set(normalize_text(a) for a in orig_authors if a)
            verify_set = set(normalize_text(a) for a in verify_authors if a)
            
            if not orig_set and not verify_set:
                is_match = True
            elif not orig_set or not verify_set:
                is_match = False
            else:
                # Calculate overlap ratio
                intersection = orig_set.intersection(verify_set)
                union = orig_set.union(verify_set)
                overlap_ratio = len(intersection) / len(union) if union else 0
                is_match = overlap_ratio >= 0.7  # 70% author overlap threshold
            
            comparison_results[field] = {
                "original": orig_value,
                "verified": verify_authors,
                "match": is_match,
                "confidence": "high" if is_match and len(orig_set) == len(verify_set) else "medium" if is_match else "low",
                "needs_review": not is_match and verify_value is not None
            }
            continue
        
        # Normalize values for comparison
        orig_normalized = normalize_text(orig_value)
        verify_normalized = normalize_text(verify_value)
        
        # Determine if values match
        if not orig_normalized and not verify_normalized:
            is_match = True
            confidence = "high"
        elif not orig_normalized or not verify_normalized:
            is_match = False
            confidence = "low"
        elif orig_normalized == verify_normalized:
            is_match = True
            confidence = "high"
        elif field == "title":
            # Use fuzzy matching for titles
            similarity = calculate_string_similarity(orig_value, verify_value)
            is_match = similarity >= SIMILARITY_THRESHOLD
            confidence = "high" if similarity >= 0.95 else "medium" if is_match else "low"
        elif field == "year":
            # Year should match exactly
            is_match = orig_normalized == verify_normalized
            confidence = "high" if is_match else "low"
        else:
            # For other fields, check substring containment
            is_match = (orig_normalized in verify_normalized or 
                       verify_normalized in orig_normalized or
                       calculate_string_similarity(orig_value, verify_value) >= 0.8)
            
            confidence = "high" if is_match else "medium" if calculate_string_similarity(orig_value, verify_value) >= 0.7 else "low"
        
        comparison_results[field] = {
            "original": orig_value,
            "verified": verify_value,
            "match": is_match,
            "confidence": confidence,
            "needs_review": not is_match and verify_value is not None,
            "similarity": calculate_string_similarity(orig_value, verify_value) if field == "title" else None
        }
    
    return comparison_results


def merge_metadata_entries(original: Dict[str, Any], verified: Dict[str, Any], 
                          comparison_results: Dict[str, Dict], auto_correct: bool = True) -> Dict[str, Any]:
    """
    Merge original BibTeX entry with verified metadata based on comparison results.
    
    Args:
        original: Original BibTeX entry
        verified: Verified metadata from external API
        comparison_results: Results from compare_metadata_fields()
        auto_correct: Whether to automatically correct high-confidence matches
        
    Returns:
        Merged entry dictionary with corrected fields
    """
    # Start with a copy of original to preserve fields not in comparison
    merged = {k: v for k, v in original.items() if not k.startswith("_")}
    
    # Add verification metadata
    merged["_verification_source"] = verified.get("source", "unknown")
    merged["_verified_doi"] = verified.get("doi")
    merged["_verification_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    if not auto_correct:
        # Just attach comparison results for manual review
        merged["_comparison_results"] = comparison_results
        merged["_verified_metadata"] = verified
        return merged
    
    # Auto-correct fields with high/medium confidence matches
    for field, result in comparison_results.items():
        # Skip if no verified data available
        if result["verified"] is None:
            continue
        
        # Auto-correct if match is confirmed with sufficient confidence
        if result["match"] and result["confidence"] in ["high", "medium"]:
            # Handle author field formatting
            if field in ["author", "authors"]:
                merged["author"] = format_authors_for_bibtex(result["verified"])
            else:
                merged[field] = result["verified"]
            merged[f"_corrected_{field}"] = True
            logger.debug(f"Auto-corrected field '{field}': {result['original']} -> {result['verified']}")
        elif result["needs_review"]:
            # Flag for manual review but keep original
            merged[f"_flag_{field}"] = True
            merged[f"_flag_reason_{field}"] = f"Discrepancy: original='{result['original']}', verified='{result['verified']}', confidence={result['confidence']}"
            logger.info(f"Field '{field}' flagged for review: {result['original']} vs {result['verified']}")
    
    return merged


# ==================== LLM Integration Functions ====================

def initialize_llm_pipeline(model_name: str, device: Optional[str] = None) -> Optional[Any]:
    """
    Initialize HuggingFace pipeline for small LLMs (<1B parameters).
    
    Args:
        model_name: Name/identifier of the model to load
        device: Target device ('cuda', 'cpu', or None for auto)
        
    Returns:
        Initialized pipeline object or None if initialization fails
    """
    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
        logger.warning("Cannot initialize LLM: transformers or torch not available")
        return None
    
    try:
        # Map user-friendly names to HuggingFace model IDs
        model_mapping = {
            "gpt2": "gpt2",
            "gpt2 (~124M params)": "gpt2",
            "distilgpt2": "distilgpt2", 
            "distilgpt2 (~82M params)": "distilgpt2",
            "Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen2.5-0.5B-Instruct (~0.5B params)*": "Qwen/Qwen2.5-0.5B-Instruct",
            "TinyLlama-1.1B-Chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        }
        
        model_id = model_mapping.get(model_name, model_name)
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading model '{model_id}' on device: {device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        # Load model with appropriate precision
        if device == "cuda" and torch.cuda.is_available():
            # Use float16 on GPU for memory efficiency
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.device_count() > 1 else None,
                low_cpu_mem_usage=True
            )
        else:
            # Use float32 on CPU or single GPU
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
        
        # Configure generation parameters for factual refinement (not creative generation)
        generation_config = GenerationConfig(
            max_new_tokens=256,
            temperature=0.1,  # Low temperature for deterministic output
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,  # Reduce repetition
            no_repeat_ngram_size=3,
            num_return_sequences=1
        )
        
        # Create pipeline
        text_generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" and torch.cuda.is_available() else -1,
            generation_config=generation_config
        )
        
        logger.info(f"Successfully initialized LLM pipeline for {model_id}")
        return text_generation_pipeline
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM pipeline: {type(e).__name__}: {e}", exc_info=True)
        return None


def build_llm_prompt_for_metadata_refinement(original: Dict[str, Any], 
                                            verified: Dict[str, Any],
                                            discrepancies: Dict[str, Dict]) -> str:
    """
    Build structured prompt for LLM to help resolve metadata discrepancies.
    
    Args:
        original: Original BibTeX entry data
        verified: Verified metadata from external API
        discrepancies: Field-by-field comparison results
        
    Returns:
        Formatted prompt string for LLM inference
    """
    # Filter to only fields that need review
    fields_needing_review = {
        k: v for k, v in discrepancies.items() 
        if v.get("needs_review") and v.get("verified") is not None
    }
    
    if not fields_needing_review:
        return ""  # No refinement needed
    
    # Format original entry for prompt
    orig_lines = []
    for field in ["author", "title", "journal", "booktitle", "year", "volume", "pages", "doi"]:
        value = original.get(field)
        if value is not None:
            orig_lines.append(f"  {field} = {{{format_authors_for_bibtex(value) if field == 'author' else value}}}")
    orig_formatted = "@article{KEY,\n" + ",\n".join(orig_lines) + "\n}"
    
    # Format verified data for prompt
    verified_lines = []
    for field in ["author", "title", "journal", "year", "volume", "pages"]:
        value = verified.get(field)
        if value is not None:
            if field == "author" and isinstance(value, list):
                value = " and ".join(value)
            verified_lines.append(f"  {field}: {value}")
    verified_formatted = "{\n" + ",\n".join(verified_lines) + "\n}"
    
    # Format discrepancies for prompt
    discrepancy_lines = []
    for field, info in fields_needing_review.items():
        discrepancy_lines.append(
            f"- {field}: original='{info['original']}' | verified='{info['verified']}' | confidence={info['confidence']}"
        )
    discrepancies_formatted = "\n".join(discrepancy_lines)
    
    # Construct the full prompt with clear instructions
    prompt = f"""You are an expert academic metadata validator. Your task is to resolve discrepancies between a BibTeX reference entry and verified metadata from Crossref/OpenAlex APIs.

### ORIGINAL BIBTEX ENTRY:
{orig_formatted}

### VERIFIED METADATA (from Crossref/OpenAlex):
{verified_formatted}

### FIELDS WITH DISCREPANCIES NEEDING RESOLUTION:
{discrepancies_formatted}

### INSTRUCTIONS:
1. Prefer verified metadata when it has high confidence and the original appears to contain errors
2. Keep original values when verified data is missing, incomplete, or clearly incorrect
3. For author names: output in BibTeX format "Last, First and Last, First" preserving all authors
4. For titles: preserve exact capitalization and special characters from verified source
5. For journal names: use the full journal name from verified source, not abbreviations
6. For year/volume/pages: use verified numeric values when available
7. Output ONLY valid BibTeX field assignments, one per line, in the format: field = {{value}},
8. Do NOT include the @article wrapper, cite key, explanations, or any text outside the field assignments
9. If no changes are needed, output: # NO_CHANGES_REQUIRED

### OUTPUT FORMAT (strictly follow):
author = {{Verified, Author and Another, Person}},
title = {{Exact Title From Verified Source}},
journal = {{Full Journal Name}},
year = {{2024}},
volume = {{12}},
pages = {{123--456}},

### BEGIN OUTPUT:
"""
    return prompt


def parse_llm_output_for_bibtex_fields(llm_output: str) -> Dict[str, str]:
    """
    Parse LLM output to extract corrected BibTeX field assignments.
    
    Args:
        llm_output: Raw text output from LLM inference
        
    Returns:
        Dictionary of field names to corrected values
    """
    if not llm_output or "# NO_CHANGES_REQUIRED" in llm_output:
        return {}
    
    corrections = {}
    
    # Split into lines and process each
    for line in llm_output.strip().split("\n"):
        line = line.strip()
        
        # Skip empty lines, comments, or non-field lines
        if not line or line.startswith("#") or line.startswith("@") or line.startswith("{"):
            continue
        
        # Parse field = {value} format
        match = re.match(r'^(\w+)\s*=\s*\{(.*)\},?\s*$', line)
        if match:
            field_name = match.group(1).strip().lower()
            field_value = match.group(2).strip()
            
            # Only accept known BibTeX fields
            if field_name in FIELDS_TO_VALIDATE:
                corrections[field_name] = field_value
                logger.debug(f"Parsed LLM correction: {field_name} = {field_value}")
    
    return corrections


def refine_metadata_with_llm(original: Dict[str, Any], verified: Dict[str, Any],
                            discrepancies: Dict[str, Dict], llm_pipeline: Any) -> Dict[str, Any]:
    """
    Use LLM to help resolve ambiguous metadata discrepancies.
    
    Args:
        original: Original BibTeX entry
        verified: Verified metadata from API
        discrepancies: Field comparison results
        llm_pipeline: Initialized HuggingFace pipeline
        
    Returns:
        Entry with LLM-refined metadata fields
    """
    if llm_pipeline is None:
        logger.warning("LLM pipeline not available, skipping refinement")
        return merge_metadata_entries(original, verified, discrepancies)
    
    try:
        # Build prompt
        prompt = build_llm_prompt_for_metadata_refinement(original, verified, discrepancies)
        
        if not prompt:
            # No discrepancies need LLM help
            return merge_metadata_entries(original, verified, discrepancies)
        
        logger.info("Running LLM refinement for metadata discrepancies")
        
        # Generate response with timeout protection
        start_time = time.time()
        result = llm_pipeline(prompt, max_new_tokens=256)
        elapsed = time.time() - start_time
        logger.info(f"LLM inference completed in {elapsed:.2f}s")
        
        # Extract generated text (handle different pipeline output formats)
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            generated_text = result.get("generated_text", "")
        else:
            generated_text = str(result)
        
        # Extract only the new part (after our prompt)
        if prompt in generated_text:
            llm_response = generated_text.split(prompt)[-1].strip()
        else:
            llm_response = generated_text.strip()
        
        # Parse corrections from LLM output
        corrections = parse_llm_output_for_bibtex_fields(llm_response)
        
        if not corrections:
            logger.info("LLM returned no valid corrections, using standard merge")
            return merge_metadata_entries(original, verified, discrepancies)
        
        # Apply LLM corrections to merged entry
        merged = merge_metadata_entries(original, verified, discrepancies)
        for field, value in corrections.items():
            merged[field] = value
            merged[f"_llm_corrected_{field}"] = True
            logger.info(f"Applied LLM correction to '{field}': {value}")
        
        # Log the raw LLM response for debugging (truncated)
        merged["_llm_response_preview"] = llm_response[:200] + ("..." if len(llm_response) > 200 else "")
        
        return merged
        
    except Exception as e:
        logger.error(f"LLM refinement failed: {type(e).__name__}: {e}", exc_info=True)
        # Fallback to standard merge without LLM
        return merge_metadata_entries(original, verified, discrepancies)


# ==================== BibTeX Parsing and Generation ====================

def parse_bibtex_file_content(file_content: bytes) -> Tuple[List[Dict], Optional[str]]:
    """
    Parse uploaded BibTeX file content into list of entry dictionaries.
    
    Args:
        file_content: Raw bytes from uploaded file
        
    Returns:
        Tuple of (list of entry dicts, error message or None)
    """
    try:
        # Decode with UTF-8, fallback to latin-1 if needed
        try:
            content = file_content.decode("utf-8")
        except UnicodeDecodeError:
            content = file_content.decode("latin-1")
        
        # Initialize parser with customizations
        parser = BibTexParser()
        parser.customization = convert_to_unicode
        parser.ignore_comments = False
        
        # Parse the content
        bib_database = bibtexparser.loads(content, parser)
        
        if not bib_database.entries:
            return [], "No valid BibTeX entries found in file"
        
        # Process each entry to ensure consistent field handling
        processed_entries = []
        for entry in bib_database.entries:
            processed = {}
            for key, value in entry.items():
                if key == "ID":
                    processed["cite_key"] = value
                elif key == "ENTRYTYPE":
                    processed["entry_type"] = value if value in VALID_ENTRY_TYPES else "misc"
                elif value is not None:
                    # Clean and store the value
                    if isinstance(value, str):
                        processed[key] = value.strip()
                    else:
                        processed[key] = value
            processed_entries.append(processed)
        
        return processed_entries, None
        
    except bibtexparser.bibdatabase.UndefinedString as e:
        return [], f"BibTeX parsing error: Undefined string reference - {e}"
    except Exception as e:
        return [], f"Failed to parse BibTeX file: {type(e).__name__}: {e}"


def generate_bibtex_entry_string(entry: Dict[str, Any], cite_key: Optional[str] = None) -> str:
    """
    Generate properly formatted BibTeX entry string from dictionary.
    
    Args:
        entry: Dictionary containing BibTeX field values
        cite_key: Optional cite key (uses entry["cite_key"] if not provided)
        
    Returns:
        Formatted BibTeX entry string
    """
    # Get cite key and entry type
    key = cite_key or entry.get("cite_key", "unknown_key")
    entry_type = entry.get("entry_type", "article")
    
    # Fields to include in output (in preferred order)
    output_fields = [
        "author", "title", "journal", "booktitle", "year", "volume", 
        "number", "pages", "doi", "url", "publisher", "note", "month",
        "issn", "isbn", "series", "edition", "chapter"
    ]
    
    # Build entry lines
    lines = [f"@{entry_type}{{{key},"]
    
    for field in output_fields:
        if field in entry and entry[field] is not None:
            value = entry[field]
            
            # Skip internal metadata fields
            if field.startswith("_"):
                continue
            
            # Format authors properly
            if field == "author" and isinstance(value, list):
                value = " and ".join(str(v).strip() for v in value if v)
            
            # Convert to string and escape special BibTeX characters in certain fields
            value_str = str(value).strip()
            if field in ["title", "journal", "booktitle", "note"]:
                # Escape braces but preserve existing brace groups
                value_str = re.sub(r'(?<!\\)([{}])', r'{\1}', value_str)
            
            # Add field to output
            lines.append(f"  {field} = {{{value_str}}},")
    
    # Remove trailing comma from last field and close entry
    if len(lines) > 1 and lines[-1].endswith(","):
        lines[-1] = lines[-1][:-1]
    lines.append("}")
    
    return "\n".join(lines)


def generate_complete_bibtex_file(entries: List[Dict[str, Any]], 
                                  metadata: Dict[str, Any]) -> str:
    """
    Generate complete BibTeX file content with header comments and all entries.
    
    Args:
        entries: List of processed entry dictionaries
        metadata: Generation metadata (timestamp, settings, etc.)
        
    Returns:
        Complete BibTeX file content as string
    """
    # Build header comment block
    header_lines = [
        "% ================================================================================",
        "% Hallucination-Validated BibTeX References",
        "% ================================================================================",
        f"% Generated: {metadata.get('timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))}",
        f"% Validation sources: {metadata.get('sources', 'Crossref, OpenAlex')}",
        f"% LLM refinement: {metadata.get('llm_model', 'Disabled')}",
        f"% Total entries: {len(entries)}",
        f"% Verified entries: {metadata.get('verified_count', 0)}",
        f"% Flagged for review: {metadata.get('flagged_count', 0)}",
        "%",
        "% Fields marked with _flag_* indicate discrepancies requiring manual review",
        "% ================================================================================",
        ""
    ]
    
    # Generate each entry
    entry_strings = []
    for entry in entries:
        cite_key = entry.get("cite_key", "unknown")
        entry_str = generate_bibtex_entry_string(entry, cite_key)
        entry_strings.append(entry_str)
    
    # Combine header and entries
    return "\n\n".join(header_lines + entry_strings)


# ==================== Streamlit Application Main Function ====================

def main():
    """Main Streamlit application entry point."""
    
    # Initialize session state for persistent data
    if "validation_results" not in st.session_state:
        st.session_state.validation_results = []
    if "llm_pipeline" not in st.session_state:
        st.session_state.llm_pipeline = None
    if "current_model" not in st.session_state:
        st.session_state.current_model = None
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    
    # Page title and description
    st.title("📚 BibTeX Hallucination Validator")
    st.markdown("""
    **Upload a `.bib` file** to validate references against **Crossref** and **OpenAlex** APIs.
    
    ✅ Detects and corrects hallucinated author names, journal titles, years, volumes, and page ranges  
    🔍 DOI-first verification with title-search fallback  
    🤖 Optional LLM refinement using models <1B parameters for ambiguous cases  
    💾 Download validated, corrected BibTeX ready for your LaTeX manuscript
    """)
    
    # ==================== Sidebar Configuration ====================
    with st.sidebar:
        st.header("⚙️ Validation Settings")
        
        # LLM Model Selection
        st.subheader("🤖 LLM Refinement (Optional)")
        llm_options = [
            "None (API-only verification)",
            "gpt2 (~124M params)",
            "distilgpt2 (~82M params)", 
            "Qwen2.5-0.5B-Instruct (~0.5B params)*",
            "TinyLlama-1.1B-Chat (~1.1B params)†"
        ]
        selected_model = st.selectbox(
            "Select model for metadata refinement:",
            options=llm_options,
            index=0,
            help="*Qwen2.5-0.5B: Best quality for <1B class, requires ~1GB VRAM\n†TinyLlama: Slightly larger but more capable, requires ~2GB VRAM"
        )
        
        use_llm = selected_model != "None (API-only verification)"
        
        # Load LLM if selected and not already loaded
        if use_llm and (st.session_state.llm_pipeline is None or st.session_state.current_model != selected_model):
            with st.status(f"🔄 Loading {selected_model}...", expanded=True) as status:
                st.write("Initializing model tokenizer...")
                st.write("Loading model weights...")
                st.write("Configuring generation parameters...")
                
                device_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
                st.write(f"Target device: {device_info}")
                
                pipeline = initialize_llm_pipeline(selected_model)
                
                if pipeline:
                    st.session_state.llm_pipeline = pipeline
                    st.session_state.current_model = selected_model
                    status.update(label=f"✅ {selected_model} loaded successfully", state="complete")
                else:
                    status.update(label=f"⚠️ Failed to load {selected_model}", state="error")
                    st.warning("Falling back to API-only verification mode")
                    use_llm = False
        
        # Advanced Options
        st.subheader("🔧 Advanced Options")
        
        timeout = st.slider(
            "API request timeout (seconds)", 
            min_value=5, max_value=60, value=10, step=5,
            help="Increase for slower network connections"
        )
        
        auto_correct = st.checkbox(
            "Auto-correct high-confidence matches", 
            value=True,
            help="Automatically fix fields where verified data confidently differs from original"
        )
        
        show_detailed_diff = st.checkbox(
            "Show field-by-field comparison", 
            value=True,
            help="Display side-by-side view of original vs verified metadata"
        )
        
        include_unverified = st.checkbox(
            "Include unverified entries in output", 
            value=True,
            help="Keep entries that couldn't be verified (with warning flags)"
        )
        
        # Rate limiting notice
        st.info("""
        **Rate Limiting Notice**  
        Crossref API: ~50 requests/minute without API key  
        OpenAlex API: ~100 requests/minute  
        Large files may take several minutes to process.
        """)
        
        # User email for Crossref API (required)
        user_email = st.text_input(
            "Your email (for Crossref API User-Agent):",
            value="validator@example.com",
            help="Required by Crossref API policy. Replace with your actual email."
        )
        if user_email and "example.com" not in user_email:
            global USER_AGENT
            USER_AGENT = f"BibTeX-Validator/1.0 (mailto:{user_email})"
        
        st.divider()
        
        # Quick stats
        st.subheader("📊 Session Stats")
        if st.session_state.validation_results:
            results = st.session_state.validation_results
            verified = sum(1 for r in results if r.get("verified"))
            flagged = sum(1 for r in results if any(
                d.get("needs_review") for d in r.get("discrepancies", {}).values()
            ))
            st.metric("✅ Verified", verified)
            st.metric("⚠️ Flagged", flagged)
            st.metric("📋 Total", len(results))
        else:
            st.caption("Upload a file to see validation statistics")
    
    # ==================== File Upload Section ====================
    st.subheader("📁 Upload BibTeX File")
    
    uploaded_file = st.file_uploader(
        "Choose a `.bib` or `.txt` file containing your references",
        type=["bib", "txt"],
        help="File should contain valid BibTeX entries (@article, @inproceedings, etc.)"
    )
    
    if uploaded_file is None:
        st.info("👆 Please upload a BibTeX file to begin validation.")
        
        # Show sample file format help
        with st.expander("📋 Example BibTeX Entry Format"):
            st.code("""
@article{Author2024Title,
  author  = {Last, First and Another, Person},
  title   = {Article Title with Proper Capitalization},
  journal = {Full Journal Name},
  year    = {2024},
  volume  = {12},
  pages   = {123--456},
  doi     = {10.1234/example.doi}
}
            """, language="bibtex")
        st.stop()
    
    # ==================== File Processing ====================
    # Parse the uploaded file
    with st.spinner("🔍 Parsing BibTeX entries..."):
        entries, parse_error = parse_bibtex_file_content(uploaded_file.getvalue())
    
    if parse_error:
        st.error(f"❌ {parse_error}")
        st.stop()
    
    if not entries:
        st.error("❌ No valid BibTeX entries found. Please check your file format.")
        st.stop()
    
    st.success(f"✅ Found **{len(entries)}** reference(s) to validate")
    
    # ==================== Validation Progress UI ====================
    st.subheader("🔄 Validation Progress")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    progress_details = st.expander("📋 Detailed Progress Log", expanded=False)
    
    validated_results = []
    
    # Process each entry sequentially with progress updates
    for idx, entry in enumerate(entries):
        # Update progress display
        progress = (idx + 1) / len(entries)
        progress_bar.progress(progress)
        
        cite_key = entry.get("cite_key", f"entry_{idx}")
        original_title = safe_string_slice(entry.get("title"), 60)
        original_doi = entry.get("doi", "")
        
        status_text.text(f"🔍 [{idx+1}/{len(entries)}] {original_title}")
        
        # Log to detailed view
        with progress_details:
            st.caption(f"`{cite_key}`: {original_title}")
        
        # ==================== Step 1: DOI-Based Lookup ====================
        verified_metadata = None
        
        if original_doi:
            clean_doi_var = clean_doi(original_doi)
            if clean_doi_var:
                # Try Crossref first (primary source)
                with progress_details:
                    st.text(f"  → Querying Crossref for DOI: {clean_doi_var}")
                verified_metadata = fetch_crossref_metadata(original_doi, timeout=timeout)
                
                # Fallback to OpenAlex if Crossref fails
                if not verified_metadata:
                    with progress_details:
                        st.text(f"  → Crossref not found, trying OpenAlex...")
                    verified_metadata = fetch_openalex_metadata(original_doi, timeout=timeout)
                    
                    if verified_metadata:
                        with progress_details:
                            st.success(f"  ✓ Found via OpenAlex")
                    else:
                        with progress_details:
                            st.warning(f"  ✗ Not found in Crossref or OpenAlex")
                else:
                    with progress_details:
                        st.success(f"  ✓ Found via Crossref")
        
        # ==================== Step 2: Title Search Fallback ====================
        if not verified_metadata and entry.get("title"):
            with progress_details:
                st.text(f"  → DOI lookup failed, searching by title...")
            
            title_results = search_crossref_by_title(
                entry["title"], 
                max_results=TITLE_SEARCH_MAX_RESULTS, 
                timeout=timeout
            )
            
            if title_results:
                with progress_details:
                    st.text(f"  → Found {len(title_results)} potential matches")
                
                # Select best match by score threshold
                best_match = None
                for result in title_results:
                    if result.get("match_score", 0) >= 0.9 and result.get("doi"):
                        best_match = result
                        break
                
                if best_match and best_match["doi"]:
                    with progress_details:
                        st.text(f"  → Best match (score={best_match['match_score']:.2f}): {best_match['title'][:50]}...")
                    # Fetch full metadata for the matched DOI
                    verified_metadata = fetch_crossref_metadata(best_match["doi"], timeout=timeout)
                    if verified_metadata:
                        with progress_details:
                            st.success(f"  ✓ Verified via title match + DOI lookup")
        
        # ==================== Step 3: Metadata Comparison ====================
        if verified_metadata:
            # Compare original vs verified fields
            discrepancies = compare_metadata_fields(entry, verified_metadata)
            
            # Count discrepancies for logging
            needs_review_count = sum(
                1 for d in discrepancies.values() if d.get("needs_review")
            )
            
            with progress_details:
                if needs_review_count == 0:
                    st.success(f"  ✓ All fields match - no corrections needed")
                else:
                    st.warning(f"  ⚠ {needs_review_count} field(s) need review")
            
            # ==================== Step 4: Apply Corrections ====================
            # Merge metadata with optional LLM refinement
            if use_llm and st.session_state.llm_pipeline and needs_review_count > 0:
                with progress_details:
                    st.text(f"  → Running LLM refinement with {selected_model}...")
                corrected_entry = refine_metadata_with_llm(
                    entry, verified_metadata, discrepancies, 
                    st.session_state.llm_pipeline
                )
            else:
                corrected_entry = merge_metadata_entries(
                    entry, verified_metadata, discrepancies, 
                    auto_correct=auto_correct
                )
            
            # Store result
            validated_results.append({
                "cite_key": cite_key,
                "original": entry,
                "verified": verified_metadata,
                "discrepancies": discrepancies,
                "corrected": corrected_entry,
                "status": "verified",
                "verification_source": verified_metadata.get("source", "unknown")
            })
            
        else:
            # No verification source found - flag for manual review
            discrepancies = {
                field: {
                    "original": entry.get(field),
                    "verified": None,
                    "match": False,
                    "confidence": "low",
                    "needs_review": True
                }
                for field in FIELDS_TO_VALIDATE if entry.get(field)
            }
            
            corrected_entry = entry.copy()
            corrected_entry["_warning"] = "⚠️ Could not verify against external sources - manual review required"
            corrected_entry["_unverified_fields"] = list(discrepancies.keys())
            
            validated_results.append({
                "cite_key": cite_key,
                "original": entry,
                "verified": None,
                "discrepancies": discrepancies,
                "corrected": corrected_entry,
                "status": "unverified",
                "verification_source": None
            })
            
            with progress_details:
                st.warning(f"  ✗ No verification source found - entry flagged for manual review")
        
        # Small delay to respect API rate limits
        time.sleep(0.05)
    
    # Final progress update
    progress_bar.progress(1.0)
    status_text.text("✨ Validation complete!")
    progress_bar.empty()
    
    # Store results in session state for persistence across reruns
    st.session_state.validation_results = validated_results
    st.session_state.processing_complete = True
    
    # ==================== Results Summary ====================
    st.divider()
    st.subheader("📊 Validation Summary")
    
    # Calculate summary statistics
    total_entries = len(validated_results)
    verified_count = sum(1 for r in validated_results if r["status"] == "verified")
    unverified_count = total_entries - verified_count
    flagged_count = sum(
        1 for r in validated_results 
        if any(d.get("needs_review") for d in r.get("discrepancies", {}).values())
    )
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📋 Total Entries", total_entries)
    col2.metric("✅ Verified", verified_count, delta=f"{verified_count/total_entries*100:.1f}%" if total_entries > 0 else None)
    col3.metric("❌ Unverified", unverified_count)
    col4.metric("⚠️ Flagged for Review", flagged_count)
    
    # Progress visualization
    if total_entries > 0:
        st.progress(verified_count / total_entries, text=f"Verification Rate: {verified_count/total_entries*100:.1f}%")
    
    # ==================== Detailed Results Table ====================
    st.subheader("📋 Entry-by-Entry Results")
    
    # Prepare data for table display
    table_data = []
    for result in validated_results:
        row = {
            "Cite Key": f"`{result['cite_key']}`",
            "Title": safe_string_slice(result["original"].get("title"), 70),
            "DOI": clean_doi(result["original"].get("doi")) or "N/A",
            "Status": "✅ Verified" if result["status"] == "verified" else "❌ Unverified",
            "Source": result.get("verification_source", "N/A") or "N/A",
            "Fields Flagged": sum(1 for d in result.get("discrepancies", {}).values() if d.get("needs_review"))
        }
        table_data.append(row)
    
    # Display interactive table
    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Cite Key": st.column_config.TextColumn("Cite Key", help="BibTeX citation key"),
            "Title": st.column_config.TextColumn("Title", width="medium"),
            "DOI": st.column_config.TextColumn("DOI", help="Digital Object Identifier"),
            "Status": st.column_config.TextColumn("Status"),
            "Source": st.column_config.TextColumn("Verification Source"),
            "Fields Flagged": st.column_config.NumberColumn("Fields Flagged", help="Number of fields needing manual review")
        }
    )
    
    # ==================== Expandable Detailed Comparison View ====================
    if show_detailed_diff:
        with st.expander("🔍 Detailed Field Comparison View", expanded=False):
            for result in validated_results:
                with st.container():
                    st.markdown(f"#### `{result['cite_key']}` — {result['original'].get('title', 'Untitled')[:80]}...")
                    
                    # Status indicator
                    if result["status"] == "verified":
                        st.success(f"✅ Verified via {result.get('verification_source', 'external source')}")
                    else:
                        st.warning(result["corrected"].get("_warning", "⚠️ Unverified - manual review required"))
                    
                    # Field comparison table
                    if result["discrepancies"]:
                        comparison_data = []
                        for field, info in result["discrepancies"].items():
                            status_icon = (
                                "✅" if info["match"] else 
                                "⚠️" if info["needs_review"] else 
                                "➖"
                            )
                            comparison_data.append({
                                "Field": field,
                                "Original": safe_string_slice(info["original"], 40),
                                "Verified": safe_string_slice(info["verified"], 40) if info["verified"] else "N/A",
                                "Status": status_icon,
                                "Confidence": info.get("confidence", "N/A")
                            })
                        
                        if comparison_data:
                            st.dataframe(
                                comparison_data,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Status": st.column_config.TextColumn("Status", width="small"),
                                    "Confidence": st.column_config.TextColumn("Confidence", width="small")
                                }
                            )
                    
                    st.divider()
    
    # ==================== Download Corrected BibTeX ====================
    st.divider()
    st.subheader("💾 Download Validated BibTeX File")
    
    # Prepare entries for output
    output_entries = []
    for result in validated_results:
        # Skip unverified entries if user chose to exclude them
        if result["status"] == "unverified" and not include_unverified:
            continue
        
        # Use corrected entry (with internal flags removed for clean output)
        entry_output = {
            k: v for k, v in result["corrected"].items() 
            if not k.startswith("_") or k in ["_warning"]  # Optionally keep warning
        }
        output_entries.append(entry_output)
    
    # Generate BibTeX content
    generation_metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": "Crossref API, OpenAlex API",
        "llm_model": selected_model if use_llm else "Disabled",
        "verified_count": verified_count,
        "flagged_count": flagged_count
    }
    
    bib_content = generate_complete_bibtex_file(output_entries, generation_metadata)
    
    # Create download button
    st.download_button(
        label="📥 Download validated_references.bib",
        data=bib_content,
        file_name="validated_references.bib",
        mime="text/plain",
        type="primary",
        help="Download corrected BibTeX file ready for use in LaTeX documents"
    )
    
    # Preview the output
    with st.expander("👀 Preview First 3 Entries", expanded=False):
        preview_entries = output_entries[:3]
        for entry in preview_entries:
            key = entry.get("cite_key", "unknown")
            preview = generate_bibtex_entry_string(entry, key)
            st.code(preview, language="bibtex")
            st.divider()
    
    # ==================== Help and Documentation ====================
    with st.expander("❓ How This Tool Works", expanded=False):
        st.markdown("""
        ### Validation Pipeline
        
        1️⃣ **Parse Input**: Extract title, DOI, authors, and other fields from your BibTeX entries
        
        2️⃣ **DOI Lookup (Primary)**: Query Crossref API using DOI - most reliable method
           - Falls back to OpenAlex API if Crossref has no record
           - Handles DOI URL formats and normalization automatically
        
        3️⃣ **Title Search (Fallback)**: If DOI lookup fails, search Crossref by title
           - Uses fuzzy matching with 85% similarity threshold
           - Auto-selects best match if confidence score ≥90%
        
        4️⃣ **Field Comparison**: Compare each metadata field between original and verified
           - Authors: Set-based comparison (order-independent, 70% overlap threshold)
           - Titles: Fuzzy token-based similarity matching
           - Year/Volume/Pages: Exact or substring matching
           - Confidence scoring: high/medium/low based on match quality
        
        5️⃣ **Correction Application**: 
           - Auto-correct: Fields with high/medium confidence matches
           - Flag for review: Discrepancies with low confidence or missing verified data
           - LLM refinement (optional): Resolve ambiguous author formatting, journal abbreviations
        
        6️⃣ **Output Generation**: Produce clean BibTeX with corrections applied and flags for manual review
        
        ### Understanding the Output
        
        - ✅ **Verified**: Entry matched external source with high confidence
        - ⚠️ **Flagged**: Specific fields differ between original and verified - review recommended  
        - ❌ **Unverified**: No matching record found in Crossref/OpenAlex - manual verification required
        
        Fields with `_flag_*` suffix in the output indicate discrepancies needing your attention.
        
        ### Best Practices
        
        🔹 Start with "API-only" mode for fastest processing of large files  
        🔹 Use LLM refinement only for entries flagged with author/journal formatting issues  
        🔹 Always manually review entries marked as unverified or with multiple flagged fields  
        🔹 Update the User-Agent email in settings to comply with Crossref API policy  
        🔹 For institutional use, consider adding API keys for higher rate limits
        """)
    
    # ==================== Footer ====================
    st.divider()
    st.caption("""
    **BibTeX Hallucination Validator v1.0** | 
    Validation sources: Crossref API, OpenAlex API | 
    LLM models: GPT-2, DistilGPT-2, Qwen2.5-0.5B (all <1.2B parameters) |
    [View Source Code](https://github.com/your-repo/bibtex-validator) | 
    [Report Issue](https://github.com/your-repo/bibtex-validator/issues)
    
    *This tool assists with reference validation but does not replace expert scholarly review. 
    Always verify critical references against original sources before publication.*
    """)


# ==================== Application Entry Point ====================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Catch-all error handler with detailed logging
        import traceback
        error_details = traceback.format_exc()
        logger.critical(f"Unhandled exception in main(): {error_details}")
        
        st.error("""
        ## ❌ Application Error
        
        An unexpected error occurred. This has been logged for debugging.
        
        **Troubleshooting steps:**
        1. Refresh the page and try uploading your file again
        2. Check that your BibTeX file is properly formatted
        3. If using LLM features, ensure you have sufficient memory (~1-2GB VRAM for Qwen2.5-0.5B)
        4. Try "API-only" mode if LLM loading fails
        
        **Error details for developers:**
        """)
        st.code(error_details, language="text")
        
        st.info("""
        💡 **Quick fix**: If you see "AttributeError" related to entry fields, 
        your BibTeX file may have non-standard field names. Try cleaning the file 
        with a BibTeX validator like [JabRef](https://www.jabref.org/) first.
        """)
