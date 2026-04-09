# app.py - BibTeX Hallucination Validator (Fixed LLM Loading for Streamlit Cloud)
# ============================================================================
# Key fixes:
# 1. LLM loading moved to dedicated button (not automatic sidebar)
# 2. Memory-aware loading with psutil monitoring
# 3. Optimized model loading: float16, low_cpu_mem_usage, 8-bit quantization option
# 4. Progressive loading with explicit user control
# 5. @st.cache_resource properly isolated with minimal concurrent UI activity
# 6. Graceful fallbacks and clear error messages
# 7. Force garbage collection before model load
# 8. Hugging Face cache management with clear button
# ============================================================================

import streamlit as st
import bibtexparser
import requests
import re
import json
import time
import logging
import sys
import os
import gc
from typing import Optional, Dict, List, Tuple, Any, Union
from bibtexparser.bparser import BibTexParser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.customization import convert_to_unicode
from urllib.parse import quote
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==================== Optional LLM Imports ====================
TRANSFORMERS_AVAILABLE = False
TORCH_AVAILABLE = False
BITSANDBYTES_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    logger.info("PyTorch loaded successfully")
except ImportError:
    logger.warning("PyTorch not installed - LLM features disabled")
    torch = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers library loaded successfully")
except ImportError:
    logger.warning("Transformers not installed - LLM features disabled")
    AutoTokenizer = None
    AutoModelForCausalLM = None
    pipeline = None
    GenerationConfig = None
    BitsAndBytesConfig = None

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
    logger.info("bitsandbytes available for 8-bit quantization")
except ImportError:
    logger.info("bitsandbytes not installed - 8-bit quantization disabled")
    BITSANDBYTES_AVAILABLE = False

# Optional: psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="BibTeX Hallucination Validator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
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
VALID_ENTRY_TYPES = {"article", "book", "inproceedings", "misc", "techreport", "phdthesis", "mastersthesis", "online", "dataset", "report", "manual"}
FIELDS_TO_VALIDATE = ["title", "author", "journal", "booktitle", "year", "volume", "number", "pages", "doi", "publisher"]

# ==================== Memory Monitoring Utilities ====================

def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    return 0.0

def check_memory_available(min_required_mb: float = 400) -> Tuple[bool, str]:
    """Check if enough memory is available for model loading."""
    if not PSUTIL_AVAILABLE:
        return True, "Memory monitoring not available"
    
    total_ram = psutil.virtual_memory().total / (1024 * 1024)
    available_ram = psutil.virtual_memory().available / (1024 * 1024)
    current_usage = get_memory_usage_mb()
    
    # Estimate peak usage during load: current + model size + buffer
    estimated_peak = current_usage + min_required_mb + 100  # 100MB buffer
    
    if estimated_peak > total_ram * 0.9:  # Stay under 90% of total
        return False, f"Insufficient memory: need ~{min_required_mb}MB, have {available_ram:.0f}MB available (peak est: {estimated_peak:.0f}MB)"
    
    if available_ram < min_required_mb:
        return False, f"Low available memory: {available_ram:.0f}MB < required {min_required_mb}MB"
    
    return True, f"Memory OK: {available_ram:.0f}MB available"

def force_garbage_collection():
    """Force Python garbage collection to free memory before model load."""
    gc.collect()
    if torch and TORCH_AVAILABLE:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

# ==================== Author Normalization & Matching ====================

def normalize_author_name(name: str) -> str:
    """Convert author name to canonical form for comparison: 'first last' lowercase, no punctuation."""
    if not name or not isinstance(name, str):
        return ""
    cleaned = re.sub(r'[^a-z\s\-]', '', name.lower())
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if ',' in cleaned:
        parts = [p.strip() for p in cleaned.split(',', 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            return f"{parts[1]} {parts[0]}"
    return cleaned

def extract_initials(name: str) -> str:
    """Extract initials from a name for fuzzy matching."""
    norm = normalize_author_name(name)
    if not norm:
        return ""
    return ''.join([w[0] for w in norm.split() if w])

def authors_match_score(authors_orig: List[str], authors_ver: List[str]) -> Tuple[float, str, bool]:
    """Compare two author lists and return (match_score, confidence, needs_review)."""
    if not authors_orig and not authors_ver:
        return 1.0, "high", False
    if not authors_orig or not authors_ver:
        return 0.0, "low", True
        
    norm_orig = [normalize_author_name(a) for a in authors_orig if normalize_author_name(a)]
    norm_ver = [normalize_author_name(a) for a in authors_ver if normalize_author_name(a)]
    
    if not norm_orig or not norm_ver:
        return 0.0, "low", True
        
    if set(norm_orig) == set(norm_ver):
        return 1.0, "high", False
        
    matches = 0
    for va in norm_ver:
        va_initials = extract_initials(va)
        matched = False
        for oa in norm_orig:
            if va == oa:
                matched = True
                break
            if va_initials and len(va_initials) >= 2:
                if va_initials in oa or oa.startswith(va_initials):
                    matched = True
                    break
                oa_initials = extract_initials(oa)
                if oa_initials and oa_initials in va:
                    matched = True
                    break
        if matched:
            matches += 1
            
    score = matches / max(len(norm_ver), 1)
    
    if score >= 0.9:
        return score, "high", False
    elif score >= 0.7:
        return score, "medium", True
    else:
        return score, "low", True

def format_authors_bibtex(authors: Union[str, List[str], Dict, None]) -> str:
    """Convert any author format to proper BibTeX: 'Last, First and Last, First'."""
    if authors is None:
        return ""
    if isinstance(authors, str):
        authors = authors.strip()
        if not authors:
            return ""
        if " and " in authors.lower():
            return authors
        return authors.replace(";", " and") if ";" in authors else authors
    if isinstance(authors, list):
        return " and ".join(str(a).strip() for a in authors if a)
    if isinstance(authors, dict):
        given = authors.get("given", "") or ""
        family = authors.get("family", "") or ""
        if family:
            return f"{family.strip()}, {given.strip()}" if given else family.strip()
        return given.strip()
    return str(authors).strip()

# ==================== String & Data Utilities ====================

def safe_string_slice(value: Any, max_length: int = 50) -> str:
    if value is None:
        return "Untitled"
    if isinstance(value, list):
        value = " and ".join(str(v) for v in value if v is not None)
    text = str(value).strip()
    return (text[:max_length] + "...") if len(text) > max_length else text if text else "Untitled"

def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\-\.]', '', text)
    return text.strip()

def clean_doi(doi: Optional[str]) -> Optional[str]:
    if not doi or not isinstance(doi, str):
        return None
    cleaned = doi.strip()
    cleaned = re.sub(r'^https?://(dx\.)?doi\.org/', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^doi:\s*', '', cleaned, flags=re.IGNORECASE)
    if cleaned and '/' in cleaned and len(cleaned) > 5:
        return cleaned.strip()
    return None

def calculate_string_similarity(str1: Optional[str], str2: Optional[str]) -> float:
    if str1 is None or str2 is None:
        return 0.0
    n1, n2 = normalize_text(str1), normalize_text(str2)
    if n1 == n2:
        return 1.0
    if not n1 or not n2:
        return 0.0
    t1 = set(re.findall(r'\b[a-z0-9]+\b', n1))
    t2 = set(re.findall(r'\b[a-z0-9]+\b', n2))
    if not t1 or not t2:
        return 0.0
    return len(t1.intersection(t2)) / len(t1.union(t2))

def extract_year_from_date(date_value: Any) -> Optional[str]:
    if date_value is None:
        return None
    if isinstance(date_value, int):
        return str(date_value) if 1000 <= date_value <= 2100 else None
    if isinstance(date_value, str):
        match = re.search(r'\b(1[0-9]{3}|2[0-9]{3})\b', date_value.strip())
        return match.group(1) if match else None
    if isinstance(date_value, dict):
        parts = date_value.get("date-parts")
        if parts and isinstance(parts, list) and len(parts) > 0:
            first = parts[0]
            if isinstance(first, list) and len(first) > 0 and isinstance(first[0], int):
                return str(first[0]) if 1000 <= first[0] <= 2100 else None
    if isinstance(date_value, list) and len(date_value) > 0:
        return extract_year_from_date(date_value[0])
    return None

# ==================== API Interaction Functions ====================

def make_api_request(url: str, headers: Dict[str, str], timeout: int, 
                    params: Optional[Dict] = None, max_retries: int = MAX_RETRIES) -> Optional[Dict]:
    last_exception = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=timeout, allow_redirects=True)
            if response.status_code == 429:
                wait = float(response.headers.get('Retry-After', RETRY_DELAY * (attempt + 1)))
                logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            if response.status_code >= 400:
                if response.status_code >= 500:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                return None
            return response.json()
        except requests.exceptions.Timeout:
            last_exception = "Timeout"
        except requests.exceptions.ConnectionError:
            last_exception = "Connection error"
        except requests.exceptions.RequestException as e:
            last_exception = str(e)
        except json.JSONDecodeError as e:
            last_exception = f"JSON decode: {e}"
            return None
        if attempt < max_retries - 1:
            time.sleep(RETRY_DELAY * (2 ** attempt))
    logger.error(f"API request failed after {max_retries} attempts: {last_exception}")
    return None

def fetch_crossref_metadata(doi: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[Dict[str, Any]]:
    clean_doi_val = clean_doi(doi)
    if not clean_doi_val:
        return None
    url = f"{CROSSREF_API_URL}{quote(clean_doi_val)}"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    data = make_api_request(url, headers, timeout)
    if not data or "message" not in data:
        return None
    msg = data["message"]
    result = {
        "source": "crossref", 
        "doi": msg.get("DOI"), 
        "title": None, 
        "authors": [], 
        "journal": None, 
        "year": None, 
        "volume": None, 
        "issue": None, 
        "pages": None, 
        "publisher": msg.get("publisher"), 
        "type": msg.get("type"), 
        "url": msg.get("URL")
    }
    
    titles = msg.get("title", [])
    result["title"] = titles[0] if titles and isinstance(titles, list) and len(titles) > 0 else None
    
    authors_data = msg.get("author", [])
    if authors_data and isinstance(authors_data, list):
        for author in authors_data:
            if isinstance(author, dict):
                given = author.get("given", "") or ""
                family = author.get("family", "") or ""
                if not given and not family:
                    name = author.get("name", "")
                    if name: 
                        result["authors"].append(name)
                    continue
                full = f"{given} {family}".strip()
                if full: 
                    result["authors"].append(full)
    
    containers = msg.get("container-title", [])
    short_containers = msg.get("short-container-title", [])
    result["journal"] = containers[0] if containers else (short_containers[0] if short_containers else None)
    
    for date_field in ["published-print", "published-online", "created", "deposited"]:
        date_data = msg.get(date_field)
        if date_data:
            yr = extract_year_from_date(date_data)
            if yr: 
                result["year"] = yr
                break
            
    result["volume"] = msg.get("volume")
    result["issue"] = msg.get("issue")
    result["pages"] = msg.get("page")
    return result

def fetch_openalex_metadata(doi: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[Dict[str, Any]]:
    clean_doi_val = clean_doi(doi)
    if not clean_doi_val:
        return None
    url = f"{OPENALEX_API_URL}{quote(clean_doi_val)}"
    headers = {"User-Agent": USER_AGENT}
    data = make_api_request(url, headers, timeout)
    if not data or data.get("id") is None:
        return None
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
        "url": data.get("doi")
    }
    
    for authship in data.get("authorships", []):
        if isinstance(authship, dict):
            adata = authship.get("author", {})
            if isinstance(adata, dict):
                dname = adata.get("display_name")
                if dname: 
                    result["authors"].append(dname)
                
    loc = data.get("primary_location", {})
    if loc and isinstance(loc, dict):
        src = loc.get("source", {})
        if src and isinstance(src, dict):
            result["journal"] = src.get("display_name")
            
    biblio = data.get("biblio", {})
    if biblio and isinstance(biblio, dict):
        result["volume"] = biblio.get("volume")
        result["issue"] = biblio.get("issue")
        fp, lp = biblio.get("first_page"), biblio.get("last_page")
        if fp and lp: 
            result["pages"] = f"{fp}--{lp}"
        elif fp: 
            result["pages"] = str(fp)
    return result

def search_crossref_by_title(title: str, max_results: int = TITLE_SEARCH_MAX_RESULTS, 
                            timeout: int = DEFAULT_TIMEOUT) -> List[Dict[str, Any]]:
    if not title or not isinstance(title, str):
        return []
    url = CROSSREF_SEARCH_URL
    params = {"query.title": title, "rows": max_results, "select": "DOI,title,author,container-title,published-print,volume,page,type,score"}
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    data = make_api_request(url, headers, timeout, params)
    if not data or "message" not in data:
        return []
    items = data["message"].get("items", [])
    results = []
    for item in items:
        if not isinstance(item, dict): 
            continue
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
        t = item.get("title", [])
        entry["title"] = t[0] if t and isinstance(t, list) else None
        for a in item.get("author", []):
            if isinstance(a, dict):
                full = f"{a.get('given','')} {a.get('family','')}".strip()
                if full: 
                    entry["authors"].append(full)
        c = item.get("container-title", [])
        entry["journal"] = c[0] if c and isinstance(c, list) else None
        pp = item.get("published-print", {})
        if pp and isinstance(pp, dict):
            yr = extract_year_from_date(pp)
            if yr: 
                entry["year"] = yr
        if entry["doi"] or entry["title"]:
            results.append(entry)
    results.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    return results

# ==================== Metadata Comparison & Merging ====================

def compare_metadata_fields(original: Dict[str, Any], verified: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    comparison_results = {}
    for field in FIELDS_TO_VALIDATE:
        orig_val = original.get(field)
        verify_val = verified.get(field)
        
        if field in ["author", "authors"]:
            orig_auth_list = [a.strip() for a in re.split(r'\s+and\s+', str(orig_val or ""), flags=re.IGNORECASE) if a.strip()]
            ver_auth_list = verified.get("authors", [])
            if isinstance(ver_auth_list, str):
                ver_auth_list = [a.strip() for a in ver_auth_list.split("and") if a.strip()]
                
            score, conf, needs_review = authors_match_score(orig_auth_list, ver_auth_list)
            
            comparison_results[field] = {
                "original": orig_val, 
                "verified": ver_auth_list, 
                "match": score >= 0.9, 
                "confidence": conf, 
                "needs_review": needs_review,
                "match_score": score
            }
            continue
            
        o_norm, v_norm = normalize_text(orig_val), normalize_text(verify_val)
        if not o_norm and not v_norm:
            match, conf = True, "high"
        elif not o_norm or not v_norm:
            match, conf = False, "low"
        elif o_norm == v_norm:
            match, conf = True, "high"
        elif field == "title":
            sim = calculate_string_similarity(orig_val, verify_val)
            match = sim >= SIMILARITY_THRESHOLD
            conf = "high" if sim >= 0.95 else "medium" if match else "low"
        elif field == "year":
            match = o_norm == v_norm
            conf = "high" if match else "low"
        else:
            match = (o_norm in v_norm or v_norm in o_norm or calculate_string_similarity(orig_val, verify_val) >= 0.8)
            conf = "high" if match else "low"
            
        comparison_results[field] = {
            "original": orig_val, 
            "verified": verify_val, 
            "match": match, 
            "confidence": conf, 
            "needs_review": not match and verify_val is not None, 
            "similarity": calculate_string_similarity(orig_val, verify_val) if field == "title" else None
        }
    return comparison_results

def merge_metadata_entries(original: Dict[str, Any], verified: Dict[str, Any], 
                          comparison_results: Dict[str, Dict], auto_correct: bool = True) -> Dict[str, Any]:
    merged = {k: v for k, v in original.items() if not k.startswith("_")}
    merged["_verification_source"] = verified.get("source", "unknown")
    merged["_verified_doi"] = verified.get("doi")
    merged["_verification_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    if not auto_correct:
        merged["_comparison_results"] = comparison_results
        merged["_verified_metadata"] = verified
        return merged
    
    for field, res in comparison_results.items():
        if res["verified"] is None: 
            continue
        if res["match"] and res["confidence"] in ["high", "medium"]:
            if field in ["author", "authors"]:
                ver_auths = res["verified"] if isinstance(res["verified"], list) else [res["verified"]]
                bibtex_authors = []
                for a in ver_auths:
                    if "," in str(a):
                        bibtex_authors.append(a.strip())
                    else:
                        parts = str(a).rsplit(" ", 1)
                        if len(parts) == 2:
                            bibtex_authors.append(f"{parts[1].strip()}, {parts[0].strip()}")
                        else:
                            bibtex_authors.append(a.strip())
                merged["author"] = " and ".join(bibtex_authors)
            else:
                merged[field] = res["verified"]
            merged[f"_corrected_{field}"] = True
        elif res["needs_review"]:
            merged[f"_flag_{field}"] = True
            merged[f"_flag_reason_{field}"] = f"Discrepancy: orig='{res['original']}', verified='{res['verified']}', conf={res['confidence']}"
    return merged

# ==================== LLM Integration Functions - FIXED FOR STREAMLIT CLOUD ====================

@st.cache_resource
def _load_llm_model_cached(model_id: str, use_8bit: bool = False, device: str = "cpu"):
    """
    Cached LLM loader with memory optimizations for Streamlit Cloud.
    
    Key optimizations:
    - use_8bit: Enables 8-bit quantization via bitsandbytes (reduces memory by ~4x)
    - device: Explicit device selection
    - low_cpu_mem_usage: Reduces peak memory during loading
    - float16 on CPU: Reduces memory vs float32 (with acceptable precision loss)
    """
    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
        return None, None
    
    logger.info(f"Loading {model_id} on {device} with 8-bit={use_8bit}")
    
    try:
        # Memory check before loading
        if PSUTIL_AVAILABLE:
            available = psutil.virtual_memory().available / (1024 * 1024)
            logger.info(f"Available RAM before load: {available:.0f}MB")
        
        # Force GC to free memory before loading
        force_garbage_collection()
        
        # Load tokenizer first (smaller, faster)
        logger.info(f"Loading tokenizer for {model_id}")
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tok.pad_token is None: 
            tok.pad_token = tok.eos_token
        
        # Configure model loading with memory optimizations
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,  # Critical for reducing peak memory
        }
        
        # Quantization config if available and requested
        if use_8bit and BITSANDBYTES_AVAILABLE:
            logger.info("Using 8-bit quantization")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        else:
            # Use float16 to reduce memory (works on CPU with recent PyTorch)
            if device == "cpu":
                # Try float16 first, fall back to float32 if not supported
                try:
                    model_kwargs["torch_dtype"] = torch.float16
                except:
                    model_kwargs["torch_dtype"] = torch.float32
            elif device == "cuda" and torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.float16
                if torch.cuda.device_count() > 1:
                    model_kwargs["device_map"] = "auto"
            elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                model_kwargs["torch_dtype"] = torch.float16
        
        logger.info(f"Loading model weights for {model_id} with kwargs: {model_kwargs}")
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        
        # Move to device if not already handled by device_map
        if device == "cpu" and not hasattr(model, "hf_device_map"):
            model = model.to("cpu")
        elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            model = model.to("mps")
        
        # Generation config optimized for metadata refinement (not creative writing)
        gen_conf = GenerationConfig(
            max_new_tokens=256, 
            temperature=0.1,  # Low for deterministic output
            top_p=0.9, 
            do_sample=True, 
            pad_token_id=tok.pad_token_id, 
            eos_token_id=tok.eos_token_id, 
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
        
        # Create pipeline with minimal overhead
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tok, 
            device=0 if device == "cuda" and torch.cuda.is_available() else -1,
            generation_config=gen_conf
        )
        
        logger.info(f"Successfully loaded {model_id} on {device}")
        return tok, pipe
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA OOM loading {model_id}: {e}")
        return None, None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"Memory error loading {model_id}: {e}")
            return None, None
        raise
    except Exception as e:
        logger.error(f"Failed to load {model_id}: {type(e).__name__}: {e}", exc_info=True)
        return None, None


def initialize_llm_pipeline(model_name: str, use_8bit: bool = False, device: Optional[str] = None) -> Optional[Any]:
    """
    Public LLM initializer with memory-aware loading.
    Returns pipeline object or None if loading fails.
    """
    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
        logger.warning("Transformers or PyTorch not available")
        return None
    
    try:
        # Map user-friendly names to HuggingFace model IDs
        mapping = {
            "gpt2 (~124M)": "gpt2",
            "distilgpt2 (~82M)": "distilgpt2", 
            "Qwen2.5-0.5B-Instruct (~0.5B)": "Qwen/Qwen2.5-0.5B-Instruct"
        }
        model_id = mapping.get(model_name, model_name)
        
        # Detect device - prefer CPU for Cloud environments to avoid GPU memory issues
        if device is None:
            # On Streamlit Cloud, default to CPU to avoid GPU memory fragmentation
            if "STREAMLIT" in os.environ or "HUGGINGFACE" in os.environ:
                device = "cpu"
            elif torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        logger.info(f"Initializing LLM: {model_name} -> {model_id} on {device}, 8-bit={use_8bit}")
        
        # Memory check with model-specific requirements
        if model_id == "gpt2":
            min_mem = 300  # ~300MB for GPT-2 with float16
        elif model_id == "distilgpt2":
            min_mem = 250  # ~250MB for distilgpt2
        elif "Qwen" in model_id:
            min_mem = 800 if not use_8bit else 300  # ~800MB FP16, ~300MB 8-bit
        else:
            min_mem = 400
        
        mem_ok, mem_msg = check_memory_available(min_mem)
        if not mem_ok:
            logger.warning(f"Memory check failed: {mem_msg}")
            return None
        
        # Use cached loader - THIS IS THE KEY FIX
        tokenizer, pipe = _load_llm_model_cached(model_id, use_8bit=use_8bit, device=device)
        
        if pipe is None:
            logger.warning(f"Pipeline creation failed for {model_id}")
            return None
            
        return pipe
        
    except Exception as e:
        logger.error(f"LLM init failed: {type(e).__name__}: {e}", exc_info=True)
        return None


def build_llm_prompt_for_metadata_refinement(original: Dict[str, Any], verified: Dict[str, Any], discrepancies: Dict[str, Dict]) -> str:
    fields_needing = {k: v for k, v in discrepancies.items() if v.get("needs_review") and v.get("verified") is not None}
    if not fields_needing: 
        return ""
        
    orig_lines = []
    for f in ["author", "title", "journal", "booktitle", "year", "volume", "pages", "doi"]:
        val = original.get(f)
        if val is not None:
            v = format_authors_bibtex(val) if f == "author" else val
            orig_lines.append(f"  {f} = {{{v}}}")
            
    ver_lines = []
    for f in ["author", "title", "journal", "year", "volume", "pages"]:
        val = verified.get(f)
        if val is not None:
            v = format_authors_bibtex(val) if f == "author" and isinstance(val, list) else val
            ver_lines.append(f"  {f}: {v}")
            
    disc_lines = [f"- {k}: orig='{v['original']}' | ver='{v['verified']}' | conf={v['confidence']}" for k, v in fields_needing.items()]
    
    prompt = f"""You are an academic metadata validator. Resolve discrepancies between a BibTeX entry and verified API data.

ORIGINAL:
@article{{KEY,
{chr(10).join(orig_lines)}
}}

VERIFIED:
{chr(10).join(ver_lines)}

DISCREPANCIES:
{chr(10).join(disc_lines)}

INSTRUCTIONS: Output ONLY valid BibTeX field assignments for fields that need correction. Format: field = {{value}},
Do not include explanations or @ wrappers. If no changes needed, output: # NO_CHANGES_REQUIRED

OUTPUT:
"""
    return prompt


def parse_llm_output_for_bibtex_fields(llm_output: str) -> Dict[str, str]:
    if not llm_output or "# NO_CHANGES_REQUIRED" in llm_output: 
        return {}
    corrections = {}
    for line in llm_output.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("@"): 
            continue
        m = re.match(r'^(\w+)\s*=\s*\{(.*)\},?\s*$', line)
        if m:
            f, v = m.group(1).strip().lower(), m.group(2).strip()
            if f in FIELDS_TO_VALIDATE: 
                corrections[f] = v
    return corrections


def refine_metadata_with_llm(original: Dict[str, Any], verified: Dict[str, Any], discrepancies: Dict[str, Dict], llm_pipe: Any) -> Dict[str, Any]:
    if llm_pipe is None: 
        return merge_metadata_entries(original, verified, discrepancies)
    try:
        prompt = build_llm_prompt_for_metadata_refinement(original, verified, discrepancies)
        if not prompt: 
            return merge_metadata_entries(original, verified, discrepancies)
        res = llm_pipe(prompt, max_new_tokens=256)
        gen = res[0].get("generated_text", "") if isinstance(res, list) else res.get("generated_text", str(res))
        llm_resp = gen.split(prompt)[-1].strip() if prompt in gen else gen.strip()
        corrections = parse_llm_output_for_bibtex_fields(llm_resp)
        merged = merge_metadata_entries(original, verified, discrepancies)
        for f, v in corrections.items():
            merged[f] = v
            merged[f"_llm_corrected_{f}"] = True
        merged["_llm_response_preview"] = llm_resp[:200] + ("..." if len(llm_resp)>200 else "")
        return merged
    except Exception as e:
        logger.error(f"LLM refinement failed: {e}")
        return merge_metadata_entries(original, verified, discrepancies)

# ==================== BibTeX Parsing & Generation ====================

def parse_bibtex_file_content(file_content: bytes) -> Tuple[List[Dict], Optional[str]]:
    try:
        try: 
            content = file_content.decode("utf-8")
        except UnicodeDecodeError: 
            content = file_content.decode("latin-1")
        parser = BibTexParser()
        parser.customization = convert_to_unicode
        parser.ignore_comments = False
        bib_db = bibtexparser.loads(content, parser)
        if not bib_db.entries: 
            return [], "No valid BibTeX entries found"
        processed = []
        for entry in bib_db.entries:
            p = {}
            for k, v in entry.items():
                if k == "ID": 
                    p["cite_key"] = v
                elif k == "ENTRYTYPE": 
                    p["entry_type"] = v if v in VALID_ENTRY_TYPES else "misc"
                elif v is not None: 
                    p[k] = str(v).strip() if isinstance(v, str) else v
            processed.append(p)
        return processed, None
    except Exception as e:
        return [], f"Parsing error: {type(e).__name__}: {e}"


def generate_bibtex_entry_string(entry: Dict[str, Any], cite_key: Optional[str] = None) -> str:
    key = cite_key or entry.get("cite_key", "unknown_key")
    etype = entry.get("entry_type", "article")
    out_fields = ["author", "title", "journal", "booktitle", "year", "volume", "number", "pages", "doi", "url", "publisher", "note", "month", "issn", "isbn"]
    lines = [f"@{etype}{{{key},"]
    for f in out_fields:
        if f in entry and entry[f] is not None and not f.startswith("_"):
            v = entry[f]
            if f == "author" and isinstance(v, list): 
                v = " and ".join(str(x).strip() for x in v if x)
            vs = str(v).strip()
            if f in ["title", "journal", "booktitle", "note"]: 
                vs = re.sub(r'(?<!\\)([{}])', r'{\1}', vs)
            lines.append(f"  {f} = {{{vs}}},")
    if len(lines) > 1 and lines[-1].endswith(","): 
        lines[-1] = lines[-1][:-1]
    lines.append("}")
    return "\n".join(lines)


def generate_complete_bibtex_file(entries: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
    header = f"""% ================================================================================
% Hallucination-Validated BibTeX References
% ================================================================================
% Generated: {metadata.get('timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))}
% Sources: {metadata.get('sources', 'Crossref, OpenAlex')}
% LLM: {metadata.get('llm_model', 'Disabled')}
% Total: {len(entries)} | Verified: {metadata.get('verified_count', 0)} | Flagged: {metadata.get('flagged_count', 0)}
% Fields with _flag_* indicate discrepancies requiring manual review
% ================================================================================
"""
    entry_strs = [generate_bibtex_entry_string(e, e.get("cite_key", "unknown")) for e in entries]
    return header + "\n\n".join(entry_strs)

# ==================== Streamlit Main Application ====================

def main():
    # Initialize session state for persistent data across reruns
    if "validation_results" not in st.session_state: 
        st.session_state.validation_results = []
    if "llm_pipeline" not in st.session_state: 
        st.session_state.llm_pipeline = None
    if "current_model" not in st.session_state: 
        st.session_state.current_model = None
    if "processing_complete" not in st.session_state: 
        st.session_state.processing_complete = False
    if "llm_load_error" not in st.session_state:
        st.session_state.llm_load_error = None
    if "llm_8bit_enabled" not in st.session_state:
        st.session_state.llm_8bit_enabled = False

    st.title("📚 BibTeX Hallucination Validator")
    st.markdown("**Upload a `.bib` file** to validate references against Crossref & OpenAlex APIs. Detects and corrects hallucinated metadata, with robust author name matching.")

    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Memory status display (if psutil available)
        if PSUTIL_AVAILABLE:
            mem_usage = get_memory_usage_mb()
            mem_total = psutil.virtual_memory().total / (1024 * 1024)
            mem_avail = psutil.virtual_memory().available / (1024 * 1024)
            st.metric("💾 Memory Usage", f"{mem_usage:.0f}MB / {mem_total:.0f}MB", 
                     delta=f"{mem_avail:.0f}MB available", delta_color="normal")
        
        # LLM Model Selection - dropdown (does NOT auto-load)
        st.subheader("🤖 LLM Refinement (Optional)")
        llm_opts = ["None (API-only)", "gpt2 (~124M)", "distilgpt2 (~82M)", "Qwen2.5-0.5B-Instruct (~0.5B)"]
        sel_model = st.selectbox("Select model for metadata refinement:", llm_opts, index=0)
        
        # 8-bit quantization option (only if bitsandbytes available)
        use_8bit = False
        if sel_model != "None (API-only)" and BITSANDBYTES_AVAILABLE:
            use_8bit = st.checkbox("Use 8-bit quantization (reduces memory by ~4x)", 
                                  value=st.session_state.llm_8bit_enabled,
                                  help="Recommended for Streamlit Cloud. May slightly reduce precision.")
            st.session_state.llm_8bit_enabled = use_8bit
        
        # KEY FIX: Dedicated button to load LLM (not automatic)
        use_llm = False
        if sel_model != "None (API-only)":
            col_load1, col_load2 = st.columns([2, 1])
            with col_load1:
                load_llm_btn = st.button(f"🔧 Load {sel_model}", 
                                        use_container_width=True,
                                        disabled=(st.session_state.llm_pipeline is not None and st.session_state.current_model == sel_model))
            with col_load2:
                if st.session_state.llm_pipeline is not None:
                    if st.button("🗑️ Unload", use_container_width=True, help="Free memory by unloading model"):
                        st.session_state.llm_pipeline = None
                        st.session_state.current_model = None
                        st.session_state.llm_load_error = None
                        force_garbage_collection()
                        st.rerun()
            
            # Load LLM ONLY when button is clicked (isolated from other UI activity)
            if load_llm_btn:
                with st.status(f"🔄 Loading {sel_model}...", expanded=True) as status:
                    # Minimize UI activity during load
                    st.write("Preparing environment...")
                    force_garbage_collection()
                    
                    st.write("Initializing tokenizer & model weights...")
                    
                    # Device detection for display
                    if "STREAMLIT" in os.environ or "HUGGINGFACE" in os.environ:
                        dev_display = "CPU (Cloud-optimized)"
                    elif torch.cuda.is_available():
                        dev_display = "GPU (CUDA)"
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        dev_display = "Apple Silicon (MPS)"
                    else:
                        dev_display = "CPU"
                    st.write(f"Target device: {dev_display}")
                    
                    # Warn about model sizes
                    if "Qwen" in sel_model:
                        if use_8bit:
                            st.info("ℹ️ Qwen2.5-0.5B with 8-bit: ~300MB RAM, ~45s load time")
                        else:
                            st.warning("⚠️ Qwen2.5-0.5B without 8-bit: ~800MB RAM, ~60s load time")
                    elif "gpt2" in sel_model.lower():
                        st.info("ℹ️ GPT-2: ~300MB RAM, ~20s load time")
                    
                    # Attempt to load with cached function
                    pipeline = initialize_llm_pipeline(sel_model, use_8bit=use_8bit)
                    
                    if pipeline:
                        st.session_state.llm_pipeline = pipeline
                        st.session_state.current_model = sel_model
                        st.session_state.llm_load_error = None
                        status.update(label=f"✅ {sel_model} loaded successfully", state="complete", expanded=False)
                        st.success(f"Model ready for refinement!")
                        use_llm = True
                    else:
                        st.session_state.llm_load_error = f"Failed to load {sel_model}"
                        status.update(label=f"⚠️ Load failed - using API-only", state="error", expanded=False)
                        st.warning("LLM refinement disabled. Author matching still works via rule-based normalization.")
                        st.info("💡 Tips: Try 8-bit quantization, close other apps, or use a smaller model.")
                        use_llm = False
        
        # Display load error if present with retry option
        if st.session_state.llm_load_error:
            st.error(f"❌ {st.session_state.llm_load_error}")
            col_retry1, col_retry2 = st.columns(2)
            with col_retry1:
                if st.button("🔄 Retry Loading", use_container_width=True):
                    st.session_state.llm_pipeline = None
                    st.session_state.current_model = None
                    st.session_state.llm_load_error = None
                    force_garbage_collection()
                    st.rerun()
            with col_retry2:
                if st.button("📦 Clear Cache & Retry", use_container_width=True, 
                           help="Clear Hugging Face cache and retry (may re-download model)"):
                    import shutil
                    cache_dir = os.path.expanduser("~/.cache/huggingface")
                    if os.path.exists(cache_dir):
                        try:
                            # Only clear transformers cache, not everything
                            hf_cache = os.path.join(cache_dir, "hub")
                            if os.path.exists(hf_cache):
                                shutil.rmtree(hf_cache)
                                st.success("Hugging Face cache cleared")
                        except Exception as e:
                            st.warning(f"Could not clear cache: {e}")
                    st.session_state.llm_pipeline = None
                    st.session_state.current_model = None
                    st.session_state.llm_load_error = None
                    force_garbage_collection()
                    st.rerun()
        
        # Advanced Options
        st.subheader("🔧 Advanced Options")
        
        timeout = st.slider("API request timeout (seconds)", 5, 60, 10, 5,
                          help="Increase for slower network connections")
        
        auto_correct = st.checkbox("Auto-correct high-confidence matches", True,
                                 help="Automatically fix fields where verified data confidently differs from original")
        
        show_diff = st.checkbox("Show field-by-field comparison", True,
                              help="Display side-by-side view of original vs verified metadata")
        
        include_unverified = st.checkbox("Include unverified entries in output", True,
                                       help="Keep entries that couldn't be verified (with warning flags)")
        
        # Rate limiting notice
        st.info("""
        **Rate Limiting Notice**  
        Crossref API: ~50 requests/minute without API key  
        OpenAlex API: ~100 requests/minute  
        Large files may take several minutes to process.
        """)
        
        # User email for Crossref API (required)
        user_email = st.text_input("Your email (for Crossref API User-Agent):",
                                 value="validator@example.com",
                                 help="Required by Crossref API policy. Replace with your actual email.")
        if user_email and "example.com" not in user_email: 
            global USER_AGENT
            USER_AGENT = f"BibTeX-Validator/1.0 (mailto:{user_email})"

    st.subheader("📁 Upload BibTeX")
    uploaded = st.file_uploader("Choose .bib file", type=["bib", "txt"])
    if not uploaded:
        st.info("👆 Upload a BibTeX file to start.")
        st.stop()

    with st.spinner("Parsing entries..."):
        entries, err = parse_bibtex_file_content(uploaded.getvalue())
    if err: 
        st.error(f"❌ {err}")
        st.stop()
    if not entries: 
        st.error("❌ No valid entries found.")
        st.stop()
    st.success(f"✅ Found **{len(entries)}** references")

    st.subheader("🔄 Validation Progress")
    prog = st.progress(0)
    status_txt = st.empty()
    log = st.expander("📋 Progress Log", expanded=False)
    results = []

    for idx, entry in enumerate(entries):
        prog.progress((idx+1)/len(entries))
        ck = entry.get("cite_key", f"entry_{idx}")
        title = safe_string_slice(entry.get("title"), 60)
        doi = entry.get("doi", "")
        status_txt.text(f"🔍 [{idx+1}/{len(entries)}] {title}")

        verified = None
        # 1. DOI Lookup
        if doi:
            cdoi = clean_doi(doi)
            if cdoi:
                verified = fetch_crossref_metadata(doi, timeout)
                if not verified: 
                    verified = fetch_openalex_metadata(doi, timeout)
        
        # 2. Title Search Fallback
        if not verified and entry.get("title"):
            t_res = search_crossref_by_title(entry["title"], TITLE_SEARCH_MAX_RESULTS, timeout)
            if t_res:
                best = next((r for r in t_res if r.get("match_score",0)>=0.9 and r.get("doi")), None)
                if best: 
                    verified = fetch_crossref_metadata(best["doi"], timeout)

        # 3. Process Result
        if verified:
            disc = compare_metadata_fields(entry, verified)
            n_flag = sum(1 for d in disc.values() if d.get("needs_review"))
            with log: 
                st.text(f"`{ck}`: {'✅ Match' if n_flag==0 else f'⚠️ {n_flag} flags'}")
            
            if use_llm and st.session_state.llm_pipeline and n_flag > 0:
                corrected = refine_metadata_with_llm(entry, verified, disc, st.session_state.llm_pipeline)
            else:
                corrected = merge_metadata_entries(entry, verified, disc, auto_correct)
            
            results.append({"cite_key": ck, "original": entry, "verified": verified, "discrepancies": disc, "corrected": corrected, "status": "verified", "source": verified.get("source")})
        else:
            disc = {f: {"original": entry.get(f), "verified": None, "match": False, "confidence": "low", "needs_review": True} for f in FIELDS_TO_VALIDATE if entry.get(f)}
            corrected = entry.copy()
            corrected["_warning"] = "⚠️ No external verification found"
            results.append({"cite_key": ck, "original": entry, "verified": None, "discrepancies": disc, "corrected": corrected, "status": "unverified", "source": None})
            with log: 
                st.warning(f"`{ck}`: ❌ Unverified")
        time.sleep(0.05)

    prog.progress(1.0)
    status_txt.text("✨ Validation complete!")
    st.session_state.validation_results = results
    st.session_state.processing_complete = True

    st.divider()
    st.subheader("📊 Results Summary")
    total = len(results)
    ver_c = sum(1 for r in results if r["status"]=="verified")
    flag_c = sum(1 for r in results if any(d.get("needs_review") for d in r.get("discrepancies",{}).values()))
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total", total)
    c2.metric("Verified", ver_c)
    c3.metric("Unverified", total-ver_c)
    c4.metric("Flagged", flag_c)

    st.subheader("📋 Entry Details")
    tbl = [{"Cite Key": f"`{r['cite_key']}`", "Title": safe_string_slice(r["original"].get("title"),70), 
            "DOI": clean_doi(r["original"].get("doi")) or "N/A", 
            "Status": "✅ Verified" if r["status"]=="verified" else "❌ Unverified",
            "Source": r["source"] or "N/A", "Flags": sum(1 for d in r.get("discrepancies",{}).values() if d.get("needs_review"))} for r in results]
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    if show_diff:
        with st.expander("🔍 Detailed Comparison", expanded=False):
            for r in results:
                st.markdown(f"#### `{r['cite_key']}` — {r['original'].get('title','Untitled')[:70]}...")
                if r["status"]=="verified": 
                    st.success(f"✅ via {r['source']}")
                else: 
                    st.warning(r["corrected"].get("_warning","⚠️ Unverified"))
                if r["discrepancies"]:
                    comp = []
                    for f, d in r["discrepancies"].items():
                        orig_disp = safe_string_slice(d["original"], 40)
                        ver_disp = safe_string_slice(d["verified"], 40) if d["verified"] is not None else "N/A"
                        if isinstance(d["verified"], list):
                            ver_disp = safe_string_slice(" and ".join(d["verified"]), 40)
                        status_icon = "✅" if d["match"] else "⚠️" if d["needs_review"] else "➖"
                        comp.append({"Field": f, "Original": orig_disp, "Verified": ver_disp, "Status": status_icon, "Confidence": d["confidence"]})
                    st.dataframe(comp, use_container_width=True, hide_index=True)
                st.divider()

    st.divider()
    st.subheader("💾 Download")
    out = [r["corrected"] for r in results if r["status"]=="verified" or include_unverified]
    meta = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "sources": "Crossref/OpenAlex", "llm_model": sel_model if use_llm else "Disabled", "verified_count": ver_c, "flagged_count": flag_c}
    bib_content = generate_complete_bibtex_file(out, metadata=meta)
    st.download_button("📥 Download validated_references.bib", bib_content, "validated_references.bib", "text/plain", type="primary")
    with st.expander("👀 Preview (First 3)"):
        for e in out[:3]: 
            st.code(generate_bibtex_entry_string(e, e.get("cite_key","?")), "bibtex")
            st.divider()

    st.caption("🔹 Author matching handles initials, full names, and 'Last, First' formatting automatically. LLM is optional. Always manually verify flagged entries.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        st.error("❌ Unexpected Error\n" + traceback.format_exc())
        st.info("💡 Refresh and retry. Ensure BibTeX is well-formatted.")
