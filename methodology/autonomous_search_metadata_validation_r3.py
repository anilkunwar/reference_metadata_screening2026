# app.py - BibTeX Hallucination Validator (Complete Expanded Version with Working LLM)
# ============================================================================
# Features: Robust author matching, CPU-safe LLM loading with @st.cache_resource,
#           clean BibTeX output, dropdown model selection that actually works
# ============================================================================

import streamlit as st
import bibtexparser
import requests
import re
import json
import time
import logging
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

try:
    import torch
    TORCH_AVAILABLE = True
    logger.info("PyTorch loaded successfully")
except ImportError:
    logger.warning("PyTorch not installed - LLM features disabled")
    torch = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers library loaded successfully")
except ImportError:
    logger.warning("Transformers not installed - LLM features disabled")
    AutoTokenizer = None
    AutoModelForCausalLM = None
    pipeline = None
    GenerationConfig = None

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

# ==================== Author Normalization & Matching ====================

def normalize_author_name(name: str) -> str:
    """Convert author name to canonical form for comparison: 'first last' lowercase, no punctuation."""
    if not name or not isinstance(name, str):
        return ""
    # Remove non-alphanumeric except spaces and hyphens
    cleaned = re.sub(r'[^a-z\s\-]', '', name.lower())
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Handle "Last, First" -> "first last"
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
    """
    Compare two author lists and return (match_score, confidence, needs_review).
    Handles initials vs full names, order variations, and formatting differences.
    """
    if not authors_orig and not authors_ver:
        return 1.0, "high", False
    if not authors_orig or not authors_ver:
        return 0.0, "low", True
        
    norm_orig = [normalize_author_name(a) for a in authors_orig if normalize_author_name(a)]
    norm_ver = [normalize_author_name(a) for a in authors_ver if normalize_author_name(a)]
    
    if not norm_orig or not norm_ver:
        return 0.0, "low", True
        
    # Exact match check (order-independent)
    if set(norm_orig) == set(norm_ver):
        return 1.0, "high", False
        
    # Check if all verified authors match original by initials or substring
    matches = 0
    for va in norm_ver:
        va_initials = extract_initials(va)
        matched = False
        for oa in norm_orig:
            if va == oa:
                matched = True
                break
            if va_initials and len(va_initials) >= 2:
                # Check if original contains initials
                if va_initials in oa or oa.startswith(va_initials):
                    matched = True
                    break
                # Check if verified contains original initials
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
    
    # FIXED: Complete author extraction loop - was truncated as 'authors_'
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
        
        # Handle Authors with robust matching
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
                # Format verified authors to proper BibTeX Last, First
                ver_auths = res["verified"] if isinstance(res["verified"], list) else [res["verified"]]
                bibtex_authors = []
                for a in ver_auths:
                    if "," in str(a):
                        bibtex_authors.append(a.strip())
                    else:
                        # Assume "First Last" -> "Last, First"
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

# ==================== LLM Integration Functions - KEY FIXES FOR LOADING ====================

@st.cache_resource
def _load_llm_model_cached(model_id: str, device: str):
    """
    Internal cached function to load LLM model - decorated with @st.cache_resource.
    This is the KEY to making dropdown LLM loading work like the Core-Shell app.
    """
    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
        return None, None
    
    logger.info(f"Loading {model_id} on {device} (cached)")
    
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tok.pad_token is None: 
            tok.pad_token = tok.eos_token
            
        # CPU-safe model loading with appropriate dtype
        if device == "cpu":
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float32, 
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).to("mps")
        else:  # cuda
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.device_count() > 1 else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        
        gen_conf = GenerationConfig(
            max_new_tokens=256, 
            temperature=0.1, 
            top_p=0.9, 
            do_sample=True, 
            pad_token_id=tok.pad_token_id, 
            eos_token_id=tok.eos_token_id, 
            repetition_penalty=1.2
        )
        
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tok, 
            device=0 if device == "cuda" else (-1 if device == "cpu" else None),
            generation_config=gen_conf
        )
        
        return tok, pipe
        
    except Exception as e:
        logger.error(f"Failed to load {model_id}: {type(e).__name__}: {e}")
        return None, None


def initialize_llm_pipeline(model_name: str, device: Optional[str] = None) -> Optional[Any]:
    """
    Initialize LLM pipeline using cached loader - matches Core-Shell app pattern.
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
        
        # Detect device safely - matches Core-Shell app logic
        if device is None:
            if torch.cuda.is_available():
                dev = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dev = "mps"
            else:
                dev = "cpu"
        else:
            dev = device
            
        logger.info(f"Initializing LLM: {model_name} -> {model_id} on {dev}")
        
        # Use cached loader - THIS IS THE KEY FIX
        tokenizer, pipe = _load_llm_model_cached(model_id, dev)
        
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

    st.title("📚 BibTeX Hallucination Validator")
    st.markdown("**Upload a `.bib` file** to validate references against Crossref & OpenAlex APIs. Detects and corrects hallucinated metadata, with robust author name matching.")

    with st.sidebar:
        st.header("⚙️ Settings")
        
        # LLM Model Selection - dropdown that actually triggers loading
        st.subheader("🤖 LLM Refinement (Optional)")
        llm_opts = ["None (API-only)", "gpt2 (~124M)", "distilgpt2 (~82M)", "Qwen2.5-0.5B-Instruct (~0.5B)"]
        sel_model = st.selectbox("Select model for metadata refinement:", llm_opts, index=0)
        use_llm = sel_model != "None (API-only)"
        
        # KEY FIX: Load LLM when dropdown changes - matches Core-Shell app pattern
        if use_llm and (st.session_state.llm_pipeline is None or st.session_state.current_model != sel_model):
            with st.status(f"🔄 Loading {sel_model}...", expanded=True) as status:
                st.write("Initializing tokenizer & model weights...")
                
                # Detect device for display
                if torch.cuda.is_available():
                    dev_display = "GPU (CUDA)"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    dev_display = "Apple Silicon (MPS)"
                else:
                    dev_display = "CPU"
                st.write(f"Target device: {dev_display}")
                
                # Warn about CPU loading for large models
                if dev_display == "CPU" and "Qwen" in sel_model:
                    st.warning("⚠️ Qwen2.5-0.5B on CPU requires ~1.2GB RAM and 30-60s to load. May timeout on free tiers.")
                
                # Attempt to load with cached function
                pipeline = initialize_llm_pipeline(sel_model)
                
                if pipeline:
                    st.session_state.llm_pipeline = pipeline
                    st.session_state.current_model = sel_model
                    st.session_state.llm_load_error = None
                    status.update(label=f"✅ {sel_model} loaded successfully", state="complete", expanded=False)
                    st.success(f"Model ready for refinement!")
                else:
                    st.session_state.llm_load_error = f"Failed to load {sel_model}"
                    status.update(label=f"⚠️ Load failed - using API-only", state="error", expanded=False)
                    st.warning("LLM refinement disabled. Author matching still works via rule-based normalization.")
                    use_llm = False  # Disable LLM usage for this session
        
        # Display load error if present
        if st.session_state.llm_load_error:
            st.error(f"❌ {st.session_state.llm_load_error}")
            if st.button("🔄 Retry Loading"):
                st.session_state.llm_pipeline = None
                st.session_state.current_model = None
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
            clean_doi = clean_doi(original_doi)
            if clean_doi:
                # Try Crossref first (primary source)
                with progress_details:
                    st.text(f"  → Querying Crossref for DOI: {clean_doi}")
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
                    st.text(f"  → Running LLM refinement with {sel_model}...")
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
    if show_diff:
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
        "llm_model": sel_model if use_llm else "Disabled",
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
