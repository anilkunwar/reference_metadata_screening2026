# app.py
import streamlit as st
import bibtexparser
import requests
import re
import json
import time
from typing import Optional, Dict, List, Tuple
from bibtexparser.bparser import BibTexParser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.customization import getnames
import warnings
warnings.filterwarnings("ignore")

# Optional: LLM imports (with graceful fallback)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ==================== Configuration ====================
st.set_page_config(
    page_title="BibTeX Hallucination Checker",
    page_icon="📚",
    layout="wide"
)

# ==================== Helper Functions ====================

def parse_bibtex_file(file) -> List[Dict]:
    """Parse uploaded .bib file and return list of entries"""
    try:
        content = file.getvalue().decode("utf-8")
        parser = BibTexParser()
        parser.customization = getnames
        bib_database = bibtexparser.loads(content, parser)
        return bib_database.entries
    except Exception as e:
        st.error(f"❌ Error parsing BibTeX file: {e}")
        return []

def format_authors(authors: List[str]) -> str:
    """Format author list for BibTeX output"""
    if not authors:
        return ""
    return " and ".join(authors)

def clean_doi(doi: str) -> Optional[str]:
    """Clean and normalize DOI string"""
    if not doi:
        return None
    # Remove URL prefixes
    doi = re.sub(r'^https?://doi\.org/', '', doi, flags=re.IGNORECASE)
    doi = re.sub(r'^doi:\s*', '', doi, flags=re.IGNORECASE)
    doi = doi.strip()
    return doi if doi else None

def fetch_crossref_metadata(doi: str, timeout: int = 10) -> Optional[Dict]:
    """Fetch metadata from Crossref API using DOI"""
    clean_doi_val = clean_doi(doi)
    if not clean_doi_val:
        return None
    
    url = f"https://api.crossref.org/works/{clean_doi_val}"
    headers = {"User-Agent": "BibTeX-Validator/1.0 (mailto:your-email@example.com)"}
    
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            message = data.get("message", {})
            return {
                "title": message.get("title", [None])[0],
                "authors": [f"{a.get('given', '')} {a.get('family', '')}".strip() 
                           for a in message.get("author", []) if a.get("family")],
                "journal": message.get("container-title", [None])[0] or message.get("short-container-title", [None])[0],
                "year": message.get("published-print", {}).get("date-parts", [[None]])[0][0] or 
                        message.get("published-online", {}).get("date-parts", [[None]])[0][0],
                "volume": message.get("volume"),
                "issue": message.get("issue"),
                "pages": message.get("page"),
                "doi": message.get("DOI"),
                "type": message.get("type")
            }
    except requests.RequestException as e:
        st.warning(f"⚠️ Crossref API error for DOI {clean_doi_val}: {e}")
    return None

def fetch_openalex_metadata(doi: str, timeout: int = 10) -> Optional[Dict]:
    """Fetch metadata from OpenAlex API as fallback"""
    clean_doi_val = clean_doi(doi)
    if not clean_doi_val:
        return None
    
    url = f"https://api.openalex.org/works/https://doi.org/{clean_doi_val}"
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            return {
                "title": data.get("title"),
                "authors": [f"{a.get('author', {}).get('display_name', '')}" 
                           for a in data.get("authorships", [])],
                "journal": data.get("primary_location", {}).get("source", {}).get("display_name"),
                "year": data.get("publication_year"),
                "volume": data.get("biblio", {}).get("volume"),
                "issue": data.get("biblio", {}).get("issue"),
                "pages": data.get("biblio", {}).get("last_page"),  # Simplified
                "doi": data.get("doi"),
                "type": data.get("type")
            }
    except requests.RequestException as e:
        st.warning(f"⚠️ OpenAlex API error: {e}")
    return None

def search_by_title(title: str, max_results: int = 3) -> List[Dict]:
    """Search Crossref by title as fallback when DOI fails"""
    if not title:
        return []
    
    url = "https://api.crossref.org/works"
    params = {
        "query.title": title,
        "rows": max_results,
        "select": "DOI,title,author,container-title,published-print,volume,page,type"
    }
    headers = {"User-Agent": "BibTeX-Validator/1.0"}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        if response.status_code == 200:
            items = response.json().get("message", {}).get("items", [])
            results = []
            for item in items:
                results.append({
                    "title": item.get("title", [None])[0],
                    "doi": item.get("DOI"),
                    "authors": [f"{a.get('given', '')} {a.get('family', '')}".strip() 
                               for a in item.get("author", []) if a.get("family")],
                    "journal": item.get("container-title", [None])[0],
                    "year": item.get("published-print", {}).get("date-parts", [[None]])[0][0],
                    "volume": item.get("volume"),
                    "pages": item.get("page"),
                    "match_score": item.get("score", 0)
                })
            return results
    except Exception as e:
        st.warning(f"⚠️ Title search error: {e}")
    return []

def compare_metadata(original: Dict, verified: Dict) -> Dict[str, Dict]:
    """Compare original and verified metadata, flagging discrepancies"""
    fields_to_check = ["title", "authors", "journal", "year", "volume", "pages"]
    discrepancies = {}
    
    for field in fields_to_check:
        orig_val = str(original.get(field, "")).lower().strip()
        verify_val = str(verified.get(field, "")).lower().strip()
        
        # Handle authors list comparison
        if field == "authors" and isinstance(original.get("authors"), list):
            orig_val = " ".join(sorted([a.lower() for a in original["authors"]]))
            verify_val = " ".join(sorted([a.lower() for a in verified.get("authors", [])]))
        
        is_match = orig_val and verify_val and (
            orig_val == verify_val or 
            orig_val in verify_val or 
            verify_val in orig_val or
            # Fuzzy match for titles (simple)
            (field == "title" and similarity_ratio(orig_val, verify_val) > 0.85)
        )
        
        discrepancies[field] = {
            "original": original.get(field),
            "verified": verified.get(field),
            "match": bool(is_match),
            "needs_review": not is_match and verify_val  # Only flag if we have verified data
        }
    
    return discrepancies

def similarity_ratio(s1: str, s2: str) -> float:
    """Simple string similarity ratio (for title matching)"""
    if not s1 or not s2:
        return 0.0
    s1, s2 = s1.lower(), s2.lower()
    if s1 == s2:
        return 1.0
    # Simple token-based overlap
    tokens1 = set(re.findall(r'\w+', s1))
    tokens2 = set(re.findall(r'\w+', s2))
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)

def generate_corrected_entry(original: Dict, verified: Dict, discrepancies: Dict, 
                            use_llm: bool = False, model_name: str = None) -> Dict:
    """Generate corrected BibTeX entry, optionally using LLM for refinement"""
    corrected = original.copy()
    
    # Auto-correct fields with high-confidence matches
    for field, info in discrepancies.items():
        if info["match"] and verified.get(field):
            corrected[field] = verified[field]
        elif info["needs_review"] and not use_llm:
            # Keep original but flag for review
            corrected[f"_flag_{field}"] = True
    
    # Optional LLM refinement (for ambiguous cases)
    if use_llm and TRANSFORMERS_AVAILABLE and discrepancies:
        corrected = llm_refine_metadata(original, verified, discrepancies, model_name)
    
    return corrected

def llm_refine_metadata(original: Dict, verified: Dict, discrepancies: Dict, 
                       model_name: str) -> Dict:
    """Use small LLM to help resolve metadata conflicts"""
    if not TRANSFORMERS_AVAILABLE:
        return {**original, **verified}  # Fallback merge
    
    try:
        # Prepare prompt for LLM
        prompt = f"""You are a scholarly metadata validator. Compare these two reference entries and output ONLY valid BibTeX author field format.

ORIGINAL:
{json.dumps(original, indent=2, ensure_ascii=False)}

VERIFIED (from Crossref/OpenAlex):
{json.dumps(verified, indent=2, ensure_ascii=False)}

DISCREPANCIES:
{json.dumps({k: v for k, v in discrepancies.items() if v.get('needs_review')}, indent=2)}

Instructions:
1. Prefer verified data when confidence is high
2. Format authors as "Last, First and Last, First"
3. Keep original if verified data is missing or clearly wrong
4. Output ONLY the corrected BibTeX entry fields that need changes, in valid BibTeX format
5. Do not add explanations

Corrected fields:"""
        
        # Load model (cached after first load)
        if not hasattr(st, "llm_pipe") or st.session_state.get("current_model") != model_name:
            with st.spinner(f"🔄 Loading {model_name}..."):
                if "qwen" in model_name.lower() and "0.5b" in model_name.lower():
                    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
                elif "gpt2" in model_name.lower() or model_name == "gpt2":
                    model_id = "gpt2"
                elif "distilgpt2" in model_name.lower():
                    model_id = "distilgpt2"
                else:
                    model_id = "gpt2"  # Default fallback
                
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                st.session_state.llm_pipe = pipeline(
                    "text-generation", 
                    model=model, 
                    tokenizer=tokenizer,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                st.session_state.current_model = model_name
        
        # Generate refinement
        result = st.session_state.llm_pipe(prompt)
        llm_output = result[0]['generated_text'].split("Corrected fields:")[-1].strip()
        
        # Parse LLM output (simple key-value extraction)
        for line in llm_output.split('\n'):
            if '=' in line and not line.strip().startswith(('%', '@', '{', '}')):
                key, _, value = line.partition('=')
                key = key.strip().strip(',').strip()
                value = value.strip().strip(',').strip('{}"')
                if key in original:
                    original[key] = value
        
        return original
        
    except Exception as e:
        st.warning(f"⚠️ LLM refinement failed: {e}. Using fallback merge.")
        return {**original, **verified}  # Safe fallback

def create_bibtex_entry(entry: Dict, cite_key: str) -> str:
    """Create properly formatted BibTeX entry string"""
    entry_type = entry.get("_type", "
