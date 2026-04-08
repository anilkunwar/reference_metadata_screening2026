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
    tokens2 = set(re
