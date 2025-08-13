# streamlit_app.py
# Mining Operations Chatbot (India) — Google-first Web RAG
# Copy-paste into a file and run:  streamlit run streamlit_app.py

import os, re, urllib.parse, io
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import requests
from bs4 import BeautifulSoup
import streamlit as st
import trafilatura
import html2text
import tldextract
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pdfminer.high_level import extract_text as pdf_extract_text

# =========================
# Config / constants
# =========================
DEFAULT_DOMAINS = [
    "dgms.gov.in","ibm.gov.in","mines.gov.in","coal.nic.in","moefcc.gov.in",
    "coalindia.in","secl-cil.in","nclcil.in","scclmines.com","nmdc.co.in","hzlindia.com",
    "vedantalimited.com","sail.co.in","hindustancopper.com","cmpdi.co.in",
    "iitism.ac.in","iitkgp.ac.in","iitg.ac.in","bis.gov.in","isstandards.in",
    "cat.com","komatsu.jp"
]
MINING_KEYWORDS = [
    "DGMS","IBM","MMDR","haul road","stripping ratio","dragline","shovel","dumper",
    "grade control","beneficiation","drilling","blasting","ventilation","opencast","underground",
    "MTBF","MTTR","availability","OEE","dispatch","mine planning"
]

EMBED_MODEL = os.getenv("EMBED_MODEL", "thenlper/gte-small")  # small, good quality
LOCAL_HF_MODEL = os.getenv("LOCAL_HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # CPU-friendly
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # optional (only if OPENAI_API_KEY set)

SYSTEM_PROMPT = """You are MINER-GPT, a safety-first assistant for mining operations in India.

Rules:
- Prefer Indian context: ₹, tonnes (t), bcm, kWh, DGMS/IBM/MMDR norms, Indian fleets.
- Cite sources you used (domains). If unsure, say what is missing.
- Be concise and practical for shift use; include bullet steps and formulas where useful.
- Safety first: Do NOT provide explosives handling or instructions violating DGMS/regulations.

Format:
1) Direct answer
2) Key steps / calculations
3) Sources (list)
"""

SAFETY_BLOCKLIST = [
    r"how .*make.*explosive", r"bypass .*interlock", r"disable .*failsafe",
    r"illegal .*mining", r"forge .*certificate", r"improvised.*explosive"
]

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}

# =========================
# Secrets / env
# =========================
def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    return st.secrets.get(name) if name in st.secrets else os.getenv(name, default)

GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")   # optional
GOOGLE_CSE_ID  = get_secret("GOOGLE_CSE_ID")    # optional (aka "cx")
SERPAPI_API_KEY = get_secret("SERPAPI_API_KEY") # optional
OPENAI_API_KEY  = get_secret("OPENAI_API_KEY")  # optional

# =========================
# Utilities
# =========================
def clean_domain(url: str) -> str:
    try:
        ext = tldextract.extract(url)
        return ".".join([p for p in [ext.domain, ext.suffix] if p])
    except Exception:
        return url

def domain_bias_query(query: str, prefer_domains: bool, limit: int = 8) -> str:
    if not prefer_domains:
        return query
    ops = " OR ".join([f"site:{d}" for d in DEFAULT_DOMAINS[:limit]])
    kw = " ".join(MINING_KEYWORDS[:6])
    return f"({query}) ({ops}) india mining {kw}"

# =========================
# Web search (Google-first)
# =========================
def google_search_json(query: str, max_results: int = 12) -> List[Dict]:
    """Google Programmable Search JSON API (needs GOOGLE_API_KEY & GOOGLE_CSE_ID)."""
    if not (GOOGLE_API_KEY and GOOGLE_CSE_ID):
        return []
    items: List[Dict] = []
    for start in range(1, min(max_results, 50) + 1, 10):
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "q": query,
            "num": min(10, max_results - len(items)),
            "start": start,
            "safe": "active",
            "lr": "lang_en"
        }
        try:
            r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            for it in data.get("items", []):
                link = it.get("link")
                title = it.get("title") or link
                if link and link.startswith("http"):
                    items.append({"title": title, "href": link})
                if len(items) >= max_results:
                    break
            if len(items) >= max_results or not data.get("items"):
                break
        except Exception:
            break
    return items

def serpapi_search(query: str, max_results: int = 12) -> List[Dict]:
    """SerpAPI fallback (needs SERPAPI_API_KEY)."""
    if not SERPAPI_API_KEY:
        return []
    try:
        params = {"engine": "google", "q": query, "num": max_results, "hl": "en", "api_key": SERPAPI_API_KEY}
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        out = []
        for res in (data.get("organic_results") or []):
            link = res.get("link")
            title = res.get("title") or link
            if link and link.startswith("http"):
                out.append({"title": title, "href": link})
            if len(out) >= max_results:
                break
        return out
    except Exception:
        return []

def ddg_html_fallback(query: str, max_results: int = 12) -> List[Dict]:
    """DuckDuckGo Lite HTML (no keys)."""
    url = "https://duckduckgo.com/html/"
    try:
        r = requests.get(url, params={"q": query}, headers=UA, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        out = []
        for a in soup.select("a.result__a, a.result__url"):
            href = a.get("href")
            title = a.get_text(" ", strip=True)
            if not href:
                continue
            if href.startswith("/l/?"):
                parsed = urllib.parse.urlparse(href)
                qs = urllib.parse.parse_qs(parsed.query)
                if "uddg" in qs and qs["uddg"]:
                    href = urllib.parse.unquote(qs["uddg"][0])
            if href.startswith("http"):
                out.append({"title": title or href, "href": href})
            if len(out) >= max_results:
                break
        return out
    except Exception:
        return []

def web_search(query: str, max_results: int, prefer_domains: bool) -> List[Dict]:
    q = domain_bias_query(query, prefer_domains)
    items = google_search_json(q, max_results=max_results)
    if items:
        return items
    items = serpapi_search(q, max_results=max_results)
    if items:
        return items
    return ddg_html_fallback(q, max_results=max_results)

# =========================
# Fetch & parse content
# =========================
def fetch_text(url: str, timeout: int = 20) -> str:
    """Fetch text from HTML or PDF. Returns '' on failure."""
    try:
        # HEAD to detect PDFs (don’t crash if blocked)
        ctype = ""
        try:
            h = requests.head(url, timeout=10, allow_redirects=True, headers=UA)
            ctype = (h.headers.get("Content-Type") or "").lower()
        except Exception:
            pass

        # PDF path
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            r = requests.get(url, timeout=timeout, headers=UA)
            r.raise_for_status()
            bio = io.BytesIO(r.content)
            txt = pdf_extract_text(bio) or ""
            return txt.strip()[:200_000]

        # HTML path
        downloaded = trafilatura.fetch_url(url, timeout=timeout)
        if not downloaded:
            # fallback: requests + trafilatura on raw html
            r = requests.get(url, timeout=timeout, headers=UA)
            r.raise_for_status()
            downloaded = r.text

        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if text and len(text.strip()) > 120:
            return text.strip()

        # last resort: html2text
        parser = html2text.HTML2Text()
        parser.ignore_links = True
        return parser.handle(downloaded)[:25_000]
    except Exception:
        return ""

def chunk(text: str, chunk_size: int = 1200, overlap: int = 120) -> List[str]:
    parts = re.split(r"(?<=[\.\?\!])\s+|\n{2,}", text)
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) < chunk_size:
            cur += (" " if cur else "") + p
        else:
            if cur: chunks.append(cur.strip())
            cur = p
    if cur: chunks.append(cur.strip())
    final = []
    for i, ch in enumerate(chunks):
        if i == 0: final.append(ch)
        else:
            tail = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
            final.append((tail + " " + ch).strip())
    return [c for c in final if c.strip()]

# =========================
# Retrieval
# =========================
@dataclass
class DocChunk:
    text: str
    source: str
    url: str

@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)

class Retriever:
    def __init__(self):
        self.model = get_embedder()
        self.index = None
        self.meta: List[DocChunk] = []

    def build(self, docs: List[DocChunk]):
        if not docs:
            self.index = None; self.meta = []; return
        self.meta = docs
        X = self.model.encode([d.text for d in docs], normalize_embeddings=True, batch_size=32)
        dim = X.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(X.astype("float32"))

    def search(self, query: str, k: int = 6) -> List[Tuple[float, DocChunk]]:
        if not self.index or not self.meta:
            return []
        qv = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(qv.astype("float32"), k)
        hits: List[Tuple[float, DocChunk]] = []
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.meta):
                hits.append((float(score), self.meta[idx]))
        return hits

# =========================
# Generation (CPU-only local by default — no accelerate needed)
# =========================
def has_openai() -> bool:
    return bool(OPENAI_API_KEY)

def gen_with_openai(system_prompt: str, user_prompt: str) -> str:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
        "temperature": 0.2
    }
    r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

@st.cache_resource(show_spinner=False)
def get_local_pipe():
    # Force plain CPU path; avoids needing accelerate
    return pipeline(
        "text-generation",
        model=LOCAL_HF_MODEL,
        torch_dtype="auto",
        max_new_tokens=700,
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.05,
        trust_remote_code=True,
        device=-1,  # CPU
    )

def gen_with_local(system_prompt: str, user_prompt: str) -> str:
    pipe = get_local_pipe()
    prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
    out = pipe(prompt)[0]["generated_text"]
    return out.split("<|assistant|>")[-1].strip()

# =========================
# Safety
# =========================
def is_unsafe(q: str) -> Tuple[bool, str]:
    low = q.lower()
    if any(re.search(pat, low) for pat in SAFETY_BLOCKLIST):
        return True, "This looks unsafe/illegal. I can only share public safety/regulatory info."
    if "blast" in low or "explosive" in low:
        if any(x in low for x in ["procedure","regulation","license","permit","dgms","mmdr"]):
            return False, ""
        return True, "Explosives/blasting instructions require licensed personnel per DGMS. I can only provide high-level regulatory info."
    return False, ""

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="MINER-GPT (India) – Web RAG (Google-first)", page_icon="⛏️", layout="wide")
st.title("⛏️ MINER-GPT (India) – Web-powered Mining Chatbot (Google-first)")

with st.sidebar:
    st.header("Settings")
    use_openai = st.toggle("Use OpenAI (set OPENAI_API_KEY)", value=has_openai())
    prefer_domains = st.toggle("Prefer India mining domains", value=False)  # default OFF for broader recall
    max_sites = st.slider("Max sites to ingest per query", 5, 20, 12)
    top_k_ctx = st.slider("Retrieved chunks (Top-K)", 3, 12, 6)
    st.markdown("---")
    st.subheader("Mining calculators")
    with st.expander("Shovel productivity (tph)"):
        b = st.number_input("Bucket (m³)", 3.0, 60.0, 10.0, 0.5)
        f = st.number_input("Fill factor (0–1)", 0.5, 1.0, 0.9, 0.01)
        d = st.number_input("Density (t/m³)", 1.4, 3.5, 2.1, 0.1)
        c = st.number_input("Cycle time (min)", 0.3, 5.0, 0.9, 0.05)
        p = st.number_input("Passes", 1, 12, 4, 1)
        tph = (b*f*d)*p*(60.0/c)
        st.info(f"Estimated **{tph:.1f} tph**")
    with st.expander("Equipment availability"):
        mtbf = st.number_input("MTBF (h)", 1.0, 500.0, 28.0, 0.5)
        mttr = st.number_input("MTTR (h)", 0.1, 48.0, 2.0, 0.1)
        avail = mtbf / max(mtbf + mttr, 1e-9)
        st.info(f"Availability ≈ **{avail:.3f}**")
    with st.expander("Cost per tonne (₹/t)"):
        lph = st.number_input("Fuel (L/h)", 5.0, 400.0, 80.0, 1.0)
        price = st.number_input("Diesel (₹/L)", 50.0, 140.0, 95.0, 1.0)
        prod = st.number_input("Production (t/h)", 10.0, 10000.0, 800.0, 10.0)
        other = st.number_input("Other ₹/h", 0.0, 100000.0, 1500.0, 100.0)
        cost = (lph*price + other)/max(prod, 1e-6)
        st.info(f"Cost ≈ **₹{cost:.2f}/t**")
    st.markdown("---")
    st.caption("Tip: add GOOGLE_API_KEY + GOOGLE_CSE_ID in Secrets for the strongest search. Try prompts like “DGMS haul road gradient 100T”, “bench height opencast India”, “IBM monthly return format”.")

st.write("Ask anything about **mining operations (India)**. I’ll search the web (HTML + PDF), retrieve the most relevant pieces, and answer with citations.")
q = st.text_input("Your question", placeholder="e.g., DGMS haul road gradient for 100T trucks?")

# =========================
# Build web context (aggressive, multi-pass)
# =========================
def build_context_from_web(query: str, max_sites: int, prefer_domains_flag: bool) -> Tuple[List[DocChunk], List[Dict], bool]:
    """
    Returns: (docs, sources_used, broadened_flag)
    - Pass 1: domain-biased (if prefer_domains_flag)
    - Pass 2: general query (no domain bias)
    - Pass 3: boosted with mining keywords
    """
    def run_once(qq: str, cap: int):
        results = web_search(qq, max_results=cap, prefer_domains=False)  # bias baked into qq
        docs, used = [], []
        for r in results:
            url = r.get("href") or r.get("link") or r.get("url")
            if not url:
                continue
            domain = clean_domain(url)
            txt = fetch_text(url)
            if len(txt) < 120:
                continue
            for ch in chunk(txt, 1200, 120)[:8]:
                docs.append(DocChunk(text=ch, source=domain, url=url))
            used.append({"title": r.get("title") or domain, "url": url, "domain": domain})
        return docs, used

    docs: List[DocChunk] = []
    used: List[Dict] = []
    broadened = False

    # Pass 1
    q1 = domain_bias_query(query, prefer_domains_flag, limit=8)
    d1, u1 = run_once(q1, max_sites)
    docs += d1; used += u1

    # Pass 2 (no domain bias) if weak
    if len(docs) < 5:
        broadened = True
        d2, u2 = run_once(query, max_sites + 4)
        seen = {x["url"] for x in used}
        used += [x for x in u2 if x["url"] not in seen]
        docs += d2

    # Pass 3 (add mining keywords) if still weak
    if len(docs) < 5:
        broadened = True
        kw = " ".join(MINING_KEYWORDS[:6])
        q3 = f"{query} india mining {kw}"
        d3, u3 = run_once(q3, max_sites + 6)
        seen = {x["url"] for x in used}
        used += [x for x in u3 if x["url"] not in seen]
        docs += d3

    return docs, used, broadened

def format_sources(srcs: List[Dict]) -> str:
    if not srcs:
        return "_No sources parsed (answer may be limited)._"
    lines = []
    for s in srcs:
        title = (s["title"] or s["domain"]).strip()
        title = re.sub(r"\s+", " ", title)
        lines.append(f"- {title} ({s['domain']})")
    return "\n".join(lines)

def build_user_prompt(query: str, hits: List[Tuple[float, DocChunk]], broadened: bool) -> str:
    ctx_lines = []
    for i, (score, d) in enumerate(hits, 1):
        ctx_lines.append(f"[{i}] {d.text}\n(Source: {d.source} | {d.url})")
    ctx_text = "\n\n---\n\n".join(ctx_lines) if ctx_lines else "(No web context parsed — answer from general knowledge, flag uncertainty.)"
    helper = (
        "Useful calculators:\n"
        "- tonnes_per_hour(bucket_m3, fill_factor, density_t_per_m3, cycle_time_min, passes)\n"
        "- availability(mtbf_h, mttr_h)\n"
        "- cost_per_tonne(fuel_lph, diesel_rs_per_l, prod_tph, other_rs_per_hr)\n"
        "If numeric, show formula, inputs and result.\n"
    )
    note = "Note: Query was broadened beyond preferred domains due to sparse sources.\n" if broadened else ""
    return (
        f"{note}User question:\n{query}\n\n"
        f"Use the context below if relevant. Quote & cite as [n]. If unsure, say what’s missing.\n\n"
        f"Context:\n{ctx_text}\n\n{helper}"
    )

if st.button("Answer", type="primary") and q:
    unsafe, why = is_unsafe(q)
    if unsafe:
        st.error(why + " Try rephrasing (e.g., ask for regulations or safety guidelines).")
        st.stop()

    with st.spinner("Searching + reading the web…"):
        docs, sources_used, broadened = build_context_from_web(q, max_sites=max_sites, prefer_domains_flag=prefer_domains)

    retriever = Retriever()
    if docs:
        retriever.build(docs)
        hits = retriever.search(q, k=top_k_ctx)
    else:
        hits = []

    user_prompt = build_user_prompt(q, hits, broadened)

    st.subheader("Sources I’m using")
    st.write(format_sources(sources_used))

    with st.spinner("Generating answer…"):
        try:
            if use_openai and has_openai():
                answer = gen_with_openai(SYSTEM_PROMPT, user_prompt)
            else:
                answer = gen_with_local(SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()

    st.subheader("Answer")
    if not docs:
        st.warning("Sources were sparse; answer may be limited. Add GOOGLE_API_KEY + GOOGLE_CSE_ID in Secrets for stronger search.")
    st.write(answer)

    if hits:
        st.subheader("Retrieved context (for transparency)")
        for i, (score, d) in enumerate(hits, 1):
            with st.expander(f"[{i}] {d.source} — score {score:.3f}"):
                st.write(d.text)
                st.caption(d.url)

st.markdown("---")
st.caption("Disclaimer: Informational only. Follow site SOPs and DGMS/IBM regulations. Live web search may reflect source inaccuracies.")
