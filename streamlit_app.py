# app.py
import os, re, time, math, html
import json
import tldextract
from typing import List, Dict, Tuple
from dataclasses import dataclass

import streamlit as st
from duckduckgo_search import DDGS
import trafilatura
import html2text
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ------------------------------
# Config
# ------------------------------
DEFAULT_DOMAINS = [
    # India govt / regulators / PSUs / institutes
    "dgms.gov.in", "ibm.gov.in", "coal.nic.in", "mines.gov.in", "moef.gov.in", "moefcc.gov.in",
    "coalindia.in", "secl-cil.in", "nclcil.in", "scclmines.com", "nmdc.co.in", "hzlindia.com",
    "vedantalimited.com", "sail.co.in", "hindustancopper.com", "bsesdelhi.com", "becnm.gov.in",
    "cmpdi.co.in", "iitg.ac.in", "iitism.ac.in", "iitkgp.ac.in",
    # OEMs & standards
    "cat.com", "komatsu.jp", "eimcoelecon.in", "isstandards.in", "bis.gov.in",
]
MINING_KEYWORDS = [
    "DGMS", "IBM", "MMDR", "MCR", "blasting", "overburden", "stripping ratio", "dragline",
    "shovel", "dumper", "grade control", "beneficiation", "haul road", "MTBF", "MTTR",
    "availability", "OEE", "HSE", "ESG", "mine planning", "drilling", "ventilation",
    "opencast", "underground", "longwall", "bord and pillar", "dispatch"
]

EMBED_MODEL = os.getenv("EMBED_MODEL", "thenlper/gte-small")  # small & good
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")      # optional if you have a key
LOCAL_HF_MODEL = os.getenv("LOCAL_HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

SYSTEM_PROMPT = """You are MINER-GPT, a safety-first assistant focused on mining operations in India.

Rules:
- Prefer Indian context: ₹, tonnes (t), bcm, kWh, km/h, DGMS/IBM/MMDR norms, Indian fleets and practices.
- Cite web sources you used (with domains) and clearly separate facts from assumptions.
- Be practical and concise for shift use; include bullet steps and formulas where useful.
- Safety first: Do NOT provide instructions that bypass interlocks, explosives handling beyond public safety guidance, or anything violating DGMS/regulations. If asked, refuse and suggest contacting the mine manager/blasting officer.
- If sources conflict, state both views and what’s commonly adopted in India.
Format:
1) Direct answer
2) Key steps / calculations
3) Sources (list)
"""

SAFETY_BLOCKLIST = [
    r"how .*make.*explosive", r"bypass .*interlock", r"disable .*fail[- ]?safe",
    r"illegal .*mining", r"forge .*certificate", r"gunpowder", r"improvised.*explosive",
]

# ------------------------------
# Simple domain tools & math
# ------------------------------
def tonnes_per_hour(bucket_m3, fill_factor, density_t_per_m3, cycle_time_min, passes):
    bucket_t = bucket_m3 * fill_factor * density_t_per_m3
    return (passes * bucket_t) * (60.0 / cycle_time_min)

def availability(mtbf_h, mttr_h):
    return mtbf_h / max(mtbf_h + mttr_h, 1e-9)

def cost_per_tonne(fuel_lph, diesel_rs_per_l, prod_tph, other_rs_per_hr=0.0):
    return (fuel_lph * diesel_rs_per_l + other_rs_per_hr) / max(prod_tph, 1e-6)

# ------------------------------
# Retrieval (search + scrape + chunk + embed)
# ------------------------------
def ddg_search(q, max_results=8, site_filters=None):
    q2 = q
    if site_filters:
        domain_q = " OR ".join([f"site:{d}" for d in site_filters])
        q2 = f"({q}) ({domain_q})"
    # Small bias to India mining
    kw = " ".join(MINING_KEYWORDS[:6])
    q2 = f"{q2} india mining {kw}"
    with DDGS() as ddgs:
        return list(ddgs.text(q2, max_results=max_results, safesearch="moderate", region="in-en"))

def clean_domain(url):
    try:
        ext = tldextract.extract(url)
        return ".".join([p for p in [ext.domain, ext.suffix] if p])
    except:
        return url

def fetch_text(url, timeout=20):
    try:
        downloaded = trafilatura.fetch_url(url, timeout=timeout)
        if not downloaded:
            return ""
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if text and len(text.strip()) > 200:
            return text.strip()
        # fallback: html2text
        h = trafilatura.fetch_url(url, timeout=timeout)
        if not h:
            return ""
        parser = html2text.HTML2Text()
        parser.ignore_links = True
        return parser.handle(h)[:20000]
    except Exception:
        return ""

def chunk(text, chunk_size=900, overlap=150):
    parts = re.split(r"(?<=[\.\?\!])\s+|\n{2,}", text)
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) < chunk_size:
            cur += (" " if cur else "") + p
        else:
            if cur: chunks.append(cur.strip())
            cur = p
    if cur: chunks.append(cur.strip())
    # overlap
    final = []
    for i, ch in enumerate(chunks):
        if i == 0: final.append(ch)
        else:
            tail = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
            final.append((tail + " " + ch).strip())
    return [c for c in final if c.strip()]

@dataclass
class DocChunk:
    text: str
    source: str   # domain or short url
    url: str

class Retriever:
    def __init__(self, embed_model=EMBED_MODEL):
        self.model = SentenceTransformer(embed_model)
        self.index = None
        self.meta: List[DocChunk] = []

    def build(self, docs: List[DocChunk]):
        if not docs:
            self.index = None
            self.meta = []
            return
        self.meta = docs
        X = self.model.encode([d.text for d in docs], normalize_embeddings=True, batch_size=32)
        dim = X.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(X.astype("float32"))

    def search(self, query, top_k=6):
        if not self.index or not self.meta:
            return []
        qv = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(qv.astype("float32"), top_k)
        hits = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0: continue
            d = self.meta[idx]
            hits.append((float(score), d))
        return hits

# ------------------------------
# Generation backends
# ------------------------------
def has_openai():
    return os.getenv("OPENAI_API_KEY") is not None and len(os.getenv("OPENAI_API_KEY")) > 5

def gen_with_openai(system_prompt, user_prompt):
    # Minimal dependency to avoid extra installs
    import requests
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
        "temperature": 0.2
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

@st.cache_resource(show_spinner=False)
def get_local_pipe():
    # Small local chat model; works on CPU
    return pipeline(
        "text-generation",
        model=LOCAL_HF_MODEL,
        torch_dtype="auto",
        device_map="auto",
        max_new_tokens=600,
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.05,
        trust_remote_code=True,
    )

def gen_with_local(system_prompt, user_prompt):
    pipe = get_local_pipe()
    prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
    out = pipe(prompt)[0]["generated_text"]
    return out.split("<|assistant|>")[-1].strip()

# ------------------------------
# Safety & policy checks
# ------------------------------
def is_unsafe(q: str) -> Tuple[bool, str]:
    low = q.lower()
    if any(re.search(pat, low) for pat in SAFETY_BLOCKLIST):
        return True, "This request appears to involve unsafe or illegal instructions."
    # explosives specific
    if "blast" in low or "explosive" in low:
        # allow generic safety/regulatory info only
        if any(x in low for x in ["procedure", "regulation", "regulations", "license", "permit", "dgms", "mmdr"]):
            return False, ""
        return True, "Explosives/blasting instructions must be handled by licensed personnel as per DGMS; I can only share public safety/regulatory info."
    return False, ""

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="MINER-GPT (India) – Web RAG", page_icon="⛏️", layout="wide")
st.title("⛏️ MINER-GPT (India) – Web-powered Mining Chatbot")

with st.sidebar:
    st.header("Settings")
    st.caption("Defaults work out-of-the-box.")
    use_openai = st.toggle("Use OpenAI (set OPENAI_API_KEY env var)", value=has_openai())
    top_sites_only = st.toggle("Prefer India mining domains", value=True)
    max_sites = st.slider("Max sites to ingest per query", 2, 12, 6)
    top_k_ctx = st.slider("Retrieved chunks (Top-K)", 3, 12, 6)
    st.markdown("---")
    st.subheader("Mining calculators")
    with st.expander("Shovel productivity (tph)"):
        b = st.number_input("Bucket (m³)", 3.0, 60.0, 10.0, 0.5)
        f = st.number_input("Fill factor (0–1)", 0.5, 1.0, 0.9, 0.01)
        d = st.number_input("Density (t/m³)", 1.4, 3.5, 2.1, 0.1)
        c = st.number_input("Cycle time (min)", 0.3, 5.0, 0.9, 0.05)
        p = st.number_input("Passes", 1, 12, 4, 1)
        st.info(f"Estimated **{tonnes_per_hour(b,f,d,c,p):.1f} tph**")
    with st.expander("Equipment availability"):
        mtbf = st.number_input("MTBF (h)", 1.0, 500.0, 28.0, 0.5)
        mttr = st.number_input("MTTR (h)", 0.1, 48.0, 2.0, 0.1)
        st.info(f"Availability ≈ **{availability(mtbf, mttr):.3f}**")
    with st.expander("Cost per tonne (₹/t)"):
        lph = st.number_input("Fuel (L/h)", 5.0, 400.0, 80.0, 1.0)
        price = st.number_input("Diesel (₹/L)", 50.0, 140.0, 95.0, 1.0)
        tph = st.number_input("Production (t/h)", 10.0, 10000.0, 800.0, 10.0)
        other = st.number_input("Other ₹/h", 0.0, 1_00_000.0, 1500.0, 100.0)
        st.info(f"Cost ≈ **₹{cost_per_tonne(lph, price, tph, other):.2f}/t**")
    st.markdown("---")
    st.caption("Tip: ask things like “bench width for 100T trucks as per Indian practice?” or “DGMS norms for haul road gradient”.")

st.write("Ask anything about **mining operations (India)**. I’ll search authoritative sources, read them, and answer with citations.")

q = st.text_input("Your question", placeholder="e.g., Recommended haul road gradient per DGMS for 100T mine trucks?")

def build_context_from_web(query: str, max_sites: int, prefer_domains: bool) -> Tuple[List[DocChunk], List[Dict]]:
    # 1) Search
    filters = DEFAULT_DOMAINS if prefer_domains else None
    results = ddg_search(query, max_results=max_sites, site_filters=filters)
    # 2) Fetch & chunk
    docs, used = [], []
    for r in results:
        url = r.get("href") or r.get("url") or r.get("link")
        if not url: continue
        domain = clean_domain(url)
        txt = fetch_text(url)
        if len(txt) < 400:  # skip weak pages
            continue
        for ch in chunk(txt, 900, 150)[:8]:
            docs.append(DocChunk(text=ch, source=domain, url=url))
        used.append({"title": r.get("title") or domain, "url": url, "domain": domain})
    return docs, used

def format_sources(srcs: List[Dict]) -> str:
    # Make a tidy list
    lines = []
    for s in srcs:
        title = (s["title"] or s["domain"]).strip()
        title = re.sub(r"\s+", " ", title)
        lines.append(f"- {title} ({s['domain']})")
    return "\n".join(lines)

def build_user_prompt(query: str, hits: List[Tuple[float, DocChunk]]) -> str:
    ctx = []
    for i, (score, d) in enumerate(hits, 1):
        ctx.append(f"[{i}] {d.text}\n(Source: {d.source} | {d.url})")
    ctx_text = "\n\n---\n\n".join(ctx)
    helper = (
        "Useful calculators available:\n"
        "- tonnes_per_hour(bucket_m3, fill_factor, density_t_per_m3, cycle_time_min, passes)\n"
        "- availability(mtbf_h, mttr_h)\n"
        "- cost_per_tonne(fuel_lph, diesel_rs_per_l, prod_tph, other_rs_per_hr)\n"
        "If a numeric question is asked, show formula, inputs and result.\n"
    )
    return (
        f"User question:\n{query}\n\n"
        f"Use the context below if relevant. Quote & cite as [n]. If unsure, say what’s missing.\n\n"
        f"Context:\n{ctx_text}\n\n{helper}"
    )

if st.button("Answer", type="primary") and q:
    unsafe, why = is_unsafe(q)
    if unsafe:
        st.error(why + " I can provide only high-level safety/regulatory guidance. Try rephrasing.")
        st.stop()

    with st.spinner("Searching authoritative India mining sources…"):
        docs, sources_used = build_context_from_web(q, max_sites=max_sites, prefer_domains=top_sites_only)

    if not docs:
        st.warning("Couldn’t find strong sources. Try a more specific query or turn off domain filter.")
        st.stop()

    retriever = Retriever(EMBED_MODEL)
    retriever.build(docs)
    hits = retriever.search(q, top_k=top_k_ctx)

    user_prompt = build_user_prompt(q, hits)

    # Show sources upfront
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
    st.write(answer)

    st.subheader("Retrieved context (for transparency)")
    for i, (score, d) in enumerate(hits, 1):
        with st.expander(f"[{i}] {d.source} — score {score:.3f}"):
            st.write(d.text)
            st.caption(d.url)

st.markdown("---")
st.caption("Disclaimer: Informational only. Always follow site SOPs and DGMS/IBM regulations. This tool performs live web search and may reflect source inaccuracies.")
