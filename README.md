# MINER-GPT (India) — Streamlit Web-RAG Chatbot

Zero-config chatbot for India-focused mining operations. It searches authoritative Indian sources (DGMS, IBM, PSUs, OEMs), scrapes pages, does lightweight RAG with FAISS + `gte-small`, and answers with citations. Runs **without any API keys** using a local Hugging Face model by default.

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
