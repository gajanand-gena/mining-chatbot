# --- add near the top with other imports ---
import requests
from bs4 import BeautifulSoup

# --- replace your ddg_search() with this robust version ---
def ddg_search(q, max_results=8, site_filters=None):
    """
    Try official duckduckgo_search first. If it fails (vqd error / 403 / None),
    fallback to scraping the lite HTML endpoint.
    """
    q2 = q
    if site_filters:
        domain_q = " OR ".join([f"site:{d}" for d in site_filters])
        q2 = f"({q}) ({domain_q})"
    kw = " ".join(MINING_KEYWORDS[:6])
    q2 = f"{q2} india mining {kw}"

    # 1) primary: API
    try:
        with DDGS(timeout=30) as ddgs:
            results = list(ddgs.text(q2, max_results=max_results, safesearch="moderate", region="in-en"))
            if results:
                return results
    except Exception:
        pass  # fall through

    # 2) fallback: lite HTML scraping
    return ddg_html_fallback(q2, max_results=max_results)

# --- add this helper anywhere below imports ---
def ddg_html_fallback(q, max_results=8):
    """
    Scrape DuckDuckGo's lite HTML results. Returns list of dicts with keys: title, href.
    """
    url = "https://duckduckgo.com/html/"
    params = {"q": q}
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        out = []
        # each result link is in a <a class="result__a"> or within result links
        for a in soup.select("a.result__a, a.result__url"):
            href = a.get("href")
            title = a.get_text(" ", strip=True)
            if href and href.startswith("http"):
                out.append({"title": title, "href": href})
            if len(out) >= max_results:
                break
        return out
    except Exception:
        return []
