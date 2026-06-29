import re
import json
import requests
import trafilatura

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"


def _fetch_html(url):
    return trafilatura.fetch_url(url)


def _extract_jsonld_product(html):
    """Extract name + price from JSON-LD structured data (instant, no AI)."""
    name, price = None, None
    for block in re.finditer(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html, re.DOTALL | re.IGNORECASE
    ):
        try:
            data = json.loads(block.group(1))
            if isinstance(data, list):
                data = next(
                    (d for d in data if isinstance(d, dict) and d.get('@type') == 'Product'),
                    {}
                )
            if not isinstance(data, dict) or data.get('@type') != 'Product':
                continue
            if data.get('name'):
                name = data['name']
            offers = data.get('offers', {})
            if isinstance(offers, list):
                offers = offers[0] if offers else {}
            if isinstance(offers, dict):
                raw = offers.get('price') or offers.get('lowPrice')
                if raw:
                    price = float(str(raw).replace(',', '.'))
            if name or price:
                break
        except Exception:
            continue
    return name, price


def _extract_title(html):
    m = re.search(r'<title[^>]*>(.*?)</title>', html, re.DOTALL | re.IGNORECASE)
    if m:
        return re.sub(r'<[^>]+>', '', m.group(1)).strip()
    m = re.search(
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
        html, re.IGNORECASE
    )
    return m.group(1).strip() if m else ''


def _extract_dimensions(text):
    """Extract W x D x H in cm. Returns (w, d, h) or (None, None, None)."""
    if not text:
        return None, None, None
    m = re.search(
        r'(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*cm',
        text, re.IGNORECASE
    )
    if m:
        def p(s): return float(s.replace(',', '.'))
        return p(m.group(1)), p(m.group(2)), p(m.group(3))
    return None, None, None


def _extract_price_ollama(text):
    if not text:
        return None
    snippet = text[:1500]
    prompt = (
        "Esti un asistent care extrage preturi din text romanesc. "
        "Raspunde DOAR cu pretul in RON ca numar (ex: 1299.00 sau 850). "
        "Daca nu gasesti un pret clar, raspunde: null\n\n"
        f"Text: {snippet}"
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 20},
            },
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        if not raw or raw.lower().startswith("null"):
            return None
        m = re.search(r'(\d{2,6}(?:[.,]\d{1,2})?)', raw)
        if m:
            v = float(m.group(1).replace(',', '.'))
            if 10 <= v <= 100_000:
                return v
    except Exception:
        pass
    return None


def scrape_product_url(url):
    result = {
        'name': '', 'price': None,
        'width_cm': None, 'depth_cm': None, 'height_cm': None,
    }

    html = _fetch_html(url)
    if not html:
        return result, 'Nu s-a putut descarca pagina (timeout sau acces refuzat).'

    # 1. JSON-LD — instant, most precise for product pages
    name, price = _extract_jsonld_product(html)
    if name:
        result['name'] = name
    if price:
        result['price'] = price

    # 2. Fallback name: page title
    if not result['name']:
        result['name'] = _extract_title(html)

    # 3. Clean text for price and dimension extraction
    clean = trafilatura.extract(html) or ''

    # 4. Price via Ollama if JSON-LD didn't find it
    if not result['price']:
        result['price'] = _extract_price_ollama(clean)

    # 5. Dimensions
    w, d, h = _extract_dimensions(clean)
    result['width_cm'], result['depth_cm'], result['height_cm'] = w, d, h

    msg = ('Produs analizat cu succes.' if result['price']
           else 'Pretul nu a putut fi extras automat — completeaza manual.')
    return result, msg
