# ESU_202502 Utils v9
# Extracted helper functions from ESU_202502_OneShot_FewShot_Images_Scaffold_v8.ipynb
# You can import this in Colab as:
#   import ESU_202502_utils_v9 as esu_utils

#  To make it easier, add to __init__.py:

# from .ESU_202502_utils_v9 import (
#    fetch_image_any,
    #     __all__ = [
#    "fetch_image_any",
   
#  etc.

import requests
import time
import random
import numpy as np
from PIL import Image
from io import BytesIO

rng = random.Random(42)

# === Patch: polite loader + backoff + cache (drop-in) ===

# Polite globals
UA_CONTACT = "FewShotColab/1.0 (contact: teacher@example.com)"  # keep a descriptive UA
HEADERS = {"User-Agent": UA_CONTACT, "Accept": "image/*,*/*;q=0.8"}
TIMEOUT = 25
THROTTLE_SEC = 0.8          # base pause between fetches (be generous)
MAX_RETRIES = 4
BACKOFF = 1.8               # exponential backoff factor
JITTER = 0.25               # random jitter added to pauses
DOWNLOAD_CACHE = {}         # url -> PIL.Image (in-memory)

# If your auto-fetch code defines these, we’ll reuse; otherwise we’ll make a new session.
session = globals().get("session") or requests.Session()
session.headers.update(HEADERS)




def load_image(url):
    """Downloads and returns a PIL image from the given URL."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise ValueError(f"Failed to download image: {url}")
    try:
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not open image from {url}. Error: {e}")


# If your auto-fetch cell had a private _load_image(), we override it here:

# Optional: shrink the target width used by Commons resolver for fewer bytes per file.
# If you have a function commons_files_to_urls(width=...), rebind a narrower default:
def _wrap_commons_files_to_urls(fn):
    def wrapped(file_titles, width=640):  # default 640 instead of 1024
        return fn(file_titles, width=width)
    return wrapped

if "commons_files_to_urls" in globals():
    commons_files_to_urls = _wrap_commons_files_to_urls(commons_files_to_urls)


# ---- Utility cell extracted from notebook ----
def fetch_image_any(src):
    """
    Accepts:
      - direct http(s) URL
      - 'commons:File:...' token
      - 'File:...' title
    Respects ALLOW_WEB_DOWNLOADS toggle.
    """
    if not ALLOW_WEB_DOWNLOADS:
        raise ValueError("Web downloads disabled via ALLOW_WEB_DOWNLOADS=False")

    # Resolve Commons tokens via Special:FilePath which is robust to mirrors
    if isinstance(src, str) and (src.startswith("commons:") or src.startswith("File:")):
        title = src.replace("commons:", "").strip()
        # Build Special:FilePath URL
        quoted = requests.utils.requote_uri(title.replace("File:", ""))
        url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{quoted}?width=1024"
        return load_image(url)

    # Otherwise treat as direct URL
    return load_image(src)

# --- Cosine logits helper (drop-in) ---
# Safe to run multiple times; works even if tensors are empty.
def cosine_logits(query, prototypes, tau: float = 1.0):
    import torch
    import torch.nn.functional as F

    if query is None or prototypes is None:
        return torch.empty(0, 0)

    if query.numel() == 0 or prototypes.numel() == 0:
        qn = query.shape[0] if query.ndim == 2 else 0
        cn = prototypes.shape[0] if prototypes.ndim == 2 else 0
        return torch.empty(qn, cn)

    # Defensive L2-normalization (your pipeline *usually* already normalizes)
    q = F.normalize(query, dim=1)
    p = F.normalize(prototypes, dim=1)

    tau = max(float(tau), 1e-8)
    return (q @ p.T) / tau



# Wikipedia image download utils:



def _load_image(url, ref="https://www.google.com/"):
    r = session.get(url, timeout=TIMEOUT, allow_redirects=True, stream=True, headers={
        **HEADERS, "Accept": "image/avif,image/webp,image/*,*/*;q=0.8", "Referer": ref
    })
    r.raise_for_status()
    content = r.raw.read()
    return Image.open(BytesIO(content)).convert("RGB")

# ----- Wikimedia Commons search -----
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

def commons_search_files(query: str, limit: int = 20):
    # 'origin=*' is harmless outside CORS, but tolerated by API
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srnamespace": 6,  # File:
        "srlimit": limit,
        "format": "json",
        "origin": "*",
    }
    r = session.get(COMMONS_API, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    hits = data.get("query", {}).get("search", [])
    return [h["title"] for h in hits if h.get("title", "").startswith("File:")]

def commons_files_to_urls(file_titles, width: int = 1024):
    if not file_titles:
        return []
    titles = "|".join(file_titles[:50])
    params = {
        "action": "query",
        "titles": titles,
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": str(width),
        "format": "json",
        "origin": "*",
    }
    r = session.get(COMMONS_API, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    urls = []
    for _, page in data.get("query", {}).get("pages", {}).items():
        info = page.get("imageinfo", [])
        if not info:
            continue
        url = info[0].get("thumburl") or info[0].get("url")
        if url:
            urls.append(url)
    return urls

# ----- Wikipedia fallback (pageimages) -----
WIKI_API = "https://en.wikipedia.org/w/api.php"

def wikipedia_pageimages_urls(query: str, limit: int = 10, width: int = 800):
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": limit,
        "prop": "pageimages",
        "piprop": "thumbnail|original",
        "pithumbsize": width,
        "format": "json",
        "origin": "*",
    }
    r = session.get(WIKI_API, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    urls = []
    for _, p in pages.items():
        # prefer thumbnail, then original
        thumb = p.get("thumbnail", {}).get("source")
        orig  = p.get("original", {}).get("source") if p.get("original") else None
        url = thumb or orig
        if url:
            urls.append(url)
    return list(dict.fromkeys(urls))

# ----- Picsum last-resort fallback (semantic-agnostic) -----
def picsum_urls(label: str, n: int, size: int = 512):
    return [f"https://picsum.photos/seed/{label}_{i}/{size}/{size}" for i in range(n)]

def fetch_images_for_label(label: str, min_needed: int) -> list:
    """Try Commons (w/ UA), then Wikipedia pageimages, then Picsum."""
    urls = []
    try:
        titles = commons_search_files(f"{label} animal", limit=20)
        if not titles:
            titles = commons_search_files(label, limit=20)
        urls = commons_files_to_urls(titles, width=1024)
    except requests.HTTPError as e:
        print(f"[INFO] Commons HTTP {e.response.status_code} for '{label}'. Falling back.")
    except Exception as e:
        print(f"[INFO] Commons search error for '{label}': {e}. Falling back.")

    if len(urls) < min_needed:
        try:
            wiki_urls = wikipedia_pageimages_urls(label, limit=20, width=800)
            urls = list(dict.fromkeys(urls + wiki_urls))
        except Exception as e:
            print(f"[INFO] Wikipedia pageimages error for '{label}': {e}. Falling back.")

    # Final safety: picsum (not semantically correct, but keeps the pipeline alive)
    if len(urls) < min_needed:
        need = min_needed - len(urls)
        urls.extend(picsum_urls(label, need, size=512))

    imgs = []
    for u in urls:
        if len(imgs) >= min_needed:
            break
        try:
            imgs.append(_load_image(u))
            time.sleep(0.2)  # be polite
        except Exception as e:
            print(f"[WARN] fetch/open failed for {u}: {e}")
    return imgs

def split_support_query(imgs, k_shot, q_per_class):
    idx = np.arange(len(imgs))
    rng.shuffle(idx)
    k = min(k_shot, len(imgs))
    q = min(q_per_class, max(0, len(imgs) - k))
    return [imgs[i] for i in idx[:k]], [imgs[i] for i in idx[k:k+q]]


def _sleep_with_jitter(base):
    time.sleep(max(0.0, base + random.uniform(-JITTER, JITTER)))

def _should_retry_status(status):
    return status in (429, 500, 502, 503, 504)

def _open_image_from_bytes(content):
    img = Image.open(BytesIO(content)).convert("RGB")
    return img



def pil_from_sources(label, urls_or_tokens, min_needed=1, search_fallback=True):
    """
    Download images from a mixed list of direct URLs or 'commons:File:...' tokens.
    If nothing usable is obtained and ENABLE_PLACEHOLDER_ON_FAIL is True, synthesize placeholders.
    """
    imgs = []

    # Try provided sources if any
    for src in (urls_or_tokens or []):
        try:
            imgs.append(fetch_image_any(src))
        except Exception as e:
            print(f"[WARN] {e}")

    # If still not enough and search fallback is implemented elsewhere, we could add it here (left as-is).
    # ...

    # Placeholder fallback
    if len(imgs) < min_needed and ENABLE_PLACEHOLDER_ON_FAIL:
        need = max(min_needed, 3)
        print(f"[INFO] Using placeholder images for '{label}' (requested {min_needed}, got {len(imgs)}).")
        imgs.extend(make_placeholder_images(label, n=need-len(imgs), size=256))

    return imgs

def load_image_polite(url, referer="https://www.google.com/"):
    # Cache first
    if url in DOWNLOAD_CACHE:
        return DOWNLOAD_CACHE[url]

    pause = THROTTLE_SEC
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(
                url,
                timeout=TIMEOUT,
                allow_redirects=True,
                stream=True,
                headers={**HEADERS, "Referer": referer},
            )
            # Quick short-circuit for retryable statuses
            if _should_retry_status(r.status_code):
                # Respect Retry-After if present
                ra = r.headers.get("Retry-After")
                if ra:
                    try:
                        pause = max(pause, float(ra))
                    except Exception:
                        pass
                if attempt < MAX_RETRIES:
                    _sleep_with_jitter(pause)
                    pause *= BACKOFF
                    continue
            r.raise_for_status()
            content = r.raw.read()
            img = _open_image_from_bytes(content)
            DOWNLOAD_CACHE[url] = img
            # be nice even on success
            _sleep_with_jitter(THROTTLE_SEC * 0.6)
            return img
        except requests.HTTPError as e:
            if getattr(e.response, "status_code", None) and _should_retry_status(e.response.status_code) and attempt < MAX_RETRIES:
                _sleep_with_jitter(pause)
                pause *= BACKOFF
                continue
            raise
        except Exception:
            if attempt < MAX_RETRIES:
                _sleep_with_jitter(pause)
                pause *= BACKOFF
                continue
            raise
