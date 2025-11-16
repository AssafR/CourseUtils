# ESU_202502 Utils v9
# Extracted helper functions from ESU_202502_OneShot_FewShot_Images_Scaffold_v8.ipynb
# You can import this in Colab as:
#   import ESU_202502_utils_v9 as esu_utils


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
