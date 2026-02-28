from __future__ import annotations

import re
from typing import Dict, Iterable
from urllib.parse import urljoin

_ARTIFACT_EXTS = {
    "bif": [".bif", ".bif.gz"],
    "net": [".net"],
    "dsc": [".dsc"],
    "rda": [".rda"],
    "rds": [".rds"],
}


def _network_id_from_filename(name: str) -> str:
    base = name
    for exts in _ARTIFACT_EXTS.values():
        for ext in exts:
            if base.lower().endswith(ext):
                base = base[: -len(ext)]
                break
    base = base.strip()
    return base.replace("-", "_")


def _artifact_key_from_filename(name: str) -> str | None:
    lower = name.lower()
    if lower.endswith(".bif.gz") or lower.endswith(".bif"):
        return "bif_gz" if lower.endswith(".bif.gz") else "bif"
    if lower.endswith(".net"):
        return "net"
    if lower.endswith(".dsc"):
        return "dsc"
    if lower.endswith(".rda"):
        return "rda"
    if lower.endswith(".rds"):
        return "rds"
    return None


def parse_bnlearn_category_html(
    html: str, *, base_url: str | None = None
) -> Dict[str, dict]:
    """
    Parse a bnlearn category HTML page and return per-network artifact URLs.
    The output maps network_id -> {"urls": {artifact: url}, "artifacts": [..]}.
    """
    url_re = re.compile(r"href=[\"\\\']([^\"\\\']+)[\"\\\']", re.IGNORECASE)
    entries: Dict[str, dict] = {}
    for match in url_re.finditer(html):
        href = match.group(1)
        if not href:
            continue
        if base_url:
            href = urljoin(base_url, href)
        name = href.rsplit("/", 1)[-1]
        key = _artifact_key_from_filename(name)
        if key is None:
            continue
        net_id = _network_id_from_filename(name)
        record = entries.setdefault(net_id, {"urls": {}, "artifacts": []})
        record["urls"][key] = href
        art = "bif" if key.startswith("bif") else key
        if art not in record["artifacts"]:
            record["artifacts"].append(art)
    return entries


def update_registry_from_html(
    registry: dict,
    html: str,
    *,
    base_url: str | None = None,
    only: Iterable[str] | None = None,
) -> dict:
    updates = parse_bnlearn_category_html(html, base_url=base_url)
    only_set = {n for n in (only or [])}
    for net_id, payload in updates.items():
        if only_set and net_id not in only_set:
            continue
        meta = registry.setdefault(net_id, {})
        urls = dict(meta.get("urls") or {})
        urls.update(payload.get("urls") or {})
        meta["urls"] = urls
        artifacts = set(meta.get("artifacts") or [])
        artifacts.update(payload.get("artifacts") or [])
        meta["artifacts"] = sorted(artifacts)
    return registry
