#!/usr/bin/env python3
"""Validate the frozen quantum-access literature data and its paired article output."""
from __future__ import annotations

import argparse
from datetime import date, datetime
from html import unescape
from html.parser import HTMLParser
import itertools
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_data/quantum_access_literature.json"
SLUG = "Quantum-Access-Resistance-Literature-Map"
TRANSLATION_KEY = "quantum-access-resistance-literature-map"
VIEWS = {"citations", "coupling", "timeline"}
SEEDS = {
    "hung2025": ("10.1109/led.2025.3614369", "On-Current Degradation in Ultra-Scaled Nanosheet FETs With S/D Underlap Doping"),
    "venugopal2004": ("10.1063/1.1631754", "Quantum mechanical analysis of channel access geometry and series resistance in nanoscale transistors"),
    "laux2004": ("10.1063/1.1695597", "Analysis of quantum ballistic electron transport in ultrasmall silicon devices including space-charge and geometric effects"),
    "ahn2023": ("10.1109/iedm45741.2023.10413710", "Geometric Scattering Describing Mode-Coupling Effects in Non-Uniform Cross-Sections for Non-Equilibrium Green’s Function and Multi Subband Boltzmann Transport Equation Solvers"),
    "luisier2006": ("10.1063/1.2244522", "Quantum transport in two- and three-dimensional nanoscale transistors: Coupled mode effects in the nonequilibrium Green’s function formalism"),
    "pourghaderi2026": ("10.1109/led.2026.3681884", "On Effectiveness of Constriction Barrier in Nanotransistors"),
}
CONTEXT = {"kim2025", "shared2021", "qdame2002", "berrada2015", "underlap2005"}
SPECIAL_EDGE = ("kim2025", "underlap2005")
EXPECTED_PAPERS = 36
BRANCHES = {"foundations", "mode-coupling", "scattering", "device-design"}
RICH_FIELDS = ("question", "method", "findings", "constriction", "limits", "reading_scope")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def doi(value):
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value.strip(), flags=re.I).lower()


def https(value):
    parsed = urlsplit(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def rich_text(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list) and value and all(isinstance(s, str) and s.strip() for s in value):
        return " ".join(value)
    return ""


def check_data(data):
    nodes, edges = data["nodes"], data["edges"]
    require(len(nodes) == EXPECTED_PAPERS and len(edges) > 21, "Expected the expanded 36-paper collection and its added directed citations")
    by_id = {n["id"]: n for n in nodes}
    require(len(by_id) == len(nodes), "Duplicate paper IDs")
    require(set(SEEDS) | CONTEXT <= set(by_id), "An original paper identity is missing")
    require(len({doi(n["doi"]) for n in nodes}) == len(nodes), "Duplicate paper DOIs")
    require(data["seed_ids"] == list(SEEDS), "The six seeds or their user-supplied order changed")
    require(set(data["context_ids"]) == set(by_id) - set(SEEDS), "Context IDs must identify every non-seed paper")
    require(len(set(data["context_ids"])) == len(data["context_ids"]), "Duplicate context paper IDs")
    require(data["edge_direction"] == "citing_to_cited", "Citation direction must be citing → cited")
    date.fromisoformat(data["snapshot_date"])
    require(data["timezone"] == "Asia/Seoul", "Snapshot timezone must be explicit")
    for n in nodes:
        key = n["id"]
        require(bool(n["seed"]) == (key in SEEDS), f"Incorrect seed flag: {key}")
        require(n["label"] and n["journal"] and n["title"], f"Incomplete bibliography: {key}")
        require(isinstance(n["authors"], list) and all(isinstance(a, str) and a for a in n["authors"]), f"Invalid author metadata: {key}")
        require(doi(n["url"]) == doi(n["doi"]) and https(n["url"]), f"Invalid DOI link: {key}")
        require(https(n["openalex_id"]) and n["openalex_id"].startswith("https://openalex.org/W"), f"Invalid OpenAlex ID: {key}")
        require(https(n["crossref_url"]) and doi(unquote(urlsplit(n["crossref_url"]).path.removeprefix("/works/"))) == doi(n["doi"]), f"Crossref URL does not identify paper: {key}")
        timestamp = datetime.fromisoformat(n["retrieved_at_utc"])
        require(timestamp.tzinfo is not None, f"Missing retrieval timezone: {key}")
        require(all(isinstance(n[k], int) and n[k] >= 0 for k in ("cited_by_count", "crossref_references", "openalex_references")), f"Invalid snapshot counts: {key}")
        refs = n["reference_dois"]
        require(refs == sorted(set(doi(r) for r in refs)), f"Reference DOIs must be normalized, unique, and sorted: {key}")
        require(len(refs) <= n["crossref_references"], f"Known DOI references exceed publisher reference count: {key}")
        for language in ("en", "ko"):
            localized = n["i18n"][language]
            require(localized.get("label") == n["label"], f"Localized author/year label differs: {key}:{language}")
            require(all(isinstance(localized.get(k), str) and localized[k].strip() for k in ("topic", "summary", "limitation")), f"Missing {language} paper descriptions: {key}")
            detailed = n["rich"][language]
            require(all(rich_text(detailed.get(k)) for k in RICH_FIELDS), f"Missing {language} paper analysis or reading scope: {key}")
        require(isinstance(n["branches"], list) and n["branches"] and set(n["branches"]) <= BRANCHES, f"Unknown or missing research branch: {key}")
        require(n["sources"] and all(s.get("label") and https(s.get("url", "")) for s in n["sources"]), f"Missing detailed-analysis source links: {key}")
        if key in SEEDS:
            expected_doi, expected_title = SEEDS[key]
            require(doi(n["doi"]) == expected_doi and n["title"] == expected_title, f"Seed DOI/title mismatch: {key}")
            require(n["authors"], f"Missing seed authors: {key}")
            require(n["seed_index"] == list(SEEDS).index(key) + 1, f"Seed order mismatch: {key}")
    require(by_id["venugopal2004"]["year"] == 2004 and by_id["underlap2005"]["year"] == 2005, "Use journal issue years for the known date discrepancies")
    latest = by_id["pourghaderi2026"]
    require("abstract" in latest["i18n"]["en"]["limitation"].lower(), "Pourghaderi2026 must retain the abstract-only evidence limit")
    require("초록" in latest["i18n"]["ko"]["limitation"], "Korean description lost the abstract-only evidence limit")
    require("nominal overlap" in " ".join(latest["i18n"]["en"].values()).lower(), "Pourghaderi2026 nominal-overlap condition is missing")
    require("abstract" in rich_text(latest["rich"]["en"]["reading_scope"]).lower(), "Detailed 2026 reading scope must remain abstract-only")
    require("초록" in " ".join(rich_text(latest["rich"]["ko"][k]) for k in ("reading_scope", "limits")), "Korean detailed 2026 analysis must retain the abstract-only limit")
    require(any("does not prove" in c for c in data["caveats"]), "Missing caution about omitted citation edges")
    require(any("qualitative" in c or ("historical" in c.lower() and "not publication-growth statistics" in c) for c in data["caveats"]), "Historical stages must retain their interpretive, non-volume scope")

    by_doi = {doi(n["doi"]): n["id"] for n in nodes}
    deposited = {(n["id"], by_doi[r]) for n in nodes for r in n["reference_dois"] if r in by_doi}
    actual = {(e["source"], e["target"]) for e in edges}
    require(len(actual) == len(edges), "Duplicate directed citation edges")
    require(actual == deposited | {SPECIAL_EDGE}, "Displayed citations differ from known DOI references plus the documented bibliographic match")
    for e in edges:
        pair = (e["source"], e["target"])
        require(pair[0] in by_id and pair[1] in by_id and pair[0] != pair[1], f"Invalid citation endpoints: {pair}")
        require(e["reference_key"] and https(e["evidence_url"]), f"Missing citation receipt: {pair}")
        require(isinstance(e["openalex_confirmed"], bool), f"Missing secondary-index confirmation state: {pair}")
        if pair == SPECIAL_EDGE:
            require(e["evidence_type"] == "publisher_fulltext_bibliographic_match" and e["reference_key"] == "435_CR30", "Kim→Trivedi must retain its DOI-less reference match provenance")
            require("omits DOI" in e.get("note", ""), "Kim→Trivedi must disclose the deposited reference's missing DOI")
        else:
            require(e["evidence_type"] == "publisher_deposited_reference_doi", f"Incorrect citation evidence type: {pair}")
            require(e["evidence_url"] == by_id[pair[0]]["crossref_url"], f"Citation receipt must identify the citing paper: {pair}")
    stages = data["stages"]
    require(len(stages) >= 5, "Historical stages are missing")
    require(all(set(s["ids"]) <= set(by_id) and s["period"] for s in stages), "Unknown paper in historical stages")
    for stage in stages:
        for language in ("en", "ko"):
            require(all(stage["i18n"][language].get(k) for k in ("label", "text")), f"Missing {language} historical-stage description: {stage['period']}")
    require(all(https(s["url"]) and s["label"] for s in data["sources"]), "Incomplete source links")
    return by_id


def check_coupling(data, by_id):
    expected_pairs = {frozenset(p) for p in itertools.combinations(data["seed_ids"], 2)}
    actual_pairs = set()
    nonzero = 0
    for row in data["bibliographic_coupling"]:
        pair = (row["source"], row["target"])
        key = frozenset(pair)
        require(key in expected_pairs and key not in actual_pairs, f"Invalid or duplicate coupling pair: {pair}")
        actual_pairs.add(key)
        left, right = (set(by_id[p]["reference_dois"]) for p in pair)
        shared = sorted(left & right)
        expected_cosine = len(shared) / math.sqrt(len(left) * len(right)) if left and right else 0.0
        require(row["shared_dois"] == shared and row["shared_reference_doi_count"] == len(shared), f"Incorrect shared reference DOI intersection: {pair}")
        require(set(shared) <= {doi(n["doi"]) for n in by_id.values()}, f"Shared-foundation paper details are absent from the expanded collection: {pair}")
        require(math.isclose(row["cosine"], round(expected_cosine, 4), abs_tol=0.000001), f"Incorrect cosine normalization: {pair}")
        require("DOI-less" in row["scope"] and "publisher" in row["scope"], f"Coupling scope must state the observed DOI subset: {pair}")
        nonzero += bool(shared)
    require(actual_pairs == expected_pairs, "Coupling data must cover every pair of the six seeds")
    require(nonzero == 12 and len(actual_pairs) - nonzero == 3, "The frozen seed-pair overlap results changed")
    return {"pairs": len(actual_pairs), "nonzero_pairs": nonzero, "zero_pairs": len(actual_pairs) - nonzero}


def check_snapshot(data, path):
    old = read_json(path)
    for key in ("snapshot_date", "timezone", "seed_ids", "edge_direction", "bibliographic_coupling"):
        require(data[key] == old[key], f"Original snapshot changed: {key}")
    original = {n["id"]: n for n in old["nodes"]}
    require(set(original) <= {n["id"] for n in data["nodes"]}, "Original snapshot paper identities changed")
    for node in data["nodes"]:
        if node["id"] not in original:
            continue
        for key, value in original[node["id"]].items():
            if key not in ("i18n", "rich", "sources", "branches"):
                require(node[key] == value, f"Original snapshot changed: {node['id']}.{key}")
    expanded_edges = {(e["source"], e["target"]): e for e in data["edges"]}
    for edge in old["edges"]:
        pair = (edge["source"], edge["target"])
        require(pair in expanded_edges and all(expanded_edges[pair].get(k) == v for k, v in edge.items()), f"Original citation receipt changed: {pair}")
    require(set(old["context_ids"]) <= set(data["context_ids"]), "Original contextual papers were removed")


def check_raw(data, by_id, directory):
    raw = {}
    for n in data["nodes"]:
        key = n["id"]
        item = read_json(directory / f"{key}.json")
        raw[key] = item
        message, openalex = item["crossref"]["message"], item["openalex"]
        refs = message.get("reference", [])
        require(doi(message["DOI"]) == doi(n["doi"]), f"Raw Crossref identity mismatch: {key}")
        known = sorted({doi(r["DOI"]) for r in refs if r.get("DOI")})
        require(n["reference_dois"] == known, f"Known reference DOIs differ from raw publisher deposit: {key}")
        require(n["crossref_references"] == message.get("reference-count", len(refs)), f"Publisher reference count changed: {key}")
        require(n["openalex_references"] == len(openalex.get("referenced_works", [])), f"OpenAlex reference count changed: {key}")
        require(n["cited_by_count"] == openalex["cited_by_count"] and n["openalex_id"] == openalex["id"], f"OpenAlex snapshot differs: {key}")
        require(n["retrieved_at_utc"] == item["retrieved_at_utc"], f"Retrieval receipt changed: {key}")
    for n in data["nodes"]:
        key = n["id"]
        message, openalex = raw[key]["crossref"]["message"], raw[key]["openalex"]
        published = message.get("published") or {}
        parts = published.get("date-parts", [[]])
        issue_year = parts[0][0] if parts and parts[0] else openalex.get("publication_year")
        if n["year"] != issue_year:
            evidence = n.get("year_evidence", {})
            require(key == "qdameiedm2002" and n["year"] == 2002 and not published and evidence.get("kind") == "conference_year" and evidence.get("year") == 2002 and https(evidence.get("source_url", "")), f"Unsupported publication-year override: {key}")
            matches = [r for r in raw["kim2025"]["crossref"]["message"]["reference"] if r.get("key") == "435_CR19" and doi(r.get("DOI", "")) == doi(n["doi"])]
            require(len(matches) == 1 and re.search(r"\b2002\b", matches[0].get("unstructured", "")) and "IEDM" in matches[0].get("unstructured", ""), "QDAME conference-year override lacks the cited publisher bibliography receipt")
    for edge in data["edges"]:
        refs = raw[edge["source"]]["crossref"]["message"].get("reference", [])
        matched = [r for r in refs if r.get("key") == edge["reference_key"]]
        require(len(matched) == 1, f"Publisher reference key not found: {edge['source']}:{edge['reference_key']}")
        if edge["evidence_type"] == "publisher_deposited_reference_doi":
            require(doi(matched[0].get("DOI", "")) == doi(by_id[edge["target"]]["doi"]), f"Publisher reference points to a different DOI: {edge['source']}→{edge['target']}")
        else:
            text = matched[0].get("unstructured", "").lower()
            require("nanoscale finfets" in text and "56" in text and "62" in text, "DOI-less Kim→Trivedi bibliographic match lacks matching title/pages")
        confirmed = by_id[edge["target"]]["openalex_id"] in raw[edge["source"]]["openalex"].get("referenced_works", [])
        require(edge["openalex_confirmed"] == confirmed, f"OpenAlex confirmation mismatch: {edge['source']}→{edge['target']}")


def check_collection(data, by_id, evidence_dir=None):
    collection = data["collection"]
    require(collection["seed_papers"] == len(data["seed_ids"]) and collection["context_papers"] == len(data["context_ids"]) and collection["all_papers"] == len(by_id) and collection["directed_citations"] == len(data["edges"]), "Collection summary disagrees with the actual dataset")
    queries = collection["forward_queries"]
    require({q["paper"] for q in queries} == set(SEEDS) | {"kim2025"} and len(queries) == len({q["paper"] for q in queries}), "Forward-query papers must cover the seeds and the contextual Kim2025 paper once each")
    cap = collection["forward_limit_per_query"]
    require(isinstance(cap, int) and cap > 0, "Invalid forward-query record cap")
    unique_ids = set()
    for query in queries:
        key = query["paper"]
        require(0 <= query["retrieved_records"] <= min(cap, query["matching_records"]), f"Invalid forward-query counts: {key}")
        require(query["snapshot_date"] == data["snapshot_date"], f"Forward-query date differs: {key}")
        parsed = parse_qs(urlsplit(query["source_url"]).query)
        filters = parsed.get("filter", [""])[0].split(",")
        require(https(query["source_url"]) and f"cites:{by_id[key]['openalex_id'].rsplit('/', 1)[-1]}" in filters and f"to_publication_date:{data['snapshot_date']}" in filters and parsed.get("per-page") == [str(cap)], f"Forward-query URL does not match its scope: {key}")
        if evidence_dir:
            receipt = read_json(evidence_dir / f"{key}-citers.json")
            response = receipt["response"]
            require(receipt["source_url"] == query["source_url"] and receipt["cited_paper"] == key, f"Forward-query receipt identity differs: {key}")
            require(response["meta"]["count"] == query["matching_records"] and len(response["results"]) == query["retrieved_records"], f"Forward-query receipt counts differ: {key}")
            unique_ids.update(r["id"] for r in response["results"])
    require(0 <= collection["forward_unique_records_retrieved"] <= sum(q["retrieved_records"] for q in queries), "Invalid unique forward-record count")
    if evidence_dir:
        require(len(unique_ids) == collection["forward_unique_records_retrieved"], "Unique forward records differ from the saved query receipts")
    return {"forward_queries": len(queries), "unique_records_retrieved": collection["forward_unique_records_retrieved"], "limit_per_query": cap, "queries_with_more_matches_than_retrieved": sum(q["matching_records"] > q["retrieved_records"] for q in queries), "query_receipts_checked": evidence_dir is not None}


def front_matter(text):
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n", text, re.S)
    require(match is not None, "Article front matter is missing")
    metadata = {m[1]: m[2].strip().strip("\"'") for m in re.finditer(r"^([a-z_]+):\s*(.*?)\s*$", match[1], re.M)}
    if "published" in metadata:
        published = re.search(r"^published:[ \t]*(true|false)[ \t]*(?:#[^\n]*)?$", match[1], re.M | re.I)
        require(published is not None, "Article published must be an unquoted YAML boolean")
        metadata["published"] = published[1].lower()
    return metadata


def check_articles():
    template = (ROOT / "_includes/interactive/quantum-access-literature.html").read_text(encoding="utf-8")
    root = re.search(r'<section\b[^>]*\bdata-qal-root\b[^>]*>', template)
    require(root is not None and "data-local-editor-skip" in root[0] and "tex2jax_ignore" in root[0], "The data-owned literature explorer must be isolated from local text editing and MathJax")
    result = {}
    for language, suffix in (("en", ""), ("ko", "-KR")):
        matches = list((ROOT / "_posts").glob(f"**/*-{SLUG}{suffix}.md"))
        require(len(matches) == 1, f"Expected one {language} literature-map article")
        text = matches[0].read_text(encoding="utf-8")
        metadata = front_matter(text)
        require(metadata.get("lang") == language, f"Incorrect article language: {matches[0].name}")
        require(metadata.get("translation_key") == TRANSLATION_KEY, "Article translation key differs")
        require(metadata.get("published") in {"true", "false"}, "The literature-map articles must declare a boolean published state")
        require(metadata.get("interactive") == "quantum-access-literature", "Missing article-specific interactive asset flag")
        require(metadata.get("topic") == "quantum-transport", "Literature map topic differs")
        require(metadata.get("title") and metadata.get("description"), "Article title/description is missing")
        require("include interactive/quantum-access-literature.html" in text, "Article does not include the shared literature map")
        result[language] = {"path": str(matches[0]), "metadata": metadata}
    for key in ("date", "topic", "categories", "tags", "translation_key", "published", "interactive"):
        require(result["en"]["metadata"].get(key) == result["ko"]["metadata"].get(key), f"Paired article metadata differs: {key}")
    return result


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.elements = []
        self.scripts = []
        self.current_script = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.elements.append((tag, attrs))
        if tag == "script":
            self.current_script = {"attrs": attrs, "text": ""}
            self.scripts.append(self.current_script)

    def handle_data(self, text):
        if self.current_script is not None:
            self.current_script["text"] += text

    def handle_endtag(self, tag):
        if tag == "script":
            self.current_script = None


def check_site(data, directory, *, published=True, include_unpublished=False):
    exported_path = directory / "assets/data/quantum-access-literature.json"
    require(read_json(exported_path) == data, "Built downloadable JSON differs from the native Jekyll dataset")
    if not published and not include_unpublished:
        routes = [f"/posts/{SLUG}{suffix}/" for suffix in ("", "-KR")]
        for route in routes:
            require(not (directory / route.lstrip("/") / "index.html").exists(), f"Unpublished article route is still generated: {route}")
        indexes = ("index.html", "sitemap.xml", "feed.xml", "assets/js/data/search.json", "topics/quantum-transport/index.html")
        for relative in indexes:
            require((directory / relative).is_file(), f"Missing discovery index required for unpublication verification: {relative}")
        checked = 0
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix not in {".html", ".xml", ".json"}:
                continue
            text = unquote(unescape(path.read_text(encoding="utf-8")).replace("\\/", "/"))
            require(not any(route in text for route in routes), f"Unpublished article still appears in generated discovery content: {path.relative_to(directory)}")
            checked += 1
        return {"publication_mode": "unpublished", "routes_absent": routes, "discovery_indexes": list(indexes), "discovery_documents_checked": checked, "download_dataset_parity": True}
    pages = {}
    for language, suffix in (("en", ""), ("ko", "-KR")):
        route = f"/posts/{SLUG}{suffix}/"
        path = directory / route.lstrip("/") / "index.html"
        require(path.exists(), f"Missing literature-map route ({'published' if published else '--unpublished preview'}): {route}")
        page = Page(path.read_text(encoding="utf-8"))
        require(any(tag == "html" and a.get("lang") == language for tag, a in page.elements), f"Rendered article language differs: {route}")
        roots = [a for _, a in page.elements if "data-qal-root" in a]
        require(len(roots) == 1 and roots[0].get("data-lang") == language, f"Missing or incorrectly localized map root: {route}")
        require("data-local-editor-skip" in roots[0] and "tex2jax_ignore" in roots[0].get("class", "").split(), f"The explorer lost its editor/MathJax boundary: {route}")
        scripts = [s for s in page.scripts if "data-qal-data" in s["attrs"]]
        require(len(scripts) == 1 and scripts[0]["attrs"].get("type") == "application/json", f"Missing native embedded data: {route}")
        require(json.loads(scripts[0]["text"]) == data, f"Embedded map data differs from the native dataset: {route}")
        views = {a["data-qal-view"] for tag, a in page.elements if tag == "button" and "data-qal-view" in a}
        panels = {a["data-qal-panel"] for _, a in page.elements if "data-qal-panel" in a}
        require(views == VIEWS and panels == VIEWS, f"Three literature views are not available: {route}")
        for attribute in ("data-qal-paper", "data-qal-context", "data-qal-pair", "data-qal-inspector", "data-qal-share", "data-qal-search", "data-qal-branch", "data-qal-scope", "data-qal-coupling-graph"):
            require(any(attribute in a for _, a in page.elements), f"Missing map interaction {attribute}: {route}")
        options = {a.get("value") for tag, a in page.elements if tag == "option"}
        require({n["id"] for n in data["nodes"]} <= options, f"Paper selector omits an expanded-collection paper: {route}")
        require(BRANCHES <= options and {"neighbors", "overview"} <= options, f"Research-branch or network-scope options are missing: {route}")
        bibliography = [a for tag, a in page.elements if tag == "a" and "data-qal-bibliography-paper" in a]
        ordered_papers = sorted(data["nodes"], key=lambda n: (n["year"], n["label"], n["id"]))
        require([a["data-qal-bibliography-paper"] for a in bibliography] == [n["id"] for n in ordered_papers], f"Bibliography must include every paper in stable year/label/ID order: {route}")
        require([doi(a.get("href", "")) for a in bibliography] == [doi(n["doi"]) for n in ordered_papers], f"Bibliography DOI links differ from their ordered paper identities: {route}")
        refs = [a.get("href", a.get("src", "")) for _, a in page.elements]
        for asset in ("/assets/css/quantum-access-literature.css", "/assets/js/quantum-access-literature.js", "/assets/data/quantum-access-literature.json"):
            urls = [r for r in refs if urlsplit(r).path == asset]
            require(urls and (directory / asset.lstrip("/")).is_file(), f"Missing literature asset/link {asset}: {route}")
            if not asset.endswith(".json"):
                require(any(re.search(r"(?:[?&])v=[a-f0-9]{12}(?:&|$)", r) for r in urls), f"Interactive asset is missing its source-content version: {asset}")
        counterpart = f"/posts/{SLUG}{'-KR' if language == 'en' else ''}/"
        require(any(urlsplit(r).path == counterpart for r in refs), f"Translation link missing: {route}")
        pages[language] = route
    return {"publication_mode": "published" if published else "unpublished_preview", "routes": pages, "embedded_dataset_parity": True, "download_dataset_parity": True, "stable_bibliography_order": True, "views": sorted(VIEWS), "selectable_papers": len(data["nodes"]), "research_branches": sorted(BRANCHES)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, help="Also inspect a production Jekyll build")
    parser.add_argument("--include-unpublished", action="store_true", help="With --site-dir, inspect articles included by Jekyll --unpublished instead of requiring unpublished routes to be absent")
    parser.add_argument("--evidence-dir", type=Path, help="Compare each paper's id.json raw Crossref/OpenAlex retrieval receipt; other query receipts are ignored")
    parser.add_argument("--snapshot", type=Path, help="Preserve the original study.json papers and citations as a subset; permit enriched descriptions and added papers")
    args = parser.parse_args()
    if args.include_unpublished and not args.site_dir:
        parser.error("--include-unpublished requires --site-dir")
    report = {"status": "FAIL", "checks_passed": []}
    try:
        data = read_json(DATA)
        by_id = check_data(data)
        report["checks_passed"].append("paper_identity_bilingual_descriptions_and_citation_provenance")
        report["coupling"] = check_coupling(data, by_id)
        report["checks_passed"].append("independent_reference_intersections_and_cosine")
        if args.snapshot:
            check_snapshot(data, args.snapshot)
            report["checks_passed"].append("original_snapshot_preservation")
        if args.evidence_dir:
            check_raw(data, by_id, args.evidence_dir)
            report["checks_passed"].append("raw_publisher_reference_keys_and_openalex_receipts")
        report["collection"] = check_collection(data, by_id, args.evidence_dir)
        report["checks_passed"].append("collection_scope_and_forward_query_counts")
        articles = check_articles()
        report["checks_passed"].append("paired_article_publication_metadata_and_editor_boundary")
        published = articles["en"]["metadata"]["published"] == "true"
        report["published"] = published
        report["dataset"] = {"snapshot_date": data["snapshot_date"], "papers": len(by_id), "seeds": len(data["seed_ids"]), "context": len(data["context_ids"]), "directed_citations": len(data["edges"])}
        report["dataset"]["citation_evidence"] = {kind: sum(e["evidence_type"] == kind for e in data["edges"]) for kind in sorted({e["evidence_type"] for e in data["edges"]})}
        report["dataset"]["openalex_confirmed_citations"] = sum(e["openalex_confirmed"] for e in data["edges"])
        report["dataset"]["bilingual_detailed_papers"] = sum(all(n.get("rich", {}).get(lang) for lang in ("en", "ko")) for n in by_id.values())
        report["articles"] = {lang: item["path"] for lang, item in articles.items()}
        if args.site_dir:
            report["site"] = check_site(data, args.site_dir.resolve(), published=published, include_unpublished=args.include_unpublished)
            report["checks_passed"].append("paired_rendered_routes_controls_assets_and_data_parity" if published or args.include_unpublished else "unpublished_routes_and_discovery_absence")
        report["status"] = "PASS"
        report["limits"] = ["A zero known-DOI intersection does not establish an absence of shared references.", "Static checks do not establish browser interaction, responsive rendering, or hosted parity."]
    except (ValueError, KeyError, TypeError, OSError) as error:
        report["error"] = str(error)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
