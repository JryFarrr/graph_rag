import ast
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# The dataset lives in ../data/Pubmed relative to the repo root
ROOT_DIR = Path(__file__).resolve().parents[2]
PUBMED_DIR = ROOT_DIR.joinpath("data", "Pubmed")
PRIMARY_CSV = PUBMED_DIR / "PubMed Multi Label Text Classification Dataset.csv"
PROCESSED_CSV = PUBMED_DIR / "PubMed Multi Label Text Classification Dataset Processed.csv"

# Prefer the smaller CSV if available; fall back to the processed one.
PUBMED_CSV = PRIMARY_CSV if PRIMARY_CSV.exists() else PROCESSED_CSV


def ensure_dataset_exists():
    if not PUBMED_CSV.exists():
        raise FileNotFoundError(f"PubMed CSV not found. Expected at {PUBMED_CSV}")


def parse_mesh_list(raw: str) -> List[str]:
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, str)]
    return []


def iter_pubmed_rows(max_rows: int | None = None):
    ensure_dataset_exists()
    with PUBMED_CSV.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            yield row
            if max_rows and idx + 1 >= max_rows:
                break


def build_article_uri(pmid: str | None) -> str | None:
    if not pmid:
        return None
    return f"http://example.org/pubmed/{pmid}"


def score_match(text: str, query: str) -> int:
    return text.lower().count(query.lower())


def search_articles_local(query_text: str, limit: int = 20, max_rows: int = 5000) -> List[Dict]:
    """
    Lightweight article search over the local CSV.
    Uses substring frequency in title + abstract as a simple relevance score.
    """
    results: List[Tuple[int, Dict]] = []
    for row in iter_pubmed_rows(max_rows=max_rows):
        title = row.get("Title", "")
        abstract = row.get("abstractText", "")
        mesh_major = parse_mesh_list(row.get("meshMajor", ""))
        score = score_match(title, query_text) + score_match(abstract, query_text)
        if score == 0:
            continue

        results.append(
            (
                score,
                {
                    "article_URI": build_article_uri(row.get("pmid")),
                    "title": title,
                    "abstractText": abstract,
                    "meshMajor": mesh_major,
                    "pmid": row.get("pmid"),
                    "score": score,
                },
            )
        )

    # Sort by score descending and keep top N
    top = sorted(results, key=lambda item: item[0], reverse=True)[:limit]
    return [item[1] for item in top]


def search_tags_local(query_text: str, limit: int = 20, max_rows: int = 5000) -> List[Dict]:
    """
    Collect tag occurrences by scanning meshMajor lists and counting matches.
    """
    counter: Counter[str] = Counter()
    for row in iter_pubmed_rows(max_rows=max_rows):
        tags = parse_mesh_list(row.get("meshMajor", ""))
        for tag in tags:
            if query_text.lower() in tag.lower():
                counter[tag] += 1

    most_common = counter.most_common(limit)
    return [{"Tag": tag, "Score": count} for tag, count in most_common]

