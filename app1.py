import ast
import csv
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from urllib.parse import quote

import streamlit as st
from langchain_openai import ChatOpenAI

from query_functions.rdf_queries import (
    RDF_DEFAULT_PATH,
    get_all_narrower_concepts,
    get_concept_triples_for_term,
    query_rdf,
    sanitize_term,
)
from query_functions.weaviate_queries import search_articles_local, search_tags_local


st.set_page_config(
    page_title="Article + MeSH Explorer",
    layout="wide",
)

LOCAL_RDF_DEFAULT = Path(os.getenv("LOCAL_RDF_FILE", RDF_DEFAULT_PATH))
ROOT_DIR = Path(__file__).resolve().parent.parent
CSV_DEFAULT_PATH = ROOT_DIR.joinpath("data", "Pubmed", "PubMed Multi Label Text Classification Dataset.csv")

SESSION_DEFAULTS = {
    "article_results": [],
    "article_uris": [],
    "final_terms": [],
    "filtered_articles": [],
    "combined_text": "",
    "tag_results": [],
    "tag_candidates": [],
    "llm_response": "",
}
for key, default in SESSION_DEFAULTS.items():
    st.session_state.setdefault(key, default)


def flatten(values: Dict[str, Sequence[str]]) -> List[str]:
    merged: set[str] = set()
    for _, items in values.items():
        merged.update(items)
    return sorted(merged)


def combine_abstracts(ranked_articles: List[Tuple[str, dict]]) -> str:
    combined_text = " ".join(
        [f"Title: {data['title']} Abstract: {data['abstract']}" for _, data in ranked_articles]
    )
    return combined_text


def summarize_with_llm(text: str, user_prompt: str) -> str:
    """Summarize filtered articles using the configured LLM endpoint."""
    llm = ChatOpenAI(
        base_url=os.getenv("LMSTUDIO_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")),
        api_key=os.getenv("LMSTUDIO_API_KEY", os.getenv("OPENAI_API_KEY", "lm-studio")),
        model=os.getenv("LMSTUDIO_MODEL", os.getenv("OPENAI_MODEL", "qwen2.5-7b-instruct")),
        temperature=0.2,
    )
    prompt = (
        "You are summarizing biomedical articles. "
        "Use the user's prompt as guidance, and ensure you cite titles where relevant.\n\n"
        f"User prompt: {user_prompt}\n\n"
        f"Articles: {text}"
    )
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


def mesh_uri(label: str) -> str:
    return f"http://example.org/mesh/{quote(label.replace(' ', '_'))}"


def build_ttl_from_csv(csv_path: Path, ttl_out: Path) -> int:
    """
    Build a minimal TTL graph from the PubMed CSV.
    Returns the number of articles written.
    """
    from rdflib import Graph, Namespace, URIRef, Literal

    EX = Namespace("http://example.org/")
    SCHEMA = Namespace("http://schema.org/")

    g = Graph()
    g.bind("ex", EX)
    g.bind("schema", SCHEMA)

    count = 0
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pmid = row.get("pmid")
            if not pmid:
                continue
            art_uri = URIRef(f"{EX}pubmed/{pmid}")
            title = row.get("Title", "").strip()
            abstract = row.get("abstractText", "").strip()
            g.add((art_uri, EX.type, EX.Article))
            if title:
                g.add((art_uri, SCHEMA.name, Literal(title)))
            if abstract:
                g.add((art_uri, SCHEMA.description, Literal(abstract)))

            try:
                mesh_values = ast.literal_eval(row.get("meshMajor", "[]"))
            except Exception:
                mesh_values = []
            for label in mesh_values if isinstance(mesh_values, list) else []:
                if not isinstance(label, str) or not label.strip():
                    continue
                term_uri = URIRef(mesh_uri(label.strip()))
                g.add((term_uri, EX.type, EX.MeSHTerm))
                g.add((art_uri, SCHEMA.about, term_uri))

            count += 1

    ttl_out.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(ttl_out, format="turtle")
    return count


def filter_articles_from_csv(article_uris: List[str], terms: List[str], csv_path: Path) -> List[Tuple[str, dict]]:
    """
    Fallback filter: match article_uris against meshMajor in the CSV directly.
    """
    if not csv_path.exists():
        return []

    term_set = {t.lower() for t in terms if t}
    results = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pmid = row.get("pmid")
            if not pmid:
                continue
            uri = f"http://example.org/pubmed/{pmid}"
            if uri not in article_uris:
                continue
            try:
                mesh_values = ast.literal_eval(row.get("meshMajor", "[]"))
            except Exception:
                mesh_values = []

            mesh_set = {m.lower() for m in mesh_values if isinstance(m, str)}
            matched = term_set.intersection(mesh_set)
            if not matched:
                continue

            results.append(
                (
                    uri,
                    {
                        "title": row.get("Title"),
                        "abstract": row.get("abstractText"),
                        "datePublished": row.get("datePublished"),
                        "access": row.get("access"),
                        "meshTerms": matched,
                    },
                )
            )

    ranked = sorted(results, key=lambda item: len(item[1]["meshTerms"]), reverse=True)
    return ranked[:10]


st.title("Graph RAG - Article and MeSH Explorer")
tab_search, tab_terms, tab_filter = st.tabs(
    ["Search Articles", "Refine Terms", "Filter & Summarize"]
)

# --- TAB 1: Search Articles (using local PubMed CSV) ---
with tab_search:
    st.header("Search Articles (local PubMed CSV)")
    query_text = st.text_input("Enter your search term (e.g., Mouth Neoplasms):", key="vector_search")

    if st.button("Search Articles", key="search_articles_btn"):
        if not query_text.strip():
            st.warning("Please enter a search term first.")
        else:
            try:
                article_results = search_articles_local(query_text, limit=20, max_rows=8000)

                st.session_state.article_uris = [
                    result.get("article_URI")
                    for result in article_results
                    if result.get("article_URI")
                ]

                st.session_state.article_results = [
                    {
                        "Title": result.get("title", "N/A"),
                        "Abstract": (result.get("abstractText", "N/A")[:120] + "..."),
                        "Score": result.get("score", 0),
                        "MeSH Terms": ", ".join(result.get("meshMajor", [])),
                    }
                    for result in article_results
                ]

                tag_results = search_tags_local(query_text, limit=20, max_rows=8000)
                st.session_state.tag_results = tag_results
                st.session_state.tag_candidates = [
                    entry["Tag"] for entry in tag_results if entry.get("Tag")
                ]
            except Exception as exc:
                st.error(f"Error during article search: {exc}")

    if st.session_state.article_results:
        st.write("**Search Results for Articles:**")
        st.dataframe(st.session_state.article_results, use_container_width=True)
    else:
        st.info("No articles found yet.")

    st.subheader("Suggested tags (from meshMajor in CSV)")
    if st.session_state.tag_results:
        st.dataframe(st.session_state.tag_results, use_container_width=True)
    else:
        st.caption("Run a search to see suggested tags.")


# --- TAB 2: Refine Terms ---
with tab_terms:
    st.header("Refine Tags / MeSH Terms")
    st.caption("Start from suggested tags, then expand via MeSH API to alternative and narrower concepts.")

    suggested_tags = st.session_state.tag_candidates
    default_select = suggested_tags[:5]
    selected_tags = st.multiselect(
        "Pick tags from search to refine",
        options=suggested_tags,
        default=default_select,
    )
    manual_seeds = st.text_input("Add extra tag/MeSH seed (comma-separated)", key="mesh_seed")
    depth = st.slider("Depth for narrower concepts", 1, 3, 2)

    if st.button("Build refined term list", key="expand_mesh_btn"):
        try:
            seeds = set(selected_tags)
            if manual_seeds.strip():
                seeds.update({sanitize_term(item) for item in manual_seeds.split(",") if item.strip()})

            if not seeds:
                st.warning("Select or type at least one seed tag/term first.")
                st.stop()

            refined_terms: set[str] = set()
            narrower_map: Dict[str, List[str]] = {}

            for seed in sorted(seeds):
                triples = get_concept_triples_for_term(seed)
                narrower = get_all_narrower_concepts(seed, depth=depth)
                narrower_map.update(narrower)

                refined_terms.update(triples)
                refined_terms.update(flatten(narrower))
                refined_terms.add(sanitize_term(seed))

            st.session_state.final_terms = sorted(t for t in refined_terms if t)
            st.success(f"Collected {len(st.session_state.final_terms)} refined terms.")
            if st.session_state.final_terms:
                st.write(st.session_state.final_terms)
            if narrower_map:
                st.caption("Narrower concept map")
                st.json(narrower_map)
        except Exception as exc:
            st.error(f"Failed to expand MeSH terms: {exc}")

    custom_terms = st.text_area(
        "Optional: override/add MeSH terms (one per line)",
        value="\n".join(st.session_state.final_terms),
        height=140,
    )
    if st.button("Update term list", key="update_terms_btn"):
        cleaned = [sanitize_term(line) for line in custom_terms.splitlines() if line.strip()]
        st.session_state.final_terms = cleaned
        st.success(f"Updated term list with {len(cleaned)} items.")


# --- TAB 3: Filter & Summarize ---
with tab_filter:
    st.header("Filter Articles using MeSH terms and RDF graph")
    csv_path_input = st.text_input(
        "Path to PubMed CSV (source to build TTL if missing)",
        value=str(CSV_DEFAULT_PATH),
    )
    rdf_path_input = st.text_input(
        "Path to local RDF .ttl file",
        value=str(LOCAL_RDF_DEFAULT),
        help="This file is queried to filter the articles with the selected MeSH terms.",
    )

    ttl_path = Path(rdf_path_input)
    if not ttl_path.exists():
        st.warning("TTL file not found. You can build it from the PubMed CSV.")
        if st.button("Build TTL from CSV", key="build_ttl_btn"):
            try:
                written = build_ttl_from_csv(Path(csv_path_input), ttl_path)
                st.success(f"Built TTL with {written} articles at {ttl_path}")
            except Exception as exc:
                st.error(f"Failed to build TTL: {exc}")

    if st.session_state.article_uris:
        st.caption("Article URIs captured from search")
        st.code("\n".join(map(str, st.session_state.article_uris)))
    else:
        st.info("Search for articles first to collect their URIs.")

    if st.session_state.final_terms:
        st.caption("Current MeSH terms")
        st.write(st.session_state.final_terms)
    else:
        st.info("Build or enter MeSH terms in the previous tab.")

    if st.button("Filter Articles", key="filter_articles_btn"):
        try:
            if not st.session_state.article_uris:
                st.warning("No article URIs available. Run a search in tab 1 first.")
                st.stop()

            final_terms = st.session_state.final_terms
            if not final_terms:
                st.warning("No MeSH terms available. Expand or add terms in tab 2 first.")
                st.stop()

            article_uris_string = ", ".join([f"<{str(uri)}>" for uri in st.session_state.article_uris])
            sparql_query = """
            PREFIX schema: <http://schema.org/>
            PREFIX ex: <http://example.org/>

            SELECT ?article ?title ?abstract ?datePublished ?access ?meshTerm
            WHERE {{
              ?article a ex:Article ;
                       schema:name ?title ;
                       schema:description ?abstract ;
                       schema:about ?meshTerm .

              OPTIONAL {{ ?article schema:datePublished ?datePublished . }}
              OPTIONAL {{ ?article ex:access ?access . }}

              ?meshTerm a ex:MeSHTerm .

              FILTER (?article IN ({article_uris}))
            }}
            """
            query_text = sparql_query.format(article_uris=article_uris_string)

            top_articles = query_rdf(Path(rdf_path_input), query_text, final_terms)
            st.session_state.filtered_articles = top_articles

            # Fallback to CSV-based filtering if SPARQL returns nothing
            if not top_articles:
                fallback = filter_articles_from_csv(
                    article_uris=st.session_state.article_uris,
                    terms=final_terms,
                    csv_path=Path(csv_path_input),
                )
                if fallback:
                    st.info("No SPARQL matches; showing CSV-based matches instead.")
                    top_articles = fallback
                    st.session_state.filtered_articles = top_articles

            if top_articles:
                st.session_state.combined_text = combine_abstracts(top_articles)
                st.success(f"Found {len(top_articles)} matching articles.")

                display_rows = []
                for article_uri, data in top_articles:
                    display_rows.append(
                        {
                            "Article": str(article_uri),
                            "Title": data["title"],
                            "Abstract": data["abstract"],
                            "Published": data["datePublished"],
                            "Access": data["access"],
                            "Matched MeSH terms": len(data["meshTerms"]),
                        }
                    )
                st.dataframe(display_rows, use_container_width=True)
            else:
                st.info("No articles found for the selected terms.")
        except Exception as exc:
            st.error(f"Error filtering articles: {exc}")

    st.subheader("Summarize filtered articles")
    user_prompt = st.text_input(
        "User prompt for the LLM",
        value="Summarize the key findings from the filtered articles.",
    )
    if st.button("Generate summary", key="summarize_btn"):
        if not st.session_state.combined_text:
            st.warning("Run the filter first to produce combined article text.")
        else:
            try:
                st.session_state.llm_response = summarize_with_llm(st.session_state.combined_text, user_prompt)
            except Exception as exc:
                st.error(f"LLM summarization failed: {exc}")

    if st.session_state.llm_response:
        st.write(st.session_state.llm_response)
