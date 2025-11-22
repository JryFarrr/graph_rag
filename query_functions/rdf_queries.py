import os
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.parse import quote

try:
    from rdflib import Graph, URIRef
except Exception:
    Graph = None
    URIRef = None

try:
    from SPARQLWrapper import JSON, SPARQLWrapper
except Exception:
    SPARQLWrapper = None
    JSON = None


ROOT_DIR = Path(__file__).resolve().parents[2]
RDF_DEFAULT_PATH = ROOT_DIR.joinpath("data", "Pubmed", "PubMedGraph.ttl")
MESH_BASE_NAMESPACE = os.getenv("MESH_BASE_NAMESPACE", "http://example.org/mesh/")


def sanitize_term(term: str) -> str:
    return term.strip().replace('"', "").replace("\n", " ")


def convert_to_uri(term: str, base_namespace: str = MESH_BASE_NAMESPACE):
    if URIRef is None:
        raise ImportError("rdflib is required to convert terms into URIs.")
    safe_term = quote(term.replace(" ", "_"))
    return URIRef(f"{base_namespace}{safe_term}")


def get_concept_triples_for_term(term: str) -> List[str]:
    if SPARQLWrapper is None or JSON is None:
        raise ImportError("Install SPARQLWrapper to expand MeSH terms.")

    term = sanitize_term(term)
    sparql = SPARQLWrapper("https://id.nlm.nih.gov/mesh/sparql")
    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
    PREFIX mesh: <http://id.nlm.nih.gov/mesh/>

    SELECT ?subject ?p ?pLabel ?o ?oLabel
    FROM <http://id.nlm.nih.gov/mesh>
    WHERE {{
        ?subject rdfs:label "{term}"@en .
        ?subject ?p ?o .
        FILTER(CONTAINS(STR(?p), "concept"))
        OPTIONAL {{ ?p rdfs:label ?pLabel . }}
        OPTIONAL {{ ?o rdfs:label ?oLabel . }}
    }}
    """
    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        triples = set()
        for result in results["results"]["bindings"]:
            obj_label = result.get("oLabel", {}).get("value", "No label")
            triples.add(sanitize_term(obj_label))

        triples.add(term)
        return list(triples)

    except Exception as exc:  # pragma: no cover
        return [term]


def get_narrower_concepts_for_term(term: str) -> List[str]:
    if SPARQLWrapper is None or JSON is None:
        raise ImportError("Install SPARQLWrapper to expand MeSH terms.")

    term = sanitize_term(term)
    sparql = SPARQLWrapper("https://id.nlm.nih.gov/mesh/sparql")
    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
    PREFIX mesh: <http://id.nlm.nih.gov/mesh/>

    SELECT ?narrowerConcept ?narrowerConceptLabel
    WHERE {{
        ?broaderConcept rdfs:label "{term}"@en .
        ?narrowerConcept meshv:broaderDescriptor ?broaderConcept .
        ?narrowerConcept rdfs:label ?narrowerConceptLabel .
    }}
    """
    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        concepts = set()
        for result in results["results"]["bindings"]:
            subject_label = result.get("narrowerConceptLabel", {}).get("value", "No label")
            concepts.add(sanitize_term(subject_label))

        return list(concepts)

    except Exception as exc:  # pragma: no cover
        return []


def get_all_narrower_concepts(term: str, depth: int = 2, current_depth: int = 1) -> Dict[str, List[str]]:
    term = sanitize_term(term)
    all_concepts: Dict[str, List[str]] = {}
    narrower_concepts = get_narrower_concepts_for_term(term)
    all_concepts[term] = narrower_concepts

    if current_depth < depth:
        for concept in narrower_concepts:
            child_concepts = get_all_narrower_concepts(concept, depth, current_depth + 1)
            all_concepts.update(child_concepts)

    return all_concepts


def query_rdf(
    local_file_path: Path,
    query: str,
    mesh_terms: Iterable[str],
    base_namespace: str = MESH_BASE_NAMESPACE,
):
    if Graph is None:
        raise ImportError("rdflib is required to query the RDF graph.")

    terms = [t for t in mesh_terms if t]
    if not terms:
        raise ValueError("The list of MeSH terms is empty or invalid.")

    if not Path(local_file_path).exists():
        raise FileNotFoundError(f"RDF file not found: {local_file_path}")

    g = Graph()
    g.parse(local_file_path, format="ttl")

    # Build a local allow-list of MeSH URIs to filter after the SPARQL query.
    mesh_uri_allowlist = {
        str(convert_to_uri(term, base_namespace))
        for term in terms
    }

    hydrated_query = query
    article_data = {}
    results = g.query(hydrated_query)

    for row in results:
        if str(row["meshTerm"]) not in mesh_uri_allowlist:
            continue
        article_uri = row["article"]
        if article_uri not in article_data:
            article_data[article_uri] = {
                "title": row.get("title"),
                "abstract": row.get("abstract"),
                "datePublished": row.get("datePublished"),
                "access": row.get("access"),
                "meshTerms": set(),
            }
        article_data[article_uri]["meshTerms"].add(str(row["meshTerm"]))

    ranked_articles = sorted(
        article_data.items(),
        key=lambda item: len(item[1]["meshTerms"]),
        reverse=True,
    )
    return ranked_articles[:10]
