import csv
from pathlib import Path
from urllib.parse import quote
from rdflib import Graph, Namespace, URIRef, Literal

CSV_PATH = Path("data/Pubmed/PubMed Multi Label Text Classification Dataset.csv")
TTL_OUT = Path("data/Pubmed/PubMedGraph.ttl")

EX = Namespace("http://example.org/")
SCHEMA = Namespace("http://schema.org/")

g = Graph()
g.bind("ex", EX)
g.bind("schema", SCHEMA)

def mesh_uri(label: str) -> URIRef:
    return URIRef(f"{EX}mesh/{quote(label.replace(' ', '_'))}")

with CSV_PATH.open("r", encoding="utf-8") as fh:
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

        # meshMajor column is a Python-list-as-string; eval it safely
        try:
            mesh_values = eval(row.get("meshMajor", "[]"))
        except Exception:
            mesh_values = []
        for label in mesh_values:
            if not isinstance(label, str) or not label.strip():
                continue
            term_uri = mesh_uri(label.strip())
            g.add((term_uri, EX.type, EX.MeSHTerm))
            g.add((art_uri, SCHEMA.about, term_uri))

g.serialize(TTL_OUT, format="turtle")
print(f"Wrote {TTL_OUT}")
