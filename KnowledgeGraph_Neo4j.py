import os
import tempfile
from pathlib import Path
from typing import Tuple, List

for env_key in (
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_ENDPOINT",
    "LANGSMITH_TRACING",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_API_KEY",
):
    os.environ.pop(env_key, None)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

# LangChain Core Runnables
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

# Text Splitters
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

# Document Loaders
from langchain_community.document_loaders import DirectoryLoader

# Embeddings (HuggingFace BGE)
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Neo4j Graph Integration
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph

# Vector Store for Neo4j
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

# LLM Graph Transformer
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Prompts
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    PromptTemplate,
)

# Pydantic Model (v1)
from pydantic import BaseModel, Field

# Chat Messages
from langchain_core.messages import AIMessage, HumanMessage


# kalau kamu juga pakai ResponseSchema, sekalian:
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema

class RAG_Graph:
    default_cypher = "MATCH (S)-[r:!MENTIONS]->(T) RETURN S, r, T LIMIT 100"

    def __init__(self) :
        load_dotenv()
        os.environ["NEO4J_URI"] = "neo4j://127.0.0.1:7687"
        os.environ["NEO4J_USERNAME"] = "neo4j"
        os.environ["NEO4J_PASSWORD"] = "password"
        # refresh_schema triggers APOC usage; disable to avoid requiring APOC plugin
        self.graph = Neo4jGraph(refresh_schema=False)
        self.llm = ChatOpenAI(
            base_url=os.getenv("LMSTUDIO_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")),
            api_key=os.getenv("LMSTUDIO_API_KEY", os.getenv("OPENAI_API_KEY", "lm-studio")),
            model=os.getenv("LMSTUDIO_MODEL", os.getenv("OPENAI_MODEL", "dqwen2.5-7b-instruct")),
            temperature=0.2,
        )



    def create_graph(self, docs, TMP_DIR):
        for source_docs in docs:
            with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix=".docx") as tmp_file:
                tmp_file.write(source_docs.read())

        loader = DirectoryLoader(TMP_DIR.as_posix(), glob="**/*.docx", show_progress = True)
        self.documents = loader.load()

        text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=0)
        texts = text_splitter.split_documents(self.documents)

        llm_transformer = LLMGraphTransformer(llm=self.llm)

        # convert documents into graph structure using current API
        try:
            # LLM must respond with proper OpenAI-compatible payload; otherwise LangChain raises TypeError
            graph_documents = llm_transformer.convert_to_graph_documents(texts)
        except TypeError as exc:
            raise RuntimeError(
                "Failed to parse LLM response while extracting graph data. "
                "Check LM Studio/OpenAI base_url, model name, and that the server is running."
            ) from exc

        self.graph.add_documents(
            graph_documents,
            baseEntityLabel= True,
            include_source=True)
        
        
