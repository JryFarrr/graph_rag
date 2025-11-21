import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Graph RAG with Neo4j",
    page_icon="🧠",
    layout="wide"
)

# CSS tambahan biar terasa modern
st.markdown("""
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #020617;  /* very dark slate */
    padding-top: 1.5rem;
}

/* Hilangkan garis radio / border default */
.sidebar-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #e5e7eb;
    padding-left: 0.5rem;
    margin-bottom: 0.8rem;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
</style>
""", unsafe_allow_html=True)

import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
from dotenv import load_dotenv, find_dotenv
import networkx as nx
from py2neo import Graph
from pyvis.network import Network
from KnowledgeGraph_Neo4j import RAG_Graph

TMP_DIR = Path(__file__).resolve().parent.parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.parent.joinpath('data', 'vector_store')
TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

header = st.container()

def streamlit_ui():
    with st.sidebar:
        st.markdown("<div class='sidebar-title'>Navigation</div>", unsafe_allow_html=True)

        selected = option_menu(
            menu_title=None,
            options=["Home", "Chat with document/RAG", "RAG with Neo4j"],
            icons=["house", "chat-dots", "diagram-3"],   # bebas diganti
            default_index=1,
            styles={
                "container": {
                    "padding": "0!important",
                    "background-color": "transparent",
                },
                "icon": {
                    "color": "#9ca3af",
                    "font-size": "18px",
                },
                "nav-link": {
                    "font-size": "14px",
                    "color": "#e5e7eb",
                    "padding": "0.55rem 0.9rem",
                    "border-radius": "0.75rem",
                    "margin": "0.15rem 0",
                    "font-weight": "500",
                },
                "nav-link-hover": {
                    "background-color": "#111827",
                },
                "nav-link-selected": {
                    "background-color": "#111827",
                    "color": "#f97316",   # warna accent (oranye modern)
                },
            },
        )

    if selected == 'Home':
        st.title("Chat with document by clicking on Chat with document/RAG")
    elif selected == 'Chat with document/RAG':
        with header:
            st.title('Chat with document/RAG')
            st.write('Upload your PDF documents and chat with them using RAG (Retrieval-Augmented Generation) technique.')
            source_docs = st.file_uploader(label='Upload a documents', type=['pdf'], accept_multiple_files=True)
            if not source_docs:
                st.warning("Please upload a document")
            else:
                query = st.chat_input("Masukkan pertanyaan untuk mulai berdiskusi dengan dokumen.")
                if query:
                    RAG(source_docs, query)
                else:
                    st.info("Silakan masukkan pertanyaan terlebih dahulu.")
    elif selected == 'RAG with Neo4j':
        with header:
            st.title('RAG with Neo4j')
            st.write('This is RAG approach  with Neo4J knowledge graph database.')
            source_docs = st.file_uploader(label='Upload a documents', type=['docx'], accept_multiple_files=True)
            if not source_docs:
                st.warning("Please upload a document")
            else:
                RAG_Neo4j(source_docs, TMP_DIR)

load_dotenv(find_dotenv())
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

def RAG_Neo4j(docs, tmp_dir):
    graph_builder = RAG_Graph()
    graph_builder.create_graph(docs, tmp_dir)
    show_graph()


def show_graph():
    st.title("Knowledge Graph Visualization")

    url = st.text_input("Enter the Neo4j Aura URL", "bolt://localhost:7687")
    username = st.text_input("Enter the Neo4j Username", "neo4j")
    password = st.text_input("Enter the Neo4j Password", "password", type="password")


    if st.button("Load Graph"):
        try:
            data = get_graph_data(url, username, password)
            G = create_networkx_graph(data)
            visualize_graph(G)


            HtmlFile = open("knowledge_graph.html", "r", encoding="utf-8")
            source_code = HtmlFile.read()
            components.html(source_code, height=600, scrolling=True)

        except Exception as e:
            st.error(f"Error loading graph: {e}")
        

def get_graph_data(url, username, password):
    graph = Graph(url, auth=(username, password))
    query = """
    MATCH (n)-[r]->(m)
    RETURN n, r, m
    LIMIT 100
    """

    data = graph.run(query).data()
    return data

def create_networkx_graph(data):
    

    G = nx.DiGraph()

    for record in data:
        n = record['n']
        m = record['m']
        r = record['r']

        G.add_node(n['id'],label=n['name'])
        G.add_node(m['id'], label=m['name'])
        G.add_edge(n['id'], m['id'], label=r['type'])
    return G

def visualize_graph(G):
    net = Network(notebook=True)
    net.from_nx(G)
    net.show("knowledge_graph.html")





def RAG(docs, query):
    if not query:
        return
    #Load documents
    for source_docs in docs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=TMP_DIR.as_posix()) as tmp_file:
            tmp_file.write(source_docs.read())
            tmp_file_path = tmp_file.name
        
        loader = DirectoryLoader(TMP_DIR.as_posix(), glob="**/*.pdf", show_progress=True)
        documents = loader.load()

        #Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        text = text_splitter.split_documents(documents)

        # Vector and embeddings
        DB_FAISS_PATH = 'vectorstore_lmstudio/faiss'
        embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

        db = FAISS.from_documents(text, embeddings)
        db.save_local(DB_FAISS_PATH)

        #setup LLM
        llm = ChatOpenAI(base_url='http://127.0.0.1:1234', api_key="lm-studio")

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
            return_source_documents=True
        )

        chat_history = []
        result = qa_chain({"question": query, "chat_history": chat_history})
        st.write(result['answer'])
        chat_history.append((query, result['answer']))
streamlit_ui()
