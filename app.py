import hashlib
from io import StringIO
from typing import Sequence

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
import ollama

import pandas as pd
import pdfplumber
import streamlit as st

# ---- default settings for generation ----
TEMPERATURE = 0.5
MAX_TOKENS = 200
CHUNK_SIZE = 150
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1"

text = ""
result = None

# ---- ChromaDB setup (persistent, survives restarts) ----
client = chromadb.PersistentClient(path="./chroma_db")


class OllamaEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return [
            ollama.embeddings(model=EMBED_MODEL, prompt=t)["embedding"]
            for t in input
        ]


collection = client.get_or_create_collection(
    name="docbot", embedding_function=OllamaEmbeddingFunction()
)


def get_doc_id(text_content: str) -> str:
    """Stable id for a document based on its content, so re-uploading the
    same file doesn't duplicate it in the collection."""
    return hashlib.md5(text_content.encode("utf-8")).hexdigest()


def document_exists(doc_id: str) -> bool:
    existing = collection.get(where={"source": doc_id})
    return len(existing["ids"]) > 0


def add_document(chunks, doc_id):
    collection.add(
        documents=chunks,
        metadatas=[{"source": doc_id}] * len(chunks),
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
    )


# ---- PDF extraction ----
def extractTextFromPdf(pdfPath) -> str:
    extracted = ""
    with pdfplumber.open(pdfPath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                extracted += page_text
    return extracted


# ---- Chunking ----
def processTextInput(text: str) -> pd.DataFrame:
    text = StringIO(text).read()
    chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    df = pd.DataFrame.from_dict({"text": chunks})
    return df


def convertToList(df: pd.DataFrame) -> Sequence[str]:
    df["col"] = df[["text"]].apply(
        lambda row: " ".join(row.dropna().astype(str)), axis=1
    )
    return df["col"].tolist()


# ---- Retrieval (Chroma handles embedding + similarity search internally) ----
def topNNeighbours(query: str, k: int = 5):
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0]


# ---- Generation via Ollama chat ----
def generate(prompt: str, temperature: float, max_tokens: int) -> str:
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    return response["message"]["content"]


# ---- Streamlit UI ----
st.title("DocBot (local: Ollama + ChromaDB)")

options = st.selectbox("Input type", ["PDF", "TEXT"])

doc_id = None

if options == "PDF":
    pdfFile = st.file_uploader("Upload file", type=["pdf"])
    if pdfFile is not None:
        text = extractTextFromPdf(pdfFile)
elif options == "TEXT":
    text = st.text_area("Paste the Document")

if text:
    doc_id = get_doc_id(text)
    st.write(f"Document ID: {doc_id}")
    df = processTextInput(text)

    if not document_exists(doc_id):
        listOfText = convertToList(df)
        with st.spinner("Indexing document..."):
            add_document(listOfText, doc_id)
        st.success("Document indexed.")
    else:
        st.info("Document already indexed — skipping re-embedding.")

if doc_id is not None:
    prompt = st.text_input("Ask a Question")
    advancedOpt = st.checkbox("Advanced Options")
    if advancedOpt:
        TEMPERATURE = st.slider("Temperature", min_value=0.0, max_value=1.0, value=TEMPERATURE)
        MAX_TOKENS = st.slider("Max Tokens", min_value=1, max_value=5000, value=MAX_TOKENS)

    if prompt:
        with st.spinner("Retrieving relevant context..."):
            augPrompts = topNNeighbours(prompt, k=15)

        context = "\n".join(augPrompts)
        joinedPrompt = (
            f"Context:\n{context}\n\n"
            f"Answer the question using only the context above.\n"
            f"Question: {prompt}\n"
        )

        with st.spinner("Generating answer..."):
            try:
                result = generate(joinedPrompt, TEMPERATURE, MAX_TOKENS)
            except Exception as e:
                st.error(
                    "Couldn't reach Ollama. Make sure the Ollama server is "
                    f"running (`ollama serve`) and models are pulled. Details: {e}"
                )
                result = None

if result is not None:
    st.write(result)
