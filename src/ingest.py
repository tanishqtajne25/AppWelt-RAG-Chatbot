import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------- CONFIG ----------------
DATA_PATH = "./data"
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

ALLOWED_DEPARTMENTS = {"hr", "finance", "general"}
# --------------------------------------


def load_documents():
    """
    Loads PDFs from data/<department>/ folders and assigns metadata.
    """

    documents = []

    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            if not file.lower().endswith(".pdf"):
                continue

            file_path = os.path.join(root, file)
            folder_name = os.path.basename(root).lower()

            if folder_name not in ALLOWED_DEPARTMENTS:
                print(f"Skipping {file} (unknown department: {folder_name})")
                continue

            print(f"Loading: {file} | Department: {folder_name}")

            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                for i, doc in enumerate(docs):
                    doc.metadata["source"] = file
                    doc.metadata["department"] = folder_name
                    doc.metadata["doc_id"] = f"{file}_{i}"


                documents.extend(docs)

            except Exception as e:
                print(f"Failed to load {file}: {e}")

    return documents


def create_vector_db():
    # Optional but STRONGLY recommended
    if os.path.exists(CHROMA_PATH):
        print("Clearing existing Chroma DB...")
        shutil.rmtree(CHROMA_PATH)

    print("Loading documents...")
    docs = load_documents()

    if not docs:
        print("No documents found. Check data folder.")
        return

    print(f"Splitting {len(docs)} pages...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("Creating vector database...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )

    print("Vector DB created successfully.")


if __name__ == "__main__":
    create_vector_db()
