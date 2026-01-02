import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- CONFIG ----------------
DATA_PATH = "./data"
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# ----------------------------------------
def load_documents():
    """
    Loads all PDFs and Text files directly from the data/ folder.
    Assigns 'General' category to everything.
    """
    documents = []
    
    # Walk through the directory
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Determine Loader
            if file.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.lower().endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                continue # Skip images/other files

            print(f"Loading: {file}")

            try:
                docs = loader.load()

                for i, doc in enumerate(docs):
                    doc.metadata["source"] = file
                    doc.metadata["category"] = "General"  # Default category for flat files
                    doc.metadata["doc_id"] = f"{file}_{i}"

                documents.extend(docs)

            except Exception as e:
                print(f"Failed to load {file}: {e}")

    return documents

def create_vector_db():
    if os.path.exists(CHROMA_PATH):
        print("Clearing existing database...")
        shutil.rmtree(CHROMA_PATH)

    print("Scanning data folder...")
    docs = load_documents()

    if not docs:
        print("No documents found in 'data/' folder!")
        return

    print(f"Processing {len(docs)} pages...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} text chunks.")

    print("Generating Embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )

    print("✅ Database Ready!")

if __name__ == "__main__":
    create_vector_db()