import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

#Config : hard coded
DATA_PATH = "./data"
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Loading the documents
def load_documents():
    """
    Traverses the data folder, loads PDFs, and assigns 'role' metadata
    based on the folder name (e.g., data/finance -> role: finance).
    """

    documents = []
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)

                # determinia specific role, HR, finance or general purpose
                # if file in hr, role is hr
                folder_name = os.path.basename(root)
                role = folder_name if folder_name in ['hr', 'finance'] else 'general'
                ## safety check
                print(f"Loading: {file} | Role: {role}")

                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()

                    ## taging every page with that specific role
                    for doc in docs:
                        doc.metadata["source"] = file
                        doc.metadata["role"] = role

                    documents.extend(docs)
                
                except Exception as e:
                    print(f"failed to load {file}: {e}")
    
    return documents

##Chroma vector DB
def create_vector_db():
    print("1--Loading Documents")
    docs = load_documents()
    ## safety check
    if not docs:
        print("No document found, upload documents to data folder")
        return
    
    print(f"SPLITTING {len(docs)} Pages")

    # text to chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   
        chunk_overlap=200  
    )

    chunks = text_splitter.split_documents(docs)
    print(f"{len(chunks)} are formed...")

    # creating vector db
    print("\nCreating Vector DB \n")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device':'cpu'}
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print("created DB successdully")

if __name__ == "__main__":
    create_vector_db()


