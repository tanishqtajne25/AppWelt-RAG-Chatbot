import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# --- OPTIONAL: GROQ IMPORTS ---
# from dotenv import load_dotenv          
# from langchain_groq import ChatGroq     

# --- CONFIGURATION ---
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1" 

# --- 1. INITIALIZE RESOURCES (Run once on startup) ---
print("Loading Resources...")

embedding_function = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'} 
)

vectorstore = Chroma(
    persist_directory=CHROMA_PATH, 
    embedding_function=embedding_function
)

# --- 2. SELECT YOUR LLM ---
# [OPTION 1] LOCAL OLLAMA (Active)
llm = ChatOllama(
    model=LLM_MODEL, 
    temperature=0,
    num_ctx=4096
)

# [OPTION 2] GROQ API (Inactive)
# load_dotenv() 
# if not os.getenv("GROQ_API_KEY"):
#     print("ERROR: GROQ_API_KEY not found in .env")
#
# llm = ChatGroq(
#     model="llama-3.3-70b-versatile", 
#     temperature=0
# )

# --- 3. RETRIEVER SETUP (MMR Enabled) ---
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

# --- CORE LOGIC FUNCTION ---
async def get_answer(user_input: str, history_data: list):
    """
    Simulates the exact main() loop from the original code.
    args:
        user_input: The string message from the user.
        history_data: A list of dicts [{'role': 'user', 'content': '...'}, ...]
    """
    
    # 1. Reconstruct Memory from JSON
    # We need to convert the raw list back into LangChain Message objects
    chat_history = []
    for msg in history_data:
        if msg['role'] == 'user':
            chat_history.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            chat_history.append(AIMessage(content=msg['content']))

    # --- STEP 1: CONTEXTUALIZE QUESTION ---
    query_text = user_input

    if len(chat_history) > 0:
        rephrase_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as it is."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])
            
        rephrase_chain = rephrase_prompt | llm
        res = await rephrase_chain.ainvoke({"chat_history": chat_history, "input": user_input})
        query_text = res.content

    # --- STEP 2: RETRIEVE DOCUMENTS ---
    docs = await retriever.ainvoke(query_text)

    # --- SAFETY GUARD: CHECK FOR EMPTY RETRIEVAL ---
    if not docs:
        return "I do not find this information in the allowed documents."

    # --- STEP 3: GENERATE ANSWER ---
    context_text = "\n\n".join([d.page_content for d in docs])
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful Sales Assistant for Vignaharta Gold.
    Use the following pieces of retrieved context to answer the question.

    If the answer is explicitly stated in the context, answer directly.
    Only say "I do not find this information in the allowed documents"
    if the information is completely absent.
    Do not introduce additional assumptions or ambiguities.

    Context:
    {context}
    """
            ),
            ("human", "{input}"),
        ]
    )

    qa_chain = qa_prompt | llm
    res = await qa_chain.ainvoke({"context": context_text, "input": query_text})
    
    return res.content