import os
import chainlit as cl
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# from dotenv import load_dotenv          
# from langchain_groq import ChatGroq     

# --- CONFIGURATION ---
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1" 

# --- OPTION 2: GROQ API (Uncomment to use) ---
# load_dotenv() 
# 
# if not os.getenv("GROQ_API_KEY"):
#     print("ERROR: GROQ_API_KEY not found in .env")
#
# llm = ChatGroq(
#     model="llama-3.3-70b-versatile", 
#     temperature=0,
#     max_context_length=4096
# )

# --- 1. INITIALIZE RESOURCES ---
print("Loading Resources...")
embedding_function = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'} 
)

vectorstore = Chroma(
    persist_directory=CHROMA_PATH, 
    embedding_function=embedding_function
)

# Global LLM instance
llm = ChatOllama(
    model=LLM_MODEL, 
    temperature=0,
    num_ctx=4096
)

# UPDATED: Removed the 'finance' filter so it searches ALL jewelry docs
# MMR (Maximal Marginal Relevance) finds the best match, then finds 
# other matches that are diverse, preventing repetitive context.
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

@cl.on_chat_start
async def start():
    # We just store an empty history list in the session
    cl.user_session.set("chat_history", [])
    # UPDATED: Welcome message
    await cl.Message(content="✨ **Welcome to Vignaharta Gold!** ✨\n\nI can help you with Gold Rates, Jewellery Schemes, and Diamond Collections.").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("chat_history")
    
    msg = cl.Message(content="")
    await msg.send()

    # --- STEP 1: CONTEXTUALIZE QUESTION (Logic Unchanged) ---
    query_text = message.content

    if len(history) > 0:
        rephrase_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])
            
        rephrase_chain = rephrase_prompt | llm
        res = await rephrase_chain.ainvoke({"chat_history": history, "input": message.content})
        query_text = res.content

    
    # --- STEP 2: RETRIEVE DOCUMENTS ---
    docs = await retriever.ainvoke(query_text)
    # --- SAFETY GUARD: CHECK FOR EMPTY RETRIEVAL ---
    if not docs:
        answer = "I do not find this information in the allowed documents."
        
        # Update History immediately
        history.append(HumanMessage(content=message.content))
        history.append(AIMessage(content=answer))
        if len(history) > 10:
            history = history[-10:]
        cl.user_session.set("chat_history", history)

        # Send response and EXIT function
        msg.content = answer
        await msg.update()
        return
    
    # --- STEP 3: GENERATE ANSWER ---
    context_text = "\n\n".join([d.page_content for d in docs])
    
    # UPDATED: Persona changed, but STRICT rules preserved exactly as you wanted
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

    
    # We manually invoke the LLM to get the final answer
    qa_chain = qa_prompt | llm
    res = await qa_chain.ainvoke({"context": context_text, "input": query_text})
    answer = res.content

# --- STEP 4: UPDATE MEMORY & UI ---
    
    # Append new interaction
    history.append(HumanMessage(content=message.content))
    history.append(AIMessage(content=answer))

    # LIMIT HISTORY: Keep only the last 10 messages (5 user + 5 AI)
    if len(history) > 10:
        history = history[-10:]

    # Save back to session
    cl.user_session.set("chat_history", history)

    msg.content = answer
    await msg.update()