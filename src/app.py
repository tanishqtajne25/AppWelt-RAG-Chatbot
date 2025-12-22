import chainlit as cl
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# --- CONFIGURATION ---
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1" 

print("Loading Resources...")
embedding_function = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'} 
)

vectorstore = Chroma(
    persist_directory=CHROMA_PATH, 
    embedding_function=embedding_function
)

# Main LLM
llm = ChatOllama(
    model=LLM_MODEL, 
    temperature=0,  # Strict mode
    num_ctx=4096
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

@cl.on_chat_start
async def start():
    cl.user_session.set("chat_history", [])
    await cl.Message(content="Hello! I am ready. Ask me anything about company policies.").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("chat_history")
    
    msg = cl.Message(content="")
    await msg.send()

    # --- STEP 1: SMART REPHRASING ---
    query_text = message.content

    if len(history) > 0:
        # Improved Prompt: Explicitly forbids answering
        rephrase_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a precise search query generator.
Your task is to rewrite the user's question to be standalone, using the chat history for context.
RULES:
1. Output ONLY the rewritten question.
2. Do NOT answer the question.
3. Do NOT provide explanations.
4. If the question is already standalone, repeat it exactly.

Example 1:
History: User "Who is the CEO?" Bot "John Doe".
Input: "How old is he?"
Output: How old is John Doe?

Example 2:
History: User "What is the budget?" Bot "$500".
Input: "Who approved it?"
Output: Who approved the budget?

Example 3 (Correction):
Input: "When do they get paid?"
Bad Output: They get paid on the 5th.
Good Output: When do employees get paid?
"""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])
        
        # We use a lower temperature chain specifically for rephrasing if possible, 
        # but here we rely on the main LLM with temp=0
        rephrase_chain = rephrase_prompt | llm
        res = await rephrase_chain.ainvoke({"chat_history": history, "input": message.content})
        
        clean_query = res.content.strip().replace('"', '')
        
        print(f"DEBUG: Original Q: {message.content}")
        print(f"DEBUG: Rephrased Q: {clean_query}")
        
        # --- DEFENSE MECHANISM ---
        # If it looks like a statement (no question mark) or is too long, reject it.
        if clean_query.endswith("?") and len(clean_query) < 150:
            query_text = clean_query
        else:
            print("DEBUG: Rephrasing failed (Not a question). Using original.")

    # --- STEP 2: RETRIEVE ---
    docs = await retriever.ainvoke(query_text)
    
    # --- STEP 3: ANSWER ---
    context_text = "\n\n".join([d.page_content for d in docs])
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful corporate assistant. Answer the user's question based strictly on the context provided below.\nIf the context does not contain the answer, say 'I cannot find this information in the documents.'\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])
    
    qa_chain = qa_prompt | llm
    res = await qa_chain.ainvoke({"context": context_text, "question": query_text})
    answer = res.content

    # --- UPDATE MEMORY & UI ---
    history.append(HumanMessage(content=message.content))
    history.append(AIMessage(content=answer))
    
    # Keep history short to avoid confusion
    if len(history) > 6:
        history = history[-6:]
    cl.user_session.set("chat_history", history)

    text_elements = []
    if docs:
        for idx, doc in enumerate(docs):
            source_name = doc.metadata.get("source", "Unknown")
            page_num = doc.metadata.get("page", "Unknown")
            role_tag = doc.metadata.get("role", "general")
            
            text_elements.append(
                cl.Text(content=doc.page_content, name=f"Source {idx+1} ({source_name})")
            )
            answer += f"\n* [Source {idx+1}] {source_name} (Page {page_num}) [{role_tag.upper()}]"
    else:
        answer += "\n\n*(No relevant documents found)*"

    msg.content = answer
    msg.elements = text_elements
    await msg.update()