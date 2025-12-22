import chainlit as cl
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import MessagesPlaceholder


# --- CONFIGURATION ---
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1" 

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

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

@cl.on_chat_start
async def start():
    # We just store an empty history list in the session
    cl.user_session.set("chat_history", [])
    await cl.Message(content="Hello! I am ready. Ask me anything about company policies.").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("chat_history")
    
    msg = cl.Message(content="")
    await msg.send()

    # --- STEP 1: CONTEXTUALIZE QUESTION (If history exists) ---
    # If we have history, we need to rewrite the user's question to include context.
    # e.g., User: "What is the budget?" -> Bot: "..."; User: "Who approved it?" -> "Who approved the budget?"
    
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
    # Search the vector DB with the (possibly rephrased) query
    docs = await retriever.ainvoke(query_text)
    
    # --- STEP 3: GENERATE ANSWER ---
    # Prepare the context text from the docs
    context_text = "\n\n".join([d.page_content for d in docs])
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a secure corporate assistant.
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
    
    # Save to history
    history.append(HumanMessage(content=message.content))
    history.append(AIMessage(content=answer))
    cl.user_session.set("chat_history", history)

    # Format sources for UI
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