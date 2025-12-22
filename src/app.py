import os
import chainlit as cl

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
CHROMA_PATH = "./chroma_db"          # must match ingest.py
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1"

print("Loading resources...")

# Embeddings
embedding_function = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
)

# Vector store
if not os.path.exists(CHROMA_PATH):
    print(f"❌ Chroma DB not found at {CHROMA_PATH}. Run ingest.py first.")
    vectorstore = None
    retriever = None
else:
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
    )

    # Use simple similarity search for now (stable)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("✅ Vectorstore and retriever loaded.")

# Main LLM
llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0,      # strict / deterministic
    num_ctx=4096,
)


@cl.on_chat_start
async def start():
    cl.user_session.set("chat_history", [])
    await cl.Message(
        content="Hello! I am ready. Ask me anything about company policies."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    # Safety: if vectorstore not ready
    if retriever is None:
        await cl.Message(
            content="Vector database not found. Please run ingest.py and restart the app."
        ).send()
        return

    history = cl.user_session.get("chat_history") or []

    msg = cl.Message(content="")
    await msg.send()

    # --- STEP 1: SMART REPHRASING ---
    query_text = message.content

    if len(history) > 0:
        rephrase_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a precise search query generator.
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
""",
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )

        rephrase_chain = rephrase_prompt | llm
        res = await rephrase_chain.ainvoke(
            {"chat_history": history, "input": message.content}
        )

        clean_query = res.content.strip().replace('"', "")

        print(f"DEBUG: Original Q: {message.content}")
        print(f"DEBUG: Rephrased Q: {clean_query}")

        # basic guard
        if clean_query.endswith("?") and len(clean_query) < 150:
            query_text = clean_query
        else:
            print("DEBUG: Rephrasing failed / not a question. Using original.")

    # --- STEP 2: RETRIEVE ---
    try:
        docs = await retriever.ainvoke(query_text)
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        docs = []

    # --- STEP 3: ANSWER ---
    if not docs:
        answer = "I cannot find this information in the company documents."
        text_elements = []
    else:
        context_text = "\n\n".join(d.page_content for d in docs)

        qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful corporate assistant. Use ONLY the context below:

    {context}

    Answer the user's question based STRICTLY on this context.
    Quote exact numbers and rules.
    Do NOT infer, assume, or add information not explicitly stated.
    If the answer is not in the context, say: "I cannot find this information in the documents.""",
            ),
            ("human", "{question}"),
        ]
    )


        qa_chain = qa_prompt | llm
        res = await qa_chain.ainvoke(
            {"context": context_text, "question": query_text}
        )
        answer = res.content

        text_elements = []
        for idx, doc in enumerate(docs):
            source_name = doc.metadata.get("source", "Unknown")
            page_num = doc.metadata.get("page", "Unknown")
            dept_tag = doc.metadata.get("department", "general")

            text_elements.append(
                cl.Text(
                    content=doc.page_content,
                    name=f"Source {idx + 1} ({source_name})",
                )
            )
            answer += (
                f"\n\n* [Source {idx+1}] {source_name} "
                f"(Page {page_num}) [{dept_tag.upper()}]"
            )

    # --- UPDATE MEMORY ---
    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": answer})

    if len(history) > 6:
        history = history[-6:]

    cl.user_session.set("chat_history", history)

    msg.content = answer
    msg.elements = text_elements
    await msg.update()

print("✅ Chainlit app ready!")
