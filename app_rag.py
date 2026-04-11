import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Config ───────────────────────────────────────────────────────
load_dotenv()
DOCS_DIR    = Path("Data")
FAISS_DIR   = "faiss_store"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o"

# ── Chargement des documents ─────────────────────────────────────
@st.cache_resource
def init_rag():
    # Chargement
    docs = []
    for path in DOCS_DIR.rglob("*"):
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
            docs.extend(loader.load())
        elif path.suffix.lower() == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
            docs.extend(loader.load())

    if not docs:
        st.error(f"Aucun document trouvé dans {DOCS_DIR}")
        st.stop()

    # Découpage
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    # Vectorisation
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    if Path(FAISS_DIR).exists():
        vectorstore = FAISS.load_local(
            FAISS_DIR, embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(FAISS_DIR)

    return vectorstore

# ── Prompt RAG ───────────────────────────────────────────────────
def build_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "Tu es un assistant expert en analyse de documents financiers et économiques. "
         "Réponds uniquement à partir du contexte fourni ci-dessous. "
         "Si tu ne trouves pas la réponse dans le contexte, dis-le clairement. "
         "Cite toujours tes sources avec précision.\n\n"
         "Contexte :\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

# ── Récupération du contexte ─────────────────────────────────────
def get_context(vectorstore, question):
    retrieved_docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join([d.page_content for d in retrieved_docs])
    sources = list(set([d.metadata.get("source", "inconnu") for d in retrieved_docs]))
    return context, sources, retrieved_docs

# ── Interface Streamlit ──────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Assistant RAG - Sanofi & Orano",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Assistant RAG - Analyse Financière")
    st.caption("Basé sur les rapports Sanofi 2025")

    # Initialisation
    with st.spinner("⏳ Chargement des documents..."):
        vectorstore = init_rag()
    st.success("✅ Documents chargés !")

    # Historique de conversation
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Affichage des messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        if "chunks" in message:
            with st.expander("📚 Sources et contexte récupéré"):
                for i, chunk in enumerate(message["chunks"]):
                    st.markdown(f"**📄 Chunk {i+1}** — `{chunk['source']}` — page {chunk['page']}")
                    st.info(chunk["content"])
                    st.divider()

    # Input utilisateur
    if question := st.chat_input("Posez votre question sur les documents..."):

        # Affiche la question
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Génère la réponse
        with st.chat_message("assistant"):
            with st.spinner("🔍 Recherche dans les documents..."):

                # Récupère le contexte
                context, sources, retrieved_docs = get_context(vectorstore, question)

                # Construit la chaîne
                llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
                prompt = build_prompt()
                chain = prompt | llm | StrOutputParser()

                # Génère la réponse
                response = chain.invoke({
                    "context": context,
                    "chat_history": st.session_state.chat_history,
                    "question": question
                })

            st.markdown(response)

            # Affiche les chunks récupérés
            with st.expander("📚 Sources et contexte récupéré"):
                for i, doc in enumerate(retrieved_docs):
                    st.markdown(
                        f"**📄 Chunk {i+1}** — "
                        f"`{doc.metadata.get('source', 'inconnu')}` — "
                        f"page {doc.metadata.get('page', '?')}"
                    )
                    st.info(doc.page_content)
                    st.divider()

        # Sauvegarde chunks pour réaffichage
        chunks_data = [
            {
                "source": doc.metadata.get("source", "inconnu"),
                "page": doc.metadata.get("page", "?"),
                "content": doc.page_content
            }
            for doc in retrieved_docs
        ]

        # Met à jour l'historique
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "chunks": chunks_data
        })
        st.session_state.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=response),
        ])

if __name__ == "__main__":
    main()