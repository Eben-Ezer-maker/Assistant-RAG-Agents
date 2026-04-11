import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

# ── Config ───────────────────────────────────────────────────────
load_dotenv()
DOCS_DIR    = Path("Data")
FAISS_DIR   = "faiss_store"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o"

# ── Chargement des documents ─────────────────────────────────────
def load_documents():
    docs = []
    for path in DOCS_DIR.rglob("*"):
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
            docs.extend(loader.load())
        elif path.suffix.lower() == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
            docs.extend(loader.load())
    if not docs:
        raise RuntimeError(f"Aucun document trouvé dans {DOCS_DIR}")
    print(f"✅ {len(docs)} pages chargées")
    return docs

# ── Découpage en chunks ──────────────────────────────────────────
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ {len(chunks)} chunks créés")
    return chunks

# ── Vectorisation et index FAISS ────────────────────────────────
def get_vectorstore(chunks=None):
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    if Path(FAISS_DIR).exists():
        print("✅ Chargement du vector store existant...")
        return FAISS.load_local(
            FAISS_DIR, embeddings,
            allow_dangerous_deserialization=True
        )
    print("⏳ Construction du vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_DIR)
    print("✅ Vector store sauvegardé !")
    return vectorstore

# ── Construction de la chaîne RAG ───────────────────────────────
def build_rag_chain(vectorstore):
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Prompt pour reformuler la question avec l'historique
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "En tenant compte de l'historique de la conversation, "
         "reformule la dernière question pour qu'elle soit autonome. "
         "Ne réponds pas, reformule seulement."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    # Prompt pour répondre avec le contexte
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Tu es un assistant expert en analyse de documents financiers et économiques. "
         "Réponds uniquement à partir du contexte fourni. "
         "Si tu ne trouves pas la réponse, dis-le clairement. "
         "Cite toujours tes sources.\n\n"
         "Contexte :\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(
        history_aware_retriever, document_chain
    )
    return rag_chain


# ── Boucle de chat terminal ──────────────────────────────────────
if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    vectorstore = get_vectorstore(chunks)
    chain = build_rag_chain(vectorstore)

    chat_history = []
    print("\n🤖 Assistant RAG prêt ! (Ctrl+C pour quitter)\n")

    while True:
        try:
            question = input("💬 Vous : ")
            if not question.strip():
                continue
            result = chain.invoke({
                "input": question,
                "chat_history": chat_history
            })
            answer = result["answer"]
            print(f"\n🤖 Assistant : {answer}")
            print("\n📚 Sources :")
            for doc in result["context"]:
                print(f"  - {doc.metadata.get('source', 'inconnu')}")
            print()

            chat_history.extend([
                HumanMessage(content=question),
                AIMessage(content=answer),
            ])
        except KeyboardInterrupt:
            print("\nAu revoir !")
            break