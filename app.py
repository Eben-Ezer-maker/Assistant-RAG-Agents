import os
import json
import requests
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from pathlib import Path

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent

load_dotenv()

# Lecture des secrets Streamlit Cloud
import os
try:
    for key, value in st.secrets.items():
        os.environ[key] = str(value)
except:
    pass

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════
DOCS_DIR    = Path("Data")
FAISS_DIR   = "faiss_store"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o"

# ════════════════════════════════════════════════════════════════
# RAG
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def init_rag():
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

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    if Path(FAISS_DIR).exists():
        vectorstore = FAISS.load_local(
            FAISS_DIR, embeddings, allow_dangerous_deserialization=True
        )
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(FAISS_DIR)

    return vectorstore


def get_context(vectorstore, question):
    retrieved_docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join([
        f"[Source: {d.metadata.get('source', 'inconnu')}, Page: {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in retrieved_docs
    ])
    sources = list(set([d.metadata.get("source", "inconnu") for d in retrieved_docs]))
    return context, sources, retrieved_docs


def rag_answer(vectorstore, question, chat_history):
    context, sources, retrieved_docs = get_context(vectorstore, question)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Tu es un assistant expert en analyse de documents financiers et économiques. "
         "Réponds uniquement à partir du contexte fourni. "
         "Si tu ne trouves pas la réponse, dis-le clairement. "
         "IMPORTANT : Cite toujours tes sources avec précision en mentionnant "
         "le nom exact du fichier source et la page. "
         "Par exemple : 'Selon le Document d'Enregistrement Universel Sanofi 2025, page 42...' "
         "ou 'D'après le Rapport Financier Semestriel Sanofi 2025, page 12...' "
         "N'écris JAMAIS 'Document fourni' mais toujours le vrai nom du fichier.\n\n"
         "Contexte :\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": question
    })
    return response, sources, retrieved_docs


# ════════════════════════════════════════════════════════════════
# OUTILS AGENTS
# ════════════════════════════════════════════════════════════════

# ✅ MODIFICATION 1 : ajout de days=7 pour forcer les résultats récents
search_tool = TavilySearchResults(
    max_results=3,
    days=7,
    description="Recherche des actualités récentes : guerre Iran, football, économie, marchés."
)

@tool
def calculatrice(expression: str) -> str:
    """Effectue des calculs mathématiques. Exemple: '2 + 2', '15 * 3'"""
    try:
        return f"Résultat : {eval(expression)}"
    except Exception as e:
        return f"Erreur : {str(e)}"

@tool
def meteo(ville: str) -> str:
    """Donne la météo actuelle d'une ville."""
    try:
        response = requests.get(f"https://wttr.in/{ville}?format=3&lang=fr", timeout=5)
        return response.text
    except Exception as e:
        return f"Erreur météo : {str(e)}"

@tool
def resumer(contenu: str) -> str:
    """Résume un texte long ou une URL (commençant par http)."""
    try:
        if contenu.startswith("http"):
            texte = requests.get(contenu, timeout=10).text[:5000]
        else:
            texte = contenu[:5000]
        llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
        return llm.invoke(f"Résume en 5 points clés en français :\n\n{texte}").content
    except Exception as e:
        return f"Erreur résumé : {str(e)}"

@tool
def generer_graphique(donnees_json: str) -> str:
    """
    Génère un graphique depuis des données manuelles.
    JSON attendu: {"type": "bar|line|pie|scatter", "titre": "...", "x": [...], "y": [...], "xlabel": "...", "ylabel": "..."}
    """
    try:
        data = json.loads(donnees_json)
        df = pd.DataFrame({"x": data["x"], "y": data["y"]})
        titre = data.get("titre", "Graphique")
        xlabel = data.get("xlabel", "")
        ylabel = data.get("ylabel", "")

        if data.get("type") == "line":
            fig = px.line(df, x="x", y="y", title=titre,
                         labels={"x": xlabel, "y": ylabel}, markers=True)
        elif data.get("type") == "pie":
            fig = px.pie(df, names="x", values="y", title=titre)
        elif data.get("type") == "scatter":
            fig = px.scatter(df, x="x", y="y", title=titre,
                            labels={"x": xlabel, "y": ylabel})
        else:
            fig = px.bar(df, x="x", y="y", title=titre,
                        labels={"x": xlabel, "y": ylabel},
                        color="y", color_continuous_scale="Blues")

        fig.update_layout(template="plotly_white", title_font_size=18)
        fig.write_json("graphique_genere.json")
        return f"GRAPHIQUE_GENERE:graphique_genere.json|{titre}"
    except Exception as e:
        return f"Erreur graphique : {str(e)}"

@tool
def analyser_excel(chemin_fichier: str) -> str:
    """Analyse un fichier Excel et retourne des statistiques descriptives."""
    try:
        df = pd.read_excel(chemin_fichier)
        df.to_json("excel_data.json", orient="records")
        analyse = [
            f"📊 {df.shape[0]} lignes, {df.shape[1]} colonnes",
            f"Colonnes : {', '.join(df.columns.tolist())}",
            f"Statistiques :\n{df.describe().to_string()}",
        ]
        return "\n".join(analyse)
    except Exception as e:
        return f"Erreur analyse : {str(e)}"


# ════════════════════════════════════════════════════════════════
# ROUTEUR
# ════════════════════════════════════════════════════════════════

# ✅ MODIFICATION 2 : routeur LLM intelligent — comprend l'intention
def router(question: str, vectorstore) -> str:
    """Routeur intelligent — LLM décide entre RAG, AGENT et DIRECT."""

    # Salutations → DIRECT (pas besoin du LLM pour ça)
    direct_keywords = [
        "bonjour", "salut", "merci", "au revoir", "bonsoir",
        "comment vas", "qui es-tu", "hello", "bonne journée"
    ]
    for keyword in direct_keywords:
        if keyword in question.lower():
            return "DIRECT"

    # LLM classifie entre RAG et AGENT
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    prompt = f"""Tu es un routeur intelligent. Classe cette question en UN seul mot parmi : RAG ou AGENT.

- RAG : si la question concerne Sanofi, ses documents financiers, sa stratégie, ses médicaments, ses résultats, son pipeline, ses rapports
- AGENT : si la question nécessite internet, actualité récente, météo, calcul mathématique, fichier excel, graphique, résumé d'URL

Question : {question}

Réponds uniquement par RAG ou AGENT, rien d'autre."""

    try:
        decision = llm.invoke(prompt).content.strip().upper()
        if decision in ["RAG", "AGENT"]:
            return decision
        return "RAG"
    except:
        return "RAG"


# ════════════════════════════════════════════════════════════════
# AGENT
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def build_agent():
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    tools = [search_tool, calculatrice, meteo, resumer,
             generer_graphique, analyser_excel]

    system_prompt = (
        "Tu es un assistant intelligent spécialisé en économie et finance. "
        "Outils disponibles :\n"
        "- 🌍 Recherche web : actualités récentes\n"
        "- 🧮 Calculatrice : calculs mathématiques\n"
        "- 🌤️ Météo : météo d'une ville\n"
        "- 📝 Résumé : texte ou URL\n"
        "- 📊 Graphique : données manuelles\n"
        "- 📂 Analyse Excel : statistiques\n\n"
        "Utilise l'outil le plus approprié. Réponds toujours en français."
    )

    return create_react_agent(llm, tools, prompt=system_prompt)


# ════════════════════════════════════════════════════════════════
# AFFICHAGE GRAPHIQUE
# ════════════════════════════════════════════════════════════════
def display_graph(json_path: str):
    try:
        import plotly.io as pio
        import time
        fig = pio.read_json(json_path)
        unique_key = f"chart_{json_path}_{int(time.time() * 1000)}"
        st.plotly_chart(fig, use_container_width=True, key=unique_key)
    except Exception as e:
        st.error(f"Erreur affichage graphique : {str(e)}")


# ════════════════════════════════════════════════════════════════
# INTERFACE STREAMLIT
# ════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="Assistant RAG + Agents",
        page_icon="🤖",
        layout="wide"
    )

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Paramètres")
        st.markdown("---")

        st.subheader("📂 Upload fichier Excel")
        uploaded_file = st.file_uploader("Glisse ton fichier ici", type=["xlsx", "xls"])
        if uploaded_file:
            excel_path = f"Data/{uploaded_file.name}"
            with open(excel_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ {uploaded_file.name} chargé !")
            st.session_state["excel_path"] = excel_path

        st.markdown("---")
        if st.button("🗑️ Effacer la conversation"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

    # ── Titre ─────────────────────────────────────────────────────
    st.title("🤖 Yao.AI ")
    st.title("Bonjour Eben-Ezer!")

    # ── Initialisation ────────────────────────────────────────────
    with st.spinner("⏳ Chargement..."):
        vectorstore = init_rag()
        agent = build_agent()
    st.success("✅ Prêt !")

    # ── Session state ─────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ── Historique ────────────────────────────────────────────────
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "graph_path" in message:
                display_graph(message["graph_path"])
            if "chunks" in message:
                with st.expander("📚 Sources et contexte récupéré"):
                    for i, chunk in enumerate(message["chunks"]):
                        st.markdown(f"**📄 Chunk {i+1}** — `{chunk['source']}` — page {chunk['page']}")
                        st.info(chunk["content"])
                        st.divider()

    # ── Input ─────────────────────────────────────────────────────
    if question := st.chat_input("Posez votre question..."):

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyse en cours..."):

                decision = router(question, vectorstore)
                message_data = {"role": "assistant"}

                # ── RAG ───────────────────────────────────────────
                if decision == "RAG":
                    st.caption("📄 *Recherche dans les documents...*")
                    response, sources, retrieved_docs = rag_answer(
                        vectorstore, question, st.session_state.chat_history
                    )
                    st.markdown(response)

                    chunks_data = [{
                        "source": doc.metadata.get("source", "inconnu"),
                        "page": doc.metadata.get("page", "?"),
                        "content": doc.page_content
                    } for doc in retrieved_docs]

                    with st.expander("📚 Sources et contexte récupéré"):
                        for i, chunk in enumerate(chunks_data):
                            st.markdown(f"**📄 Chunk {i+1}** — `{chunk['source']}` — page {chunk['page']}")
                            st.info(chunk["content"])
                            st.divider()

                    message_data["content"] = response
                    message_data["chunks"] = chunks_data

                # ── AGENT ─────────────────────────────────────────
                elif decision == "AGENT":
                    st.caption("🤖 *Agent en action...*")
                    question_agent = question
                    if any(w in question.lower() for w in ["excel", "analyser", "fichier", "statistique"]):
                        if "excel_path" in st.session_state:
                            question_agent = f"{question} — chemin du fichier : {st.session_state['excel_path']}"
                    result = agent.invoke({
                        "messages": st.session_state.chat_history + [HumanMessage(content=question_agent)]
                    })

                    response = result["messages"][-1].content

                    # Cherche le signal graphique dans tous les messages
                    graph_path = None
                    graph_title = "Graphique"
                    for msg in result["messages"]:
                        msg_content = str(getattr(msg, "content", ""))
                        if "GRAPHIQUE_GENERE:" in msg_content:
                            parts = msg_content.split("GRAPHIQUE_GENERE:")[1].split("|")
                            graph_path = parts[0].strip()
                            graph_title = parts[1].strip() if len(parts) > 1 else "Graphique"
                            break

                    if graph_path and Path(graph_path).exists():
                        response = f"✅ Graphique généré : **{graph_title}**"

                    st.markdown(response)

                    if graph_path and Path(graph_path).exists():
                        display_graph(graph_path)
                        message_data["graph_path"] = graph_path

                    message_data["content"] = response

                # ── DIRECT ────────────────────────────────────────
                else:
                    st.caption("💬 *Réponse directe...*")
                    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
                    response = llm.invoke(
                        f"Réponds en français : {question}"
                    ).content
                    st.markdown(response)
                    message_data["content"] = response

            # Badge routing
            badge = {"RAG": "📄 RAG", "AGENT": "🤖 Agent", "DIRECT": "💬 Direct"}
            st.caption(f"*Traité par : {badge.get(decision, decision)}*")

        st.session_state.messages.append(message_data)
        st.session_state.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=message_data["content"]),
        ])


if __name__ == "__main__":
    main()
