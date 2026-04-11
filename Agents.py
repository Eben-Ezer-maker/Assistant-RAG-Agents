import os
import json
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from pathlib import Path
 
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
 
load_dotenv()
 
# ── Outil 1 : Recherche Web Tavily ───────────────────────────────
search_tool = TavilySearchResults(
    max_results=3,
    description=(
        "Utile pour rechercher des actualités récentes : "
        "guerre en Iran, football, économie, marchés financiers."
    )
)
 
# ── Outil 2 : Calculatrice ───────────────────────────────────────
@tool
def calculatrice(expression: str) -> str:
    """Effectue des calculs mathématiques. Exemple: '2 + 2', '15 * 3', '100 / 4'"""
    try:
        result = eval(expression)
        return f"Résultat : {result}"
    except Exception as e:
        return f"Erreur de calcul : {str(e)}"
 
# ── Outil 3 : Météo ──────────────────────────────────────────────
@tool
def meteo(ville: str) -> str:
    """Donne la météo actuelle d'une ville."""
    try:
        url = f"https://wttr.in/{ville}?format=3&lang=fr"
        response = requests.get(url, timeout=5)
        return response.text
    except Exception as e:
        return f"Impossible de récupérer la météo pour {ville} : {str(e)}"
 
# ── Outil 4 : Résumé de texte ou URL ────────────────────────────
@tool
def resumer(contenu: str) -> str:
    """
    Résume un texte long ou le contenu d'une URL.
    Passe soit un texte directement, soit une URL commençant par http.
    """
    try:
        if contenu.startswith("http"):
            response = requests.get(contenu, timeout=10)
            texte = response.text[:5000]
        else:
            texte = contenu[:5000]
 
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        prompt = f"Résume ce texte en 5 points clés en français :\n\n{texte}"
        result = llm.invoke(prompt)
        return result.content
    except Exception as e:
        return f"Erreur lors du résumé : {str(e)}"
 
# ── Outil 5 : Générateur de graphique depuis données manuelles ──
@tool
def generer_graphique(donnees_json: str) -> str:
    """
    Génère un graphique à partir de données manuelles.
    Le paramètre doit être un JSON avec :
    - 'type' : 'bar', 'line', 'pie', 'scatter'
    - 'titre' : titre du graphique
    - 'x' : liste des labels
    - 'y' : liste des valeurs
    - 'xlabel' : label axe x (optionnel)
    - 'ylabel' : label axe y (optionnel)
 
    Exemple:
    {
        "type": "bar",
        "titre": "Chiffre d affaires Sanofi",
        "x": ["2022", "2023", "2024"],
        "y": [42, 45, 48],
        "xlabel": "Année",
        "ylabel": "Milliards EUR"
    }
    """
    try:
        data = json.loads(donnees_json)
        type_graphique = data.get("type", "bar")
        titre = data.get("titre", "Graphique")
        x = data.get("x", [])
        y = data.get("y", [])
        xlabel = data.get("xlabel", "")
        ylabel = data.get("ylabel", "")
 
        df = pd.DataFrame({"x": x, "y": y})
 
        if type_graphique == "bar":
            fig = px.bar(df, x="x", y="y", title=titre,
                         labels={"x": xlabel, "y": ylabel},
                         color="y", color_continuous_scale="Blues")
        elif type_graphique == "line":
            fig = px.line(df, x="x", y="y", title=titre,
                          labels={"x": xlabel, "y": ylabel},
                          markers=True)
        elif type_graphique == "pie":
            fig = px.pie(df, names="x", values="y", title=titre)
        elif type_graphique == "scatter":
            fig = px.scatter(df, x="x", y="y", title=titre,
                             labels={"x": xlabel, "y": ylabel})
        else:
            fig = px.bar(df, x="x", y="y", title=titre)
 
        fig.update_layout(template="plotly_white", title_font_size=18)
 
        output_path = "graphique_genere.json"
        fig.write_json(output_path)
        return f"GRAPHIQUE_GENERE:{output_path}|{titre}"
 
    except Exception as e:
        return f"Erreur lors de la génération du graphique : {str(e)}"
 
# ── Outil 6 : Analyse de fichier Excel ──────────────────────────
@tool
def analyser_excel(chemin_fichier: str) -> str:
    """
    Analyse un fichier Excel et retourne des statistiques descriptives.
    Passe le chemin complet du fichier Excel.
    Exemple: 'Data/mon_fichier.xlsx'
    """
    try:
        df = pd.read_excel(chemin_fichier)
 
        analyse = []
        analyse.append(f"📊 **Aperçu** : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        analyse.append(f"\n**Colonnes disponibles** : {', '.join(df.columns.tolist())}")
        analyse.append(f"\n**Statistiques descriptives** :\n{df.describe().to_string()}")
 
        missing = df.isnull().sum()
        if missing.any():
            analyse.append(f"\n**Valeurs manquantes** :\n{missing[missing > 0].to_string()}")
 
        df.to_json("excel_data.json", orient="records")
        analyse.append("\n✅ Données chargées. Tu peux maintenant générer un dashboard.")
 
        return "\n".join(analyse)
    except Exception as e:
        return f"Erreur lors de l'analyse : {str(e)}"
 
# ── Outil 7 : Dashboard depuis Excel ────────────────────────────
@tool
def dashboard_excel(config_json: str) -> str:
    """
    Génère un graphique depuis les données Excel déjà chargées.
    Le paramètre est un JSON avec :
    - 'colonne_x' : nom de la colonne pour l'axe X
    - 'colonne_y' : nom de la colonne pour l'axe Y
    - 'type' : 'bar', 'line', 'pie', 'scatter', 'histogram'
    - 'titre' : titre du graphique
 
    Exemple:
    {
        "colonne_x": "Annee",
        "colonne_y": "Chiffre_affaires",
        "type": "bar",
        "titre": "Evolution du CA"
    }
    """
    try:
        config = json.loads(config_json)
        col_x = config.get("colonne_x")
        col_y = config.get("colonne_y")
        type_graph = config.get("type", "bar")
        titre = config.get("titre", "Dashboard")
 
        if not Path("excel_data.json").exists():
            return "Aucune donnée Excel chargée. Utilisez d'abord l'outil analyser_excel."
 
        df = pd.read_json("excel_data.json")
 
        if type_graph == "bar":
            fig = px.bar(df, x=col_x, y=col_y, title=titre,
                         color=col_y, color_continuous_scale="Viridis")
        elif type_graph == "line":
            fig = px.line(df, x=col_x, y=col_y, title=titre, markers=True)
        elif type_graph == "pie":
            fig = px.pie(df, names=col_x, values=col_y, title=titre)
        elif type_graph == "scatter":
            fig = px.scatter(df, x=col_x, y=col_y, title=titre)
        elif type_graph == "histogram":
            fig = px.histogram(df, x=col_x, title=titre)
        else:
            fig = px.bar(df, x=col_x, y=col_y, title=titre)
 
        fig.update_layout(template="plotly_white", title_font_size=18)
 
        output_path = "dashboard_genere.json"
        fig.write_json(output_path)
        return f"GRAPHIQUE_GENERE:{output_path}|{titre}"
 
    except Exception as e:
        return f"Erreur lors de la génération du dashboard : {str(e)}"
 
 
# ── Construction de l'agent ──────────────────────────────────────
def build_agent():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
 
    tools = [
        search_tool,
        calculatrice,
        meteo,
        resumer,
        generer_graphique,
        analyser_excel,
        dashboard_excel,
    ]
 
    system_prompt = (
        "Tu es un assistant intelligent et polyvalent spécialisé en économie et finance. "
        "Tu as accès aux outils suivants :\n"
        "- 🌍 Recherche web (Tavily) : actualités récentes\n"
        "- 🧮 Calculatrice : calculs mathématiques\n"
        "- 🌤️ Météo : météo actuelle d'une ville\n"
        "- 📝 Résumé : résume un texte ou une URL\n"
        "- 📊 Générateur de graphique : crée des graphiques depuis des données manuelles\n"
        "- 📂 Analyse Excel : analyse un fichier Excel\n"
        "- 📈 Dashboard Excel : génère des graphiques depuis les données Excel\n\n"
        "Utilise toujours l'outil le plus approprié. "
        "Pour les graphiques, génère toujours un JSON valide. "
        "Réponds toujours en français."
    )
 
    agent = create_react_agent(llm, tools, prompt=system_prompt)
    return agent
 
 
# ── Test en terminal ─────────────────────────────────────────────
if __name__ == "__main__":
    agent = build_agent()
    chat_history = []
 
    print("\n🤖 Agent prêt ! (Ctrl+C pour quitter)")
    print("💡 Essaie :")
    print("   - 'Météo à Paris'")
    print("   - 'Résume cet article : https://...'")
    print("   - 'Fais un graphique bar : CA Sanofi 2022=42, 2023=45, 2024=48'")
    print("   - 'Analyse le fichier Data/mon_fichier.xlsx'\n")
 
    while True:
        try:
            question = input("💬 Vous : ")
            if not question.strip():
                continue
 
            result = agent.invoke({
                "messages": chat_history + [HumanMessage(content=question)]
            })
 
            answer = result["messages"][-1].content
            print(f"\n🤖 Agent : {answer}\n")
 
            chat_history.extend([
                HumanMessage(content=question),
                AIMessage(content=answer),
            ])
 
        except KeyboardInterrupt:
            print("\nAu revoir !")
            break
 