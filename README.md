
# 🤖 Assistant Intelligent Multi-Compétences (RAG + Agents)

> Projet réalisé dans le cadre du **DU Sorbonne Data Analytics — Génération IA 2026**

---

## 📌 Description

Cet assistant intelligent combine deux approches complémentaires :

- **RAG (Retrieval-Augmented Generation)** : répond à des questions basées sur des documents financiers récents (rapports Sanofi 2025)
- **Agents & Outils** : effectue des actions dynamiques (recherche web, météo, calculs, graphiques, analyse Excel)

Un **routeur intelligent** décide automatiquement quelle approche utiliser selon la question posée.

---

## 🏗️ Architecture

```
Question utilisateur
        ↓
   🔀 Routeur intelligent
        ↓
┌───────────────────────────────────────┐
│  📄 RAG                               │
│  → Questions sur documents Sanofi     │
│  → Rapports financiers 2025           │
├───────────────────────────────────────┤
│  🤖 Agent (7 outils)                  │
│  → Recherche web (Tavily)             │
│  → Météo                              │
│  → Calculatrice                       │
│  → Résumé texte/URL                   │
│  → Graphiques Plotly                  │
│  → Analyse Excel                      │
│  → Dashboard Excel                    │
├───────────────────────────────────────┤
│  💬 Réponse directe                   │
│  → Conversation simple                │
└───────────────────────────────────────┘
```

---

## 📁 Structure du projet

```
Assistant-RAG-Agents/
├── app.py                  ← Application principale (RAG + Agents + Streamlit)
├── agents.py               ← Script de test des agents en terminal
├── rag_langchain.py        ← Script RAG en terminal
├── Data/                   ← Documents sources (PDFs, Excel)
│   ├── 2025-sanofi-urd.pdf
│   ├── sanofi-rapport-financier-semestriel-2025.pdf
│   └── SWI_Sanofi_Amendement_URD_2025_VMEL.pdf
├── faiss_store/            ← Index vectoriel FAISS (auto-généré)
├── requirements.txt        ← Dépendances Python
├── .env                    ← Variables d'environnement (non versionné)
└── README.md
```

---

## 🚀 Installation et lancement

### 1. Cloner le projet

```bash
git clone https://github.com/TON_PSEUDO/Assistant-RAG-Agents.git
cd Assistant-RAG-Agents
```

### 2. Créer un environnement virtuel

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les clés API

Créer un fichier `.env` à la racine :

```env
OPENAI_API_KEY=sk-...votre_clé_openai...
TAVILY_API_KEY=tvly-...votre_clé_tavily...
```

> 🔑 **OpenAI** : [platform.openai.com/api-keys](https://platform.openai.com/api-keys)  
> 🔑 **Tavily** : [app.tavily.com](https://app.tavily.com) (gratuit — 1000 crédits/mois)

### 5. Ajouter vos documents

Placez vos fichiers `.pdf` ou `.txt` dans le dossier `Data/`.

### 6. Lancer l'application

```bash
streamlit run app.py
```

Ouvrir [http://localhost:8501](http://localhost:8501) dans le navigateur.

---

## 🧩 Fonctionnalités

### 📄 Partie 1 — RAG

- Ingestion et indexation de documents PDF/TXT
- Vectorisation avec **OpenAI text-embedding-3-small**
- Index vectoriel **FAISS** (sauvegardé localement)
- Récupération des 4 chunks les plus pertinents
- Réponse générée par **GPT-4o**
- Affichage des sources et chunks récupérés
- Mémoire conversationnelle

### 🤖 Partie 2 — Agents & Outils

| Outil | Description |
|---|---|
| 🌍 **Tavily Search** | Recherche web en temps réel |
| 🧮 **Calculatrice** | Calculs mathématiques |
| 🌤️ **Météo** | Météo actuelle d'une ville |
| 📝 **Résumé** | Résume un texte ou une URL |
| 📊 **Graphique** | Génère des graphiques depuis des données manuelles |
| 📂 **Analyse Excel** | Statistiques descriptives d'un fichier Excel |
| 📈 **Dashboard Excel** | Génère des graphiques depuis un fichier Excel |

### 🔀 Partie 3 — Intégration RAG + Agents

Routeur intelligent basé sur :
- **Mots clés prioritaires** (Sanofi, rapport, bilan → RAG / météo, actualité → Agent)
- **LLM en fallback** pour les cas ambigus

### 💬 Partie 4 — Interface & Mémoire

- Interface **Streamlit** avec sidebar
- **Upload Excel** via la sidebar
- **Mémoire conversationnelle** (historique des échanges)
- **Badge de routage** visible sur chaque réponse
- Bouton pour effacer la conversation

---

## 💡 Exemples d'utilisation

### Questions RAG (documents Sanofi)
```
Quel est le chiffre d'affaires de Sanofi en 2025 ?
Quelle est la stratégie de Sanofi pour les prochaines années ?
Quels sont les principaux risques identifiés dans le rapport ?
```

### Questions Agent (web & outils)
```
Quelle est la situation en Iran aujourd'hui ?
Météo à Paris ?
Combien fait 1250 * 8.5 ?
Résume cet article : https://...
```

### Graphiques & Excel
```
Fais un graphique bar : CA Sanofi 2022=42Md, 2023=45Md, 2024=48Md
Analyse le fichier Data/mon_fichier.xlsx et génère un dashboard
```

---

## 🛠️ Technologies utilisées

| Technologie | Usage |
|---|---|
| **Python 3.12** | Langage principal |
| **LangChain 1.2+** | Framework LLM |
| **LangGraph** | Agents ReAct |
| **OpenAI GPT-4o** | Modèle de langage |
| **FAISS** | Index vectoriel |
| **Streamlit** | Interface web |
| **Plotly** | Visualisations interactives |
| **Pandas** | Analyse de données |
| **Tavily** | Recherche web |

---

## 👨‍💻 Auteur

**N'Guessan Ebenezer**  
DU Sorbonne Data Analytics — Génération IA 2026

---

## 📄 Licence

Projet académique — Usage éducatif uniquement.