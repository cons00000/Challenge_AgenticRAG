# NOVAGEN Biopharma — RAG Agentique Multimodal
## Étude de cas IA — Mention Master 2

**Groupe 3 :** De Boni Florian · Sakgi Sarra · Piquet Constance · Martin Célestine · Larbi Ylias

---

## Vue d'ensemble du projet

Ce dépôt implémente le système **HERMES** — un pipeline RAG (Retrieval-Augmented Generation) agentique multimodal pour l'environnement pharmaceutique souverain de NOVAGEN Biopharma. Le prototype est construit et évalué sur le dataset public **`vidore/vidore_v3_pharmaceuticals`** (HuggingFace, CC-BY 4.0), avant un éventuel déploiement on-premise sur les données internes de NOVAGEN.

---

## Structure des fichiers

```
projet/
│
├── RAG_Text/
│   ├── textual.ipynb                        ← Bloc 2 Q5 — Approche A (BioBERT)
│   ├── embeddings_textual_corpus.json       ← Cache embeddings corpus BioBERT
│   └── embeddings_textual_queries.json      ← Cache embeddings requêtes BioBERT
│
├── RAG_VDR/
│   ├── CLIP/
│   │   ├── visual_vdr.ipynb                 ← Bloc 2 Q6 — Approche B (CLIP ViT-B/32)
│   │   ├── embeddings_visual_corpus.json    ← Cache embeddings corpus CLIP
│   │   └── embeddings_visual_queries.json   ← Cache embeddings requêtes CLIP
│   │
│   └── Colqwen/
│       ├── visualise_vdr_colqwen.ipynb       ← Bloc 2 Q6 — Approche B (ColQwen2.5)
│       ├── colqwen_corpus_embeddings.pt     ← Cache embeddings corpus ColQwen (PyTorch)
│       └── colqwen_query_embeddings.pt      ← Cache embeddings requêtes ColQwen (PyTorch)
│
├── Analyse/
│   └── analyse.ipynb                        ← Bloc 2 Q4 — Analyse exploratoire
│
├── RAG_Hybride/
│   ├── hybride_novagen.ipynb            ← Q7/Q8 avec CLIP dense (lit les .json)
│   └── hybride_novagen_colqwen.ipynb    ← Q7/Q8 avec ColQwen MaxSim (lit les .pt)
│
└── RAG_Agentique/
    ├── agentic.py                           ← Bloc 3 — Agent ReAct HERMES
    └── trace_hermes.md                      ← Trace d'exécution Q10 annotée
```

---

## Description détaillée de chaque fichier

### `RAG_Text/textual.ipynb` — *Bloc 2 Q5 — Baseline textuel (Approche A)*
Pipeline de retrieval textuel complet basé sur **BioBERT** (`pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`). Ce notebook est autonome : il charge le dataset depuis HuggingFace, encode les 2 313 pages et 2 184 requêtes, calcule le nDCG@5, analyse qualitativement les 10 pires résultats, et sauvegarde les embeddings.

**Résultat obtenu :** nDCG@5 = **0.1029** (médiane 0.0, 78.5 % des requêtes à zéro).

**Fichiers produits :**
- `embeddings_textual_corpus.json` — dict `{corpus_id: [float, ...]}`, 2 313 entrées, dim 768
- `embeddings_textual_queries.json` — dict `{query_id: [float, ...]}`, 2 184 entrées, dim 768

---

### `RAG_VDR/CLIP/visual_vdr.ipynb` — *Bloc 2 Q6 — Visual Document Retrieval (Approche B, CLIP)*
Pipeline VDR basé sur **CLIP ViT-B/32** (`sentence-transformers/clip-ViT-B-32`). Traite les pages comme des images (dim 512). Ce notebook est autonome : charge le dataset, encode le corpus image par image, calcule le nDCG@5, et sauvegarde les embeddings.

> **Note :** Cette approche utilise CLIP (modèle généraliste). Elle sert de point de comparaison intermédiaire entre le pur textuel BioBERT et le spécialisé ColQwen2.5. Voir `visualise_vdr_colqwen.ipynb` pour l'approche VDR de référence.

**Fichiers produits :**
- `embeddings_visual_corpus.json` — dict `{corpus_id: [float, ...]}`, dim 512
- `embeddings_visual_queries.json` — dict `{query_id: [float, ...]}`, dim 512

---

### `RAG_VDR/Colqwen/visualise_vdr_colqwen.ipynb` — *Bloc 2 Q6 — VDR ColQwen2.5 (Approche B, référence)*
Pipeline VDR de référence basé sur **ColQwen2.5-v0.2** (`vidore/colqwen2.5-v0.2`), modèle SOTA sur ViDoRe V3. Utilise la late interaction (MaxSim) pour comparer les embeddings multi-vectoriels. Ce notebook est autonome : charge le dataset, encode le corpus en images, calcule le nDCG@5, visualise les bounding boxes pour valider que le modèle « regarde » les bonnes zones.

> **Note :** Requiert un GPU avec ≥ 6 Go VRAM. Le notebook vérifie la présence des `.pt` avant de recharger le modèle.

**Fichiers produits :**
- `colqwen_corpus_embeddings.pt` — tenseur PyTorch `{embeddings: [...], corpus_ids: [...]}`
- `colqwen_query_embeddings.pt` — tenseur PyTorch `{embeddings: [...], query_ids: [...]}`

---

### `Analyse/analyse.ipynb` — *Bloc 2 Q4 — Analyse exploratoire du dataset*
Analyse descriptive complète du dataset ViDoRe V3 Pharmaceuticals. Ce notebook est autonome et ne requiert aucun fichier préalable. Il charge les trois subsets (corpus, queries, qrels), produit des visualisations sur la distribution des types de documents (52 doc_id), la nature des requêtes annotées, et la distribution des scores de pertinence dans les qrels. Identifie 5 requêtes pour lesquelles le RAG textuel est anticipé en échec, avec visualisation des pages concernées.

**Aucun fichier produit** (notebook exploratoire pur, pas de sauvegarde).

---

### `Bloc2_Q7Q8/hybride_novagen.ipynb` — *Bloc 2 Q7+Q8 — Analyse différentielle et hybride (CLIP)*
Analyse différentielle entre l'Approche A (BioBERT) et l'Approche B (CLIP). Ce notebook est autonome pour les calculs mais **lit les fichiers de cache JSON** produits par `textual.ipynb` et `visual_vdr.ipynb` pour éviter de réencoder. Identifie les 20 requêtes les plus divergentes entre les deux approches, les catégorise (tableaux / figures / narratif / mise en page), et implémente deux stratégies hybrides : **RRF** (Reciprocal Rank Fusion) et **WSF** (Weighted Score Fusion avec sweep α).

**Dépendances :**
- `RAG_Text/embeddings_textual_corpus.json`
- `RAG_Text/embeddings_textual_queries.json`
- `RAG_VDR/CLIP/embeddings_visual_corpus.json`
- `RAG_VDR/CLIP/embeddings_visual_queries.json`

---

### `RAG_Hybride/hybride_novagen_colqwen.ipynb` — *Bloc 2 Q7+Q8 — Analyse différentielle et hybride (ColQwen2.5)*
Version de référence de l'analyse différentielle, utilisant **ColQwen2.5** à la place de CLIP comme modèle VDR. Compare trois systèmes : BioBERT (textuel), CLIP (dense VDR), et ColQwen2.5 (MaxSim). Ce notebook est autonome pour les calculs mais **lit les fichiers de cache** produits par les notebooks précédents. Inclut un cache intermédiaire `colqwen_results_q7q8.pt` pour les scores MaxSim (calcul long : ~40 min sur CPU).

**Résultats obtenus :** BioBERT nDCG@5 = 0.1032 · ColQwen nDCG@5 = **0.5718**

**Dépendances :**
- `RAG_Text/embeddings_textual_corpus.json`
- `RAG_Text/embeddings_textual_queries.json`
- `RAG_VDR/CLIP/embeddings_visual_corpus.json`
- `RAG_VDR/CLIP/embeddings_visual_queries.json`
- `RAG_VDR/Colqwen/colqwen_corpus_embeddings.pt`
- `RAG_VDR/Colqwen/colqwen_query_embeddings.pt`

**Fichier cache intermédiaire produit :**
- `colqwen_results_q7q8.pt` — scores MaxSim pré-calculés pour toutes les paires (query, corpus)

---

### `RAG_Agentique/agentic.py` — *Bloc 3 — Agent ReAct HERMES*
Implémentation de la couche agentique complète du système HERMES. Ce script est autonome et s'exécute depuis la racine du projet. Il charge les embeddings BioBERT (`.json`) et ColQwen2.5 (`.pt`) pour alimenter les deux backends de retrieval, et expose un agent **ReAct** (Reasoning + Acting) via LangChain avec **Mistral API** comme LLM de génération.

**Architecture interne :**
- `TextRetriever` — retrieval BioBERT par similarité cosinus
- `ColQwenRetriever` — retrieval ColQwen2.5 par MaxSim (late interaction)
- `modal_router(query)` — détection automatique de la modalité optimale (lexical signal text/visual)
- `make_tools(...)` — fabrique les outils ReAct : `modal_router`, `text_retrieval`, `visual_retrieval`, `verify_claim`
- `AgentTrace` — dataclass de journalisation des étapes Thought/Act/Observe (traçabilité GxP)
- `build_agent(...)` — constructeur de l'agent ReAct
- `run_agent_with_trace(...)` — boucle d'exécution avec capture de la trace complète

**Variable d'environnement requise :** `MISTRAL_API_KEY`

**Dépendances fichiers :**
- `RAG_Text/embeddings_textual_corpus.json`
- `RAG_Text/embeddings_textual_queries.json`
- `RAG_VDR/Colqwen/colqwen_corpus_embeddings.pt`
- `RAG_VDR/Colqwen/colqwen_query_embeddings.pt`

**Fichier produit :**
- `RAG_Agentique/trace_hermes.md` — trace d'exécution ReAct au format Markdown

---

### `RAG_Agentique/trace_hermes.md` — *Bloc 3 Q10 — Trace d'exécution annotée*
Trace d'exécution complète de l'agent HERMES sur la requête : *"En comparant les cas FAERS rapportés sur la buprénorphine, quels sont les facteurs de risque associés aux caries ?"*. Illustre le cycle complet ReAct : `modal_router` → `text_retrieval` → `visual_retrieval` → réponse finale synthétisée avec sources citées. Sert de livrable pour la question 10 du Bloc 3.

---

## Pourquoi les fichiers sont-ils dupliqués entre dossiers ?

**Chaque notebook est conçu pour être exécuté en standalone**, sans dépendre d'un autre notebook déjà lancé. C'est un choix délibéré de robustesse : chaque notebook recharge le dataset depuis HuggingFace et réimporte les bibliothèques nécessaires. En conséquence, les fichiers d'embeddings (`.json` et `.pt`) sont **produits par un notebook et consommés par d'autres**, mais jamais générés de novo dans les notebooks consommateurs — ils vérifient la présence du cache et le chargent directement.

La duplication apparente concerne principalement :
- `colqwen_corpus_embeddings.pt` / `colqwen_query_embeddings.pt` : produits par `visualise_vdr_colqwen.ipynb`, lus par `hybride_novagen_colqwen.ipynb` et `agentic.py`
- `embeddings_textual_*.json` : produits par `textual.ipynb`, lus par les deux notebooks Q7/Q8 et `agentic.py`

---

## Prérequis techniques

```bash
pip install datasets sentence-transformers colpali-engine torch torchvision \
            scikit-learn numpy pandas matplotlib seaborn tqdm pillow \
            langchain mistralai vidore-benchmark
```

**GPU recommandé :** ≥ 6 Go VRAM pour ColQwen2.5. Les notebooks BioBERT et CLIP tournent sur CPU.

**Variable d'environnement :**
```bash
export MISTRAL_API_KEY="votre-clé"   # Linux/macOS
$env:MISTRAL_API_KEY = "votre-clé"   # Windows PowerShell
```

---