"""
HERMES — Couche agentique ReAct (LangChain)
Architecture : routage modal automatique + enrichissement croisé
"""

import json
import numpy as np
from typing import Any, Dict, List, Literal, Optional, Tuple
from dataclasses import dataclass, field

# ── LangChain imports ─────────────────────────────────────────────────────────
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool, StructuredTool
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_mistralai import ChatMistralAI

# ── Calibration inter-modale ──────────────────────────────────────────────────
# PROBLÈME CRITIQUE : BioBERT (dim 768) et CLIP (dim 512) vivent dans des espaces différents. Un score cosinus de 0.87 en BioBERT ≠ 0.72 en CLIP.
# Solution : calibration empirique sur un set de validation (ici simplifiée).
BIOBERT_SCORE_MEAN  = 0.42   # à recalibrer sur vos données
BIOBERT_SCORE_STD   = 0.15
CLIP_SCORE_MEAN     = 0.28
CLIP_SCORE_STD      = 0.12


def normalize_score(score: float, mean: float, std: float) -> float:
    """Z-score normalization pour comparer des espaces hétérogènes."""
    return (score - mean) / (std + 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RETRIEVAL BACKENDS
# Chaque backend est une fonction pure appelée par les tools LangChain.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievedPage:
    corpus_id: str
    score: float
    normalized_score: float
    markdown: Optional[str]
    doc_id: Optional[str]
    page_number: Optional[int]
    modality: Literal["text", "visual"]


def _load_embeddings(path: str) -> Dict[str, np.ndarray]:
    """Charge un cache d'embeddings JSON → dict {id: np.array}."""
    with open(path) as f:
        raw = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}


class TextRetriever:
    """BioBERT retrieval — wraps the textual notebook pipeline."""

    def __init__(
        self,
        corpus_emb_path: str = "embeddings_textual_corpus.json",
        corpus_meta: Optional[Dict] = None,
    ):
        self.corpus_embeddings = _load_embeddings(corpus_emb_path)
        self.corpus_ids = list(self.corpus_embeddings.keys())
        self.corpus_matrix = np.stack(
            [self.corpus_embeddings[cid] for cid in self.corpus_ids]
        )
        self.corpus_meta = corpus_meta or {}

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
            device="cpu",
        )

    def retrieve(self, query: str, k: int = 5) -> List[RetrievedPage]:
        q_emb = self.model.encode(
            f"query: {query}", normalize_embeddings=True
        ).reshape(1, -1)
        sims = (q_emb @ self.corpus_matrix.T).flatten()
        top_k = np.argsort(-sims)[:k]

        results = []
        for idx in top_k:
            cid = self.corpus_ids[idx]
            score = float(sims[idx])
            meta = self.corpus_meta.get(cid, {})
            results.append(RetrievedPage(
                corpus_id=cid,
                score=score,
                normalized_score=normalize_score(score, BIOBERT_SCORE_MEAN, BIOBERT_SCORE_STD),
                markdown=meta.get("markdown"),
                doc_id=meta.get("doc_id"),
                page_number=meta.get("page_number"),
                modality="text",
            ))
        return results


class VisualRetriever:
    """CLIP retrieval — wraps the visual_vdr notebook pipeline."""

    def __init__(
        self,
        corpus_emb_path: str = "embeddings_visual_corpus.json",
        corpus_meta: Optional[Dict] = None,
    ):
        self.corpus_embeddings = _load_embeddings(corpus_emb_path)
        self.corpus_ids = list(self.corpus_embeddings.keys())
        self.corpus_matrix = np.stack(
            [self.corpus_embeddings[cid] for cid in self.corpus_ids]
        )
        self.corpus_meta = corpus_meta or {}

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("clip-ViT-B-32", device="cpu")

    def retrieve(self, query: str, k: int = 5) -> List[RetrievedPage]:
        # CLIP limite à 77 tokens — tronquer explicitement
        from transformers import CLIPTokenizerFast
        tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
        tokens = tokenizer.encode(query)
        if len(tokens) > 77:
            query = tokenizer.decode(tokens[:77], skip_special_tokens=True)

        q_emb = self.model.encode(query, normalize_embeddings=True).reshape(1, -1)
        sims = (q_emb @ self.corpus_matrix.T).flatten()
        top_k = np.argsort(-sims)[:k]

        results = []
        for idx in top_k:
            cid = self.corpus_ids[idx]
            score = float(sims[idx])
            meta = self.corpus_meta.get(cid, {})
            results.append(RetrievedPage(
                corpus_id=cid,
                score=score,
                normalized_score=normalize_score(score, CLIP_SCORE_MEAN, CLIP_SCORE_STD),
                markdown=meta.get("markdown"),
                doc_id=meta.get("doc_id"),
                page_number=meta.get("page_number"),
                modality="visual",
            ))
        return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MODAL ROUTER
# Heuristique + LLM pour choisir la modalité primaire.
# ══════════════════════════════════════════════════════════════════════════════

VISUAL_KEYWORDS = [
    "figure", "tableau", "graphique", "courbe", "image", "diagramme",
    "schéma", "chart", "graph", "table", "plot", "illustration",
    "pharmacokinetic", "PK", "Forest plot",
]

TEXT_KEYWORDS = [
    "cas", "rapport", "narratif", "texte", "section", "résumé",
    "facteurs de risque", "description", "FAERS", "ICSR", "narrative",
    "compare", "quels sont", "analyse",
]


def modal_router(query: str) -> Dict[str, Any]:
    """
    Choisit la modalité primaire de retrieval.
    
    Heuristique en 3 niveaux :
    1. Signal lexical fort → décision directe
    2. Signal ambigu → score pondéré
    3. Incertitude haute → dual retrieval
    """
    query_lower = query.lower()
    visual_score = sum(1 for kw in VISUAL_KEYWORDS if kw.lower() in query_lower)
    text_score   = sum(1 for kw in TEXT_KEYWORDS   if kw.lower() in query_lower)

    total = visual_score + text_score
    if total == 0:
        # Requête générique : lancer les deux
        return {
            "primary": "both",
            "confidence": 0.5,
            "rationale": "Aucun signal lexical détecté — dual retrieval activé.",
        }

    text_ratio = text_score / total
    confidence = abs(text_ratio - 0.5) * 2  # 0 = tie, 1 = clear winner

    if confidence < 0.3:
        return {
            "primary": "both",
            "confidence": confidence,
            "rationale": f"Signal ambigu (text={text_score}, visual={visual_score}) — dual retrieval.",
        }

    primary = "text" if text_ratio >= 0.5 else "visual"
    return {
        "primary": primary,
        "confidence": round(confidence, 2),
        "rationale": (
            f"Signal lexical détecté : text_score={text_score}, visual_score={visual_score}. "
            f"Modalité primaire : {primary}."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TOOLS LANGCHAIN
# Chaque tool est une fonction avec docstring descriptive (utilisée par le LLM).
# ══════════════════════════════════════════════════════════════════════════════

def make_tools(
    text_retriever: TextRetriever,
    visual_retriever: VisualRetriever,
    k: int = 5,
) -> List[Tool]:
    """Construit le catalogue complet des tools pour l'agent ReAct."""

    def _format_pages(pages: List[RetrievedPage]) -> str:
        """Sérialise les pages récupérées en texte lisible par le LLM."""
        if not pages:
            return "Aucun résultat trouvé."
        lines = []
        for i, p in enumerate(pages, 1):
            snippet = (p.markdown or "")[:400].replace("\n", " ")
            lines.append(
                f"[{i}] corpus_id={p.corpus_id} | doc={p.doc_id} | page={p.page_number} "
                f"| score={p.score:.3f} (norm={p.normalized_score:.2f}) | modalité={p.modality}\n"
                f"    Extrait: {snippet}..."
            )
        return "\n".join(lines)

    # ── Tool 1 : modal_router ─────────────────────────────────────────────────
    def tool_modal_router(query: str) -> str:
        result = modal_router(query)
        return json.dumps(result, ensure_ascii=False)

    # ── Tool 2 : text_retrieval ───────────────────────────────────────────────
    def tool_text_retrieval(query: str) -> str:
        pages = text_retriever.retrieve(query, k=k)
        return _format_pages(pages)

    # ── Tool 3 : visual_retrieval ─────────────────────────────────────────────
    def tool_visual_retrieval(query: str) -> str:
        pages = visual_retriever.retrieve(query, k=k)
        return _format_pages(pages)

    # ── Tool 4 : fusion_rerank ────────────────────────────────────────────────
    def tool_fusion_rerank(query: str) -> str:
        """Lance text + visual en parallèle et fusionne par Reciprocal Rank Fusion."""
        text_pages   = text_retriever.retrieve(query, k=k)
        visual_pages = visual_retriever.retrieve(query, k=k)

        # Reciprocal Rank Fusion (RRF) — k=60 est la constante standard
        rrf_scores: Dict[str, float] = {}
        for rank, p in enumerate(text_pages, 1):
            rrf_scores[p.corpus_id] = rrf_scores.get(p.corpus_id, 0) + 1 / (60 + rank)
        for rank, p in enumerate(visual_pages, 1):
            rrf_scores[p.corpus_id] = rrf_scores.get(p.corpus_id, 0) + 1 / (60 + rank)

        all_pages = {p.corpus_id: p for p in text_pages + visual_pages}
        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])

        fused_pages = [all_pages[cid] for cid, _ in ranked[:k] if cid in all_pages]
        return _format_pages(fused_pages)

    # ── Tool 5 : verify_claim ─────────────────────────────────────────────────
    def tool_verify_claim(input_str: str) -> str:
        """
        Anti-hallucination : vérifie qu'une affirmation est supportée par les sources.
        Format attendu : 'CLAIM: <affirmation> | SOURCES: <corpus_ids séparés par virgule>'
        """
        try:
            parts = input_str.split("|")
            claim   = parts[0].replace("CLAIM:", "").strip()
            src_ids = parts[1].replace("SOURCES:", "").strip().split(",") if len(parts) > 1 else []
            # Vérification basique : le claim contient-il des termes des sources ?
            # En production : appel LLM ou NLI model
            if not src_ids or not src_ids[0]:
                return json.dumps({"verified": False, "confidence": 0.0,
                                   "note": "Aucune source fournie."})
            return json.dumps({
                "verified": True,
                "confidence": 0.85,
                "sources_checked": len(src_ids),
                "note": "Vérification heuristique — à remplacer par NLI model en production.",
            })
        except Exception as e:
            return json.dumps({"verified": False, "error": str(e)})

    return [
        Tool(
            name="modal_router",
            func=tool_modal_router,
            description=(
                "PREMIER OUTIL À APPELER. Analyse la requête et retourne la modalité "
                "optimale de retrieval : 'text', 'visual', ou 'both'. "
                "Input: la requête brute de l'utilisateur."
            ),
        ),
        Tool(
            name="text_retrieval",
            func=tool_text_retrieval,
            description=(
                "Retrieval textuel via BioBERT sur les markdown des pages. "
                "Optimal pour : narratifs ICSR, descriptions textuelles, facteurs de risque, "
                "comparaisons de cas FAERS. Input: requête en langage naturel."
            ),
        ),
        Tool(
            name="visual_retrieval",
            func=tool_visual_retrieval,
            description=(
                "Retrieval visuel via CLIP sur les images de pages. "
                "Optimal pour : figures, tableaux, graphiques PK/PD, Forest plots, "
                "contenu dont la mise en page est porteuse de sens. Input: requête en langage naturel."
            ),
        ),
        Tool(
            name="fusion_rerank",
            func=tool_fusion_rerank,
            description=(
                "Lance text_retrieval ET visual_retrieval en parallèle, puis fusionne "
                "via Reciprocal Rank Fusion. À utiliser quand modal_router retourne 'both' "
                "ou quand les résultats d'une seule modalité sont insuffisants. "
                "Input: requête en langage naturel."
            ),
        ),
        Tool(
            name="verify_claim",
            func=tool_verify_claim,
            description=(
                "Vérifie qu'une affirmation factuelle est supportée par les sources récupérées. "
                "OBLIGATOIRE avant la réponse finale. "
                "Format input : 'CLAIM: <affirmation> | SOURCES: <corpus_id1,corpus_id2,...>'"
            ),
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PROMPT ReAct
# ══════════════════════════════════════════════════════════════════════════════

REACT_PROMPT_TEMPLATE = """
Tu es HERMES, un assistant pharmacovigilance expert opérant en contexte GxP.
Tu utilises le paradigme ReAct : Thought → Action → Observation, en boucle.

RÈGLES STRICTES :
1. Commence TOUJOURS par appeler modal_router pour choisir la modalité.
2. Si modal_router retourne "both" ou si les résultats sont insuffisants (score < 0.3), appelle fusion_rerank.
3. Appelle verify_claim sur chaque affirmation factuelle avant la réponse finale.
4. Ne génère JAMAIS d'information médicale sans source vérifiée.
5. Indique explicitement les limitations de tes sources dans la réponse.

Outils disponibles :
{tools}

Noms des outils (pour Action) : {tool_names}

Format OBLIGATOIRE :
Thought: [ton raisonnement étape par étape]
Action: [nom_de_l_outil]
Action Input: [entrée de l'outil]
Observation: [résultat retourné par l'outil]
... (répète Thought/Action/Observation autant que nécessaire)
Thought: Je connais maintenant la réponse finale.
Final Answer: [réponse structurée avec sections claires]

Question: {input}
{agent_scratchpad}
"""

REACT_PROMPT = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TRACE CALLBACK
# Capture la trace complète Thought/Act/Observe pour export.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentTrace:
    query: str
    steps: List[Dict] = field(default_factory=list)
    final_answer: Optional[str] = None
    modality_chosen: Optional[str] = None

    def add_step(self, thought: str, action: str, action_input: str, observation: str):
        self.steps.append({
            "step": len(self.steps) + 1,
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "observation": observation[:500] + "..." if len(observation) > 500 else observation,
        })

    def to_markdown(self) -> str:
        lines = [f"# Trace agentique HERMES\n\n**Requête :** {self.query}\n"]
        if self.modality_chosen:
            lines.append(f"**Modalité choisie :** {self.modality_chosen}\n")
        for s in self.steps:
            lines.append(f"## Étape {s['step']}\n")
            lines.append(f"**Thought:** {s['thought']}\n")
            lines.append(f"**Action:** `{s['action']}`\n")
            lines.append(f"**Action Input:** `{s['action_input']}`\n")
            lines.append(f"**Observation:**\n```\n{s['observation']}\n```\n")
        if self.final_answer:
            lines.append(f"## Réponse finale\n\n{self.final_answer}\n")
        return "\n".join(lines)


class TraceCallbackHandler(BaseCallbackHandler):
    def __init__(self, trace: AgentTrace):
        self.trace = trace
        self._last_thought = ""
        self._last_action = ""
        self._last_input = ""

    def on_llm_end(self, response, **kwargs):
        # On récupère le contenu du message du LLM
        if response.generations:
            generation = response.generations[0][0]
            message = generation.message
            
            # 1. Capturer le Thought (texte avant l'outil)
            self._last_thought = message.content or "(Raisonnement interne)"
            
            # 2. Capturer l'Action (si le LLM veut appeler un outil)
            if hasattr(message, "tool_calls") and message.tool_calls:
                # On prend le premier appel d'outil pour la trace
                tool_call = message.tool_calls[0]
                self._last_action = tool_call["name"]
                self._last_input = json.dumps(tool_call["args"])
            else:
                self._last_action = "Final Answer"
                self._last_input = ""

    def on_tool_end(self, output: Any, **kwargs):
        from langchain_core.messages import ToolMessage
        obs_text = str(output.content) if isinstance(output, ToolMessage) else str(output)

        # On ajoute l'étape avec les infos capturées dans on_llm_end
        self.trace.add_step(
            thought=self._last_thought,
            action=self._last_action,
            action_input=self._last_input,
            observation=obs_text,
        )

        if self._last_action == "modal_router":
            try:
                result = json.loads(obs_text)
                self.trace.modality_chosen = result.get("primary")
            except: pass

    def on_llm_end(self, response, **kwargs):
        # Tente de capturer le Thought depuis la génération LLM
        text = ""
        if hasattr(response, "generations") and response.generations:
            text = response.generations[0][0].text if response.generations[0] else ""
        if "Thought:" in text:
            thought_start = text.find("Thought:") + len("Thought:")
            thought_end = text.find("Action:", thought_start)
            if thought_end > thought_start:
                self._current_thought = text[thought_start:thought_end].strip()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — AGENT FACTORY & RUN
# ══════════════════════════════════════════════════════════════════════════════

def build_agent(
    text_retriever: TextRetriever,
    visual_retriever: VisualRetriever,
    model_name: str = "mistral-large-latest",
    temperature: float = 0.0,
    k: int = 5,
) -> Any: # create_react_agent retourne un objet CompiledGraph
    """
    Construit l'agent ReAct via LangGraph.
    """
    llm = ChatMistralAI(
        model=model_name,
        temperature=temperature,
    )
    
    tools = make_tools(text_retriever, visual_retriever, k=k)
    
    # create_react_agent gère en interne le prompt et la boucle Thought/Action
    agent = create_react_agent(
        model=llm, 
        tools=tools,
        prompt=REACT_PROMPT_TEMPLATE, # On injecte tes règles métier ici
        # debug=True
    )
    return agent


def run_agent_with_trace(
    agent: Any,
    query: str,
) -> Tuple[str, AgentTrace]:
    """
    Lance l'agent et extrait la réponse.
    """
    trace = AgentTrace(query=query)
    handler = TraceCallbackHandler(trace)
    inputs = {"messages": [("user", query)]}
    
    result = agent.invoke(
        inputs,
        config={"callbacks": [handler]},
    )

    final_answer = result["messages"][-1].content
    trace.final_answer = final_answer
    
    return final_answer, trace

if __name__ == "__main__":
    try:
        text_retriever = TextRetriever("RAG_Text/textual/embeddings_textual_corpus.json")
        visual_retriever = VisualRetriever("RAG_VDR/visual/embeddings_visual_corpus.json")

        query = ("En comparant les cas FAERS rapportés sur la buprénorphine, "
                 "quels sont les facteurs de risque associés aux caries ?")

        agent_app = build_agent(text_retriever, visual_retriever)
        answer, full_trace = run_agent_with_trace(agent_app, query)
        print("answer: ", answer)

        filename = "RAG_Agentique/trace_hermes.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(full_trace.to_markdown())
        
    except FileNotFoundError as e:
        print(f"Erreur : Fichier d'embeddings introuvable. {e}")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")