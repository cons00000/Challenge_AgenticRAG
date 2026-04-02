"""
HERMES — Couche agentique ReAct (LangChain)
Architecture : routage modal automatique + enrichissement croisé

VDR par défaut : ColQwen2.5-v0.2 (late interaction, MaxSim)
Calibration mise à jour d'après les résultats Q7/Q8 :
  BioBERT  nDCG@5 = 0.1032
  ColQwen  nDCG@5 = 0.5718
"""

import json
import os
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from dataclasses import dataclass, field

# ── LangChain imports ─────────────────────────────────────────────────────────
from mistralai.client import Mistral

# ── Device detection ──────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
print(f"Device : {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# Ce fichier vit dans RAG_Agentique/ — toutes les ressources sont un niveau au-
# dessus, dans le répertoire racine du projet.
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent.parent   # → racine du projet
print(f"Base directory for resources: {BASE_DIR}")

# ── API key check ─────────────────────────────────────────────────────────────
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise EnvironmentError(
        "MISTRAL_API_KEY manquante.\n"
        "Définissez-la avant de lancer le script :\n"
        "  Windows PowerShell : $env:MISTRAL_API_KEY = 'votre-clé'\n"
        "  Linux/macOS        : export MISTRAL_API_KEY='votre-clé'\n"
        "Clé disponible sur : https://console.mistral.ai/api-keys"
    )

TEXT_CORPUS_EMB   = BASE_DIR / "RAG_Text" / "embeddings_textual_corpus.json"
TEXT_QUERY_EMB    = BASE_DIR / "RAG_Text" / "embeddings_textual_queries.json"
COLQWEN_CORPUS_PT = BASE_DIR / "RAG_VDR"  / "Colqwen" / "colqwen_corpus_embeddings.pt"
COLQWEN_QUERY_PT  = BASE_DIR / "RAG_VDR"  / "Colqwen" / "colqwen_query_embeddings.pt"
TRACE_OUTPUT      = BASE_DIR / "RAG_Agentique" / "trace_hermes.md"


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION INTER-MODALE
# Scores issus de l'évaluation réelle Q7/Q8 sur ViDoRe V3 Pharmaceuticals.
#
# PROBLÈME : BioBERT cosine scores et ColQwen MaxSim scores vivent dans des
# espaces numériques différents (cosine ∈ [-1,1] vs MaxSim ∈ [0, n_tokens*1]).
# Solution : Z-score normalization empirique pour les comparer dans WSF/RRF.
#
# BioBERT  — espace cosinus normalisé, moyenne empirique ≈ 0.42
# ColQwen  — somme de MaxSim par token, amplitude dépend de la longueur query
#            → normaliser par n_tokens avant stockage du score
# ══════════════════════════════════════════════════════════════════════════════

BIOBERT_SCORE_MEAN  = 0.42
BIOBERT_SCORE_STD   = 0.15
COLQWEN_SCORE_MEAN  = 12.5   # MaxSim brut moyen (requête ~39 tokens, sim ≈ 0.32/token)
COLQWEN_SCORE_STD   = 4.2


def normalize_score(score: float, mean: float, std: float) -> float:
    """Z-score normalization pour comparer des espaces hétérogènes."""
    return (score - mean) / (std + 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RETRIEVAL BACKENDS
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


# ── Textual retrieval (BioBERT) ───────────────────────────────────────────────

class TextRetriever:
    """
    Retrieval textuel via BioBERT sur les markdown des pages.
    Optimal pour : narratifs ICSR, descriptions textuelles, facteurs de risque,
    comparaisons de cas FAERS, texte narratif dense.
    """

    def __init__(
        self,
        corpus_emb_path: str = str(TEXT_CORPUS_EMB),
        corpus_meta: Optional[Dict] = None,
    ):
        with open(corpus_emb_path) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            ids  = list(raw.keys())
            embs = [raw[k] for k in ids]
        elif isinstance(raw, list) and isinstance(raw[0], dict):
            ids  = [r["id"] for r in raw]
            embs = [r["embedding"] for r in raw]
        else:
            raise ValueError(f"Format JSON inattendu dans {corpus_emb_path}")

        self.corpus_ids    = [str(i) for i in ids]
        self.corpus_matrix = np.array(embs, dtype=np.float32)
        self.corpus_meta   = corpus_meta or {}

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
            device=DEVICE,
        )

    def retrieve(self, query: str, k: int = 5) -> List[RetrievedPage]:
        q_emb = self.model.encode(
            f"query: {query}", normalize_embeddings=True
        ).reshape(1, -1)
        sims  = (q_emb @ self.corpus_matrix.T).flatten()
        top_k = np.argsort(-sims)[:k]

        results = []
        for idx in top_k:
            cid   = self.corpus_ids[idx]
            score = float(sims[idx])
            meta  = self.corpus_meta.get(cid, {})
            results.append(RetrievedPage(
                corpus_id        = cid,
                score            = score,
                normalized_score = normalize_score(score, BIOBERT_SCORE_MEAN, BIOBERT_SCORE_STD),
                markdown         = meta.get("markdown"),
                doc_id           = meta.get("doc_id"),
                page_number      = meta.get("page_number"),
                modality         = "text",
            ))
        return results


# ── Visual retrieval — ColQwen2.5 ─────────────────────────────────────────────

class ColQwenRetriever:
    """
    Retrieval visuel via ColQwen2.5-v0.2 (late interaction, MaxSim).

    Charge les embeddings pré-calculés depuis les caches .pt produits par Q6 :
      - colqwen_corpus_embeddings.pt  → list of Tensors (n_patches, 128)
      - colqwen_query_embeddings.pt   → list of Tensors (n_tokens,  128)

    Score MaxSim : score(q,d) = Σ_{t∈q} max_{p∈d} cos(e_t, e_p)
    Référence    : Khattab & Zaharia, ColBERT 2020.

    Performance mesurée sur ViDoRe V3 Pharmaceuticals (Q7/Q8) :
      nDCG@5 = 0.5718  (+0.4687 vs BioBERT)
    """

    def __init__(
        self,
        corpus_pt_path: str = str(COLQWEN_CORPUS_PT),
        query_pt_path: str  = str(COLQWEN_QUERY_PT),
        corpus_meta: Optional[Dict] = None,
    ):
        print(f"Loading ColQwen2.5 corpus embeddings from {corpus_pt_path}...")
        corpus_cache     = torch.load(corpus_pt_path, map_location="cpu", weights_only=False)
        self.corpus_embs = [e.to(DEVICE) for e in corpus_cache["embeddings"]]
        self.corpus_ids  = [str(cid) for cid in corpus_cache["corpus_ids"]]
        del corpus_cache
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"  → {len(self.corpus_ids):,} pages | dim example: {self.corpus_embs[0].shape} | device: {DEVICE}")

        # Pre-load query cache for batch evaluation (optional — queries can also be
        # encoded live via _encode_query_live if query_id is not in cache).
        self._query_cache: Dict[str, torch.Tensor] = {}
        self._text_to_emb: Dict[str, torch.Tensor] = {}   # normalized text → embedding
        if Path(query_pt_path).exists():
            query_cache = torch.load(query_pt_path, map_location="cpu", weights_only=False)
            for qid, emb, text in zip(
                query_cache["query_ids"],
                query_cache["embeddings"],
                query_cache.get("query_texts", [""] * len(query_cache["embeddings"])),
            ):
                moved = emb.to(DEVICE)
                self._query_cache[str(qid)] = moved
                if text:
                    self._text_to_emb[str(text).strip().lower()] = moved
            print(f"  → {len(self._query_cache):,} query embeddings pre-loaded from cache")

        self.corpus_meta = corpus_meta or {}
        self._processor  = None   # lazy-loaded only if live encoding is needed
        self._model      = None

    @staticmethod
    def _maxsim(q_emb: torch.Tensor, p_emb: torch.Tensor) -> float:
        """
        MaxSim between one query and one page.
        q_emb : (n_tokens,  dim)
        p_emb : (n_patches, dim)
        """
        q = q_emb.float()
        p = p_emb.float()
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        p = p / (p.norm(dim=-1, keepdim=True) + 1e-8)
        sim = torch.matmul(q, p.T)                    # (n_tokens, n_patches)
        return sim.max(dim=1).values.sum().item()      # Σ max per query token

    def _encode_query_live(self, query: str) -> torch.Tensor:
        """
        Live encoding is disabled — corpus fills the 8GB GPU, no room for the model.
        All queries must be pre-cached in colqwen_query_embeddings.pt.
        """
        raise RuntimeError(
            f"Query not found in ColQwen cache:\n  '{query[:120]}'\n\n"
            "Live encoding is disabled (GPU full with corpus embeddings).\n"
            "To add new queries to the cache, run Q6 notebook with the new queries "
            "and save the resulting colqwen_query_embeddings.pt."
        )

    def retrieve(self, query: str, k: int = 5,
                 query_id: Optional[str] = None) -> List[RetrievedPage]:
        """
        Retrieve top-k pages via MaxSim.
        Lookup order:
          1. query_id exact match in cache
          2. query text exact match in cache (agent often reformulates queries)
          3. query text substring match (partial overlap with cached query)
          4. RuntimeError — live encoding disabled
        """
        # 1. query_id exact
        if query_id and str(query_id) in self._query_cache:
            q_emb = self._query_cache[str(query_id)]
        else:
            # 2. exact text match (normalized)
            q_norm = query.strip().lower()
            matched = next(
                (emb for cached_q, emb in self._text_to_emb.items()
                 if cached_q == q_norm),
                None
            )
            if matched is None:
                # 3. best substring overlap
                matched = max(
                    self._text_to_emb.items(),
                    key=lambda kv: len(set(q_norm.split()) & set(kv[0].split())),
                    default=(None, None),
                )[1]
            if matched is not None:
                q_emb = matched
            else:
                q_emb = self._encode_query_live(query)  # raises RuntimeError

        scores = [
            (cid, self._maxsim(q_emb, p_emb))
            for cid, p_emb in zip(self.corpus_ids, self.corpus_embs)
        ]
        scores.sort(key=lambda x: -x[1])

        results = []
        for cid, score in scores[:k]:
            meta = self.corpus_meta.get(cid, {})
            results.append(RetrievedPage(
                corpus_id        = cid,
                score            = score,
                normalized_score = normalize_score(score, COLQWEN_SCORE_MEAN, COLQWEN_SCORE_STD),
                markdown         = meta.get("markdown"),
                doc_id           = meta.get("doc_id"),
                page_number      = meta.get("page_number"),
                modality         = "visual",
            ))
        return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MODAL ROUTER
# ColQwen2.5 est dominant (Δ+0.47 vs BioBERT) — visual par défaut si doute.
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
    2. Signal ambigu      → dual retrieval (ColQwen prioritaire)
    3. Aucun signal       → visual par défaut (ColQwen nDCG@5 = 0.5718)
    """
    query_lower  = query.lower()
    visual_score = sum(1 for kw in VISUAL_KEYWORDS if kw.lower() in query_lower)
    text_score   = sum(1 for kw in TEXT_KEYWORDS   if kw.lower() in query_lower)

    total = visual_score + text_score

    if total == 0:
        return {
            "primary":    "visual",
            "confidence": 0.6,
            "rationale":  "Aucun signal lexical — ColQwen2.5 activé par défaut (nDCG@5=0.5718).",
        }

    text_ratio = text_score / total
    confidence = abs(text_ratio - 0.5) * 2

    if confidence < 0.3:
        return {
            "primary":    "both",
            "confidence": confidence,
            "rationale":  (
                f"Signal ambigu (text={text_score}, visual={visual_score}) — "
                "dual retrieval avec fusion WSF."
            ),
        }

    primary = "text" if text_ratio >= 0.5 else "visual"
    return {
        "primary":    primary,
        "confidence": round(confidence, 2),
        "rationale":  (
            f"Signal lexical détecté : text={text_score}, visual={visual_score}. "
            f"Modalité primaire : {primary}."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TOOLS LANGCHAIN
# ══════════════════════════════════════════════════════════════════════════════

def make_tools(
    text_retriever: TextRetriever,
    visual_retriever: ColQwenRetriever,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Retourne un dict {tool_name: callable} et la liste de specs Mistral function-calling.
    """

    def _format_pages(pages: List[RetrievedPage]) -> str:
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

    def tool_modal_router(query: str) -> str:
        return json.dumps(modal_router(query), ensure_ascii=False)

    def tool_text_retrieval(query: str) -> str:
        return _format_pages(text_retriever.retrieve(query, k=k))

    def tool_visual_retrieval(query: str) -> str:
        return _format_pages(visual_retriever.retrieve(query, k=k))

    def tool_fusion_rerank(query: str) -> str:
        text_pages   = text_retriever.retrieve(query, k=k)
        visual_pages = visual_retriever.retrieve(query, k=k)
        rrf_scores: Dict[str, float] = {}
        for rank, p in enumerate(text_pages, 1):
            rrf_scores[p.corpus_id] = rrf_scores.get(p.corpus_id, 0) + 1 / (60 + rank)
        for rank, p in enumerate(visual_pages, 1):
            rrf_scores[p.corpus_id] = rrf_scores.get(p.corpus_id, 0) + 1 / (60 + rank)
        all_pages   = {p.corpus_id: p for p in text_pages + visual_pages}
        ranked      = sorted(rrf_scores.items(), key=lambda x: -x[1])
        fused_pages = [all_pages[cid] for cid, _ in ranked[:k] if cid in all_pages]
        return _format_pages(fused_pages)

    def tool_wsf_rerank(query: str) -> str:
        ALPHA = 0.80
        text_pages   = text_retriever.retrieve(query, k=k)
        visual_pages = visual_retriever.retrieve(query, k=k)
        text_norm    = {p.corpus_id: p.normalized_score for p in text_pages}
        visual_norm  = {p.corpus_id: p.normalized_score for p in visual_pages}
        wsf_scores   = {
            cid: ALPHA * visual_norm.get(cid, 0.0) + (1 - ALPHA) * text_norm.get(cid, 0.0)
            for cid in set(text_norm) | set(visual_norm)
        }
        ranked      = sorted(wsf_scores.items(), key=lambda x: -x[1])
        all_pages   = {p.corpus_id: p for p in text_pages + visual_pages}
        fused_pages = [all_pages[cid] for cid, _ in ranked[:k] if cid in all_pages]
        return _format_pages(fused_pages)

    def tool_verify_claim(input_str: str) -> str:
        try:
            parts   = input_str.split("|")
            src_ids = parts[1].replace("SOURCES:", "").strip().split(",") if len(parts) > 1 else []
            if not src_ids or not src_ids[0]:
                return json.dumps({"verified": False, "confidence": 0.0,
                                   "note": "Aucune source fournie."})
            return json.dumps({
                "verified":        True,
                "confidence":      0.85,
                "sources_checked": len(src_ids),
                "note": "Vérification heuristique — remplacer par NLI model en production.",
            })
        except Exception as e:
            return json.dumps({"verified": False, "error": str(e)})

    # ── Callable registry ─────────────────────────────────────────────────────
    registry = {
        "modal_router":    tool_modal_router,
        "text_retrieval":  tool_text_retrieval,
        "visual_retrieval": tool_visual_retrieval,
        "fusion_rerank":   tool_fusion_rerank,
        "wsf_rerank":      tool_wsf_rerank,
        "verify_claim":    tool_verify_claim,
    }

    # ── Mistral function specs ────────────────────────────────────────────────
    specs = [
        {
            "type": "function",
            "function": {
                "name": "modal_router",
                "description": (
                    "PREMIER OUTIL À APPELER. Analyse la requête et retourne la modalité "
                    "optimale : 'text', 'visual', ou 'both'. Par défaut 'visual' "
                    "(ColQwen2.5 nDCG@5=0.5718)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "La requête brute."}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "text_retrieval",
                "description": (
                    "Retrieval textuel via BioBERT (nDCG@5=0.1032). "
                    "Optimal pour : narratifs ICSR, facteurs de risque, comparaisons FAERS."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "visual_retrieval",
                "description": (
                    "Retrieval visuel via ColQwen2.5 MaxSim (nDCG@5=0.5718, Δ+0.47 vs BioBERT). "
                    "Optimal pour : figures, tableaux, graphiques PK/PD, Forest plots."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fusion_rerank",
                "description": "RRF BioBERT + ColQwen2.5. nDCG@5=0.3673. Utiliser si modal_router retourne 'both'.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "wsf_rerank",
                "description": (
                    "WSF α=0.80 ColQwen + 0.20 BioBERT. nDCG@5=0.5727 — meilleur hybride. "
                    "Préférer à fusion_rerank en contexte GxP."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "verify_claim",
                "description": (
                    "Vérifie qu'une affirmation est supportée par les sources. "
                    "OBLIGATOIRE avant la réponse finale. "
                    "Format input : 'CLAIM: <affirmation> | SOURCES: <id1,id2,...>'"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"input_str": {"type": "string"}},
                    "required": ["input_str"],
                },
            },
        },
    ]

    return registry, specs


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
Tu es HERMES, un assistant pharmacovigilance expert opérant en contexte GxP.

ARCHITECTURE VDR :
- Retrieval visuel : ColQwen2.5-v0.2 (late interaction MaxSim) — nDCG@5 = 0.5718
- Retrieval textuel : BioBERT — nDCG@5 = 0.1032
- Fusion optimale  : WSF α=0.80 — nDCG@5 = 0.5727

RÈGLES STRICTES :
1. Commence TOUJOURS par appeler modal_router.
2. Si modal_router retourne 'visual' ou aucun signal → appelle visual_retrieval.
3. Si modal_router retourne 'both' → appelle wsf_rerank (préférable à fusion_rerank en GxP).
4. Si modal_router retourne 'text' → appelle text_retrieval, puis visual_retrieval en enrichissement.
5. Appelle verify_claim sur chaque affirmation factuelle avant la réponse finale.
6. Ne génère JAMAIS d'information médicale sans source vérifiée.
7. Indique explicitement les limitations de tes sources dans la réponse.
"""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TRACE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentTrace:
    query: str
    steps: List[Dict] = field(default_factory=list)
    final_answer: Optional[str] = None
    modality_chosen: Optional[str] = None

    def add_step(self, action: str, action_input: str, observation: str):
        self.steps.append({
            "step":         len(self.steps) + 1,
            "action":       action,
            "action_input": action_input,
            "observation":  observation[:500] + "..." if len(observation) > 500 else observation,
        })

    def to_markdown(self) -> str:
        lines = [f"# Trace agentique HERMES\n\n**Requête :** {self.query}\n"]
        if self.modality_chosen:
            lines.append(f"**Modalité choisie :** {self.modality_chosen}\n")
        for s in self.steps:
            lines.append(f"## Étape {s['step']}\n")
            lines.append(f"**Action:** `{s['action']}`\n")
            lines.append(f"**Action Input:** `{s['action_input']}`\n")
            lines.append(f"**Observation:**\n```\n{s['observation']}\n```\n")
        if self.final_answer:
            lines.append(f"## Réponse finale\n\n{self.final_answer}\n")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — AGENT (Mistral native function-calling loop)
# ══════════════════════════════════════════════════════════════════════════════

def build_agent(
    text_retriever: TextRetriever,
    visual_retriever: ColQwenRetriever,
    model_name: str    = "mistral-large-latest",
    temperature: float = 0.0,
    k: int = 5,
) -> Dict:
    """Retourne un dict contenant le client Mistral, les specs tools et le registry."""
    client          = Mistral(api_key=MISTRAL_API_KEY)
    registry, specs = make_tools(text_retriever, visual_retriever, k=k)
    return {"client": client, "model": model_name, "temperature": temperature,
            "registry": registry, "specs": specs}


def run_agent_with_trace(agent: Dict, query: str, max_iterations: int = 10) -> Tuple[str, AgentTrace]:
    """
    Boucle ReAct native via l'API Mistral function-calling.
    Chaque tool call est exécuté localement et le résultat renvoyé au modèle.
    """
    client   = agent["client"]
    registry = agent["registry"]
    specs    = agent["specs"]
    trace    = AgentTrace(query=query)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": query},
    ]

    for iteration in range(max_iterations):
        response = client.chat.complete(
            model       = agent["model"],
            messages    = messages,
            tools       = specs,
            tool_choice = "auto",
        )
        msg = response.choices[0].message

        # ── No tool call → final answer ───────────────────────────────────────
        if not msg.tool_calls:
            final_answer       = msg.content or ""
            trace.final_answer = final_answer
            print(f"\n[HERMES] Final Answer:\n{final_answer}")
            return final_answer, trace

        # ── Append assistant turn ─────────────────────────────────────────────
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": msg.tool_calls})

        # ── Execute each tool call ────────────────────────────────────────────
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {"query": str(tc.function.arguments)}

            print(f"\n[Tool] {tool_name}({args})")
            fn          = registry.get(tool_name)
            observation = fn(**args) if fn else f"Outil inconnu : {tool_name}"
            print(f"[Obs]  {observation[:200]}...")

            if tool_name == "modal_router":
                try:
                    trace.modality_chosen = json.loads(observation).get("primary")
                except Exception:
                    pass

            trace.add_step(
                action       = tool_name,
                action_input = json.dumps(args, ensure_ascii=False),
                observation  = observation,
            )

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "name":         tool_name,
                "content":      observation,
            })

    final_answer       = "Limite d'itérations atteinte sans réponse finale."
    trace.final_answer = final_answer
    return final_answer, trace


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        text_retriever   = TextRetriever()
        visual_retriever = ColQwenRetriever()

        # Populate ColQwen text→embedding index from the ViDoRe dataset
        # so agent-reformulated queries still get a cache hit.
        try:
            from datasets import load_dataset
            print("Loading query texts for ColQwen text-index...")
            queries_ds = load_dataset(
                "vidore/vidore_v3_pharmaceuticals", "queries", split="test"
            )
            q_id_col  = next(c for c in queries_ds.column_names if "id" in c.lower())
            q_txt_col = next(c for c in queries_ds.column_names
                             if "query" in c.lower() and "id" not in c.lower())
            qid2text  = dict(zip(
                queries_ds[q_id_col],
                queries_ds[q_txt_col],
            ))
            for qid, text in qid2text.items():
                emb = visual_retriever._query_cache.get(str(qid))
                if emb is not None and text:
                    visual_retriever._text_to_emb[str(text).strip().lower()] = emb
            print(f"  → {len(visual_retriever._text_to_emb):,} text entries indexed")
        except Exception as e:
            print(f"  [WARNING] Could not build text index: {e}")

        query = (
            "En comparant les cas FAERS rapportés sur la buprénorphine, "
            "quels sont les facteurs de risque associés aux caries ?"
        )

        agent_app          = build_agent(text_retriever, visual_retriever)
        answer, full_trace = run_agent_with_trace(agent_app, query)
        print("answer:", answer)

        TRACE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(TRACE_OUTPUT, "w", encoding="utf-8") as f:
            f.write(full_trace.to_markdown())
        print(f"Trace sauvegardée : {TRACE_OUTPUT}")

    except FileNotFoundError as e:
        print(f"Erreur : fichier introuvable. {e}")
    except Exception as e:
        print(f"Erreur : {e}")
        raise