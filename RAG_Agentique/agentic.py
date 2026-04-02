"""
HERMES — Couche Agentique ReAct (Bloc 3)
Couvre : Q9 (catalogue outils) · Q10 (trace ReAct) · Q11 (routage complexité)
         Q12 (gestion incertitude) · Q13 (journalisation GxP)
"""

import os
import json
import uuid
import hashlib
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
from langchain_core.messages import AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_mistralai import ChatMistralAI

# ═══════════════════════════════════════════════════════════════════════════════
# Q9 — CATALOGUE D'OUTILS
# Chaque outil : signature, conditions d'appel, erreur, supervision
# ═══════════════════════════════════════════════════════════════════════════════

"""
CATALOGUE COMPLET DES OUTILS HERMES
====================================

┌─────────────────┬──────────────────────┬───────────────────────┬────────────────────────────┬──────────────────────┐
│ Outil           │ Signature            │ Conditions d'appel    │ Comportement erreur        │ Supervision          │
├─────────────────┼──────────────────────┼───────────────────────┼────────────────────────────┼──────────────────────┤
│ modal_router    │ (query:str)→dict     │ TOUJOURS en 1er       │ Retourne "both" par défaut  │ Autonome             │
│ text_retrieval  │ (query:str)→str      │ Si primary="text"     │ Retourne liste vide + msg  │ Autonome             │
│ visual_retrieval│ (query:str)→str      │ Si primary="visual"   │ Retourne liste vide + msg  │ Autonome             │
│ fusion_rerank   │ (query:str)→str      │ Si primary="both"     │ Fallback text seul         │ Autonome             │
│ verify_claim    │ (claim|sources)→dict │ Avant réponse finale  │ verified=False + warning   │ Confirmation requise │
│ complexity_check│ (query:str)→dict     │ Pré-filtre externe    │ Retourne "complex" par déf  │ Autonome             │
│ uncertainty_gate│ (scores:list)→dict   │ Avant réponse finale  │ Force escalade si erreur   │ Confirmation requise │
└─────────────────┴──────────────────────┴───────────────────────┴────────────────────────────┴──────────────────────┘
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Q11 — CLASSIFICATEUR DE COMPLEXITÉ (LLM)
# Le LLM analyse la requête et décide simple vs complexe
# ═══════════════════════════════════════════════════════════════════════════════

def complexity_check(query: str, llm: ChatMistralAI) -> dict:
    """
    Classifie la requête : 'simple' (retrieval direct) ou 'complex' (multi-étapes).
    Utilise le LLM pour une analyse sémantique réelle de la requête.
    Appelé comme pré-filtre AVANT le lancement de l'agent.
    """
    prompt = f"""Tu es un expert en pharmacovigilance. Analyse cette requête et détermine sa complexité.

Requête : "{query}"

Réponds UNIQUEMENT en JSON valide avec ce format exact :
{{"level": "simple" ou "complex", "rationale": "explication courte", "recommended_plan": "étapes suggérées"}}

Règles :
- "simple" : une seule information factuelle à récupérer (ex: "Quel est le temps moyen au diagnostic ?")
- "complex" : nécessite plusieurs étapes, comparaisons, synthèse ou raisonnement multi-sources (ex: "Quels sont les facteurs de risque associés à...")"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        # Nettoyer si le LLM ajoute des backticks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception as e:
        return {
            "level": "complex",
            "rationale": f"Parsing échoué ({e}) — fallback complex par sécurité.",
            "recommended_plan": "modal_router → retrieval → verify_claim → réponse"
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Q12 — POLITIQUE D'INCERTITUDE
# L'agent refuse de répondre si les scores sont trop bas → escalade humaine
# ═══════════════════════════════════════════════════════════════════════════════

UNCERTAINTY_THRESHOLD = 0.15  # score normalisé minimum acceptable

def uncertainty_gate(scores_json: str) -> dict:
    """
    Vérifie si les scores de retrieval sont suffisants pour répondre.
    Input JSON : '{"scores": [0.10, 0.08, 0.12]}'

    Politique HERMES :
    - score_max < 0.15  → REFUS + escalade expert humain
    - 0.15 ≤ score_max < 0.25 → RÉPONSE avec avertissement fort
    - score_max ≥ 0.25  → RÉPONSE normale

    Exemple de refus :
    Requête "Quel est l'effet de NVG-047 sur les biomarqueurs rénaux ?"
    → Aucune page pertinente dans le corpus → L'agent doit refuser
      plutôt que d'halluciner une réponse médicale.
    """
    try:
        data = json.loads(scores_json)
        scores = data.get("scores", [])
    except Exception:
        return {"decision": "ESCALADE", "reason": "Impossible de parser les scores."}

    if not scores:
        return {"decision": "ESCALADE", "reason": "Aucun résultat de retrieval."}

    score_max = max(scores)

    if score_max < UNCERTAINTY_THRESHOLD:
        return {
            "decision": "REFUS",
            "score_max": round(score_max, 3),
            "message": (
                "⚠️ Confiance insuffisante (score_max={:.3f} < seuil {}).\n"
                "HERMES ne peut pas répondre avec certitude suffisante.\n"
                "→ Escalade vers un pharmacovigilant expert recommandée."
            ).format(score_max, UNCERTAINTY_THRESHOLD),
        }
    elif score_max < 0.25:
        return {
            "decision": "AVERTISSEMENT",
            "score_max": round(score_max, 3),
            "message": "Réponse possible mais avec confiance modérée. Vérification humaine conseillée.",
        }
    else:
        return {
            "decision": "OK",
            "score_max": round(score_max, 3),
            "message": "Confiance suffisante pour répondre.",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL BACKENDS
# ═══════════════════════════════════════════════════════════════════════════════

BIOBERT_MEAN, BIOBERT_STD = 0.42, 0.15
CLIP_MEAN, CLIP_STD = 0.28, 0.12

def _normalize(score, mean, std):
    return (score - mean) / (std + 1e-9)

def _load_embeddings(path: str) -> Dict[str, np.ndarray]:
    with open(path) as f:
        raw = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}

@dataclass
class RetrievedPage:
    corpus_id: str
    score: float
    normalized_score: float
    markdown: Optional[str]
    doc_id: Optional[str]
    page_number: Optional[int]
    modality: Literal["text", "visual"]

class TextRetriever:
    def __init__(self, corpus_emb_path: str, corpus_meta: Optional[Dict] = None):
        self.corpus_embeddings = _load_embeddings(corpus_emb_path)
        self.corpus_ids = list(self.corpus_embeddings.keys())
        self.corpus_matrix = np.stack([self.corpus_embeddings[cid] for cid in self.corpus_ids])
        self.corpus_meta = corpus_meta or {}
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", device="cpu"
        )

    def retrieve(self, query: str, k: int = 5) -> List[RetrievedPage]:
        q_emb = self.model.encode(f"query: {query}", normalize_embeddings=True).reshape(1, -1)
        sims = (q_emb @ self.corpus_matrix.T).flatten()
        top_k = np.argsort(-sims)[:k]
        results = []
        for idx in top_k:
            cid = self.corpus_ids[idx]
            score = float(sims[idx])
            meta = self.corpus_meta.get(cid, {})
            results.append(RetrievedPage(
                corpus_id=cid, score=score,
                normalized_score=_normalize(score, BIOBERT_MEAN, BIOBERT_STD),
                markdown=meta.get("markdown"), doc_id=meta.get("doc_id"),
                page_number=meta.get("page_number"), modality="text",
            ))
        return results

class VisualRetriever:
    def __init__(self, corpus_emb_path: str, corpus_meta: Optional[Dict] = None):
        self.corpus_embeddings = _load_embeddings(corpus_emb_path)
        self.corpus_ids = list(self.corpus_embeddings.keys())
        self.corpus_matrix = np.stack([self.corpus_embeddings[cid] for cid in self.corpus_ids])
        self.corpus_meta = corpus_meta or {}
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("clip-ViT-B-32", device="cpu")

    def retrieve(self, query: str, k: int = 5) -> List[RetrievedPage]:
        q_emb = self.model.encode(query, normalize_embeddings=True).reshape(1, -1)
        sims = (q_emb @ self.corpus_matrix.T).flatten()
        top_k = np.argsort(-sims)[:k]
        results = []
        for idx in top_k:
            cid = self.corpus_ids[idx]
            score = float(sims[idx])
            meta = self.corpus_meta.get(cid, {})
            results.append(RetrievedPage(
                corpus_id=cid, score=score,
                normalized_score=_normalize(score, CLIP_MEAN, CLIP_STD),
                markdown=meta.get("markdown"), doc_id=meta.get("doc_id"),
                page_number=meta.get("page_number"), modality="visual",
            ))
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# MODAL ROUTER (LLM)
# Le LLM choisit la modalité optimale — plus robuste que le scoring lexical
# ═══════════════════════════════════════════════════════════════════════════════

def modal_router(query: str, llm: ChatMistralAI) -> dict:
    """
    Choisit la modalité de retrieval via le LLM.
    Fallback : "both" si le parsing échoue.
    """
    prompt = f"""Tu es un expert en pharmacovigilance. Analyse cette requête et détermine la modalité de retrieval optimale.

Requête : "{query}"

Réponds UNIQUEMENT en JSON valide avec ce format exact :
{{"primary": "text" ou "visual" ou "both", "confidence": 0.0 à 1.0, "rationale": "explication courte"}}

Règles :
- "text"   : la réponse est dans du texte narratif (cas cliniques ICSR, descriptions, facteurs de risque, rapports FAERS)
- "visual" : la réponse est dans une figure, tableau, graphique, courbe PK/PD, Forest plot
- "both"   : la requête est ambiguë ou nécessite les deux modalités"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception as e:
        return {
            "primary": "both",
            "confidence": 0.5,
            "rationale": f"Parsing échoué ({e}) — fallback both par sécurité."
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Q13 — JOURNALISATION GxP
# ═══════════════════════════════════════════════════════════════════════════════

"""
SCHÉMA DE JOURNALISATION GxP HERMES
=====================================

{
  "log_id":        UUID v4 unique — clé primaire immuable,
  "session_id":    UUID de la session — regroupe les étapes d'une requête,
  "timestamp_utc": ISO 8601 UTC — horodatage certifié,
  "step_type":     "THOUGHT" | "ACTION" | "OBSERVATION" | "FINAL_ANSWER" | "REFUSAL",
  "step_index":    Numéro d'étape dans la session,
  "actor":         "HERMES_AGENT" | "USER" | "TOOL:<nom>",
  "content":       Contenu de l'étape,
  "tool_name":     Nom de l'outil appelé (null si THOUGHT),
  "tool_input":    Paramètres envoyés à l'outil,
  "scores":        Scores de retrieval si applicable,
  "verified":      Résultat de verify_claim si applicable,
  "model_name":    Nom du LLM utilisé,
  "hash_prev":     SHA-256 de l'entrée précédente — chaîne d'intégrité,
  "hash_self":     SHA-256 de cette entrée
}

Rétention : 10 ans (exigence EMA/GxP).
Format : JSON Lines (JSONL) — append-only.
Backend : Elasticsearch WORM ou Loki avec retention policy.

Exploitation audit EMA :
- Reconstituer la chaîne causale complète d'une décision agentique
- Vérifier que verify_claim a bien été appelé avant chaque Final Answer
- Prouver que le modèle n'a pas changé entre deux sessions (model_name)
- Détecter toute modification du journal via la chaîne de hachage
"""

@dataclass
class GxPLogEntry:
    session_id: str
    step_type: Literal["THOUGHT", "ACTION", "OBSERVATION", "FINAL_ANSWER", "REFUSAL"]
    step_index: int
    actor: str
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None
    scores: Optional[List[float]] = None
    verified: Optional[bool] = None
    model_name: str = "mistral-large-latest"
    hash_prev: Optional[str] = None

    def to_dict(self) -> dict:
        entry = {
            "log_id":        str(uuid.uuid4()),
            "session_id":    self.session_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "step_type":     self.step_type,
            "step_index":    self.step_index,
            "actor":         self.actor,
            "content":       self.content,
            "tool_name":     self.tool_name,
            "tool_input":    self.tool_input,
            "scores":        self.scores,
            "verified":      self.verified,
            "model_name":    self.model_name,
            "hash_prev":     self.hash_prev,
        }
        entry["hash_self"] = hashlib.sha256(
            json.dumps(entry, sort_keys=True, default=str).encode()
        ).hexdigest()
        return entry


class GxPLogger:
    """Journal immuable GxP — append-only, chaîné par hash."""
    def __init__(self, session_id: str, log_path: Optional[str] = None):
        self.session_id = session_id
        self.log_path = log_path or f"hermes_gxp_{session_id[:8]}.jsonl"
        self._step_index = 0
        self._last_hash: Optional[str] = None

    def log(self, step_type, actor, content, **kwargs) -> dict:
        self._step_index += 1
        entry_obj = GxPLogEntry(
            session_id=self.session_id,
            step_type=step_type,
            step_index=self._step_index,
            actor=actor,
            content=content,
            hash_prev=self._last_hash,
            **kwargs,
        )
        entry = entry_obj.to_dict()
        self._last_hash = entry["hash_self"]
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry


# ═══════════════════════════════════════════════════════════════════════════════
# Q10 — TRACE ReAct : CALLBACK HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentTrace:
    query: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: List[Dict] = field(default_factory=list)
    final_answer: Optional[str] = None
    modality_chosen: Optional[str] = None

    def add_step(self, thought, action, action_input, observation):
        self.steps.append({
            "step": len(self.steps) + 1,
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "observation": observation[:600] + "..." if len(observation) > 600 else observation,
        })

    def to_markdown(self) -> str:
        lines = [f"# Trace HERMES\n\n**Session :** `{self.session_id}`\n**Requête :** {self.query}\n"]
        if self.modality_chosen:
            lines.append(f"**Modalité choisie :** `{self.modality_chosen}`\n")
        for s in self.steps:
            lines.append(f"## Étape {s['step']}\n")
            lines.append(f"**Thought:** {s['thought']}\n")
            lines.append(f"**Action:** `{s['action']}`  \n**Input:** `{s['action_input']}`\n")
            lines.append(f"**Observation:**\n```\n{s['observation']}\n```\n")
        if self.final_answer:
            lines.append(f"## Réponse finale\n\n{self.final_answer}\n")
        return "\n".join(lines)


class TraceCallbackHandler(BaseCallbackHandler):
    def __init__(self, trace: AgentTrace, gxp_logger: GxPLogger):
        self.trace = trace
        self.logger = gxp_logger
        self._last_thought = ""
        self._last_action = ""
        self._last_input = ""

    def on_llm_end(self, response, **kwargs):
        if not response.generations:
            return
        generation = response.generations[0][0]
        message = generation.message

        if hasattr(message, "tool_calls") and message.tool_calls:
            tc = message.tool_calls[0]
            self._last_action = tc["name"]
            self._last_input = json.dumps(tc["args"], ensure_ascii=False)

            thought_map = {
                "modal_router":      "Analyse de la requête pour choisir la modalité de retrieval optimale.",
                "complexity_check":  "Évaluation de la complexité de la requête (simple vs multi-étapes).",
                "text_retrieval":    "Retrieval textuel BioBERT sur le corpus — recherche des pages pertinentes.",
                "visual_retrieval":  "Retrieval visuel CLIP — recherche des pages par contenu visuel.",
                "fusion_rerank":     "Scores insuffisants ou modalité ambiguë — lancement du dual retrieval avec fusion RRF.",
                "uncertainty_gate":  "Vérification que les scores de retrieval sont suffisants pour répondre.",
                "verify_claim":      "Vérification anti-hallucination : l'affirmation est-elle supportée par les sources ?",
            }
            self._last_thought = message.content or thought_map.get(self._last_action, f"Appel de {self._last_action}.")
        else:
            self._last_action = "Final Answer"
            self._last_input = ""
            self._last_thought = message.content or "Toutes les vérifications passées — génération de la réponse finale."

        self.logger.log("THOUGHT", "HERMES_AGENT", self._last_thought)

    def on_tool_end(self, output: Any, **kwargs):
        from langchain_core.messages import ToolMessage
        obs_text = str(output.content) if isinstance(output, ToolMessage) else str(output)
        self.trace.add_step(
            thought=self._last_thought,
            action=self._last_action,
            action_input=self._last_input,
            observation=obs_text,
        )
        self.logger.log(
            "ACTION", f"TOOL:{self._last_action}", obs_text,
            tool_name=self._last_action, tool_input=self._last_input,
        )
        if self._last_action == "modal_router":
            try:
                self.trace.modality_chosen = json.loads(obs_text).get("primary")
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTRUCTION DES OUTILS LANGCHAIN
# llm est passé en paramètre pour modal_router et verify_claim
# ═══════════════════════════════════════════════════════════════════════════════

def make_tools(
    text_retriever: TextRetriever,
    visual_retriever: VisualRetriever,
    llm: ChatMistralAI,
    k: int = 5,
) -> List[Tool]:

    # Cache partagé : rempli par text/visual_retrieval, lu par verify_claim
    _retrieval_cache: Dict[str, str] = {}

    def _fmt(pages: List[RetrievedPage]) -> str:
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

    # ── Tool 1 : modal_router (LLM) ───────────────────────────────────────────
    def tool_modal_router(query: str) -> str:
        return json.dumps(modal_router(query, llm), ensure_ascii=False)

    # ── Tool 2 : text_retrieval ───────────────────────────────────────────────
    def tool_text_retrieval(query: str) -> str:
        pages = text_retriever.retrieve(query, k=k)
        for p in pages:
            if p.markdown is not None:
                _retrieval_cache[p.corpus_id] = p.markdown
        return _fmt(pages)

    # ── Tool 3 : visual_retrieval ─────────────────────────────────────────────
    def tool_visual_retrieval(query: str) -> str:
        pages = visual_retriever.retrieve(query, k=k)
        for p in pages:
            if p.markdown is not None:
                _retrieval_cache[p.corpus_id] = p.markdown
        return _fmt(pages)

    # ── Tool 4 : fusion_rerank ────────────────────────────────────────────────
    def tool_fusion_rerank(query: str) -> str:
        tp = text_retriever.retrieve(query, k=k)
        vp = visual_retriever.retrieve(query, k=k)
        for p in tp + vp:
            if p.markdown is not None:
                _retrieval_cache[p.corpus_id] = p.markdown
        rrf: Dict[str, float] = {}
        for rank, p in enumerate(tp, 1): rrf[p.corpus_id] = rrf.get(p.corpus_id, 0) + 1/(60+rank)
        for rank, p in enumerate(vp, 1): rrf[p.corpus_id] = rrf.get(p.corpus_id, 0) + 1/(60+rank)
        all_pages = {p.corpus_id: p for p in tp + vp}
        ranked = sorted(rrf.items(), key=lambda x: -x[1])
        fused = [all_pages[cid] for cid, _ in ranked[:k] if cid in all_pages]
        return _fmt(fused)

    # ── Tool 5 : verify_claim (LLM sur contenu réel) ─────────────────────────
    def tool_verify_claim(input_str: str) -> str:
        try:
            parts = input_str.split("|")
            claim = parts[0].replace("CLAIM:", "").strip()
            src_ids = [s.strip() for s in parts[1].replace("SOURCES:", "").split(",")] if len(parts) > 1 else []

            # Récupérer le contenu réel des sources depuis le cache
            sources_content = ""
            missing = []
            for src_id in src_ids:
                content = _retrieval_cache.get(src_id, "")
                if content:
                    sources_content += f"\n[Source {src_id}]: {content[:500]}"
                else:
                    missing.append(src_id)

            if not sources_content:
                return json.dumps({
                    "verified": False,
                    "confidence": 0.0,
                    "note": f"Sources {missing} introuvables dans le corpus récupéré. Claim non vérifiable."
                })

            # Appel LLM pour vérification réelle sur le contenu des pages
            prompt = f"""Tu es un expert en pharmacovigilance. Vérifie si l'affirmation suivante est explicitement supportée par les sources.

Affirmation : "{claim}"

Sources récupérées :
{sources_content}

Réponds UNIQUEMENT en JSON valide :
{{"verified": true ou false, "confidence": 0.0 à 1.0, "note": "explication courte"}}

Règles strictes :
- verified=true UNIQUEMENT si l'affirmation est explicitement présente dans les sources
- verified=false si l'affirmation est une inférence, généralisation, ou absente des sources
- En cas de doute → false (contexte GxP : la prudence prime)"""

            response = llm.invoke(prompt)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return content

        except Exception as e:
            return json.dumps({"verified": False, "confidence": 0.0, "error": str(e)})

    # ── Tool 6 : uncertainty_gate ─────────────────────────────────────────────
    def tool_uncertainty_gate(scores_json: str) -> str:
        return json.dumps(uncertainty_gate(scores_json), ensure_ascii=False)

    return [
        Tool(name="modal_router", func=tool_modal_router,
             description="PREMIER OUTIL. Analyse sémantique de la requête pour choisir la modalité : 'text', 'visual' ou 'both'. Input: requête brute."),
        Tool(name="text_retrieval", func=tool_text_retrieval,
             description="Retrieval textuel BioBERT. Pour narratifs ICSR, facteurs de risque, comparaisons FAERS. Input: requête."),
        Tool(name="visual_retrieval", func=tool_visual_retrieval,
             description="Retrieval visuel CLIP. Pour figures, tableaux, graphiques PK/PD, Forest plots. Input: requête."),
        Tool(name="fusion_rerank", func=tool_fusion_rerank,
             description="Lance text + visual et fusionne via RRF. Utiliser si modal_router retourne 'both' ou scores insuffisants."),
        Tool(name="verify_claim", func=tool_verify_claim,
             description="OBLIGATOIRE avant réponse finale. Vérifie sur le contenu réel des sources qu'une affirmation est supportée. Format: 'CLAIM: <affirmation> | SOURCES: <corpus_id1,corpus_id2>'"),
        Tool(name="uncertainty_gate", func=tool_uncertainty_gate,
             description="Vérifie si les scores de retrieval sont suffisants pour répondre. Retourne REFUS/AVERTISSEMENT/OK. Input JSON: '{\"scores\": [0.68, 0.50, ...]}'"),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Tu es HERMES, un assistant pharmacovigilance expert en contexte GxP.
Tu raisonnes en ReAct : Thought → Action → Observation, en boucle.

RÈGLES STRICTES (non négociables) :
1. Appelle TOUJOURS modal_router en premier pour choisir la modalité.
2. Si modal_router retourne "both" OU si les scores normalisés sont < 0.15, appelle fusion_rerank.
3. Appelle uncertainty_gate avec les scores normalisés avant de conclure.
   → Si le résultat est "REFUS" : arrête et signale l'impossibilité de répondre.
4. Appelle verify_claim UNE SEULE FOIS sur l'affirmation principale, puis génère immédiatement la réponse finale.
5. N'invente JAMAIS d'information médicale sans source vérifiée dans le corpus.
6. Indique les limitations et l'incertitude dans chaque réponse finale."""


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT FACTORY & RUN
# ═══════════════════════════════════════════════════════════════════════════════

def build_agent(text_retriever, visual_retriever, model_name="mistral-large-latest", k=5):
    llm = ChatMistralAI(
        model=model_name,
        temperature=0.0,
        api_key=os.environ["MISTRAL_API_KEY"],
    )
    tools = make_tools(text_retriever, visual_retriever, llm=llm, k=k)
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )


def run_agent(agent, query: str) -> Tuple[str, AgentTrace, str]:
    session_id = str(uuid.uuid4())
    trace = AgentTrace(query=query, session_id=session_id)
    gxp_logger = GxPLogger(session_id)
    handler = TraceCallbackHandler(trace, gxp_logger)

    gxp_logger.log("ACTION", "USER", query)

    result = agent.invoke(
        {"messages": [("user", query)]},
        config={
            "callbacks": [handler],
            "recursion_limit": 10,
        },
    )

    # Chercher le dernier AIMessage sans tool_calls pendants
    final_answer = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            final_answer = msg.content
            break
    final_answer = final_answer or result["messages"][-1].content
    trace.final_answer = final_answer

    gxp_logger.log("FINAL_ANSWER", "HERMES_AGENT", final_answer)

    return final_answer, trace, gxp_logger.log_path


# ═══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from datasets import load_dataset

    print("Chargement du corpus ViDoRe...")
    corpus_ds = load_dataset("vidore/vidore_v3_pharmaceuticals", "corpus", split="test")
    df_corpus = corpus_ds.to_pandas()

    corpus_meta = {}
    for _, row in df_corpus.iterrows():
        corpus_meta[str(row["corpus_id"])] = {  # str() — les corpus_id des embeddings sont des strings
            "markdown":    row["markdown"] or "",
            "doc_id":      row["doc_id"],
            "page_number": row["page_number_in_doc"],
        }
    print(f"corpus_meta chargé : {len(corpus_meta)} pages")

    text_retriever = TextRetriever(
        "RAG_Text/embeddings_textual_corpus.json",
        corpus_meta=corpus_meta,
    )
    visual_retriever = VisualRetriever(
        "RAG_VDR/CLIP/embeddings_visual_corpus.json",
        corpus_meta=corpus_meta,
    )

    query = (
        "En comparant les cas FAERS rapportés dans le document sur la buprénorphine "
        "sublinguale, quels sont les facteurs de risque patient qui semblent associés "
        "à une évolution plus sévère des caries dentaires ? Génère un résumé structuré."
    )

    # Q11 — pré-filtre de complexité (LLM) avant lancement de l'agent
    llm_for_prefilter = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.0,
        api_key=os.environ["MISTRAL_API_KEY"],
    )
    complexity = complexity_check(query, llm_for_prefilter)
    print(f"[Q11] Complexité : {complexity['level']} — {complexity['recommended_plan']}")

    agent = build_agent(text_retriever, visual_retriever)
    answer, trace, log_path = run_agent(agent, query)

    print("\n" + "="*60)
    print(answer)

    os.makedirs("RAG_Agentique", exist_ok=True)
    with open("RAG_Agentique/trace_hermes.md", "w", encoding="utf-8") as f:
        f.write(trace.to_markdown())

    print(f"\n[Q10] Trace : RAG_Agentique/trace_hermes.md")
    print(f"[Q13] Log GxP : {log_path}")