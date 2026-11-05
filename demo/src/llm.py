from __future__ import annotations
import json
import os
import re
import requests
from src.config import settings

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

class LocalLLMError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Provider layer cho pipeline LLM (predict / relevance / translate / describe)
#
# LLM_PROVIDER=ollama : raw /api/generate của Ollama trên host (như mặc định cũ).
# LLM_PROVIDER=openai : mọi endpoint tương thích OpenAI /chat/completions
#                       (Ollama /v1, vLLM, LM Studio, OpenRouter, Together...).
#                       Không cần key với server local; key đọc từ biến môi trường
#                       LLM_API_KEY_ENV (mặc định LLM_API_KEY), không ghi vào file.
# Chế độ Direct LLM (Z.ai) vẫn dùng zai_chat riêng bên dưới.
# ---------------------------------------------------------------------------

def _api_key() -> str:
    return os.environ.get(settings.llm_api_key_env, "") or settings.llm_api_key


def openai_chat(messages: list[dict], temperature: float = 0.0, json_mode: bool = False,
                num_predict: int | None = None) -> str:
    payload = {
        "model": settings.llm_model or settings.ollama_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": num_predict or settings.llm_num_predict,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json"}
    key = _api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        r = requests.post(f"{settings.llm_base_url}/chat/completions", headers=headers,
                          json=payload, timeout=180)
        r.raise_for_status()
        body = r.json()
        content = (body["choices"][0]["message"].get("content") or "").strip()
        if not content:
            raise LocalLLMError(f"Provider openai ({settings.llm_base_url}) trả content rỗng.")
        return content
    except LocalLLMError:
        raise
    except (requests.RequestException, KeyError, IndexError, TypeError) as exc:
        raise LocalLLMError(
            f"Không gọi được LLM provider openai tại {settings.llm_base_url} "
            f"(model={payload['model']}). Kiểm tra server, model và biến môi trường "
            f"{settings.llm_api_key_env}. Chi tiết: {exc}"
        ) from exc


def chat(messages: list[dict], temperature: float = 0.0, json_mode: bool = False,
         num_predict: int | None = None) -> str:
    """Điều phối theo LLM_PROVIDER; mọi hàm pipeline của demo gọi qua đây."""
    if settings.llm_provider == "openai":
        return openai_chat(messages, temperature=temperature, json_mode=json_mode,
                           num_predict=num_predict)
    return ollama_chat(messages, temperature=temperature, json_mode=json_mode,
                       num_predict=num_predict)


def zai_chat(messages: list[dict], json_mode: bool = False) -> str:
    if not settings.zai_api_key:
        raise LocalLLMError("Chưa cấu hình ZAI_API_KEY trong .env để gọi GLM.")
    payload = {
        "model": settings.zai_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": settings.zai_num_predict,
        "thinking": {"type": "disabled"},
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    try:
        response = requests.post(
            f"{settings.zai_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.zai_api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"].get("content", "")
        if not content.strip():
            raise LocalLLMError("Z.ai trả content rỗng.")
        return content
    except (requests.RequestException, KeyError, IndexError, TypeError) as exc:
        raise LocalLLMError(f"Không gọi được Z.ai GLM tại {settings.zai_base_url}: {exc}") from exc


def direct_predict_causality(entity_1: dict, entity_2: dict, context: str, relations: list[dict]) -> dict:
    relation_lines = [
        f"- {relation['source_name']} --[{relation['type']}]--> {relation['target_name']}"
        for relation in relations
    ]
    edge_text = "\n".join(f"{index + 1}. {line}" for index, line in enumerate(relation_lines))
    prompt = (
        "Classify the causal relation between two entities using only the supplied context "
        "and direct knowledge-graph edges.\n"
        "Return JSON only with exactly these fields: label (causal or non-causal), "
        "explanation (concise English), evidence_edges (list of edge numbers).\n"
        "Knowledge graph connectivity alone is not proof of causality. Do not invent evidence.\n\n"
        f"Entity 1: {entity_1['name']} ({entity_1['kind']}, {entity_1['id']})\n"
        f"Entity 2: {entity_2['name']} ({entity_2['kind']}, {entity_2['id']})\n"
        f"Context: {context}\nDirect KG edges:\n{edge_text or '(No direct edge)'}"
    )
    messages = [
        {"role": "system", "content": "You are a conservative causal-relation classifier. Return valid JSON only, in one line, with no reasoning outside JSON."},
        {"role": "user", "content": prompt},
    ]
    result = None
    last_error = None
    for attempt in range(2):
        try:
            result = _parse_json(zai_chat(messages, json_mode=True))
            break
        except LocalLLMError as error:
            last_error = error
            messages[-1]["content"] += "\nRetry: output the compact JSON object now."
    if result is None:
        raise last_error or LocalLLMError("Z.ai direct inference failed.")
    label = str(result.get("label", "non-causal")).lower().strip()
    if label not in {"causal", "non-causal"}:
        label = "non-causal"
    evidence_edges = result.get("evidence_edges", [])
    return {
        "label": label,
        "explanation": str(result.get("explanation", "No explanation returned.")),
        "evidence_edges": evidence_edges if isinstance(evidence_edges, list) else [],
        "model": settings.zai_model,
        "provider": "z.ai",
    }


def zai_translate_to_vietnamese(text: str) -> str:
    if not text.strip():
        return ""
    return zai_chat([
        {"role": "system", "content": "Translate into polished natural Vietnamese. Preserve biomedical names, IDs and relation codes. Return only the translation."},
        {"role": "user", "content": text},
    ]).strip()


def ollama_chat(messages: list[dict], temperature: float = 0.0, json_mode: bool = False, num_predict: int | None = None) -> str:
    prompt = "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages)
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict or settings.ollama_num_predict,
        },
    }
    if json_mode:
        payload["format"] = "json"
    try:
        r = requests.post(f"{settings.ollama_base_url}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        body = r.json()
        if "response" not in body:
            raise LocalLLMError(f"Ollama không trả response text: {body}")
        return body["response"]
    except requests.RequestException as e:
        raise LocalLLMError(
            f"Không gọi được Ollama tại {settings.ollama_base_url}. Hãy kiểm tra Ollama local đang chạy và đã có model '{settings.ollama_model}'. Chi tiết: {e}"
        ) from e

def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise LocalLLMError(f"LLM không trả JSON hợp lệ: {text[:500]}")


# ---------------------------------------------------------------------------
# Prompt style "paper": template tcma của repo chính (lib/prompts.py), tái
# triển khai ở đây để demo giữ standalone (không kéo lib/ + torch vào image).
# Trả về đúng 1 từ causal/non-causal rồi map sang nhãn demo (uncertain không dùng).
# ---------------------------------------------------------------------------

_PAPER_PREFIX = {
    "comagc": "Given the following information, classify the relation between the gene and disease pair. If there is a cause-effect relationship, say 'causal'; if not, say 'non-causal'.\n",
    "ade": "Given the following information, classify the relation between the drug and side effect pair. If there is a cause-effect relationship, say 'causal'; if not, say 'non-causal'.\n",
    "gene": "Given the following information, classify the relation between the gene pair. If there is a cause-effect relationship, state 'causal'; otherwise, state 'non-causal'.\n",
    "semeval": "Given the following information, classify the relation between the entity pair. If there is a cause-effect relationship, say 'causal'; if not, say 'non-causal'.\n",
}
_PAPER_TAIL = {
    "gemma": "The relation between {e1} and {e2} is",
    "llama": "The relation between {e1} and {e2} is:",
    "mistral": "[Relation]:",
}
_NEG_RE = re.compile(r"non[\s\-_]?causal|not[\s\-]+causal|no[\s\-]+causal|\bno\b|\bfalse\b|\bnegative\b|\bnot\b")
_POS_RE = re.compile(r"causal|\btrue\b|\bpositive\b|\byes\b")


def _paper_prompt(entity_1: str, entity_2: str, context: str, ranked_paths: list[dict], top_k: int) -> str:
    """Giống lib/prompts.build_prompt(template='tcma'): cặp + context + top-k path dạng 'stops'."""
    stops = []
    for p in ranked_paths[:top_k]:
        names = [n.get("name", "") for n in p.get("path_nodes", [])]
        if names:
            stops.append(" - ".join(names))
    mpath = "[Relation paths]: " + "".join(s + ". " for s in stops) + "\n"
    tail = _PAPER_TAIL.get(settings.paper_style, "[Relation]:")
    if "{e1}" in tail:
        tail = tail.format(e1=entity_1, e2=entity_2)
    return (
        _PAPER_PREFIX.get(settings.paper_task, _PAPER_PREFIX["comagc"])
        + f"[Pair]: {entity_1} and {entity_2}\n"
        + f"[Context sentence]: {' '.join(context.split())}\n"
        + mpath
        + tail
    )


def _verbalize(text: str) -> tuple[str, bool]:
    """Verbalizer robust của repo (lib/prompts.verbalize_robust): nhãn xuất hiện sớm nhất."""
    t = text.lower()
    n, p = _NEG_RE.search(t), _POS_RE.search(t)
    if n and (not p or n.start() <= p.start()):
        return "non-causal", True
    if p:
        return "causal", True
    return "non-causal", False


def _predict_causality_paper(entity_1: str, entity_2: str, context: str, ranked_paths: list[dict], top_k: int) -> dict:
    prompt = _paper_prompt(entity_1, entity_2, context, ranked_paths, top_k)
    messages = [
        {"role": "system", "content": "You are a strict classifier. Reply with exactly one word: causal or non-causal. Do not explain."},
        {"role": "user", "content": prompt},
    ]
    raw = chat(messages, temperature=0.0, json_mode=False, num_predict=16)
    label, parsed = _verbalize(raw)
    return {
        "label": label,
        "explanation": f"Paper-style zero-shot ({settings.paper_task}, tcma template): model replied '{raw.strip()[:60]}' -> {label}.",
        "evidence_paths": list(range(1, min(top_k, len(ranked_paths)) + 1)),
        "model": settings.llm_model or settings.ollama_model,
        "prompt_style": "paper",
        "parsed_ok": parsed,
    }


def describe_input_vi(entity_1: str, entity_2: str, context: str) -> str:
    """Return a short Vietnamese description of the selected input."""
    for attempt in range(3):
        strict_instruction = " Chỉ dùng chữ Latin/Vietnamese; nếu không biết thuật ngữ thì giữ nguyên tiếng Anh." if attempt else ""
        description = chat([
            {
                "role": "system",
                "content": f"Viết một mô tả tiếng Việt ngắn, tối đa hai câu, về cặp thực thể và context. Không dự đoán causal, không dùng ground truth, không thêm thông tin ngoài input và không viết ký tự Trung.{strict_instruction}",
            },
            {
                "role": "user",
                "content": f"Entity 1: {entity_1}\nEntity 2: {entity_2}\nContext: {context}",
            },
        ], temperature=0.0, num_predict=128).strip()
        if not CJK_RE.search(description):
            return description
        description = translate_to_vietnamese(description, strict=True).strip()
        if not CJK_RE.search(description):
            return description
    raise LocalLLMError("LLM vẫn trả ký tự Trung sau 3 lần tạo mô tả input.")

def predict_causality(entity_1: str, entity_2: str, context: str, ranked_paths: list[dict], top_k: int = 3) -> dict:
    if settings.causal_prompt_style == "paper":
        return _predict_causality_paper(entity_1, entity_2, context, ranked_paths, top_k)
    selected = ranked_paths[:top_k]
    path_lines = [f"Path {i+1}: {p.get('path_text','')} (rank_score={p.get('rank_score', 0):.4f})" for i, p in enumerate(selected)]
    user = f"""Task: classify whether this entity pair is described as having a causal relationship in the supplied context.
Return a JSON object with exactly these fields:
- label: one of \"causal\", \"non-causal\", \"uncertain\"
- explanation: concise English explanation, distinguishing context evidence from KG path evidence
- evidence_paths: list of path numbers used, e.g. [1, 2]

Important: Knowledge graph connectivity alone does not prove causality. Do not claim that indirect associations prove causation. If context and KG evidence are insufficient, return uncertain.

Entity 1: {entity_1}
Entity 2: {entity_2}
Context (English): {context}
Ranked Hetionet paths:
{chr(10).join(path_lines) if path_lines else '(No path found in the local KG)'}
"""
    result = _parse_json(chat([
        {"role": "system", "content": "You are a careful research assistant for knowledge-based causal relation classification. Be conservative and return only valid JSON."},
        {"role": "user", "content": user},
    ], temperature=0, json_mode=True))
    label = str(result.get("label", "uncertain")).lower().strip()
    if label not in {"causal", "non-causal", "uncertain"}:
        label = "uncertain"
    explanation = str(result.get("explanation", "No explanation returned."))
    evid = result.get("evidence_paths", [])
    if not isinstance(evid, list):
        evid = []
    return {"label": label, "explanation": explanation, "evidence_paths": evid, "model": settings.llm_model or settings.ollama_model, "prompt_style": "demo"}

def score_paths_relevance(entity_1: str, entity_2: str, context: str, paths: list[dict]) -> list[int]:
    """LLM weak-labels paths 0/1/2 for ranker training; these are pseudo-labels, not gold labels."""
    if not paths:
        return []
    lines = [f"{i+1}. {p.get('path_text', '')}" for i, p in enumerate(paths)]
    prompt = f"""Rate how useful each knowledge graph path is for deciding whether the pair below has a causal relation, using context. Return JSON only: {{\"scores\":[integer,...]}}. Each score must be 0 (not useful), 1 (somewhat useful), or 2 (strongly useful). Do not rate causal truth; rate relevance as evidence. Keep exactly {len(paths)} scores in input order.
Pair: {entity_1} | {entity_2}
Context: {context}
Paths:
{chr(10).join(lines)}"""
    result = _parse_json(chat([
        {"role": "system", "content": "You assign relevance ratings to biomedical knowledge graph paths. Return valid JSON only."},
        {"role": "user", "content": prompt},
    ], temperature=0, json_mode=True))
    scores = result.get("scores", [])
    if not isinstance(scores, list) or len(scores) != len(paths):
        raise LocalLLMError("Số relevance score không khớp số path; chạy lại hoặc dùng nhóm path ít hơn.")
    return [max(0, min(2, int(s))) for s in scores]

def translate_to_vietnamese(text: str, strict: bool = False) -> str:
    if not text.strip():
        return ""
    for attempt in range(3):
        use_strict = strict or attempt > 0
        instruction = (
            "Use only Latin/Vietnamese characters, digits and normal punctuation. "
            "If you do not know a medical term, copy that English term exactly. "
            if use_strict
            else ""
        )
        translated = chat([
            {"role": "system", "content": f"Return exactly one concise Vietnamese translation sentence and then stop. Never output analysis, repeated text, or Chinese characters. {instruction}Preserve gene symbols, disease names, entity IDs, relation identifiers and JSON-like identifiers; do not invent claims."},
            {"role": "user", "content": text},
        ], temperature=0.0, json_mode=False, num_predict=settings.ollama_translation_num_predict).strip()
        if not CJK_RE.search(translated):
            return translated
    # Keep the Vietnamese/Latin portion instead of failing the whole cached record
    # when the model appends an isolated CJK fragment after an otherwise usable translation.
    sanitized = CJK_RE.sub("", translated)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized or text.strip()
