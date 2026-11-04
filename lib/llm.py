"""Backend LLM cho Module 3 (zero-shot) và RankGPT.

  mock    : giả lập, không cần model/API (dùng cho --dry-run)
  hf      : chạy local bằng HuggingFace transformers (cách bài báo làm: greedy, max_new_tokens=6, prompt thô)
  openai  : mọi endpoint tương thích OpenAI (OpenAI, OpenRouter, Together, Groq, vLLM, Ollama, LM Studio...)
            -> khoá API đọc từ biến môi trường (--api-key-env), KHÔNG ghi vào file.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from . import paths


class FatalAPIError(RuntimeError):
    """Lỗi không nên retry (key sai, hết credit, model không tồn tại...)."""


class BaseLLM:
    name = "base"

    def generate(self, prompts: List[str]) -> List[str]:
        raise NotImplementedError

    def chat_many(self, conversations: List[List[dict]], max_tokens: int = 100) -> List[str]:
        """Nhiều hội thoại nhiều lượt (dùng cho RankGPT). Chỉ backend mock/openai hỗ trợ."""
        raise NotImplementedError(f"backend {self.name} không hỗ trợ chat nhiều lượt; dùng --backend openai")

    def generate_scored(self, prompts: List[str]) -> List[dict]:
        """Module 1: mỗi prompt -> {'text': câu trả lời, 'tokens': [[token, logprob], ...]} (log-prob chuẩn hoá log-softmax)."""
        raise NotImplementedError(f"backend {self.name} không hỗ trợ log-prob")


# ---------------------------------------------------------------------------
class MockLLM(BaseLLM):
    """Trả lời giả (xác định theo hash prompt) để kiểm tra toàn bộ pipeline mà không tốn tài nguyên."""
    name = "mock"

    def generate(self, prompts):
        outs = []
        for p in prompts:
            h = int(hashlib.md5(p.encode("utf-8")).hexdigest(), 16)
            if "[Relation paths]" in p and h % 3 != 0:
                outs.append(" causal")
            else:
                outs.append(" non-causal" if h % 2 else " causal. Knowledge")
        return outs

    def chat_many(self, conversations, max_tokens=100):
        import re
        outs = []
        for conv in conversations:
            n = len(re.findall(r"^\[\d+\]", "\n".join(m["content"] for m in conv if m["role"] == "user"), flags=re.M)) or 1
            outs.append(" > ".join(f"[{i}]" for i in range(n, 0, -1)))     # đảo thứ tự: để kiểm tra pipeline
        return outs

    def generate_scored(self, prompts):
        outs = []
        for p in prompts:
            h = int(hashlib.md5(p.encode("utf-8")).hexdigest(), 16)
            lab = " causal" if h % 3 else " non-causal"
            lp = -0.05 - (h % 997) / 997.0                                   # log-prob giả trong (-1.05, -0.05]
            outs.append({"text": lab, "tokens": [[lab, lp]]})
        return outs


# ---------------------------------------------------------------------------
class HFLocalLLM(BaseLLM):
    name = "hf"

    def __init__(self, model_name: str, max_new_tokens: int = 6, quant: str = "none", dtype: str = "auto",
                 device_map: str = "auto", hf_token_env: str = "HF_TOKEN", trust_remote_code: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        token = os.environ.get(hf_token_env) or None
        print(f"## Loading HF model {model_name} (quant={quant}, dtype={dtype}) ...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=trust_remote_code)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = {"token": token, "trust_remote_code": trust_remote_code}
        if dtype != "auto":
            kw["dtype"] = getattr(torch, dtype)
        else:
            kw["dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        if quant in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=(quant == "4bit"), load_in_8bit=(quant == "8bit"),
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        if torch.cuda.is_available():
            kw["device_map"] = device_map
        else:
            print("!! CẢNH BÁO: không thấy CUDA -> chạy CPU (rất chậm với model 7B)", flush=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
        except TypeError:                       # transformers cũ (<4.56) dùng torch_dtype thay cho dtype
            kw["torch_dtype"] = kw.pop("dtype")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def generate(self, prompts):
        outs = []
        for p in prompts:                       # batch size 1 = đúng cách của tác giả (llm.inference)
            enc = self.tok(p, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                out = self.model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                          max_new_tokens=self.max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.eos_token_id)
            outs.append(self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        return outs

    def generate_scored(self, prompts):
        """Đúng cách tác giả (llm.inference_with_score): log-softmax transition scores của từng token sinh ra (greedy)."""
        outs = []
        for p in prompts:
            enc = self.tok(p, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                out = self.model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                          max_new_tokens=self.max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.eos_token_id, return_dict_in_generate=True, output_scores=True)
                ts = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
            gen = out.sequences[0, enc["input_ids"].shape[1]:]
            toks = []
            for t, s in zip(gen.tolist(), ts.tolist()):
                if t == self.tok.eos_token_id:
                    break
                toks.append([self.tok.decode([t]), float(s)])
            outs.append({"text": self.tok.decode(gen, skip_special_tokens=True), "tokens": toks})
        return outs


# ---------------------------------------------------------------------------
class OpenAICompatLLM(BaseLLM):
    name = "openai"

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1", api_key_env: str = "OPENAI_API_KEY",
                 mode: str = "chat", max_new_tokens: int = 6, temperature: float = 0.0, concurrency: int = 4,
                 timeout: int = 60, max_retries: int = 6, cache: bool = True, extra_body: dict | None = None,
                 system_prompt: str | None = None):
        key = os.environ.get(api_key_env, "")
        if not key and "localhost" not in base_url and "127.0.0.1" not in base_url:
            raise RuntimeError(f"Chưa đặt biến môi trường {api_key_env}. Ví dụ PowerShell:  $env:{api_key_env}='sk-...'")
        self.key, self.model, self.mode = key, model, mode
        self.url_chat = base_url.rstrip("/") + "/chat/completions"
        self.url = self.url_chat if mode == "chat" else base_url.rstrip("/") + "/completions"
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        self.concurrency, self.timeout, self.max_retries = concurrency, timeout, max_retries
        self.extra_body = extra_body or {}
        self.system_prompt = system_prompt
        self.cache_path = paths.CACHE / "llm_cache.jsonl" if cache else None
        self._lock = threading.Lock()
        self._cache = {}
        if self.cache_path and self.cache_path.exists():
            for line in open(self.cache_path, encoding="utf-8"):
                try:
                    r = json.loads(line)
                    self._cache[r["k"]] = r["v"]
                except Exception:
                    pass

    def _key(self, prompt):
        s = json.dumps([self.url, self.model, self.mode, self.max_new_tokens, self.temperature, self.system_prompt,
                        self.extra_body, prompt], sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def _post_retry(self, url, body):
        """POST có retry (429/5xx) - trả về JSON; lỗi 400/401/402/403/404 -> FatalAPIError (không retry)."""
        import requests
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        err = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(url, headers=headers, json=body, timeout=self.timeout)
                if r.status_code in (400, 401, 402, 403, 404):
                    raise FatalAPIError(f"HTTP {r.status_code} từ {self.url} (model={self.model}): {r.text[:300]}")
                if r.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                j = r.json()
                if "choices" not in j:
                    raise RuntimeError(f"phản hồi không có 'choices': {str(j)[:200]}")
                return j
            except FatalAPIError:
                raise
            except Exception as e:  # noqa: BLE001
                err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Gọi API thất bại sau {self.max_retries} lần: {err}")

    def _cache_put(self, k, v):
        if self.cache_path:
            with self._lock:
                self._cache[k] = v
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")

    def _call(self, prompt, messages=None, max_tokens=None):
        mt = max_tokens or self.max_new_tokens
        k = self._key(prompt if messages is None else json.dumps(messages, sort_keys=True) + f"|{mt}")
        if k in self._cache:
            return self._cache[k]
        if self.mode == "chat" or messages is not None:
            if messages is not None:
                msgs = messages
            else:
                msgs = ([{"role": "system", "content": self.system_prompt}] if self.system_prompt else []) + \
                       [{"role": "user", "content": prompt}]
            body = {"model": self.model, "messages": msgs, "max_tokens": mt, "temperature": self.temperature}
            url = self.url_chat
        else:
            body = {"model": self.model, "prompt": prompt, "max_tokens": mt, "temperature": self.temperature}
            url = self.url
        body.update(self.extra_body)
        j = self._post_retry(url, body)
        ch = j["choices"][0]
        text = (ch["message"].get("content") or "") if "message" in ch else (ch.get("text") or "")
        self._cache_put(k, text)
        return text

    # -- Module 1: cần log-prob của token sinh ra ------------------------------------------------------------
    def _call_scored(self, prompt):
        """Trả {'text': str, 'tokens': [[token, logprob], ...]}. Yêu cầu provider trả logprobs đầy đủ & nhất quán với text."""
        from .relevancy import check_logprob_integrity
        k = "lp|" + self._key(prompt)
        if k in self._cache:
            return self._cache[k]
        if self.mode == "chat":
            msgs = ([{"role": "system", "content": self.system_prompt}] if self.system_prompt else []) + \
                   [{"role": "user", "content": prompt}]
            body = {"model": self.model, "messages": msgs, "max_tokens": self.max_new_tokens, "temperature": self.temperature,
                    "logprobs": True, "top_logprobs": 1}
            url = self.url_chat
        else:
            body = {"model": self.model, "prompt": prompt, "max_tokens": self.max_new_tokens, "temperature": self.temperature,
                    "logprobs": 1}
            url = self.url
        body.update(self.extra_body)
        if "openrouter" in url and "provider" not in body:          # chỉ định tuyến tới provider CÓ hỗ trợ logprobs
            body["provider"] = {"require_parameters": True}
        j = self._post_retry(url, body)
        ch = j["choices"][0]
        text = (ch["message"].get("content") or "") if "message" in ch else (ch.get("text") or "")
        lp = ch.get("logprobs") or {}
        if "content" in lp and lp["content"]:                         # định dạng chat
            tokens = [[t["token"], float(t["logprob"])] for t in lp["content"]]
        elif lp.get("tokens"):                                        # định dạng completion
            tokens = [[t, float(x)] for t, x in zip(lp["tokens"], lp.get("token_logprobs") or [])]
        else:
            tokens = []
        if not tokens:
            raise FatalAPIError(f"Provider '{j.get('provider')}' không trả logprobs cho model {self.model}. "
                                "Module 1 cần log-prob: dùng --model mistralai/mistral-nemo (đã kiểm tra có logprobs trên OpenRouter) "
                                "hoặc --backend hf (chạy local, đúng cách bài báo).")
        problem = check_logprob_integrity(text, tokens)
        if problem:
            raise FatalAPIError(f"Logprobs của provider '{j.get('provider')}' không nhất quán với văn bản ({problem}); "
                                f"text={text!r} tokens={tokens[:4]}. Đổi --model/provider (--extra-body) hoặc dùng --backend hf.")
        v = {"text": text, "tokens": tokens, "provider": j.get("provider")}
        self._cache_put(k, v)
        return v

    def generate_scored(self, prompts):
        if self.concurrency <= 1:
            return [self._call_scored(p) for p in prompts]
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(self._call_scored, prompts))

    def generate(self, prompts):
        if self.concurrency <= 1:
            return [self._call(p) for p in prompts]
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(self._call, prompts))

    def chat_many(self, conversations, max_tokens=100):
        fn = lambda c: self._call("", messages=c, max_tokens=max_tokens)  # noqa: E731
        if self.concurrency <= 1:
            return [fn(c) for c in conversations]
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(fn, conversations))


# ---------------------------------------------------------------------------
def add_llm_args(ap):
    g = ap.add_argument_group("LLM backend")
    g.add_argument("--backend", choices=["mock", "hf", "openai"], default="mock",
                   help="mock = dry-run; hf = local transformers; openai = API tương thích OpenAI")
    g.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.1", help="tên model HF hoặc id model của API")
    g.add_argument("--max-new-tokens", type=int, default=6, help="bài báo dùng 6")
    # hf
    g.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none", help="[hf] lượng tử hoá bitsandbytes")
    g.add_argument("--dtype", default="auto", help="[hf] auto|float16|bfloat16|float32")
    # api
    g.add_argument("--base-url", default="https://api.openai.com/v1", help="[openai] vd https://openrouter.ai/api/v1")
    g.add_argument("--api-key-env", default="OPENAI_API_KEY", help="[openai] tên biến môi trường chứa API key")
    g.add_argument("--api-mode", choices=["chat", "completion"], default="chat")
    g.add_argument("--concurrency", type=int, default=4, help="[openai] số request song song")
    g.add_argument("--system-prompt", default="", help="[openai, chat] system prompt, vd ép trả lời đúng 1 từ (dùng đồng nhất cho mọi option)")
    g.add_argument("--extra-body", default="", help='[openai] JSON trộn vào body request, vd OpenRouter: {"provider":{"order":["Novita"],"allow_fallbacks":false}}')
    return ap


def make_llm(args) -> BaseLLM:
    if args.backend == "mock":
        return MockLLM()
    if args.backend == "hf":
        return HFLocalLLM(args.model, args.max_new_tokens, args.quant, args.dtype)
    extra = json.loads(args.extra_body) if getattr(args, "extra_body", "") else None
    return OpenAICompatLLM(args.model, args.base_url, args.api_key_env, args.api_mode, args.max_new_tokens,
                           concurrency=args.concurrency, extra_body=extra,
                           system_prompt=(getattr(args, "system_prompt", "") or None))





