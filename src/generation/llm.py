"""
Open-source LLM wrapper — multi-backend, same interface.

Supported backends (tried in priority order if backend="auto")
───────────────────────────────────────────────────────────────
  ollama       Local Ollama server (llama3.2, mistral, phi3, …)
               Requires: `ollama serve` + `ollama pull <model>`

  openai       OpenAI API (gpt-4o-mini, gpt-3.5-turbo, …)
               Requires: OPENAI_API_KEY env var

  anthropic    Anthropic API (claude-3-haiku, claude-3-sonnet, …)
               Requires: ANTHROPIC_API_KEY env var

  huggingface  HuggingFace Inference API (serverless, free tier)
               Requires: HF_API_TOKEN env var

  transformers Local HuggingFace model via 🤗 transformers
               Runs CPU-only on small models (phi-2, TinyLlama, …)
               No API key needed — downloads model on first run.

  auto         Try ollama → openai → anthropic → huggingface → transformers
               Uses the first backend that succeeds.

Environment variables (set in .env or shell)
──────────────────────────────────────────────
  OPENAI_API_KEY        for openai backend
  ANTHROPIC_API_KEY     for anthropic backend
  HF_API_TOKEN          for huggingface backend
  LLM_BACKEND           overrides config (e.g. "openai")
  LLM_MODEL             overrides config model name
"""
import logging
import os
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── backend priority for "auto" mode ─────────────────────────────────────────
_AUTO_ORDER = ["ollama", "openai", "anthropic", "huggingface", "transformers"]

# ── sensible default model per backend ────────────────────────────────────────
_DEFAULT_MODEL = {
    "ollama":       "llama3.2",
    "openai":       "gpt-4o-mini",
    "anthropic":    "claude-3-haiku-20240307",
    "huggingface":  "mistralai/Mistral-7B-Instruct-v0.2",
    "transformers": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


class LLMWrapper:
    def __init__(
        self,
        model_name: str = "auto",
        backend: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 60,
    ):
        # Allow env-var override so docker / CI can inject without code changes
        self.backend     = os.environ.get("LLM_BACKEND", backend)
        self.model_name  = os.environ.get("LLM_MODEL", model_name)
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout
        self._client     = None
        self._available  = False
        self._active_backend: str = "none"
        self._init()

    # ── init / backend detection ──────────────────────────────────────────────

    def _init(self):
        order = _AUTO_ORDER if self.backend == "auto" else [self.backend]
        for b in order:
            ok = getattr(self, f"_try_{b}", lambda: False)()
            if ok:
                self._available = True
                self._active_backend = b
                # resolve "auto" model name now that we know the backend
                if self.model_name in ("auto", "llama3.2") and b != "ollama":
                    self.model_name = _DEFAULT_MODEL[b]
                logger.info("LLM backend: %s  model: %s", b, self.model_name)
                return
        logger.warning(
            "No LLM backend available — retrieval-only mode. "
            "To enable: start Ollama, or set OPENAI_API_KEY / ANTHROPIC_API_KEY / HF_API_TOKEN."
        )

    # ── individual backend probes ─────────────────────────────────────────────

    def _try_ollama(self) -> bool:
        try:
            import ollama
            models = ollama.list()
            available = [m["model"] for m in models.get("models", [])]
            if not available:
                logger.debug("Ollama running but no models pulled.")
                return False
            if self.model_name not in ("auto",) and self.model_name not in available:
                # Model not pulled — try the first available model instead
                logger.info("Model %r not in Ollama; using %r", self.model_name, available[0])
                self.model_name = available[0]
            elif self.model_name == "auto":
                self.model_name = available[0]
            self._client = ollama
            return True
        except Exception as e:
            logger.debug("Ollama probe failed: %s", e)
            return False

    def _try_openai(self) -> bool:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return False
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, timeout=self.timeout)
            # cheap connectivity probe
            client.models.list()
            self._client = client
            if self.model_name == "auto":
                self.model_name = _DEFAULT_MODEL["openai"]
            return True
        except Exception as e:
            logger.debug("OpenAI probe failed: %s", e)
            return False

    def _try_anthropic(self) -> bool:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return False
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            self._client = client
            if self.model_name == "auto":
                self.model_name = _DEFAULT_MODEL["anthropic"]
            return True
        except Exception as e:
            logger.debug("Anthropic probe failed: %s", e)
            return False

    def _try_huggingface(self) -> bool:
        token = os.environ.get("HF_API_TOKEN", "")
        if not token:
            return False
        try:
            from huggingface_hub import InferenceClient
            model = (self.model_name if self.model_name != "auto"
                     else _DEFAULT_MODEL["huggingface"])
            client = InferenceClient(model=model, token=token, timeout=self.timeout)
            # quick probe
            client.text_generation("hi", max_new_tokens=1)
            self._client = client
            self.model_name = model
            return True
        except Exception as e:
            logger.debug("HuggingFace Inference API probe failed: %s", e)
            return False

    def _try_transformers(self) -> bool:
        """Local pipeline — CPU-safe with a small model."""
        try:
            from transformers import pipeline as hf_pipeline
            model = (self.model_name if self.model_name not in ("auto", "llama3.2")
                     else _DEFAULT_MODEL["transformers"])
            logger.info("Loading local transformers model %r (first run downloads ~600 MB) …", model)
            pipe = hf_pipeline(
                "text-generation",
                model=model,
                device_map="auto",
                max_new_tokens=self.max_tokens,
                temperature=max(self.temperature, 1e-6),  # HF requires > 0
                do_sample=self.temperature > 0,
            )
            self._client = pipe
            self.model_name = model
            return True
        except Exception as e:
            logger.debug("Transformers local probe failed: %s", e)
            return False

    # ── generation ────────────────────────────────────────────────────────────

    def generate(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Returns:
            text        – generated answer string
            latency_ms  – wall-clock time in ms
            model       – model name used
            backend     – backend used
        """
        if not self._available:
            return {
                "text": (
                    "[LLM offline — showing top retrieved passage]\n\n"
                    + user_prompt.split("QUESTION:")[-1].strip()
                ),
                "latency_ms": 0.0,
                "model": "none",
                "backend": "none",
            }

        t0 = time.perf_counter()
        try:
            text = self._call_backend(system_prompt, user_prompt)
            lat  = (time.perf_counter() - t0) * 1000
            logger.debug("LLM %.1f ms  backend=%s  model=%s", lat, self._active_backend, self.model_name)
            return {
                "text": text,
                "latency_ms": round(lat, 1),
                "model": self.model_name,
                "backend": self._active_backend,
            }
        except Exception as e:
            lat = (time.perf_counter() - t0) * 1000
            logger.error("LLM generation error (%s): %s", self._active_backend, e)
            return {
                "text": f"[LLM error: {e}]",
                "latency_ms": round(lat, 1),
                "model": self.model_name,
                "backend": self._active_backend,
            }

    def _call_backend(self, system: str, user: str) -> str:
        b = self._active_backend

        if b == "ollama":
            resp = self._client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )
            return resp["message"]["content"].strip()

        if b == "openai":
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content.strip()

        if b == "anthropic":
            import anthropic
            resp = self._client.messages.create(
                model=self.model_name,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return resp.content[0].text.strip()

        if b == "huggingface":
            prompt = f"[SYSTEM]\n{system}\n\n[USER]\n{user}\n\n[ASSISTANT]"
            return self._client.text_generation(
                prompt,
                max_new_tokens=self.max_tokens,
                temperature=max(self.temperature, 0.01),
                stop_sequences=["[USER]", "[SYSTEM]"],
            ).strip()

        if b == "transformers":
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ]
            out = self._client(messages)
            # pipeline returns list[{"generated_text": ...}]
            generated = out[0].get("generated_text", "")
            # strip the input prompt if the model echoes it
            if isinstance(generated, list):
                # chat template returns list of dicts
                assistant_msgs = [m for m in generated if m.get("role") == "assistant"]
                return assistant_msgs[-1]["content"].strip() if assistant_msgs else ""
            return generated.strip()

        return "[Unknown backend]"

    # ── utilities ─────────────────────────────────────────────────────────────

    def list_models(self) -> List[str]:
        if self._active_backend == "ollama" and self._available:
            try:
                return [m["model"] for m in self._client.list().get("models", [])]
            except Exception:
                pass
        if self._active_backend == "openai" and self._available:
            try:
                return [m.id for m in self._client.models.list().data
                        if "gpt" in m.id]
            except Exception:
                pass
        return []

    @property
    def available(self) -> bool:
        return self._available
