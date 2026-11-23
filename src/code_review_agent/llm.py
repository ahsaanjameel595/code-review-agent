import os
from dotenv import load_dotenv
from litellm import Router

load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL")  # optional: huggingface/<provider>/<org>/<model>


def _build_model_list() -> list[dict]:
    models: list[dict] = []

    if GROQ_KEY:
        groq = lambda name, model: {
            "model_name": name,
            "litellm_params": {"model": model, "api_key": GROQ_KEY},
        }
        models += [
            groq("smart", "groq/openai/gpt-oss-120b"),
            groq("smart_b", "groq/llama-3.3-70b-versatile"),
            groq("smart_c", "groq/openai/gpt-oss-20b"),
            groq("fast", "groq/llama-3.1-8b-instant"),
        ]

    if HF_TOKEN and HF_MODEL:
        models.append(
            {
                "model_name": "smart_hf",
                "litellm_params": {"model": HF_MODEL, "api_key": HF_TOKEN},
            }
        )

    if not models:
        raise RuntimeError("No API key found in .env")
    return models


def build_fallback(models: list[dict]) -> list[dict]:
    names = {m["model_name"] for m in models}
    smart = [n for n in ("smart_b", "smart_c", "smart_hf") if n in names]
    fast = [n for n in ("smart_c", "smart_hf") if n in names]
    fallbacks = []
    if "smart" in names and smart:
        fallbacks.append({"smart": smart})
    if "fast" in names and fast:
        fallbacks.append({"fast": fast})
    return fallbacks


models = _build_model_list()

router = Router(
    model_list=models,
    fallbacks=build_fallback(models),
    num_retries=2,
    timeout=60,
    allowed_fails=2,
    cooldown_time=60,
)

# Simple in-memory token counter (used for the "cost per task" metric)
usage_log: dict = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "by_model": {}}


def _record(resp) -> None:
    u = getattr(resp, "usage", None)
    if not u:
        return
    model_used = getattr(resp, "model", "unknown")
    usage_log["calls"] += 1
    usage_log["prompt_tokens"] += u.prompt_tokens or 0
    usage_log["completion_tokens"] += u.completion_tokens or 0
    per = usage_log["by_model"].setdefault(model_used, {"calls": 0, "tokens": 0})
    per["calls"] += 1
    per["tokens"] += u.total_tokens or 0


async def chat(
    messages: list[dict],
    model: str = "smart",
    tools: list[dict] | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,  # reasoning models (gpt-oss) also spend tokens thinking
):
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if tools:
        kwargs["tools"] = tools

    resp = await router.acompletion(**kwargs)
    _record(resp)
    return resp
