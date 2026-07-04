#!/usr/bin/env python3
"""
inference_bench.py

General local inference benchmark tool: measures prefill speed, decode speed, and TTFT.
Supports three backends:

  ollama    Ollama native API
  openai    Any OpenAI-compatible /v1/chat/completions endpoint
               covers ds4-server (DeepSeek V4 Flash), llama.cpp server, vLLM,
               and LM Studio's /v1 endpoint
  lmstudio  LM Studio's native /api/v0 endpoint (provides more precise built-in stats;
               suitable for both GGUF and MLX models run inside LM Studio)

Usage example:
    # Gemma4 MLX model on Ollama
    python3 inference_bench.py --backend ollama --model gemma4:31b-mlx --context-size 64k 128k 256k

    # DeepSeek R1 run on Ollama
    python3 inference_bench.py --backend ollama --model deepseek-r1:70b

    # DeepSeek V4 Flash run on ds4.c (openai compatible endpoint)
    python3 inference_bench.py --backend openai --host http://127.0.0.1:8080 --model deepseek-v4-flash

    # Qwen3.5 GGUF model run on LM Studio
    python3 inference_bench.py --backend lmstudio --model qwen/qwen3.5-35b-a3b

    # Gemma4 MLX model on LM Studio
    python3 inference_bench.py --backend lmstudio --model gemma-4-26b-a4b-it-qat --context-size 32k 64k 128k 256k

Dependencies:
    pip install requests
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import random
import statistics
import string
import sys
import threading
import time
import unicodedata

import requests

# ---------------------------------------------------------------------------
# ANSI Terminal Colors
# ---------------------------------------------------------------------------
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_BLUE = "\033[34m"
C_CYAN = "\033[36m"
C_RED = "\033[31m"
C_MAGENTA = "\033[35m"

def color(text: str, ansi_code: str) -> str:
    return f"{ansi_code}{text}{C_RESET}"

# ---------------------------------------------------------------------------
# Global Stage Labels (Pre-colored for console logging)
# ---------------------------------------------------------------------------
LABEL_PREFILL = color("[Prefill]", C_YELLOW)
LABEL_DECODE = color("[Decode]", C_YELLOW)
LABEL_REQ_SENT = color("[Request Sent]", C_YELLOW)


# ---------------------------------------------------------------------------
# Spinner Class: Displays smooth loading animation on a single line in terminal
# ---------------------------------------------------------------------------

class Spinner:
    def __init__(self, prefix="", message="Waiting...", show=True):
        self.prefix = prefix
        self.message = message
        self.show = show
        self.chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.idx = 0
        self.active = False
        self.thread = None

    def __enter__(self):
        if self.show:
            self.active = True
            self.thread = threading.Thread(target=self._spin, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.show and self.active:
            self.active = False
            if self.thread:
                self.thread.join(timeout=1.0)
            # Clear current line
            sys.stdout.write("\r" + " " * 120 + "\r")
            sys.stdout.flush()

    def _spin(self):
        while self.active:
            sys.stdout.write(f"\r{self.prefix}{self.chars[self.idx]} {self.message}")
            sys.stdout.flush()
            self.idx = (self.idx + 1) % len(self.chars)
            time.sleep(0.1)

    def update(self, message):
        self.message = message


# ---------------------------------------------------------------------------
# Terminal Width Math: East Asian Width support for correct spacing in aligned tables
# ---------------------------------------------------------------------------

def display_width(s: str) -> int:
    width = 0
    for ch in s:
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def pad(s: str, target_width: int, align: str = "right") -> str:
    s = str(s)
    gap = max(target_width - display_width(s), 0)
    if align == "left":
        return s + " " * gap
    if align == "center":
        left = gap // 2
        return " " * left + s + " " * (gap - left)
    return " " * gap + s  # right


# ---------------------------------------------------------------------------
# Prompt Generation: Shuffles sentence lists and injects nonces to bypass Prefix Caching
# ---------------------------------------------------------------------------

FILLER_SENTENCES = [
    "Artificial intelligence technology is developing at an exponential rate.",
    "Machine learning models are reshaping the workflow of various traditional industries.",
    "From deep natural language understanding to computer vision, boundaries are constantly broken.",
    "Autonomous driving and intelligent recommendation systems rely heavily on neural networks.",
    "Large language models demonstrate strong emergent abilities as parameters scale up.",
    "Reinforcement learning from human feedback helps align agent behaviors with human values.",
    "The integration of edge computing accelerates the local deployment of small models.",
    "Hardware accelerators like GPUs and specialized NPUs are crucial for low-latency inference."
]


def build_prompt(approx_tokens: int) -> str:
    # 1. Shuffle sentences to scramble the prefix hash chain completely
    sentences = FILLER_SENTENCES.copy()
    random.shuffle(sentences)

    # 2. Add random noise word into sentences
    text_blocks = []
    for s in sentences:
        noise = "".join(random.choices(string.ascii_lowercase, k=4))
        text_blocks.append(f"{s} (ref:{noise})")

    base_text = " ".join(text_blocks)

    # 3. Scale text to match approx_tokens
    target_chars = approx_tokens * 4
    repeated_text = (base_text * (target_chars // len(base_text) + 1))[:target_chars]

    # 4. Generate long-generation instructions to force max tokens output
    instruction = (
        "\n\nBased on the text above, write an extremely detailed and highly descriptive technical "
        "analysis report of at least 800 words. Expand comprehensively on the industrial impact, "
        "future trends, technological challenges, and concrete implementation details for each topic, "
        "ensuring the generated output is as exhaustive and lengthy as possible:"
    )

    return repeated_text + instruction


NAN = float("nan")


def _is_nan(x):
    return x != x


def format_val(val: float, precision: int = 1, suffix: str = "") -> str:
    """Safely formats floating point values, converting NANs/Nones into 'N/A'"""
    if val is None or _is_nan(val):
        return "N/A"
    return f"{val:.{precision}f}{suffix}"


def parse_context_size(val_str: str) -> int:
    if not val_str:
        return None
    val_str = val_str.strip().lower()
    if val_str.endswith("k"):
        try:
            return int(float(val_str[:-1]) * 1024)
        except ValueError:
            pass
    if val_str.endswith("m"):
        try:
            return int(float(val_str[:-1]) * 1024 * 1024)
        except ValueError:
            pass
    try:
        return int(val_str)
    except ValueError:
        import argparse
        raise argparse.ArgumentTypeError(f"Invalid context-size value: '{val_str}'. Should be an integer or a string like '16k'.")


# ---------------------------------------------------------------------------
# Backend Adapters: Each returns a dict in a unified format:
#   prompt_tokens, prefill_tok_s, prefill_reliable,
#   gen_tokens, decode_tok_s, ttft_s, wall_time_s
# ---------------------------------------------------------------------------

def get_ollama_downloaded_models(host):
    try:
        url = f"{host.rstrip('/')}/api/tags"
        resp = requests.get(url, timeout=1.5)
        if resp.status_code == 200:
            data = resp.json()
            models = []
            for m in data.get("models", []):
                name = m.get("name")
                if name:
                    models.append(name)
            return models
    except Exception:
        pass
    return None


def run_ollama(host, model, prompt, gen_tokens, temperature, show_progress, prefix="", context_size=None):
    url = f"{host.rstrip('/')}/api/generate"
    options = {"num_predict": gen_tokens, "temperature": temperature}
    if context_size is not None:
        options["num_ctx"] = context_size
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": options,
    }

    t_start = time.perf_counter()
    t_first_token = None
    final_chunk = None
    tokens_received = 0

    with Spinner(prefix, f"{LABEL_PREFILL} Waiting for response...", show=show_progress) as spinner:
        with requests.post(url, json=payload, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if t_first_token is None and chunk.get("response"):
                    t_first_token = time.perf_counter()
                if chunk.get("response"):
                    tokens_received += 1
                    if show_progress:
                        elapsed = time.perf_counter() - t_start
                        if t_first_token:
                            time_diff = time.perf_counter() - t_first_token
                            spd = tokens_received / time_diff if time_diff > 0 else 0.0
                            spinner.update(f"{LABEL_DECODE} Generating: {tokens_received}/{gen_tokens} tokens, decode ≈ {spd:.1f} tok/s, elapsed {elapsed:.1f}s")
                        else:
                            spinner.update(f"{LABEL_PREFILL} Processing, elapsed {elapsed:.1f}s")
                if chunk.get("done"):
                    final_chunk = chunk
                    break

    t_end = time.perf_counter()
    if final_chunk is None:
        raise RuntimeError("No output received. Please verify the model name and service status.")

    prompt_eval_count = final_chunk.get("prompt_eval_count", 0)
    prompt_eval_duration = final_chunk.get("prompt_eval_duration", 0) / 1e9
    eval_count = final_chunk.get("eval_count", 0)
    eval_duration = final_chunk.get("eval_duration", 0) / 1e9

    MIN_RELIABLE = 0.02
    reliable = prompt_eval_duration >= MIN_RELIABLE
    prefill_tok_s = (prompt_eval_count / prompt_eval_duration) if reliable and prompt_eval_duration > 0 else NAN

    return {
        "prompt_tokens": prompt_eval_count,
        "prefill_tok_s": prefill_tok_s,
        "prefill_reliable": reliable,
        "gen_tokens": eval_count,
        "decode_tok_s": (eval_count / eval_duration) if eval_duration > 0 else NAN,
        "ttft_s": prompt_eval_duration if prompt_eval_duration > 0 else None,
        "wall_time_s": t_end - t_start,
    }


def run_openai_compatible(host, model, prompt, gen_tokens, temperature, show_progress, prefix=""):
    url = f"{host.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    t_first_token = None
    t_last_token = None
    tokens_received = 0
    usage = None

    with Spinner(prefix, f"{LABEL_PREFILL} Waiting for response...", show=show_progress) as spinner:
        with requests.post(url, json=payload, stream=True) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)

                if chunk.get("usage"):
                    usage = chunk["usage"]

                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    reasoning_content = delta.get("reasoning_content")
                    if content or reasoning_content:
                        now = time.perf_counter()
                        if t_first_token is None:
                            t_first_token = now
                        t_last_token = now
                        tokens_received += 1
                        if show_progress:
                            elapsed = now - t_start
                            time_diff = now - t_first_token if t_first_token else 0.0
                            spd = tokens_received / time_diff if time_diff > 0 else 0.0
                            spinner.update(f"{LABEL_DECODE} Generating: ~{tokens_received}/{gen_tokens} tokens, decode ≈ {spd:.1f} tok/s, elapsed {elapsed:.1f}s")

    t_end = time.perf_counter()
    if t_first_token is None:
        raise RuntimeError("No streaming content received. Please check backend config or model status.")

    prompt_tokens = usage.get("prompt_tokens") if usage else None
    completion_tokens = usage.get("completion_tokens") if usage else tokens_received
    cached_tokens = 0
    if usage:
        cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)

    ttft = t_first_token - t_start
    reliable = cached_tokens == 0
    prefill_tok_s = (prompt_tokens / ttft) if (prompt_tokens and ttft > 0 and reliable) else NAN

    decode_span = (t_last_token - t_first_token) if (t_last_token and completion_tokens and completion_tokens > 1) else None
    decode_tok_s = ((completion_tokens - 1) / decode_span) if decode_span and decode_span > 0 else NAN

    return {
        "prompt_tokens": prompt_tokens if prompt_tokens is not None else float("nan"),
        "prefill_tok_s": prefill_tok_s,
        "prefill_reliable": reliable,
        "gen_tokens": completion_tokens,
        "decode_tok_s": decode_tok_s,
        "ttft_s": ttft,
        "wall_time_s": t_end - t_start,
    }
def get_lmstudio_models_info(host):
    try:
        url = f"{host.rstrip('/')}/api/v1/models"
        resp = requests.get(url, timeout=1.5)
        if resp.status_code == 200:
            data = resp.json()
            downloaded = []
            loaded = {}
            for m in data.get("models", []):
                key = m.get("key")
                if key:
                    downloaded.append(key)
                    if m.get("loaded_instances"):
                        inst_ids = [inst.get("id") for inst in m["loaded_instances"] if inst.get("id")]
                        if inst_ids:
                            loaded[key] = inst_ids
            return downloaded, loaded
    except Exception:
        pass
    return None, None


def unload_ollama_model(host, model_key):
    try:
        url = f"{host.rstrip('/')}/api/generate"
        payload = {"model": model_key, "keep_alive": 0}
        resp = requests.post(url, json=payload, timeout=10.0)
        return resp.status_code == 200
    except Exception:
        return False


def load_ollama_model(host, model_key, context_size=None):
    try:
        url = f"{host.rstrip('/')}/api/generate"
        payload = {
            "model": model_key,
            "prompt": "",
            "keep_alive": -1
        }
        if context_size is not None:
            payload["options"] = {"num_ctx": context_size}
        resp = requests.post(url, json=payload, timeout=180.0)
        return resp.status_code == 200
    except Exception:
        return False


def unload_lmstudio_model(host, instance_id):
    url = f"{host.rstrip('/')}/api/v1/models/unload"
    payload = {"instance_id": instance_id}
    resp = requests.post(url, json=payload, timeout=120.0)
    resp.raise_for_status()
    return True


def load_lmstudio_model(host, model_key, context_length=None):
    url = f"{host.rstrip('/')}/api/v1/models/load"
    payload = {"model": model_key}
    if context_length is not None:
        payload["context_length"] = context_length
    resp = requests.post(url, json=payload, timeout=120.0)
    resp.raise_for_status()
    return True


def ping_lmstudio_model(host, model_key):
    try:
        url = f"{host.rstrip('/')}/api/v0/chat/completions"
        payload = {
            "model": model_key,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "temperature": 0.0,
        }
        resp = requests.post(url, json=payload, timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


def get_model_metadata(backend, host, model_key):
    res = None
    if backend == "lmstudio":
        try:
            url = f"{host.rstrip('/')}/api/v1/models"
            resp = requests.get(url, timeout=1.5)
            if resp.status_code == 200:
                data = resp.json()
                for m in data.get("models", []):
                    if m.get("key") == model_key:
                        quant = m.get("quantization", {})
                        quant_name = quant.get("name") if isinstance(quant, dict) else quant
                        
                        active_ctx = None
                        loaded_instances = m.get("loaded_instances", [])
                        if loaded_instances:
                            active_ctx = loaded_instances[0].get("config", {}).get("context_length")
                        
                        res = {
                            "params": m.get("params_string"),
                            "size_bytes": m.get("size_bytes"),
                            "format": m.get("format"),
                            "quant": quant_name,
                            "max_context": m.get("max_context_length"),
                            "active_context": active_ctx,
                            "arch": m.get("architecture"),
                        }
                        break
        except Exception:
            pass
    elif backend == "ollama":
        try:
            url = f"{host.rstrip('/')}/api/tags"
            resp = requests.get(url, timeout=1.5)
            if resp.status_code == 200:
                data = resp.json()
                for m in data.get("models", []):
                    name = m.get("name")
                    if name == model_key or (":" in name and name.rsplit(":", 1)[0] == model_key):
                        details = m.get("details", {})
                        
                        active_ctx = None
                        max_ctx = None
                        try:
                            show_url = f"{host.rstrip('/')}/api/show"
                            show_resp = requests.post(show_url, json={"name": name}, timeout=2.0)
                            if show_resp.status_code == 200:
                                show_data = show_resp.json()
                                params_text = show_data.get("parameters", "")
                                import re
                                match = re.search(r"num_ctx\s+(\d+)", params_text)
                                if match:
                                    active_ctx = int(match.group(1))
                                
                                model_info = show_data.get("model_info", {})
                                for key, val in model_info.items():
                                    if key.endswith(".context_length"):
                                        max_ctx = int(val)
                                        break
                        except Exception:
                            pass
                            
                        res = {
                            "params": details.get("parameter_size"),
                            "size_bytes": m.get("size"),
                            "format": details.get("format"),
                            "quant": details.get("quantization_level"),
                            "max_context": max_ctx,
                            "active_context": active_ctx,
                            "arch": details.get("family"),
                        }
                        break
        except Exception:
            pass
    
    if not res:
        res = {
            "params": "N/A",
            "size_bytes": None,
            "format": "N/A",
            "quant": "N/A",
            "max_context": None,
            "active_context": None,
            "arch": "N/A",
        }

    # 替换所有的 None 值为 "N/A" 以保持卡片显示的一致性，但不做任何猜测
    for k, v in res.items():
        if v is None and k != "size_bytes" and k != "max_context" and k != "active_context":
            res[k] = "N/A"

    return res


def run_lmstudio(host, model, prompt, gen_tokens, temperature, show_progress, prefix=""):
    url = f"{host.rstrip('/')}/api/v0/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen_tokens,
        "temperature": temperature,
        "stream": False,
    }

    t_start = time.perf_counter()
    with Spinner(prefix, f"{LABEL_REQ_SENT} Waiting for full response from LM Studio...", show=show_progress) as spinner:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    t_end = time.perf_counter()

    stats = data.get("stats") or {}
    usage = data.get("usage") or {}

    prompt_tokens = usage.get("prompt_tokens", NAN)
    completion_tokens = usage.get("completion_tokens", NAN)
    ttft = stats.get("time_to_first_token")
    decode_tok_s = stats.get("tokens_per_second", NAN)

    prefill_tok_s = (prompt_tokens / ttft) if (ttft and ttft > 0 and prompt_tokens == prompt_tokens) else NAN

    return {
        "prompt_tokens": prompt_tokens,
        "prefill_tok_s": prefill_tok_s,
        "prefill_reliable": True,
        "gen_tokens": completion_tokens,
        "decode_tok_s": decode_tok_s,
        "ttft_s": ttft,
        "wall_time_s": t_end - t_start,
    }


BACKENDS = {
    "ollama": (run_ollama, "http://localhost:11434"),
    "openai": (run_openai_compatible, "http://localhost:8000"),
    "lmstudio": (run_lmstudio, "http://localhost:1234"),
}


def summarize(results):
    def avg(key):
        vals = [r[key] for r in results if r.get(key) is not None and not _is_nan(r[key])]
        return statistics.mean(vals) if vals else NAN

    reliable_count = sum(1 for r in results if r.get("prefill_reliable"))
    return {
        "prompt_tokens": avg("prompt_tokens"),
        "prefill_tok_s": avg("prefill_tok_s"),
        "prefill_reliable_ratio": f"{reliable_count}/{len(results)}",
        "gen_tokens": avg("gen_tokens"),
        "decode_tok_s": avg("decode_tok_s"),
        "ttft_s": avg("ttft_s"),
        "wall_time_s": avg("wall_time_s"),
    }


def main():
    parser = argparse.ArgumentParser(description="Universal local LLM inference benchmark tool")
    parser.add_argument("--backend", choices=list(BACKENDS.keys()), required=True,
                        help="ollama / openai (ds4-server, llama.cpp, vLLM, LM Studio v1) / lmstudio")
    parser.add_argument("--model", required=True, help="Model identifier used in the backend")
    parser.add_argument("--host", default=None, help="Service URL, uses backend default if omitted")
    parser.add_argument("--prompt-len", type=int, nargs="+", default=[128, 1024, 4096],
                        help="Prompt lengths (approx tokens) to test, default: 128 1024 4096")
    parser.add_argument("--gen-tokens", type=int, default=1024, help="Tokens to generate in each pass, default: 1024")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repetitions to average over, default: 1")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature, default: 0")
    parser.add_argument("--quiet", action="store_true", help="Disable inline spinner, output summary table only")
    parser.add_argument("--context-size", type=parse_context_size, nargs="+", default=None,
                        help="Dynamic context window size list (e.g. 16384, 16k). If omitted, uses default config.")
    args = parser.parse_args()

    run_fn, default_host = BACKENDS[args.backend]
    host = args.host or default_host
    show_progress = not args.quiet
    current_vram_ctx = None

    if args.backend == "lmstudio":
        # 校验下载状态，只做一次
        downloaded, _ = get_lmstudio_models_info(host)
        if downloaded is not None:
            if args.model not in downloaded:
                print(color(f"❌ Error: Model '{args.model}' is not downloaded/available in LM Studio.", C_RED))
                sys.exit(1)

    if args.backend == "ollama":
        downloaded = get_ollama_downloaded_models(host)
        if downloaded is not None:
            match_set = set()
            for name in downloaded:
                match_set.add(name)
                if ":" in name:
                    base, tag = name.rsplit(":", 1)
                    if tag == "latest":
                        match_set.add(base)
            if args.model not in match_set:
                print(color(f"❌ Error: Model '{args.model}' is not installed in Ollama.", C_RED))
                if downloaded:
                    print(color(f"Installed models: {', '.join(downloaded)}", C_YELLOW))
                sys.exit(1)

    context_sizes = args.context_size if args.context_size else [None]
    if args.backend == "openai":
        if args.context_size is not None:
            print(color("⚠️ Warning: Backend 'openai' (e.g. ds4-server, vLLM) does not support dynamic context resizing. The test will run under the server's pre-configured default context.", C_YELLOW))
        context_sizes = [None]
    
    meta = get_model_metadata(args.backend, host, args.model)
    max_ctx = meta.get("max_context") if meta else None
    
    processed_sizes = []
    seen = set()
    for ctx in context_sizes:
        actual_ctx = ctx
        if max_ctx and ctx is not None and ctx > max_ctx:
            def format_ctx_simple(num):
                if num >= 1024 and num % 1024 == 0:
                    return f"{int(num / 1024)}K"
                return f"{num:,}"
            req_desc = format_ctx_simple(ctx)
            max_desc = format_ctx_simple(max_ctx)
            print(color(f"⚠️ Warning: Requested context size {req_desc} exceeds model's maximum supported context {max_desc}. Truncating to {max_desc}.", C_YELLOW))
            actual_ctx = max_ctx
        
        if actual_ctx not in seen:
            seen.add(actual_ctx)
            processed_sizes.append(actual_ctx)
            
    context_sizes = processed_sizes
    total_groups = len(context_sizes) * len(args.prompt_len)
    print(color(f"🚀 Backend: {args.backend}    Model: {args.model}    Host: {host}", C_BOLD + C_CYAN))
    print(color(f"Running {total_groups} groups of matrix benchmarks (context sizes = {len(context_sizes)}, prompt lengths = {len(args.prompt_len)}), repeats={args.repeats}\n", C_CYAN))

    all_rows = []
    current_group = 0

    try:
        for ctx_size in context_sizes:
            if args.backend == "lmstudio":
                downloaded, loaded = get_lmstudio_models_info(host)
                if downloaded is not None:
                    need_reload = args.model not in loaded
                    if not need_reload and ctx_size is not None:
                        if meta and meta.get("active_context") != ctx_size:
                            need_reload = True
                    
                    if need_reload:
                        old_ctx = meta.get("active_context") if meta else None
                        def format_ctx_simple(num):
                            if not num:
                                return None
                            if num >= 1024 and num % 1024 == 0:
                                return f"{int(num / 1024)}K"
                            return f"{num:,}"
                        old_ctx_desc = format_ctx_simple(old_ctx) if old_ctx else "unknown"

                        for loaded_model_key, instance_ids in loaded.items():
                            for inst_id in instance_ids:
                                print(color(f"🔴 Info: Unloading active model instance '{inst_id}' (context: {old_ctx_desc}) to free VRAM...", C_BLUE))
                                try:
                                    unload_lmstudio_model(host, inst_id)
                                except Exception as e:
                                    print(color(f"❌ Error: Failed to unload model instance '{inst_id}': {e}", C_RED))
                                    sys.exit(1)
                                
                                start_unload = time.perf_counter()
                                while time.perf_counter() - start_unload < 15.0:
                                    _, current_loaded = get_lmstudio_models_info(host)
                                    if not current_loaded or loaded_model_key not in current_loaded:
                                        break
                                    time.sleep(0.5)
                        
                        ctx_desc = f"{ctx_size}" if ctx_size else "default"
                        if ctx_size and ctx_size >= 1024 and ctx_size % 1024 == 0:
                            ctx_desc = f"{int(ctx_size / 1024)}K"
                        print(color(f"🟡 Info: Loading target model '{args.model}' (context: {ctx_desc}) into LM Studio memory...", C_BLUE))
                        try:
                            load_lmstudio_model(host, args.model, context_length=ctx_size)
                        except Exception as e:
                            print(color(f"❌ Error: Failed to trigger loading model '{args.model}' in LM Studio: {e}", C_RED))
                            sys.exit(1)
                        
                        start_time = time.perf_counter()
                        loaded_ok = False
                        with Spinner("  ", f"Waiting for model '{args.model}' to load and initialize...", show=show_progress) as loading_spinner:
                            while time.perf_counter() - start_time < 120.0:
                                _, current_loaded = get_lmstudio_models_info(host)
                                if current_loaded and args.model in current_loaded:
                                    if ping_lmstudio_model(host, args.model):
                                        loaded_ok = True
                                        break
                                time.sleep(1.0)
                        if not loaded_ok:
                            print(color(f"❌ Error: Loading or warming up model '{args.model}' timed out after 120 seconds.", C_RED))
                            sys.exit(1)
                        print(color(f"🟢 Info: Model '{args.model}' (context: {ctx_desc}) is loaded, initialized and ready.", C_BLUE))
                    else:
                        ctx_desc = f"{ctx_size}" if ctx_size else "default"
                        if ctx_size and ctx_size >= 1024 and ctx_size % 1024 == 0:
                            ctx_desc = f"{int(ctx_size / 1024)}K"
                        print(color(f"🟢 Info: Model '{args.model}' (context: {ctx_desc}) with matching config is already loaded and ready.", C_BLUE))

            if args.backend == "ollama":
                need_unload = False
                need_load = False
                try:
                    ps_resp = requests.get(f"{host.rstrip('/')}/api/ps", timeout=2.0)
                    if ps_resp.status_code == 200:
                        active_models = [m.get("name") for m in ps_resp.json().get("models", [])]
                        is_active = args.model in active_models or any(m.rsplit(":", 1)[0] == args.model for m in active_models if ":" in m)
                        if not is_active:
                            need_load = True
                        elif current_vram_ctx != ctx_size:
                            need_unload = True
                            need_load = True
                except Exception:
                    pass
                
                def format_ctx_simple(num):
                    if not num:
                        return "default"
                    if num >= 1024 and num % 1024 == 0:
                        return f"{int(num / 1024)}K"
                    return f"{num:,}"

                ctx_desc = format_ctx_simple(ctx_size)
                if need_unload:
                    old_ctx_desc = format_ctx_simple(current_vram_ctx)
                    print(color(f"🔴 Info: Unloading active model instance '{args.model}' (context: {old_ctx_desc}) to free VRAM...", C_BLUE))
                    unload_ollama_model(host, args.model)
                    time.sleep(1.0)

                if need_load:
                    print(color(f"🟡 Info: Loading target model '{args.model}' (context: {ctx_desc}) into Ollama memory...", C_BLUE))
                    loaded_ok = False
                    with Spinner("  ", f"Waiting for model '{args.model}' to load and initialize...", show=show_progress) as loading_spinner:
                        loaded_ok = load_ollama_model(host, args.model, context_size=ctx_size)
                    if not loaded_ok:
                        print(color(f"❌ Error: Loading or warming up model '{args.model}' in Ollama failed.", C_RED))
                        sys.exit(1)
                    print(color(f"🟢 Info: Model '{args.model}' (context: {ctx_desc}) is loaded, initialized and ready.", C_BLUE))
                else:
                    print(color(f"🟢 Info: Model '{args.model}' (context: {ctx_desc}) with matching config is already loaded and ready.", C_BLUE))
                
                current_vram_ctx = ctx_size

            for target_len in args.prompt_len:
                current_group += 1
                ctx_label = f"{ctx_size}" if ctx_size else "Default"
                if ctx_size and ctx_size >= 1024 and ctx_size % 1024 == 0:
                    ctx_label = f"{int(ctx_size / 1024)}K"
                print(color(f"[{current_group}/{total_groups}] Testing context ≈ {ctx_label}, prompt length ≈ {target_len} tokens", C_BOLD + C_BLUE))
                
                results = []
                for i in range(args.repeats):
                    prompt = build_prompt(target_len)
                    prefix = f"  Repeat {i + 1}/{args.repeats}: "
                    try:
                        if args.backend == "ollama":
                            r = run_fn(host, args.model, prompt, args.gen_tokens, args.temperature, show_progress, prefix=prefix, context_size=ctx_size)
                        else:
                            r = run_fn(host, args.model, prompt, args.gen_tokens, args.temperature, show_progress, prefix=prefix)
                        results.append(r)
                        prefill_str = format_val(r["prefill_tok_s"], 1, " tok/s")
                        decode_str = format_val(r["decode_tok_s"], 1, " tok/s")
                        ttft_str = format_val(r["ttft_s"], 2, "s")
                        wall_time_str = format_val(r["wall_time_s"], 2, "s")
                        
                        success_tag = color("Success", C_GREEN)
                        print(f"{prefix}{success_tag} → Prefill {color(prefill_str, C_GREEN)}, Decode {color(decode_str, C_GREEN)}, TTFT {color(ttft_str, C_CYAN)}, Wall Time {wall_time_str}")
                    except Exception as e:
                        failed_tag = color("Failed", C_RED)
                        print(f"{prefix}{failed_tag}: {e}")
                
                if not results:
                    print(color(f"  All runs failed for prompt length {target_len}, skipping\n", C_RED))
                    continue
                
                s = summarize(results)
                all_rows.append((ctx_size, target_len, s))
                pf = format_val(s["prefill_tok_s"], 1, " tok/s")
                dc = format_val(s["decode_tok_s"], 1, " tok/s")
                print(f"  {color('Group Avg:', C_BOLD + C_GREEN)} Prefill {color(pf, C_GREEN)}, Decode {color(dc, C_GREEN)}\n")

        # ---------------- Summary Table (Fully Colorized & Safely Padded) ----------------
        headers = ["Ctx/Prompt", "Prefill Toks", "Prefill Spd", "TTFT(s)", "Decode Toks", "Decode Spd", "Wall Time(s)", "Reliability"]
        widths = [14, 12, 16, 10, 12, 14, 12, 12]
        total_width = sum(widths) + 3 * (len(widths) - 1)

        meta = get_model_metadata(args.backend, host, args.model)
        print(color("\nSummary Results", C_BOLD + C_GREEN))
        print(color("=" * total_width, C_BLUE))
        
        w_half = total_width // 2 - 1
        left_1 = f" 🛠️  Backend:  {args.backend}"
        right_1 = f"🏷️  Model:  {args.model}"
        print(f"{pad(left_1, w_half, 'left')} | {right_1}")
        
        if meta:
            size_str = "N/A"
            if meta.get("size_bytes"):
                size_bytes = meta["size_bytes"]
                if size_bytes >= 1024 ** 3:
                    size_str = f"{size_bytes / (1024 ** 3):.2f} GB"
                elif size_bytes >= 1024 ** 2:
                    size_str = f"{size_bytes / (1024 ** 2):.2f} MB"
                else:
                    size_str = f"{size_bytes / 1024:.2f} KB"
            # 格式化上下文大小展示
            max_ctx = meta.get("max_context")
            
            def format_ctx(num):
                if not num:
                    return None
                if num >= 1024 and num % 1024 == 0:
                    return f"{int(num / 1024)}K"
                return f"{num:,}"
                
            max_str = format_ctx(max_ctx)
            context_str = max_str if max_str else "N/A"

            left_2 = f" 🎛️  Params:   {meta.get('params') or 'N/A'}"
            right_2 = f"💾  Size:   {size_str}"
            print(f"{pad(left_2, w_half, 'left')} | {right_2}")
            
            fmt = meta.get("format")
            fmt_str = fmt.upper() if fmt else "N/A"
            left_3 = f" 📦  Format:   {fmt_str}"
            right_3 = f"🧩  Quant:  {meta.get('quant') or 'N/A'}"
            print(f"{pad(left_3, w_half, 'left')} | {right_3}")

            left_4 = f" 🌐  Context:  {context_str}"
            right_4 = f"🧠  Arch:   {meta.get('arch') or 'N/A'}"
            print(f"{pad(left_4, w_half, 'left')} | {right_4}")
            print(color("=" * total_width, C_BLUE))
        print(" | ".join(color(pad(h, w, "center"), C_BOLD + C_CYAN) for h, w in zip(headers, widths)))
        print(color("-" * total_width, C_BLUE))

        for ctx_size, target_len, s in all_rows:
            pf = format_val(s["prefill_tok_s"], 1, " tok/s")
            dc = format_val(s["decode_tok_s"], 1, " tok/s")
            ttft = format_val(s["ttft_s"], 3)
            pt = format_val(s["prompt_tokens"], 0)
            gp = format_val(s["gen_tokens"], 0)
            wt = format_val(s["wall_time_s"], 2)
            
            ctx_desc = "Default"
            if ctx_size:
                if ctx_size >= 1024 and ctx_size % 1024 == 0:
                    ctx_desc = f"{int(ctx_size / 1024)}K"
                else:
                    ctx_desc = f"{ctx_size}"
            
            ctx_prompt_str = f"{ctx_desc} / {target_len}"
            row = [ctx_prompt_str, pt, pf, ttft, gp, dc, wt, s["prefill_reliable_ratio"]]
            
            colored_row = [
                color(pad(row[0], widths[0], "right"), C_YELLOW),
                color(pad(row[1], widths[1], "right"), C_CYAN),
                color(pad(row[2], widths[2], "right"), C_GREEN),
                color(pad(row[3], widths[3], "right"), C_CYAN),
                color(pad(row[4], widths[4], "right"), C_CYAN),
                color(pad(row[5], widths[5], "right"), C_GREEN),
                color(pad(row[6], widths[6], "right"), C_CYAN),
                color(pad(row[7], widths[7], "right"), C_BLUE if "0/" not in row[7] else C_RED)
            ]
            print(" | ".join(colored_row))

        print(color("=" * total_width, C_BLUE))
        print()
    finally:
        if args.backend == "lmstudio":
            try:
                _, current_loaded = get_lmstudio_models_info(host)
                if current_loaded:
                    for loaded_model_key, instance_ids in current_loaded.items():
                        for inst_id in instance_ids:
                            print(color(f"🔴 Info: Cleaning up VRAM... Unloaded active model instance '{inst_id}' from LM Studio.", C_BLUE))
                            unload_lmstudio_model(host, inst_id)
            except Exception:
                pass
        elif args.backend == "ollama":
            try:
                ps_resp = requests.get(f"{host.rstrip('/')}/api/ps", timeout=2.0)
                if ps_resp.status_code == 200:
                    active_models = [m.get("name") for m in ps_resp.json().get("models", [])]
                    is_active = args.model in active_models or any(m.rsplit(":", 1)[0] == args.model for m in active_models if ":" in m)
                    if is_active:
                        print(color(f"🔴 Info: Cleaning up VRAM... Unloaded active model '{args.model}' from Ollama.", C_BLUE))
                        unload_ollama_model(host, args.model)
            except Exception:
                pass


if __name__ == "__main__":
    main()