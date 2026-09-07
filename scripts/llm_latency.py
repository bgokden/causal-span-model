"""CPU latency of the multi-task LLM (GGUF) on the [causal] task, via llama-server.

Starts llama-server on CPU (-ngl 0) with N threads, sends [causal] prompts constrained
by a JSON grammar (llama.cpp grammars/json.gbnf), and reports median/p90 ms per sentence,
sentences per hour, and peak server RSS. This is the deploy-relevant number: a single
quantized model serving all three reasongraph tasks on commodity CPU.

Run:
    uv run python scripts/llm_latency.py \\
        --gguf outputs/gguf/qwen3-1.7b-multitask-Q4_K_M.gguf --threads 4
"""

import argparse
import json
import os
import statistics
import subprocess
import time
import urllib.request

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA = os.path.expanduser("~/llama.cpp")


def _post(url, body, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _wait_ready(port, proc, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("llama-server exited early")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError("llama-server did not become ready")


def _peak_rss_mb(pid):
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-sentences", type=int, default=40)
    ap.add_argument("--n-predict", type=int, default=96)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--ctx", type=int, default=1024)
    args = ap.parse_args(argv)

    df = pd.read_csv(os.path.join(REPO, "cnc_eval", "dev_grouped.csv"))
    sentences = [str(t) for t in df["text"]][:args.n_sentences]
    with open(os.path.join(LLAMA, "grammars", "json.gbnf")) as fh:
        grammar = fh.read()

    server = os.path.join(LLAMA, "build", "bin", "llama-server")
    cmd = [server, "-m", args.gguf, "-t", str(args.threads), "-c", str(args.ctx),
           "-ngl", "0", "--port", str(args.port), "--no-webui"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_ready(args.port, proc)
        url = f"http://127.0.0.1:{args.port}/completion"
        # warmup (excluded from timing)
        _post(url, {"prompt": f"[causal] {sentences[0]}", "n_predict": args.n_predict,
                    "temperature": 0, "grammar": grammar})
        lat_ms, out_tokens = [], []
        for s in sentences:
            t0 = time.perf_counter()
            r = _post(url, {"prompt": f"[causal] {s}", "n_predict": args.n_predict,
                            "temperature": 0, "grammar": grammar})
            lat_ms.append((time.perf_counter() - t0) * 1000)
            out_tokens.append(r.get("tokens_predicted", 0))
        peak = _peak_rss_mb(proc.pid)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    med = statistics.median(lat_ms)
    result = {
        "gguf": os.path.basename(args.gguf), "threads": args.threads,
        "n_sentences": len(sentences),
        "median_ms": round(med, 1),
        "p90_ms": round(statistics.quantiles(lat_ms, n=10)[8], 1),
        "mean_ms": round(statistics.mean(lat_ms), 1),
        "sentences_per_hour": round(3_600_000 / med),
        "median_out_tokens": statistics.median(out_tokens),
        "peak_rss_mb": round(peak) if peak else None,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
