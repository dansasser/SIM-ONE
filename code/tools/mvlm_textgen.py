#!/usr/bin/env python3
"""
Quick CLI for testing a local HuggingFace-style MVLM (e.g., GPT-2 family) or
the project's MVLMEngine wrapper. Generates text from a prompt using a model
directory on disk — no network access required.

Examples
  # Using the project's MVLMEngine (simple, safe defaults)
  python code/tools/mvlm_textgen.py --model-dir models/mvlm_gpt2/mvlm_final \
    --prompt "Give three bullet points on SIM-ONE governance:"

  # Using raw Transformers for more control (deterministic greedy decode)
  python code/tools/mvlm_textgen.py --engine hf --greedy --max-new-tokens 96 \
    --model-dir models/mvlm_gpt2/mvlm_final \
    --prompt "Write a concise executive summary about SIM-ONE governance:"
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path


def _load_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    # Read from stdin if available
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("No prompt provided. Use --prompt, --prompt-file, or pipe stdin.")


def _in_venv() -> bool:
    # Heuristics: venv if sys.prefix != sys.base_prefix or VIRTUAL_ENV set
    return getattr(sys, 'base_prefix', sys.prefix) != sys.prefix or bool(os.environ.get('VIRTUAL_ENV'))


def _venv_python(venv_dir: Path) -> Path:
    if os.name == 'nt':
        return venv_dir / 'Scripts' / 'python.exe'
    return venv_dir / 'bin' / 'python'


def _ensure_dependencies() -> None:
    """Ensure transformers/torch/safetensors are importable.

    - If not in a virtualenv and deps missing: create a local venv under code/.mvlm_venv,
      install deps there, and re-exec the script using that interpreter.
    - If already in a venv and deps missing: install into current venv and continue.
    """
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        import safetensors  # noqa: F401
        return
    except Exception:
        pass

    bootstrapped = os.environ.get('MVLM_TEXTGEN_BOOTSTRAPPED') == '1'
    if not _in_venv() and not bootstrapped:
        # Create a dedicated local venv under the repo's code/ directory
        code_dir = Path(__file__).resolve().parents[1]
        venv_dir = code_dir / '.mvlm_venv'
        venv_dir.mkdir(exist_ok=True)
        print(f"[mvlm_textgen] Creating virtual environment at {venv_dir} ...", file=sys.stderr)
        subprocess.run([sys.executable, '-m', 'venv', str(venv_dir)], check=True)
        python_bin = _venv_python(venv_dir)
        print("[mvlm_textgen] Installing dependencies (transformers, torch, safetensors) ...", file=sys.stderr)
        subprocess.run([str(python_bin), '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
        subprocess.run([str(python_bin), '-m', 'pip', 'install', 'transformers', 'torch', 'safetensors'], check=True)
        # Re-exec this script under the new venv
        env = os.environ.copy()
        env['MVLM_TEXTGEN_BOOTSTRAPPED'] = '1'
        cmd = [str(python_bin), str(Path(__file__).resolve())] + sys.argv[1:]
        print(f"[mvlm_textgen] Re-launching under venv: {' '.join(cmd)}", file=sys.stderr)
        rc = subprocess.run(cmd, env=env).returncode
        sys.exit(rc)

    # Already in a venv (or bootstrapped): try installing into current env
    print("[mvlm_textgen] Installing missing dependencies into current environment ...", file=sys.stderr)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'transformers', 'torch', 'safetensors'], check=True)


def run_engine(args: argparse.Namespace) -> str:
    # Ensure project root (code/) is on sys.path so imports work regardless of CWD
    code_dir = Path(__file__).resolve().parents[1]
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    _ensure_dependencies()
    from mcp_server.neural_engine.mvlm_engine import MVLMEngine

    engine = MVLMEngine(model_path=args.model_dir)
    prompt = _load_prompt(args)
    return engine.generate_text(prompt)


def run_hf(args: argparse.Namespace) -> str:
    _ensure_dependencies()
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

    model_dir = args.model_dir
    prompt = _load_prompt(args)

    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    # Ensure pad token exists to avoid warnings; GPT-2 typically lacks a pad token
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)

    inputs = tok(prompt, return_tensors="pt")
    do_sample = not args.greedy
    # Optional reproducibility
    if args.seed is not None:
        try:
            import torch
            torch.manual_seed(int(args.seed))
        except Exception:
            pass

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        temperature=args.temperature if do_sample else None,
        top_p=args.top_p if do_sample else None,
        pad_token_id=tok.eos_token_id,
    )
    # Anti-repetition controls
    if args.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = args.repetition_penalty
    if args.no_repeat_ngram_size is not None:
        gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    if args.top_k is not None and do_sample:
        gen_kwargs["top_k"] = args.top_k

    # Pass full inputs (includes attention_mask) to avoid pad/eos warning
    out_ids = model.generate(
        **inputs,
        **gen_kwargs,
    )
    text = tok.decode(out_ids[0], skip_special_tokens=True)
    # If the decoded text includes the original prompt, return only the suffix
    if text.startswith(prompt):
        return text[len(prompt):].lstrip()
    return text


def main() -> None:
    p = argparse.ArgumentParser(description="Generate text from a local HF model (GPT-2 style) or MVLMEngine.")
    p.add_argument("--model-dir", required=True, help="Path to local HF model directory (config.json, model.safetensors, tokenizer files).")
    p.add_argument("--prompt", help="Inline prompt string.")
    p.add_argument("--prompt-file", help="Path to a file containing the prompt.")
    p.add_argument("--engine", choices=["engine", "hf"], default="engine", help="Use MVLMEngine wrapper ('engine') or raw Transformers ('hf').")
    p.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens (HF engine only).")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (HF engine only; ignored if --greedy).")
    p.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling (HF engine only; ignored if --greedy).")
    p.add_argument("--top-k", type=int, help="Top-k sampling (HF engine only; ignored if --greedy).")
    p.add_argument("--repetition-penalty", type=float, help="Penalty >1.0 discourages repetition (HF engine only).")
    p.add_argument("--no-repeat-ngram-size", type=int, help="Disallow repeating n-grams (e.g., 3) (HF engine only).")
    p.add_argument("--greedy", action="store_true", help="Disable sampling (deterministic HF generation).")
    p.add_argument("--seed", type=int, help="Random seed for reproducible sampling (HF engine only).")

    args = p.parse_args()

    if args.engine == "engine":
        out = run_engine(args)
    else:
        out = run_hf(args)

    print(out)


if __name__ == "__main__":
    main()
