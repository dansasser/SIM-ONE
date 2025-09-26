#!/usr/bin/env python3
"""
SIM-ONE Dataset Builder (Instruction-Tuning JSONL)

Generates instruction/input/output JSONL examples from local SIM-ONE docs and code
so you can fine-tune a LoRA adapter on deterministic, repository-grounded tasks.

Outputs (by default): code/data/simone_instruct_train.jsonl

Categories (deterministic):
- Laws: five laws of governance → list as bullets / JSON
- Security features: extract from SECURITY.md and README → list/JSON
- Env vars: parse .env.example → list keys, render name→default JSON
- Endpoints: parse main.py → list paths + RBAC roles
- Project status: extract completed implementations (headings/lists)
- Protocol registry: parse protocol manifests names

Usage
  python code/tools/dataset/build_simone_instruct.py \
    --output code/data/simone_instruct_train.jsonl

Notes
- This builder relies on simple parsing; it avoids subjective summarization so that
  outputs are deterministic and high-precision for tuning.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
MCP = CODE / "mcp_server"


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""


def laws_examples() -> List[Dict]:
    # Ground truth from top-level docs (README/MANIFESTO often include)
    laws = [
        "Architectural Intelligence",
        "Cognitive Governance",
        "Truth Foundation",
        "Energy Stewardship",
        "Deterministic Reliability",
    ]
    bullets = "\n".join([f"- {x}" for x in laws])
    as_json = json.dumps({"laws": laws})
    return [
        {
            "instruction": "List the Five Laws of SIM-ONE Cognitive Governance as bullets.",
            "input": "",
            "output": bullets,
        },
        {
            "instruction": "Return the Five Laws of SIM-ONE Cognitive Governance as JSON with key 'laws'.",
            "input": "",
            "output": as_json,
        },
    ]


def env_examples() -> List[Dict]:
    env_file = MCP / ".env.example"
    text = read_text(env_file)
    keys: List[str] = []
    kv: Dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.strip().startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            keys.append(k.strip())
            kv[k.strip()] = v.strip()

    return [
        {
            "instruction": "List all environment variable keys used by the SIM-ONE MCP server.",
            "input": text,
            "output": "\n".join(sorted(keys)),
        },
        {
            "instruction": "Convert the SIM-ONE MCP server .env example into a JSON object mapping env var names to default values.",
            "input": text,
            "output": json.dumps(kv, indent=2),
        },
    ]


def security_examples() -> List[Dict]:
    sec = read_text(ROOT / "SECURITY.md")
    feats = [
        "API key authentication and RBAC",
        "CORS and security headers",
        "Rate limiting",
        "Advanced input validation",
        "Exception sanitization",
        "Audit logging",
    ]
    return [
        {
            "instruction": "From the SIM-ONE SECURITY.md, list the implemented security features as bullets.",
            "input": sec,
            "output": "\n".join([f"- {x}" for x in feats]),
        }
    ]


def endpoints_examples() -> List[Dict]:
    main_py = read_text(MCP / "main.py")
    # Regex for path decorators
    paths = re.findall(r"@app\.(get|post|delete|put|patch)\(\"([^\"]+)\"", main_py)
    # Roles via RoleChecker in dependencies
    role_lines = re.findall(r"Depends\(RoleChecker\(\[(.*?)\]\)\)\)\]", main_py)
    # Build a simple output: list paths
    path_list = sorted({f"/{p[1].lstrip('/')}" for p in paths})
    return [
        {
            "instruction": "List all FastAPI endpoint paths exposed by the SIM-ONE MCP server.",
            "input": main_py,
            "output": "\n".join(path_list),
        }
    ]


def status_examples() -> List[Dict]:
    status = read_text(MCP / "project_status.md")
    # Extract current status line
    m = re.search(r"Current Status:\s*\*\*(.*?)\*\*", status)
    current = m.group(1) if m else ""
    return [
        {
            "instruction": "From the project status report, state the current production readiness percentage.",
            "input": status,
            "output": current,
        }
    ]


def protocol_examples() -> List[Dict]:
    # Discover protocol manifests
    names: List[str] = []
    for manifest in (MCP / "protocols").rglob("protocol.json"):
        try:
            data = json.loads(read_text(manifest))
            n = data.get("name")
            if n:
                names.append(n)
        except Exception:
            continue
    names = sorted(set(names))
    return [
        {
            "instruction": "List the names of all cognitive protocols shipped with SIM-ONE.",
            "input": "",
            "output": "\n".join([f"- {n}" for n in names]),
        },
        {
            "instruction": "Return a JSON array of cognitive protocol names shipped with SIM-ONE.",
            "input": "",
            "output": json.dumps(names),
        },
    ]


def _match_roles(decorator: str) -> List[str]:
    m = re.search(r"RoleChecker\(\[(.*?)\]\)", decorator)
    if not m:
        return []
    inner = m.group(1)
    roles = [r.strip().strip("'\"") for r in inner.split(',') if r.strip()]
    return roles


def endpoint_role_examples() -> List[Dict]:
    main_py = read_text(MCP / "main.py")
    # Find decorator lines; capture HTTP method, path, and full decorator text
    exs: List[Dict] = []
    mapping: Dict[str, List[str]] = {}
    for m in re.finditer(r"@app\.(get|post|delete|put|patch)\(\s*\"([^\"]+)\"(.*?)\)\n", main_py, re.DOTALL):
        method, path, rest = m.group(1), m.group(2), m.group(3)
        roles = _match_roles(rest)
        # Normalize path
        norm = f"/{path.lstrip('/')}"
        mapping.setdefault(norm, roles)

    if mapping:
        exs.append({
            "instruction": "Return a JSON object mapping FastAPI endpoint paths to required roles (if any).",
            "input": main_py,
            "output": json.dumps(mapping, indent=2),
        })
        # Also produce a bullet list variant
        bullets = []
        for pth in sorted(mapping.keys()):
            roles = mapping[pth]
            role_str = ', '.join(roles) if roles else 'public'
            bullets.append(f"- {pth}: {role_str}")
        exs.append({
            "instruction": "List endpoint paths with their access roles (public if none).",
            "input": main_py,
            "output": "\n".join(bullets),
        })
    return exs


def config_rate_limit_examples() -> List[Dict]:
    cfg = read_text(MCP / "config.py")
    rate_limits = dict(re.findall(r"(RATE_LIMIT_[A-Z_]+):\s*str\s*=\s*os\.getenv\(\"(RATE_LIMIT_[A-Z_]+)\", \"([^\"]+)\"\)", cfg))
    if not rate_limits:
        return []
    return [
        {
            "instruction": "Extract default per-endpoint rate limits from the SIM-ONE config and return as JSON mapping.",
            "input": cfg,
            "output": json.dumps(rate_limits, indent=2),
        }
    ]


def governance_flag_examples() -> List[Dict]:
    cfg = read_text(MCP / "config.py")
    flags = {}
    for name in ["GOV_ENABLE", "GOV_MIN_QUALITY", "GOV_REQUIRE_COHERENCE"]:
        m = re.search(rf"{name}:\s*.*?=\s*os\.getenv\(\"{name}\",\s*\"?([^\"]*)\"?\)" , cfg)
        if m:
            flags[name] = m.group(1)
    if not flags:
        return []
    return [{
        "instruction": "From the SIM-ONE config, report the governance flags and their defaults as JSON.",
        "input": cfg,
        "output": json.dumps(flags, indent=2),
    }]


def env_groups_examples() -> List[Dict]:
    # Group env keys by commented sections in .env.example
    env_text = read_text(MCP / ".env.example")
    groups: Dict[str, List[str]] = {}
    current = "root"
    for line in env_text.splitlines():
        if line.strip().startswith('#'):
            header = line.strip('# ').strip()
            if header:
                current = header
                groups.setdefault(current, [])
        elif '=' in line and not line.strip().startswith('#'):
            k = line.split('=', 1)[0].strip()
            groups.setdefault(current, []).append(k)
    return [{
        "instruction": "Group SIM-ONE environment variables by their section headers and return as JSON mapping.",
        "input": env_text,
        "output": json.dumps(groups, indent=2),
    }]


def compose_examples() -> List[Dict]:
    exs: List[Dict] = []
    try:
        import yaml  # type: ignore
    except Exception:
        # PyYAML not installed; skip compose-derived examples gracefully
        return exs
    for comp in [CODE / "docker-compose.prod.yml", CODE / "docker-compose.override.yml"]:
        if not comp.exists():
            continue
        try:
            data = yaml.safe_load(read_text(comp)) or {}
            services = data.get('services', {})
            svc_names = sorted(services.keys())
            exs.append({
                "instruction": f"List service names defined in {comp.name} as bullets.",
                "input": read_text(comp),
                "output": "\n".join([f"- {s}" for s in svc_names]),
            })
            # Build mapping service->exposed ports (if any)
            port_map: Dict[str, List[str]] = {}
            for s, cfg in services.items():
                ports = cfg.get('ports') or []
                port_map[s] = ports
            exs.append({
                "instruction": f"Return a JSON mapping of service names to their published ports from {comp.name}.",
                "input": read_text(comp),
                "output": json.dumps(port_map, indent=2),
            })
        except Exception:
            continue
    return exs


def build_dataset() -> List[Dict]:
    ds: List[Dict] = []
    ds += laws_examples()
    ds += env_examples()
    ds += security_examples()
    ds += endpoints_examples()
    ds += status_examples()
    ds += protocol_examples()
    ds += endpoint_role_examples()
    ds += config_rate_limit_examples()
    ds += governance_flag_examples()
    ds += env_groups_examples()
    ds += compose_examples()
    return ds


def main():
    ap = argparse.ArgumentParser(description="Build SIM-ONE instruction-tuning dataset (JSONL)")
    ap.add_argument("--output", default=str(CODE / "data" / "simone_instruct_train.jsonl"))
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = build_dataset()
    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(ds)} examples to {out_path}")


if __name__ == "__main__":
    main()
