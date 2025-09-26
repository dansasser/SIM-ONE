#!/usr/bin/env python3
"""
Generate synthetic, strict JSON/bullet instruction-tuning data for SIM-ONE agent tasks.

Tasks covered (schemas align with agent expectations):
- MTP extraction (entities + relations) → strict JSON
- ESL sentiment classification → strict JSON
- REP simple reasoning (modus ponens) → strict JSON
- Summarizer (exactly 5 bullets) → strict JSON

Usage (examples)
  python code/tools/dataset/generate_synthetic_agent_data.py \
    --out code/data/synthetic_agent_data.jsonl \
    --mtp 1000 --esl 500 --rep 400 --sum 300 --seed 42

Result: JSONL with {instruction, input, output} per line.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def set_seed(seed: int):
    random.seed(seed)


# ------------------------ MTP (entities/relations) ------------------------ #

PEOPLE = [
    "John", "Alice", "Bob", "Carol", "Eve", "Mallory", "Trent", "Jules",
]
ORGS = [
    "Microsoft", "OpenAI", "Acme Corp", "Globex", "Initech", "Umbrella", "Wayne Enterprises",
]
PLACES = [
    "Seattle", "San Francisco", "New York", "Berlin", "Paris", "London", "Austin", "Tokyo",
]


def mtp_templates() -> List[Tuple[str, Dict[str, str]]]:
    # Return list of (sentence, relation_type) pairs; relation_type drives JSON
    examples: List[Tuple[str, Dict[str, str]]] = []
    for person in PEOPLE:
        org = random.choice(ORGS)
        place = random.choice(PLACES)
        place2 = random.choice([p for p in PLACES if p != place])
        examples.append((f"{person} works at {org} in {place}.", {"t1": "works_at", "t2": "located_in"}))
        examples.append((f"{person} moved from {place} to {place2}.", {"t1": "moved_from", "t2": "moved_to"}))
        examples.append((f"{person} lives in {place}.", {"t1": "located_in"}))
        examples.append((f"{person} studies at {org} in {place}.", {"t1": "studies_at", "t2": "located_in"}))
        examples.append((f"{person} founded {org}.", {"t1": "founded"}))
    return examples


def mtp_entity_type(token: str) -> str:
    if token in PEOPLE:
        return "person"
    if token in ORGS:
        return "organization"
    if token in PLACES:
        return "place"
    if token.isdigit():
        return "date"
    return "other"


def gen_mtp(n: int) -> List[Dict]:
    data: List[Dict] = []
    tmpl = mtp_templates()
    random.shuffle(tmpl)
    i = 0
    while i < n and tmpl:
        sentence, rel_map = tmpl.pop()
        # crude token detection based on membership; robust enough for synthetic
        ents = []
        for token in PEOPLE + ORGS + PLACES:
            if token in sentence:
                ents.append({"text": token, "type": mtp_entity_type(token)})
        rels = []
        # Build relations based on recognized tokens in sentence
        if "works_at" in rel_map.values() and any(p for p in PEOPLE if p in sentence) and any(o for o in ORGS if o in sentence):
            p = next(p for p in PEOPLE if p in sentence)
            o = next(o for o in ORGS if o in sentence)
            rels.append({"source": p, "type": "works_at", "target": o})
        if "studies_at" in rel_map.values() and any(p for p in PEOPLE if p in sentence) and any(o for o in ORGS if o in sentence):
            p = next(p for p in PEOPLE if p in sentence)
            o = next(o for o in ORGS if o in sentence)
            rels.append({"source": p, "type": "studies_at", "target": o})
        if "founded" in rel_map.values() and any(p for p in PEOPLE if p in sentence) and any(o for o in ORGS if o in sentence):
            p = next(p for p in PEOPLE if p in sentence)
            o = next(o for o in ORGS if o in sentence)
            rels.append({"source": p, "type": "founded", "target": o})
        if "located_in" in rel_map.values() and any(p for p in PEOPLE if p in sentence) and any(pl for pl in PLACES if pl in sentence):
            p = next(p for p in PEOPLE if p in sentence)
            pl = next(pl for pl in PLACES if pl in sentence)
            rels.append({"source": p, "type": "located_in", "target": pl})
        if "moved_from" in rel_map.values() and any(p for p in PEOPLE if p in sentence):
            p = next(p for p in PEOPLE if p in sentence)
            # choose first place occurrence as from, second as to
            all_places = [pl for pl in PLACES if pl in sentence]
            if len(all_places) >= 2:
                rels.append({"source": p, "type": "moved_from", "target": all_places[0]})
                rels.append({"source": p, "type": "moved_to", "target": all_places[1]})

        instruction = (
            "You are a JSON generator. Return ONLY JSON, no prose.\n"
            "Extract entities and relations using this schema:\n"
            "{ \"entities\": [{ \"text\": string, \"type\": \"person|organization|place|date|other\" }],\n"
            "  \"relations\": [{ \"source\": string, \"type\": string, \"target\": string }] }\n"
            "Text: {user_input}\n"
            "Response JSON:"
        )
        data.append({
            "instruction": instruction,
            "input": sentence,
            "output": json.dumps({"entities": ents, "relations": rels}, ensure_ascii=False),
        })
        i += 1
    return data


# ------------------------ ESL (sentiment) ------------------------ #

POS = ["thrilled", "excellent", "proud", "happy", "joyful", "fantastic", "great"]
NEG = ["angry", "frustrating", "unacceptable", "terrible", "sad", "awful", "horrible"]
NEU = ["fine", "okay", "acceptable", "ordinary", "standard", "common"]

EMO_MAP = {
    "positive": ["joy", "pride"],
    "negative": ["anger", "frustration"],
    "neutral": ["calm"],
}


def gen_esl(n: int) -> List[Dict]:
    out: List[Dict] = []
    while len(out) < n:
        choice = random.choice(["positive", "negative", "neutral"])
        if choice == "positive":
            word = random.choice(POS)
            intensity = round(random.uniform(0.75, 0.95), 2)
        elif choice == "negative":
            word = random.choice(NEG)
            intensity = round(random.uniform(0.75, 0.95), 2)
        else:
            word = random.choice(NEU)
            intensity = round(random.uniform(0.45, 0.6), 2)
        text = f"This is {word}."
        det = {"emotion": random.choice(EMO_MAP[choice]), "intensity": intensity}
        instruction = (
            "You are a JSON generator. Return ONLY JSON, no prose.\n"
            "Classify valence ∈ {positive, neutral, negative} and intensity ∈ [0..1].\n"
            "Schema: { \"valence\": string, \"intensity\": number, \"detected_emotions\": [ { \"emotion\": string, \"intensity\": number } ] }\n"
            "Text: {user_input}\n"
            "Response JSON:"
        )
        out.append({
            "instruction": instruction,
            "input": text,
            "output": json.dumps({"valence": choice, "intensity": intensity, "detected_emotions": [det]}, ensure_ascii=False),
        })
    return out


# ------------------------ REP (simple reasoning) ------------------------ #

X_CATS = [
    ("men", "mortal"),
    ("dogs", "mammals"),
    ("cats", "mammals"),
    ("birds", "animals"),
    ("sparrows", "birds"),
]
NAMES = PEOPLE


def singular(plural: str) -> str:
    if plural.endswith("s"):
        return plural[:-1]
    return plural


def gen_rep(n: int) -> List[Dict]:
    out: List[Dict] = []
    used = 0
    while used < n:
        (x, y) = random.choice(X_CATS)
        a = random.choice(NAMES)
        s = singular(x)
        facts = [f"All {x} are {y}", f"{a} is a {s}"]
        rules = [[facts, f"{a} is {y}"]]
        instruction = (
            "You are a JSON generator. Return ONLY JSON, no prose.\n"
            "Given facts and rules, derive conclusions. Keep it short.\n"
            "Schema: { \"conclusions\": [string], \"reasoning\": [string], \"valid\": boolean }\n"
            "Facts: {facts}\nRules: {rules}\nResponse JSON:"
        )
        conclusion = f"{a} is {y}"
        reasoning = [
            f"All {x} are {y}",
            f"{a} is a {s}",
            f"Therefore, {conclusion}",
        ]
        out.append({
            "instruction": instruction,
            "input": json.dumps({"facts": facts, "rules": rules}, ensure_ascii=False),
            "output": json.dumps({"conclusions": [conclusion], "reasoning": reasoning, "valid": True}, ensure_ascii=False),
        })
        used += 1
    return out


# ------------------------ Summarizer (5 bullets) ------------------------ #

TOPICS = {
    "SIM-ONE governance": [
        "Architectural coordination yields intelligence.",
        "Protocols govern every process.",
        "Reasoning stays grounded in truth.",
        "Compute is used efficiently.",
        "Outputs remain predictable and consistent.",
    ],
    "Security posture": [
        "RBAC with API keys.",
        "Strict CORS and headers.",
        "Advanced validation and sanitized errors.",
        "Rate limiting and audit logging.",
        "Production metrics and health checks.",
    ],
    "Core endpoints": [
        "Public health endpoints (/health).",
        "Governed /execute with RBAC.",
        "Discovery: /protocols and /templates.",
        "Sessions with isolation.",
        "Metrics for monitoring.",
    ],
}


def gen_sum(n: int) -> List[Dict]:
    out: List[Dict] = []
    topics = list(TOPICS.keys())
    i = 0
    while i < n:
        topic = random.choice(topics)
        bullets = TOPICS[topic][:5]
        instruction = (
            "You are a JSON generator. Return ONLY JSON, no prose.\n"
            "Produce exactly 5 bullet points summarizing the topic.\n"
            "Schema: { \"bullets\": [string, string, string, string, string] }\n"
            "Topic: {topic}\n"
            "Response JSON:"
        )
        out.append({
            "instruction": instruction,
            "input": topic,
            "output": json.dumps({"bullets": bullets}, ensure_ascii=False),
        })
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic strict JSON instruction-tuning data for SIM-ONE")
    ap.add_argument("--out", default="code/data/synthetic_agent_data.jsonl")
    ap.add_argument("--mtp", type=int, default=1000, help="Number of MTP examples")
    ap.add_argument("--esl", type=int, default=500, help="Number of ESL examples")
    ap.add_argument("--rep", type=int, default=400, help="Number of REP examples")
    ap.add_argument("--sum", type=int, default=300, help="Number of Summarizer examples")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    ds: List[Dict] = []
    ds.extend(gen_mtp(args.mtp))
    ds.extend(gen_esl(args.esl))
    ds.extend(gen_rep(args.rep))
    ds.extend(gen_sum(args.sum))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(ds)} examples to {out_path}")


if __name__ == "__main__":
    main()

