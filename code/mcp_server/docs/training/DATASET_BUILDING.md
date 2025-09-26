SIM-ONE Dataset Building (Deterministic Instruction Pairs)

Overview
- Build a high-precision, repository-grounded instruction dataset (JSONL) for LoRA tuning.
- Avoids subjective summaries by generating deterministic extraction/structuring tasks from local docs/code.

Script
- Location: code/tools/dataset/build_simone_instruct.py
- Output: code/data/simone_instruct_train.jsonl

Run
  python code/tools/dataset/build_simone_instruct.py --output code/data/simone_instruct_train.jsonl

Included Task Types
- Laws (Five Laws) → bullets and JSON
- Env vars (from .env.example) → list and JSON mapping
- Security features (from SECURITY.md) → bullets
- Endpoints (from main.py) → list of paths
- Project status → current readiness string
- Protocol names (from protocol manifests) → bullets and JSON

Next Steps
- Append your own hand-crafted instruction/input/output lines for tasks like REP/ESL/MTP outputs, structured summaries, editors, critics.
- Then train a LoRA adapter with code/tools/mvlm_lora_train.py and load it using MVLM_LORA_ADAPTER_PATH.

