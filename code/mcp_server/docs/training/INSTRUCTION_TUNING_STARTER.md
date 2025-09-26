SIM-ONE MVLM — Instruction Tuning Starter (LoRA)

Overview
- Goal: Make the base GPT‑2–style MVLM follow instructions and output structured, non-repetitive text suitable for the agentic system.
- Method: LoRA (parameter-efficient fine‑tune) on a small domain dataset; adapters are small, swappable, and loadable via MVLM_LORA_ADAPTER_PATH.

Data Format (JSONL)
- One JSON object per line with keys: instruction, input, output.
- Example:
  {"instruction": "Summarize SIM‑ONE governance in 5 bullets.", "input": "", "output": "- Law 1 ...\n- Law 2 ...\n..."}
  {"instruction": "Extract entities and relations.", "input": "John works at Microsoft in Seattle.", "output": "{"entities": [...], "relations": [...]}"}

Training Script
- Location: code/tools/mvlm_lora_train.py
- Example (CPU-friendly):
  python code/tools/mvlm_lora_train.py \
    --base-model models/mvlm_gpt2/mvlm_final \
    --train-file data/instruct_train.jsonl \
    --output-dir models/mvlm_gpt2_lora \
    --epochs 1 --batch-size 2 --lr 5e-5 --max-seq-len 512

After Training
- Set in .env: MVLM_LORA_ADAPTER_PATH=models/mvlm_gpt2_lora
- Restart the server (or use /admin/models/activate if switching base alias). MVLMEngine will load the adapter.

Decoding Controls
- Control sampling globally via env in .env: temperature/top‑p/top‑k/repetition_penalty/no_repeat_ngram_size/greedy/max_new_tokens/seed.
- Tune defaults per protocol via governance policy if needed.

Quality Tips
- Start with 2–20k examples targeted at your tasks (governance, REP/ESL/MTP, summarization, editing).
- Prefer concise outputs, consistent structure, and JSON where applicable.
- Validate with governance metrics; promote adapter only if quality/coherence improves and remains stable.

