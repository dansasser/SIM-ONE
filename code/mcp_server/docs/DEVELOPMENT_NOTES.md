Development Notes — MVLM Integration and Model Switching

MVLM Model (Local, HuggingFace)
- Format: HuggingFace directory with `config.json`, `model.safetensors`, tokenizer files.
- Current trained model path: `models/mvlm_gpt2/mvlm_final` (example from this repo).
- Backend: set `NEURAL_ENGINE_BACKEND=mvlm` to use the local MVLM engine.

Environment Variables
- `MVLM_MODEL_DIRS` (optional): Comma‑separated alias:path pairs.
  - Example: `MVLM_MODEL_DIRS="main:models/mvlm_gpt2/mvlm_final,enhanced:/opt/models/mvlm_v2"`
- `ACTIVE_MVLM_MODEL` (optional): Alias from `MVLM_MODEL_DIRS` to select.
  - Example: `ACTIVE_MVLM_MODEL="main"`
- `LOCAL_MODEL_PATH` (fallback): Used if aliases are not provided.

Requirements
- Ensure the following packages are installed (already added to requirements):
  - `transformers`, `torch`, `safetensors`
  - For llama‑cpp legacy local engine: `llama-cpp-python` (unchanged)

How Selection Works
1) If `ACTIVE_MVLM_MODEL` + `MVLM_MODEL_DIRS` is set, MVLMEngine loads that directory.
2) Otherwise, it falls back to `LOCAL_MODEL_PATH`.
3) If Transformers/Torch are not available or load fails, it returns deterministic stub outputs (no crash) and logs a warning.

Runtime Behavior
- The engine uses conservative generation defaults (max_new_tokens=256, temperature=0.7, top_p=0.9) for predictability.
- Async wrappers run generation in a thread pool to avoid blocking the loop.
- On errors, the engine logs the exception and returns a safe stub response.

Switching Models (Manual)
- Update `.env`:
  - Set `MVLM_MODEL_DIRS` with both model paths.
  - Set `ACTIVE_MVLM_MODEL` to the desired alias.
  - Keep `NEURAL_ENGINE_BACKEND=mvlm`.
- Restart the server to pick up the new model.

Future Enhancements
- Add an admin endpoint to switch `ACTIVE_MVLM_MODEL` at runtime (RBAC‑protected, audited).
- Introduce the SKC Swarm backend (`NEURAL_ENGINE_BACKEND=swarm`) per SKC_SWARM_ENGINEERING_PLAN.md.

