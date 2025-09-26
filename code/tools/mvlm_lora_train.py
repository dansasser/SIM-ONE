#!/usr/bin/env python3
"""
Minimal LoRA instruction-tuning starter for a local GPT-2–style model.

Usage (CPU-friendly example):
  python code/tools/mvlm_lora_train.py \
    --base-model models/mvlm_gpt2/mvlm_final \
    --train-file data/instruct_train.jsonl \
    --output-dir models/mvlm_gpt2_lora \
    --epochs 1 --batch-size 2 --lr 5e-5 --max-seq-len 512

Data format (JSONL): one object per line with fields:
  {"instruction": "...", "input": "...", "output": "..."}

Notes
- For GPU/accelerated training, install a CUDA build of torch and consider `accelerate` configs.
- Resulting adapter can be loaded by setting `MVLM_LORA_ADAPTER_PATH`.
"""

import argparse
import json
from pathlib import Path
from typing import Dict


def _load_dataset(path: Path):
    try:
        from datasets import Dataset
    except Exception as e:
        raise SystemExit(f"Install datasets: pip install datasets. Error: {e}")
    records = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj: Dict = json.loads(line)
            instr = obj.get('instruction', '').strip()
            inp = obj.get('input', '').strip()
            out = obj.get('output', '').strip()
            # Simple prompt template
            prompt = f"Instruction: {instr}\n"
            if inp:
                prompt += f"Input: {inp}\n"
            prompt += "Response:"
            # For causal LM fine-tune, we join prompt + response into a single sequence
            text = prompt + " " + out
            records.append({"text": text})
    return Dataset.from_list(records)


def main():
    p = argparse.ArgumentParser(description="LoRA instruction-tuning for GPT-2 style models")
    p.add_argument('--base-model', required=True, help='Path to local HF base model directory')
    p.add_argument('--train-file', required=True, help='Path to training JSONL (instruction/input/output per line)')
    p.add_argument('--val-file', help='Optional validation JSONL')
    p.add_argument('--output-dir', required=True, help='Directory to save the LoRA adapter')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--max-seq-len', type=int, default=512)
    p.add_argument('--lora-r', type=int, default=8)
    p.add_argument('--lora-alpha', type=int, default=16)
    p.add_argument('--lora-dropout', type=float, default=0.05)
    args = p.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, __version__ as hf_version
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        raise SystemExit(f"Install peft: pip install peft. Error: {e}")

    base = Path(args.base-model if hasattr(args, 'base-model') else args.base_model)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(base), local_files_only=True)

    ds_train = _load_dataset(Path(args.train_file))
    ds_val = _load_dataset(Path(args.val_file)) if args.val_file else None

    def tokenize_fn(batch):
        return tok(batch['text'], truncation=True, max_length=args.max_seq_len)

    t_train = ds_train.map(tokenize_fn, batched=True, remove_columns=['text'])
    t_val = ds_val.map(tokenize_fn, batched=True, remove_columns=['text']) if ds_val else None

    peft_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=['c_attn','q_attn','v_attn'] if hasattr(model, 'transformer') else None)
    model = get_peft_model(model, peft_cfg)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    # Build TrainingArguments kwargs compatible with older Transformers versions
    ta_kwargs = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=50,
        eval_steps=200 if t_val is not None else None,
        save_steps=200,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        report_to=[],
    )

    # evaluation_strategy is not available in some older versions; fall back gracefully
    try:
        args_tr = TrainingArguments(
            **{k: v for k, v in ta_kwargs.items() if v is not None},
            evaluation_strategy='steps' if t_val is not None else 'no',
        )
    except TypeError:
        # Older API: use evaluate_during_training=True/False
        if t_val is not None:
            ta_kwargs['evaluate_during_training'] = True
        args_tr = TrainingArguments(**{k: v for k, v in ta_kwargs.items() if v is not None})

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=t_train,
        eval_dataset=t_val,
        data_collator=collator,
    )
    trainer.train()
    # Save PEFT adapter
    model.save_pretrained(str(out_dir))
    print(f"Saved LoRA adapter to: {out_dir}")


if __name__ == '__main__':
    main()
