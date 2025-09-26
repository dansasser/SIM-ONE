from typing import List, Dict, Tuple


def build_json_prompt(schema_hint: str, task_text: str) -> Tuple[List[Dict[str, str]], str]:
    """Return (openai_messages, mvlm_string) for a strict JSON task.

    - OpenAI: system enforces JSON-only; user carries schema + task text
    - MVLM: flattened string prompt with explicit JSON-only instruction
    """
    system = (
        "You are a JSON generator. Return ONLY valid JSON, no prose. "
        "Do not include explanations or extra keys."
    )
    user = f"Schema: {schema_hint}\nTask: {task_text}\nResponse JSON:"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    mvlm = f"{system}\n{user}"
    return messages, mvlm


def build_bullets_prompt(count: int, task_text: str) -> Tuple[List[Dict[str, str]], str]:
    schema = '{ "bullets": [' + ', '.join(['string'] * count) + '] }'
    return build_json_prompt(schema, f"Produce exactly {count} bullet points summarizing: {task_text}")

