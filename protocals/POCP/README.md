
# Punctuated Output Control Protocol (POCP)

[![Protocol Status](https://img.shields.io/badge/Status-Release_Candidate_v1.1-green.svg)](./)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Author: Daniel T. Sasser II](https://img.shields.io/badge/Author-Daniel_T._Sasser_II-orange.svg)](https://dansasser.me)

**An open protocol for fine-grained, token-aware control over punctuation in Large Language Model (LLM) outputs.**

---

## The Problem: When Prompts Aren't Enough

Have you ever instructed an LLM, "Do not use em dashes," only for it to produce text filled with them? This common frustration occurs because a model's stylistic habits are deeply ingrained at the token level. High-level instructions in a prompt are often not enough to override the model's powerful, learned preference for certain punctuation.

### Why Tokenization Matters

Modern LLMs operate on **tokens**, not individual characters. The em dash (`—`), unlike two hyphens (`--`), is often represented as a single, atomic token in most modern tokenizers. This means:

* **It's statistically preferred:** The model has learned that this single token is a highly efficient and common way to structure fluent sentences.
* **Prompt instructions are often too weak:** A high-level semantic instruction like "don't use em dashes" may not be strong enough to suppress the model's low-level statistical preference for selecting that specific token.

This is why POCP goes beyond simple prompt engineering—it operates where style is actually enforced: in the token probabilities and their transformations during inference.

---

## The Solution: A Two-Layer Protocol

The **Punctuated Output Control Protocol (POCP)** solves this problem by providing a structured, two-layer enforcement mechanism. Instead of just hoping the model follows instructions, POCP gives you the power to *guarantee* it.

The protocol is split into two layers, allowing for adoption by different users with different levels of system access:

```

\+---------------------------------+
|      User's Application         |
|---------------------------------|
|  (Applies POCP-Compat Rules)    |
|               |                 |
|       [Post-Decoding Text]      |
|               ^                 |
\+---------------------------------+
|      Large Language Model       |
|---------------------------------|
| (Applies POCP-Core Rules)       |
|               |                 |
|     [Pre-Decoding Logits] \<-----+
|               |                 |
|       [Token Generation]        |
\+---------------------------------+

````

---

## Which Specification Is For Me?

To make the protocol as useful as possible, we've split the specification into two detailed documents. Choose the one that matches your use case.

### For Application Developers & API Users
If you are using an LLM via an API and want to control the output in your application, you need the **Compatibility Layer**. It provides a practical, robust method for applying punctuation rules to the text *after* it has been generated.

➡️ **Read the specification: [`POCP-Compat.md`](./POCP-Compat.md)**

### For LLM Builders & Researchers
If you are building your own LLM or have access to the model's inference engine, you can implement the powerful **Core Layer**. This is the most efficient method, allowing you to control the output by modifying the token probabilities *before* they are generated.

➡️ **Read the specification: [`POCP-Core.md`](./POCP-Core.md)**

---

## Quick Start Example

POCP works by applying a simple JSON rule set to the generation process. This allows you to define a clear "style guide" for the AI's output.

```json
{
  "disable": ["EM_DASH", "ELLIPSIS"],
  "substitute": {
    "EM_DASH": "COMMA"
  },
  "limit": {
    "EXCLAMATION_POINT": 1
  },
  "fallback": "FLAG"
}
````

-----

## Project Information

### Contributions

This is an open protocol, and contributions are welcome. Please feel free to open an issue to discuss ideas or submit a pull request with improvements.

### License

This project is licensed under the terms of the **MIT License**. See the [`LICENSE`](../../LICENSE) file for full terms.

### Author

**Daniel T. Sasser II** — [dansasser.me](https://dansasser.me)  
Part of the Gorombo Agent Ecosystem
