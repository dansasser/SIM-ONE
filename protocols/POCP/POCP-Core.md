
# POCP-Core: The Pre-Decoding Enforcement Layer

**Version:** 1.0  
**Status:** Release Candidate  
**Author:** Daniel T. Sasser II  
**Project:** Gorombo Agent Framework

---

## 1. Introduction & Target Audience

This document specifies the **Punctuated Output Control Protocol - Core Layer (POCP-Core)**. This is the definitive implementation of the POCP standard, designed for LLM providers and researchers who control the model's inference engine. This protocol operates at the **pre-decoding** stage, modifying token probabilities *before* a token is selected, ensuring natively compliant and fluent output.

## 2. The Technical Rationale: Why Tokenization Matters

The necessity for a pre-decoding protocol like POCP-Core is rooted in the mechanics of modern tokenizers. LLMs operate on **tokens**, not characters. Punctuation marks that may seem similar to a human user, such as an em dash (`—`) and two hyphens (`--`), are treated as distinct entities with their own token IDs.

The em dash is often a **single, atomic token** in many vocabularies. Due to its prevalence in vast training corpora, models develop a strong statistical preference for this token to create stylistically fluent sentences. This low-level preference is often strong enough to override high-level semantic instructions given in a prompt.

POCP-Core is designed to apply stylistic constraints at the correct level of abstraction: the token logits. It directly manipulates the probability distribution of the vocabulary, providing a deterministic method of control that prompt engineering alone cannot guarantee.

---

## 3. Pre-Decoding Process Flow

1.  **Rule Ingestion:** Parse and validate the session's JSON rule set.
2.  **Tokenizer Introspection:** Map named punctuation rules (e.g., `"EM_DASH"`) to their integer Token IDs.
3.  **Constraint Injection (Logit Processing):** For each step in the decoding process, before a token is sampled, apply rules by modifying the raw logit scores.
4.  **Validation & Fallback (Optional):** For complex substitution rules, a lookahead or a validation model can be used to check the likely coherence of a substitution. If it fails, the fallback logic is triggered (e.g., reverting the logit modification for that step).
5.  **Standard Decoding:** Proceed with token sampling using the modified logit scores.

## 4. Rule Implementation at the Logit Level

The JSON rule set is translated into logit operations as follows:

* `"disable": string[]`
    * **Implementation:** Apply **token masking** by setting the logit for the corresponding token ID to negative infinity (`-inf`).
* `"substitute": { [key: string]: string }`
    * **Implementation:**
        1.  The disabled token is masked (logit set to `-inf`).
        2.  The substitute token receives a **positive bias**, increasing its logit score.
* `"fallback": string`
    * **Implementation:** The fallback rule dictates the behavior of the Constraint Injection step.
        * `"FLAG"`: Not applicable at the Core level, as this is a post-processing action.
        * `"REVERT"`: If validation fails, the logit modifications for the disabled and substitute tokens are not applied for the current step.
        * `"OMIT"`: Requires more complex logic to mask both the original and substitute tokens and potentially boost tokens that would allow the sentence to "move on" without punctuation.
* `"limit": { [key:string]: number }`
    * **Implementation:** Requires a stateful process to count generated tokens. Once a limit is reached, the system dynamically applies **token masking** to that token ID for all subsequent generation steps.

## 5. Cross-Model Compatibility

The current specification assumes a static, known tokenizer vocabulary. Future versions of POCP-Core should define a standard for a **dynamic token mapping registry**. This would allow a system to query a model's tokenizer for its specific punctuation token IDs, ensuring the protocol can be applied across different LLMs without manual reconfiguration.

## 6. Conclusion

POCP-Core provides a robust framework for enforcing punctuation control at the token level, addressing the limitations of traditional prompt engineering. By manipulating token logits before decoding, it ensures that stylistic guidelines are adhered to more effectively. As the landscape of LLMs evolves, ongoing refinement and adaptation of this protocol will be essential for maintaining its relevance and efficacy.

---
