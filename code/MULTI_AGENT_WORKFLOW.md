# Multi-Agent "Writing Team" Workflow Design

This document outlines the design for a multi-agent cognitive workflow, modeled after a human writing team. This workflow will be implemented within the mCP Server to demonstrate advanced, collaborative agentic capabilities.

## 1. Overview

The goal is to create a system where a team of specialized agents collaborates to produce a high-quality written document from a simple topic. This moves beyond a simple "one-shot" generation and introduces concepts of drafting, critique, and revision.

## 2. Agent Roles and Responsibilities

The workflow will consist of five distinct agent roles, each implemented as a specialized protocol.

### 2.1. The Ideator

-   **Protocol Name:** `IdeatorProtocol`
-   **Input:** A `topic` string (e.g., "The future of AI governance").
-   **Responsibility:** To brainstorm and generate a structured list of initial ideas, key points, and potential angles related to the topic.
-   **Output:** A dictionary containing a list of strings, e.g., `{"ideas": ["Discuss the role of regulation.", "Compare different governance models.", "Explore the concept of AI constitutionalism."]}`.
-   **Underlying Engine:** Neural Engine (LLM).

### 2.2. The Drafter

-   **Protocol Name:** `DrafterProtocol`
-   **Input:** The output of the `IdeatorProtocol` (`{"ideas": [...]}`).
-   **Responsibility:** To take the structured ideas and write a coherent first draft of a document.
-   **Output:** A dictionary containing the draft text, e.g., `{"draft_text": "The future of AI governance is a complex topic..."}`.
-   **Underlying Engine:** Neural Engine (LLM).

### 2.3. The Revisor

-   **Protocol Name:** `RevisorProtocol`
-   **Input:** The output of the `DrafterProtocol` (`{"draft_text": "..."}`) and the `CriticProtocol` (`{"feedback": [...]}`).
-   **Responsibility:** To revise the draft based on the feedback provided by the Critic.
-   **Output:** A dictionary containing the revised draft, e.g., `{"revised_draft_text": "Governing the future of artificial intelligence is a complex topic..."}`.
-   **Underlying Engine:** Neural Engine (LLM).

### 2.4. The Critic

-   **Protocol Name:** `CriticProtocol`
-   **Input:** The output of the `DrafterProtocol` (`{"draft_text": "..."}`).
-   **Responsibility:** To analyze the draft and provide a structured list of constructive feedback. The feedback should be specific and actionable.
-   **Output:** A dictionary containing a list of feedback points, e.g., `{"feedback": ["The introduction is weak, it should be more engaging.", "The section on regulation lacks specific examples."]}`.
-   **Underlying Engine:** Neural Engine (LLM).

### 2.5. The Summarizer

-   **Protocol Name:** `SummarizerProtocol` (enhancement of the existing protocol)
-   **Input:** The final, revised draft from the `RevisorProtocol` (`{"revised_draft_text": "..."}`).
-   **Responsibility:** To create a concise, polished summary of the final document.
-   **Output:** A dictionary containing the summary, e.g., `{"summary": "This document explores AI governance..."}`.
-   **Underlying Engine:** Neural Engine (LLM).

## 3. Workflow Orchestration

The workflow will be executed sequentially, with a loop for the revision process. This requires an enhancement to the `OrchestrationEngine` to handle conditional logic and loops.

**Proposed Workflow:**

1.  **`IdeatorProtocol`** runs first, taking the initial topic.
2.  **`DrafterProtocol`** runs next, taking the ideas from the Ideator.
3.  **`CriticProtocol`** runs, taking the draft from the Drafter.
4.  **`RevisorProtocol`** runs, taking the original draft and the critic's feedback.
5.  **(Optional Loop):** For a more advanced implementation, the output of the Revisor could be passed back to the Critic for another round of feedback. This loop could run a fixed number of times (e.g., 2 revision cycles).
6.  **`SummarizerProtocol`** runs last, taking the final revised draft.

**Data Flow:**

A central `workflow_context` object will be maintained by the `OrchestrationEngine`. Each protocol will read its necessary inputs from this context and write its output back into the context under its own key.

**Example Context Object (mid-workflow):**
```json
{
  "topic": "The future of AI governance",
  "IdeatorProtocol": {
    "ideas": ["..."]
  },
  "DrafterProtocol": {
    "draft_text": "..."
  },
  "CriticProtocol": {
    "feedback": ["..."]
  }
}
```

This design provides a clear and powerful structure for implementing a collaborative, multi-agent system on the mCP server.
