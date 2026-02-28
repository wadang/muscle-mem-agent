# HIPPO Agent  
**HCIII Team, Lenovo**

HIPPO Agent is a **memory-augmented computer-use agent** designed for long-horizon GUI tasks. Our primary focus is **single-run reliability (pass@1)**: persisting and reusing task-critical facts and intermediate state across steps to reduce forgetting, backtracking, and repeated exploration.

> 🏆 **Latest Result**: HIPPO Agent reached a new SOTA on OSWorld-Verified with **74.5%**, becoming the **first non-multiple-rollout agent** to achieve and exceed **human-level performance (72.36%)**.

---

## Key Features

- **Memory-first execution**: task-level working memory persists relevant information across steps, reducing repeated exploration and fragile long-context dependence.  
- **Multi-turn tool calling**: we move from chat-style execution toward structured tool calls to improve controllability and observability.  
- **Determinism / process control**: lightweight guardrails and control points reduce randomness and improve single-run predictability.  
- **Task feasibility pre-check**: feasibility validation flags high-risk / infeasible tasks early to prevent error amplification.  
- **Curated toolset**: simplify and rename tools to avoid ambiguity; add web utilities and split click tools to improve robustness.

---

## System Overview

![HIPPO Agent Architecture](Architecture.png)

HIPPO Agent follows a structured agent architecture:

1. **Planner / Controller**  
   Dynamic planning and process control. Planning is triggered when beneficial and can be updated during execution, instead of enforcing a rigid upfront plan for every task.

2. **Grounding & Execution**  
   Converts abstract actions into concrete UI interactions (click / drag / type / scroll / key operations). It can be integrated with existing grounders and UI detection stacks.

3. **Memory System (core)**  
   Task-level working memory stores key facts, state, progress, and checkpoints, with selective retrieval for subsequent steps.

4. **Guardrails / Verification**  
   Lightweight rule-based checks plus feasibility validation to improve stability and reduce premature termination or drift.

---

## Memory System (Core)

### 1) Working Memory (task-level)
Across multi-step tasks, HIPPO Agent stores and retrieves essential information:

- **Typical content**: credentials/config choices, observed UI state, critical paths, completed subgoals, extracted values, TODO checkpoints, risk notes  
- **Benefits**: fewer repeated OCR/searches, less “why did we click that?” forgetting, reduced token pressure and attention dilution

> Practical note: tool naming matters. Avoid semantic collisions with in-app concepts (e.g., “Notes”) to reduce model confusion.

### 2) Selective Retrieval
To avoid drowning the planner in noise, retrieval should be **selective and structured**:
- inject only the minimal relevant memory per step rather than replaying long histories  
- store/retrieve by categories such as “facts / state / todos / risks” to reduce misreads

### 3) Memory-driven Context Engineering
- keep only necessary recent observations (e.g., a small number of recent screenshots/states)  
- persist long-range information in memory; during planning/verification/convergence, prefer memory reads over scanning long histories

---

## Guardrails & Verification

We use lightweight rule-based checks at critical control points, for example:

- **Finish verification**: if the agent claims completion, verify evidence of effective UI operations and results; otherwise continue or retry.  
- **Plan quality check**: if a generated plan is too short (e.g., `< 4` items), force replanning multiple times and select a better plan.  
- **Task feasibility validation**: assess feasibility prior to execution to mitigate error amplification from infeasible tasks.

---

## Tooling Strategy

We treat tools as a primary control surface and curate them for reliability:

- remove confusing overlap (e.g., keep only one of Python/Bash when both can solve the same task class)  
- add `web_search / web_fetch` to expand knowledge and reduce hallucinations  
- split click tools (e.g., “click UI element” vs “click image region”) so different backends/strategies can be used for different interaction types

---

## Acknowledgements

HIPPO Agent was developed on top of the Agent-S3 codebase. We sincerely thank the Agent-S3 authors and maintainers for open-sourcing their work.

We also thank the OSWorld team for creating, maintaining, and supporting the OSWorld benchmark and its evaluation infrastructure, which has been invaluable for our development and benchmarking efforts.
