# HIPPO Agent  
**HCIII Team, Lenovo**

HIPPO Agent 是一个面向长链路 GUI 任务的 **Memory-augmented Computer-Use Agent**。我们聚焦 **single-run / pass@1（单次运行成功率）** 与执行过程的稳定性：将关键事实与中间状态跨 step 保存与复用，减少遗忘、回退与重复探索。

> 🏆 **最新成绩**：HIPPO Agent 以 **74.5%** 的成绩成为 OSWorld-Verified 新 SOTA，并成为**首个非 Multiple rollout** 达到并超过**人类水平（72.36%）**的 Agent。

---

## 亮点（Highlights）

- **Memory-first 设计**：引入“短期记忆（task-level working memory）”，在跨 step 持续保存关键信息与中间结论，减少重复探索与上下文遗忘。  
- **Tool-calling first**：整体执行范式从多轮对话式迁移到多轮工具调用式，降低歧义、提升可控性与可观测性。  
- **Determinism / 过程控制**：通过规则化校验与关键节点门控降低随机性，提升单次运行的可预测性与一致性。  
- **Task feasibility pre-check**：加入任务可行性验证（feasibility validation），尽早识别高风险/不可行任务，避免错误扩散。  
- **Curated toolset**：精简与重命名工具，减少语义冲突；补充 Web 能力并细分点击工具，提高鲁棒性。

---

## 系统概览（System Overview）

![HIPPO Agent 架构图](Architecture.png)

HIPPO Agent 采用结构化的 agent 框架：

1. **Planner / Controller**  
   动态计划与过程控制：计划在“必要时触发”，并可在执行中更新，而非强制每次起手制定刚性 plan。

2. **Grounding & Execution**  
   将抽象动作落地为可执行 UI 操作（click / drag / type / scroll / key 等），可对接现有的 grounder 与 UI 检测体系。

3. **Memory System（核心）**  
   任务内短期记忆：以结构化方式记录关键事实、状态、进度与检查点，并在后续 step 按需检索。

4. **Guardrails / Verification**  
   轻量规则校验 + 可行性验证，在关键控制点提升稳定性并降低误终止与跑偏风险。

---

## 记忆系统（Memory System）

### 1) Working Memory（短期记忆 / Task-level）
在同一个任务的多 step 过程中，将关键知识与中间结果写入短期记忆，后续步骤随时读取。

- **典型内容**：账号/配置、已观察到的 UI 状态、关键路径、已完成子目标、从页面抽取的数据、待办检查点、风险点等  
- **收益**：减少重复 OCR/搜索，避免“走到一半忘了为什么点这里”，降低 token 消耗与注意力稀释

> 实践经验：记忆工具命名会影响模型理解与稳定性。建议避免与应用内概念（例如 “Notes”）发生语义冲突。

### 2) Memory Retrieval（选择性召回）
为避免噪声淹没推理，我们 **按需、结构化** 召回：
- 每步只注入最相关的记忆片段，而不是回放长历史  
- 可以将记忆按 “facts / state / todos / risks” 分类存储与读取，降低误读概率

### 3) Memory-driven Context Engineering（记忆驱动的上下文管理）
- 仅保留必要的近期观测（例如最近少量截图/状态）  
- 长程信息写入记忆，计划/验证/收敛阶段优先读取记忆而不是翻历史记录

---

## 护栏与验证（Guardrails & Verification）

我们采用轻量、规则化的校验策略来提升单次运行稳定性，例如：

- **结束动作校验**：当 agent 声称任务完成时，校验是否存在有效 UI 操作与结果证据；否则要求继续执行或重试。  
- **计划质量校验**：当触发计划时，如计划条目过少（例如 `< 4`），可强制重新规划多次并择优，以减少“随口一 plan 就跑偏”。  
- **任务可行性验证**：执行前判定任务是否可行，降低不可行任务造成的错误扩散。

---

## 工具策略（Tooling Strategy）

我们将工具视为核心控制面（control surface），并对工具体系做了面向稳定性的取舍：

- 去掉功能重叠且易混淆的工具（例如 Python 与 Bash 在同一类任务上二选一）  
- 补充 `web_search / web_fetch`，用于外部知识获取与降低幻觉  
- 细分点击能力（例如 “点击 UI 元素” 与 “点击图像区域”），便于针对不同交互类型使用不同后端策略


## 致谢（Acknowledgements）

HIPPO Agent 基于 Agent-S3 的代码基础进行开发。我们衷心感谢 Agent-S3 的作者与维护者开源其工作。

同时感谢 OSWorld 团队创建并维护 OSWorld 基准与评测基础设施，为我们的开发与评测提供了重要支持。
