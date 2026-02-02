from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from muscle_mem.agents.tools.registry import tool_action


TODO_STATUSES = ("pending", "in_progress", "completed")

TODO_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "content": {"type": "string"},
        "activeForm": {"type": "string"},
        "status": {"type": "string", "enum": list(TODO_STATUSES)},
    },
    "required": ["content", "activeForm", "status"],
    "additionalProperties": False,
}

TODO_WRITE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {"type": "array", "items": TODO_ITEM_SCHEMA, "maxItems": 20}
    },
    "required": ["items"],
    "additionalProperties": False,
}


@dataclass
class TodoRenderConfig:
    checkbox_pending: str = "[ ]"
    checkbox_completed: str = "[x]"
    empty_text: str = "[ ] No todos yet"
    include_status: bool = False


class TodoManager:
    def __init__(
        self,
        *,
        allow_alias_fields: bool = False,
        default_active_form: bool = True,
        render_config: Optional[TodoRenderConfig] = None,
        max_items: int = 20,
    ) -> None:
        self.items: List[Dict[str, str]] = []
        self.allow_alias_fields = allow_alias_fields
        self.default_active_form = default_active_form
        self.render_config = render_config or TodoRenderConfig()
        self.max_items = max_items

    def update(self, items: List[Dict[str, Any]]) -> str:
        if not isinstance(items, list):
            raise ValueError("Todo items must be a list")

        cleaned: List[Dict[str, str]] = []
        seen_ids: set[str] = set()
        in_progress = 0

        for index, raw in enumerate(items):
            if not isinstance(raw, dict):
                raise ValueError("Each todo must be an object")

            todo_id = str(raw.get("id") or index + 1)
            if todo_id in seen_ids:
                raise ValueError(f"Duplicate todo id: {todo_id}")
            seen_ids.add(todo_id)

            if self.allow_alias_fields:
                content = (
                    raw.get("content")
                    or raw.get("task")
                    or raw.get("title")
                    or raw.get("text")
                )
            else:
                content = raw.get("content")

            content = str(content or "").strip()
            if not content:
                raise ValueError("Todo content cannot be empty")

            status = str(raw.get("status") or TODO_STATUSES[0]).lower()
            if status not in TODO_STATUSES:
                raise ValueError(f"Status must be one of {', '.join(TODO_STATUSES)}")
            if status == "in_progress":
                in_progress += 1

            active_form = raw.get("activeForm") or raw.get("active_form")
            active_form = str(active_form or "").strip()
            if not active_form:
                if self.default_active_form:
                    active_form = content
                else:
                    raise ValueError("Todo activeForm cannot be empty")

            cleaned.append(
                {
                    "id": todo_id,
                    "content": content,
                    "status": status,
                    "active_form": active_form,
                }
            )

            if len(cleaned) > self.max_items:
                raise ValueError("Todo list is limited to 20 items")

        if in_progress > 1:
            raise ValueError("Only one task can be in_progress at a time")

        self.items = cleaned
        return self.render()

    def render(self) -> str:
        if not self.items:
            return self.render_config.empty_text

        lines: List[str] = []
        for todo in self.items:
            mark = (
                self.render_config.checkbox_completed
                if todo["status"] == "completed"
                else self.render_config.checkbox_pending
            )
            if self.render_config.include_status:
                lines.append(f"{mark} {todo['content']} ({todo['status']})")
            else:
                lines.append(f"{mark} {todo['content']}")
        return "\n".join(lines)

    def stats(self) -> Dict[str, int]:
        return {
            "total": len(self.items),
            "completed": sum(todo["status"] == "completed" for todo in self.items),
            "in_progress": sum(todo["status"] == "in_progress" for todo in self.items),
        }


class TodoToolProvider:
    def __init__(self, todo_manager: TodoManager, owner: Optional[Any] = None) -> None:
        self.todo_manager = todo_manager
        self.owner = owner
        self.last_board_view: Optional[str] = None
        self.last_summary: Optional[str] = None

    @tool_action
    def TodoWrite(self, items: List[Dict[str, Any]]):
        """
        使用此工具为当前任务会话创建并管理结构化的任务清单。它能帮助你追踪进度、拆解复杂任务，并确保执行的系统性和彻底性。
        同时，它也便于记录任务的整体进展情况。

        ## 何时使用此工具
        在以下场景中应主动使用此工具：

        1. 复杂的多步骤任务 - 当任务需要 3 个及以上独立步骤或操作时
        2. 非平凡的复杂任务 - 需要精心规划或多项操作的任务
        3. 指令明确要求使用任务清单 - 当指令中直接要求你使用任务清单时
        4. 指令包含多个子任务 - 当指令中包含多项待办事项（无论是编号列表还是逗号分隔）
        5. 收到新指令后 - 立即将任务需求拆解为待办项
        6. 开始执行某个任务前 - 在动手之前先将其标记为 in_progress。理想情况下，同一时刻只应有一个任务处于 in_progress 状态
        7. 完成某个任务后 - 将其标记为已完成，并添加在执行过程中发现的后续任务

        ## 何时不使用此工具

        在以下情况下跳过此工具：
        1. 只有单一、直接的任务
        2. 任务非常简单，追踪它没有任何组织价值
        3. 任务可在 3 个简单步骤内完成
        4. 任务纯粹是信息查询类

        注意：如果只有一个简单任务需要完成，直接执行即可，无需使用此工具。

        ## 使用任务清单的示例

        <example>
        User: 在应用设置中添加一个深色模式切换按钮。完成后记得运行测试和构建！
        Assistant: 我来帮你在应用设置中添加深色模式切换功能。先创建一个任务清单来追踪此实现。
        *创建包含以下项目的任务清单：*
        1. 在"设置"页面创建深色模式切换组件
        2. 添加深色模式状态管理（context/store）
        3. 实现支持深色主题的 CSS-in-JS 样式
        4. 更新现有组件以支持主题切换
        5. 运行测试和构建流程，处理可能出现的失败或错误
        *开始执行第一个任务*

        <reasoning>
        使用任务清单的原因：
        1. 添加深色模式是一个多步骤功能，涉及 UI、状态管理和样式修改
        2. 指令明确要求随后运行测试和构建
        3. 推断出测试和构建需要通过，因此将"确保测试和构建成功"作为最后一项任务
        </reasoning>
        </example>

        <example>
        User: 帮我把项目中所有的 getCwd 函数重命名为 getCurrentWorkingDirectory
        Assistant: 我先搜索一下代码库，找出所有 'getCwd' 的出现位置。
        *使用 grep 或搜索工具定位代码库中 getCwd 的所有实例*
        Assistant: 发现 'getCwd' 在 8 个不同文件中出现了 15 次。我来创建一个任务清单追踪这些修改。
        *创建任务清单，为每个需要更新的文件列出具体项目*

        <reasoning>
        使用任务清单的原因：
        1. 首先通过搜索了解任务范围
        2. 发现多个文件中存在多处引用后，判定这是一个包含多个步骤的复杂任务
        3. 任务清单有助于系统性地追踪和更新每个实例
        4. 这种方法可以防止遗漏任何引用，保持代码一致性
        </reasoning>
        </example>


        <example>
        User: 我需要为电商网站实现以下功能：用户注册、商品目录、购物车和结账流程。
        Assistant: 我来帮你实现这些功能。首先把所有功能添加到任务清单中。
        *根据项目架构，创建任务清单，将每个功能拆分为具体任务*
        Assistant: 从用户注册开始。这将涉及创建数据库模型、API 端点，以及与现有认证系统集成的前端表单。

        <reasoning>
        使用任务清单的原因：
        1. 指令中提供了多个复杂功能（以逗号分隔的列表）
        2. 任务清单有助于将这些大型功能组织成可管理的任务
        3. 这种方法便于追踪整个实现过程的进度
        </reasoning>
        </example>

        <example>
        User: 帮我优化一下 React 应用，渲染很慢，有性能问题。
        Assistant: 我来帮你优化 React 应用。首先检查一下代码库，识别潜在的性能瓶颈。
        *审查组件结构、渲染模式、状态管理和数据获取*
        Assistant: 分析代码库后，发现了几个性能问题。我来创建一个任务清单追踪优化工作。
        *创建任务清单，包含以下项目：1) 为 ProductList 中的昂贵计算实现 memoization，2) 为 Dashboard 中的长列表添加虚拟化，3) 优化 Gallery 组件中的图片加载，4) 修复 ShoppingCart 中的状态更新循环，5) 审查包体积并实现代码分割*
        从为 ProductList 组件中的昂贵计算实现 memoization 开始。

        <reasoning>
        使用任务清单的原因：
        1. 首先检查代码库以识别具体的性能问题
        2. 基于分析，发现了多个优化机会
        3. 性能优化是一个需要多个步骤的非平凡任务
        4. 任务清单有助于系统性地追踪不同组件的改进
        5. 这种系统化方法确保所有性能瓶颈都得到处理
        </reasoning>
        </example>

        ## 不使用任务清单的示例

        <example>
        User: 用 Python 怎么打印 'Hello World'？
        Assistant: 在 Python 中，可以用这行简单的代码打印 "Hello World"：

        python
        print("Hello World")

        执行后会在控制台输出 "Hello World"。

        <reasoning>
        不使用任务清单的原因：这是一个单一、简单的任务，一步即可完成。对于如此直接的请求，无需追踪多个任务或步骤。
        </reasoning>
        </example>

        <example>
        User: git status 命令是干什么的？
        Assistant: git status 命令显示工作目录和暂存区的当前状态。它展示哪些更改已暂存、哪些未暂存，以及哪些文件未被 Git 追踪。这是在提交前了解仓库当前状态的基础命令。

        <reasoning>
        不使用任务清单的原因：这是一个信息查询请求，没有实际的编码任务需要执行。指令只是要求解释，而不是执行多个步骤或任务。
        </reasoning>
        </example>

        <example>
        User: 给 calculateTotal 函数加个注释，说明它的作用。
        Assistant: 好的，我来给 calculateTotal 函数添加注释说明其功能。
        *使用 ${EDIT_TOOL_NAME} 工具为 calculateTotal 函数添加注释*

        <reasoning>
        不使用任务清单的原因：这是一个单一、直接的任务，只涉及代码中的一个位置。添加注释不需要追踪多个步骤或系统性组织。
        </reasoning>
        </example>

        <example>
        User: 帮我运行一下 npm install，看看什么情况。
        Assistant: 我来帮你运行 npm install 命令。

        *执行：npm install*

        命令执行成功。输出如下：
        [npm install 命令输出]

        所有依赖已根据 package.json 文件安装完成。

        <reasoning>
        不使用任务清单的原因：这是一个单命令执行，立即就有结果。没有多个步骤需要追踪或组织，对于这种直接的任务，任务清单毫无必要。
        </reasoning>
        </example>

        ## 任务状态与管理

        1. **任务状态**：使用以下状态追踪进度：
        - pending：尚未开始
        - in_progress：正在执行（同一时刻限制为一个任务）
        - completed：已成功完成

        **重要**：任务描述必须包含两种形式：
        - content：祈使句形式，描述需要做什么（如："运行测试"、"构建项目"）
        - activeForm：进行时形式，执行时显示（如："正在运行测试"、"正在构建项目"）

        2. **任务管理**：
        - 执行过程中实时更新任务状态
        - 完成后立即标记为已完成（不要批量更新）
        - 任何时刻有且仅有一个任务处于 in_progress 状态（不能多也不能少）
        - 完成当前任务后再开始新任务
        - 将不再相关的任务从清单中彻底移除

        3. **任务完成的判定标准**：
        - 只有在完全完成任务后才能标记为 completed
        - 如果遇到错误、阻塞或无法完成，保持任务为 in_progress
        - 遇到阻塞时，创建新任务描述需要解决的问题
        - 以下情况禁止标记为已完成：
            - 测试失败
            - 实现不完整
            - 遇到未解决的错误
            - 找不到必要的文件或依赖

        4. **任务拆分原则**：
        - 创建具体、可执行的项目
        - 将复杂任务拆分为更小、可管理的步骤
        - 使用清晰、描述性的任务名称
        - 始终提供两种形式：
            - content: "修复认证 bug"
            - activeForm: "正在修复认证 bug"

        拿不准时，就用这个工具。主动进行任务管理能确保所有要求都被成功完成。
        """

        if items is None:
            raise ValueError("TodoWrite requires an items list")
        try:
            board_view = self.todo_manager.update(items)
        except Exception as exc:
            return str(exc)
        stats = self.todo_manager.stats()
        if stats["total"] == 0:
            summary = "No todos have been created."
        else:
            summary = (
                f"Status updated: {stats['completed']} completed, "
                f"{stats['in_progress']} in progress."
            )
        self.last_board_view = board_view
        self.last_summary = summary
        if self.owner is not None:
            if hasattr(self.owner, "last_todo_board_view"):
                setattr(self.owner, "last_todo_board_view", board_view)
            if hasattr(self.owner, "last_todo_summary"):
                setattr(self.owner, "last_todo_summary", summary)
        return f"{board_view}\n\n{summary}".strip()


TodoToolProvider.TodoWrite.tool_input_schema = TODO_WRITE_INPUT_SCHEMA
