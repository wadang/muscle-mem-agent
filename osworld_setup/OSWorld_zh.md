# OSWorld 安装与 `Muscle_mem_agent` 部署

> 目标：安装 **OSWorld**（使用 **uv** + Python **3.12**），部署 **Muscle_mem_agent**，切换 AWS AMI，并在 **AWS** provider 上运行评测任务。

---

## 1) 安装 OSWorld（uv + Python 3.12）

进入 OSWorld 仓库目录：

```bash
cd OSWorld
```

同步依赖并创建虚拟环境（网络较慢时可增大超时）：

```bash
UV_HTTP_TIMEOUT=300 uv sync
```

如果你的 Python 安装不包含 `pip`，可先引导安装：

```bash
source OSWorld/.venv/bin/activate
python -m ensurepip --upgrade
```

> `uv sync` 通常会在 `OSWorld/.venv/` 下创建虚拟环境。

---

## 2) 将 `Muscle_mem_agent` 部署到 OSWorld

### 步骤 1：激活 OSWorld 虚拟环境

可在任意目录执行，但要确保路径指向 OSWorld 的 `.venv`：

```bash
source OSWorld/.venv/bin/activate
```

> 建议：安装和运行都在同一个环境中完成，避免依赖不一致。

### 步骤 2：安装 `Muscle_mem_agent`（editable 模式）

进入你的 agent 仓库目录并安装：

```bash
cd Muscle_mem_agent
python -m pip install -e .
```

---

## 3) 将运行脚本复制到 OSWorld

将 `osworld_setup/` 下的运行入口脚本复制到 OSWorld（按你的目录结构调整路径）：

```bash
cp osworld_setup/*.py ../OSWorld/
```

> 核心目的：让 OSWorld 可以直接执行 `run_muscle_mem_agent.py`。

---

## 4) 切换 AWS AMI

编辑文件：

- `desktop_env/providers/aws/manager.py`

将 AMI 设置为：

```python
"ami-0b505e9d0d99ba88c"
```

> 提示：可在文件中搜索 `ami-` 快速定位相关字段。

---

## 5) 运行（在 OSWorld 目录下）

进入 OSWorld：

```bash
cd OSWorld
```

执行：

```bash
uv run python run_muscle_mem_agent.py \
  --provider_name "aws" \
  --headless \
  --num_envs 4 \
  --max_steps 100 \
  --domain "all" \
  --test_all_meta_path evaluation_examples/test_nogdrive.json \
  --result_dir "results_nogdrive_last_v1" \
  --model_provider "anthropic" \
  --model_url "xxx" \
  --model_api_key "xxx" \
  --model "claude-4-5-opus" \
  --model_temperature 1.0 \
  --ground_provider "openai" \
  --ground_url "xxx" \
  --ground_model "qwen3-vl-plus" \
  --ground_api_key "xxx" \
  --grounding_width 1000 \
  --grounding_height 1000 \
  --image_ground_model "doubao-seed-1-8-251228" \
  --image_ground_provider "openai" \
  --image_ground_url "xxx" \
  --image_ground_api_key "xxx" \
  --tavily-api-key "xxx" \
  --jina-api-key "xxx"
```
