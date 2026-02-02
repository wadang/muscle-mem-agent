# osworld 安装

cd OSWorld

UV_HTTP_TIMEOUT=300 uv sync

python -m ensurepip --upgrade

# Deploying Muscle_mem_agent in OSWorld

# Step 1: Set up Muscle_mem_agent

source OSWorld/.venv/bin/activate

进入 Muscle_mem_agent 目录

python -m pip install -e .


# Step 2: Copying Over Run Files

cp osworld_setup/*.py to osworld

# Step 3: Switch the AMI 

Switch image AMI for the AWS provider in `desktop_env/providers/aws/manager.py` is set to `"ami-0b505e9d0d99ba88c"`.

# Step 4: run


