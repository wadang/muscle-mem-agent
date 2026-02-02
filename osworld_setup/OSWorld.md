# Deploying motor_mem_agent in OSWorld

# Step 1: Set up motor_mem_agent

Follow the local `README.md` in this repo to set up motor_mem_agent.

# Step 2: Copying Over Run Files

If you haven't already, please follow the [OSWorld environment setup](https://github.com/xlang-ai/OSWorld/blob/main/README.md). We've provided the relevant OSWorld run files for evaluation in this `osworld_setup` folder. Please copy this over to your OSWorld folder. `run_local.py` is for if you want to run locally on VMWare and `run.py` and `lib_run_single.py` are for if you want to run on AWS. All run commands in order are provided in the `run.sh`. Copy over the files in `osworld_setup/bbon` as well. 

# Step 3: Switch the AMI 

Switch image AMI for the AWS provider in `desktop_env/providers/aws/manager.py` is set to `"ami-0b505e9d0d99ba88c"`.

# Step 4: Generating Facts


