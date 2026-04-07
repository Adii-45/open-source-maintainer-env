import os
import json
from openai import OpenAI
from github_env import OpenSourceMaintainerEnv, MaintainerAction

# ==========================================
# Strict Logging Functions (Required by Grader)
# ==========================================
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # The grader breaks if there are newlines in the action string
    action_clean = action.replace('\n', ' ').replace('\r', '') if action else "none"
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ==========================================
# 1. Setup API Client
# ==========================================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ==========================================
# 2. System Prompt (The Agent's Brain)
# ==========================================
SYSTEM_PROMPT = """
You are an autonomous Open Source Repository Maintainer agent.
You will be given an observation containing a ticket (issue or pull request).
You must decide the best action to take.

You must respond strictly in JSON format matching this schema:
{
    "decision": "add_labels" | "close_duplicate" | "approve_pr" | "request_changes",
    "labels_to_add": ["bug", "enhancement", "duplicate", "help-wanted", "invalid"], 
    "comment": "your explanation or review comment" 
}
"""

# ==========================================
# 3. The Evaluation Loop
# ==========================================
env = OpenSourceMaintainerEnv()

# Variables to track for the final [END] log
step_count = 0
rewards_list = []

# REQUIRED GRADER LOG: Start
log_start(task="maintainer_eval", env="opensource-maintainer-env", model=MODEL_NAME)

while True:
    step_count += 1
    obs = env.reset()
    
    # Construct the prompt
    user_prompt = f"Ticket Type: {obs.type}\nTitle: {obs.title}\nBody: {obs.body}\n"
    if obs.code_diff:
        user_prompt += f"Code Diff:\n{obs.code_diff}\n"

    action_str = ""
    current_reward = 0.0
    is_done = False
    error_msg = None

    try:
        # Call the LLM
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}, 
            temperature=0.1 
        )
        
        # Parse the output
        llm_output = response.choices[0].message.content
        action_dict = json.loads(llm_output)
        action = MaintainerAction(**action_dict)
        
        # Save a clean string version of the action for the grader log
        action_str = json.dumps(action_dict)
        
        # Feed action to environment
        new_obs, reward, done, info = env.step(action)
        current_reward = float(reward)
        is_done = bool(done)

    except Exception as e:
        error_msg = str(e).replace('\n', ' ')
        is_done = True # Force stop on error so we don't loop infinitely

    rewards_list.append(current_reward)

    # REQUIRED GRADER LOG: Step
    log_step(step=step_count, action=action_str, reward=current_reward, done=is_done, error=error_msg)

    # Move to the next task, or break if done/errored
    if not env.next_task() or error_msg:
        break

# Calculate final math for the grader
total_score = sum(rewards_list) / len(rewards_list) if rewards_list else 0.0
is_success = total_score > 0.5 

# REQUIRED GRADER LOG: End
log_end(success=is_success, steps=step_count, score=total_score, rewards=rewards_list)
