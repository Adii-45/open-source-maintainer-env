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
# 2. System Prompt
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

while True:
    obs = env.reset()
    
    # 🎯 FIX: Get the actual task ID (e.g., "TASK_1_EASY")
    task_id = env.tasks[env.current_task_idx]["id"]
    
    # 🎯 FIX: Start a brand NEW evaluation log for this specific task
    log_start(task=task_id, env="opensource-maintainer-env", model=MODEL_NAME)
    
    user_prompt = f"Ticket Type: {obs.type}\nTitle: {obs.title}\nBody: {obs.body}\n"
    if obs.code_diff:
        user_prompt += f"Code Diff:\n{obs.code_diff}\n"

    action_str = ""
    current_reward = 0.01  
    is_done = False
    error_msg = None

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}, 
            temperature=0.1 
        )
        
        llm_output = response.choices[0].message.content
        action_dict = json.loads(llm_output)
        action = MaintainerAction(**action_dict)
        action_str = json.dumps(action_dict)
        
        new_obs, reward, done, info = env.step(action)
        current_reward = float(reward)
        is_done = bool(done)

    except Exception as e:
        error_msg = str(e).replace('\n', ' ')
        is_done = True 

    # Log the step (always step=1 since each task is a single action for a maintainer)
    log_step(step=1, action=action_str, reward=current_reward, done=is_done, error=error_msg)

    # 🎯 FIX: End the evaluation log for THIS specific task
    is_success = current_reward > 0.5 
    log_end(success=is_success, steps=1, score=current_reward, rewards=[current_reward])

    # Move to the next task, or break if done
    if not env.next_task() or error_msg:
        break
