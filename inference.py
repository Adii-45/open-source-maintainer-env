import os
import json
from openai import OpenAI
from github_env import OpenSourceMaintainerEnv, MaintainerAction

# ==========================================
# 1. Setup API Client (Strictly matching checklist format)
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
# REQUIRED BY GRADER: Print START before the loop begins
print("START")
print("🚀 Starting Agentic Evaluation...\n")

# Initialize the environment we just built
env = OpenSourceMaintainerEnv()

while True:
    # REQUIRED BY GRADER: Print STEP at the beginning of each iteration
    print("STEP")
    obs = env.reset()
    task_id = env.tasks[env.current_task_idx]["id"]
    print("=" * 40)
    print(f"📋 Running {task_id}")
    print("=" * 40)
    print(f"Observation Title: {obs.title}")
    
    # Construct the prompt for the LLM based on what the environment sees
    user_prompt = f"Ticket Type: {obs.type}\nTitle: {obs.title}\nBody: {obs.body}\n"
    if obs.code_diff:
        user_prompt += f"Code Diff:\n{obs.code_diff}\n"

    print("🤖 Agent is thinking...")
    try:
        # Call the LLM
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}, # Force JSON output
            temperature=0.1 # Keep it low so the agent is deterministic and analytical
        )
        
        # Parse the LLM's JSON output
        llm_output = response.choices[0].message.content
        action_dict = json.loads(llm_output)
        
        # Convert the raw JSON back into our strict Pydantic Action model
        action = MaintainerAction(**action_dict)
        
        print(f"➡️  Agent Decision: {action.decision}")
        if action.labels_to_add: 
            print(f"🏷️  Labels: {action.labels_to_add}")
        if action.comment: 
            print(f"💬 Comment: {action.comment}")

        # Feed the action back into the environment to get graded
        new_obs, reward, done, info = env.step(action)
        print(f"\n🎯 Reward: {reward} / 1.0")
        print(f"📝 Grader Feedback: {info['feedback']}\n")

    except Exception as e:
        print(f"❌ Error during LLM call or parsing: {e}")

    # Move to the next task, or break if we are done
    if not env.next_task():
        print("✅ All tasks completed!")
        # REQUIRED BY GRADER: Print END when all tasks are finished
        print("END")
        break