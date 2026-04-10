import uvicorn
from fastapi import FastAPI, HTTPException
from typing import Optional, Dict, Any
from github_env import OpenSourceMaintainerEnv, MaintainerAction

app = FastAPI(title="OpenEnv Server")
env = OpenSourceMaintainerEnv()

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "opensource-maintainer-env"}

@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {"id": "TASK_1_EASY", "difficulty": "easy"},
            {"id": "TASK_2_MEDIUM", "difficulty": "medium"},
            {"id": "TASK_3_HARD", "difficulty": "hard"},
        ]
    }

@app.post("/reset")
async def reset(payload: Optional[Dict[str, Any]] = None):
    payload = payload or {}
    obs = env.reset(**payload)
    return {"observation": obs.dict()}

@app.post("/step")
async def step(action: MaintainerAction):
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.dict(),
            "reward": float(reward),
            "done": bool(done),
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/state")
@app.get("/state")
async def state(payload: Optional[Dict[str, Any]] = None):
    return env.state()

@app.post("/grader")
async def grader(payload: Dict[str, Any]):
    task_id = payload.get("task_id")
    action_data = payload.get("action", {})
    try:
        env.reset(task_id=task_id)
        action = MaintainerAction(**action_data)
        _, reward, done, info = env.step(action)
        return {
            "score": float(reward),
            "feedback": info.get("feedback", ""),
            "task_id": task_id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == '__main__':
    main()
