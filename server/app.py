import uvicorn
from fastapi import FastAPI, HTTPException
from typing import Optional, Dict, Any
from github_env import OpenSourceMaintainerEnv, MaintainerAction

app = FastAPI(title="OpenEnv Server")
env = OpenSourceMaintainerEnv()

@app.post("/reset")
async def reset(payload: Optional[Dict[str, Any]] = None):
    # 🚀 FIX: Safely unpack the payload and pass it to the environment!
    payload = payload or {}
    obs = env.reset(**payload)
    return {"observation": obs}

@app.post("/step")
async def step(action: MaintainerAction):
    # FastAPI automatically parses the AI's JSON into your MaintainerAction model
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs,
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

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == '__main__':
    main()
