from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from environment.env import FinancialAssistantEnv

app = FastAPI(title="OpenEnv Personal Finance Assistant")
env = FinancialAssistantEnv()

class StepRequest(BaseModel):
    action: Dict[str, Any]

@app.get("/reset")
def reset_env(task_id: str = "1"):
    try:
        obs = env.reset(task_id)
        return {"observation": obs.dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step_env(req: StepRequest):
    try:
        obs, reward, done, info = env.step(req.action)
        return {
            "observation": obs.dict() if obs else None,
            "reward": reward.dict(),
            "done": done,
            "info": info
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def state_env():
    return {"state": env.state()}
