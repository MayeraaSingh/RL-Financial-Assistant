from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv
import threading

from environment.env import FinancialAssistantEnv

# Load environment variables
load_dotenv()

app = FastAPI(title="OpenEnv Personal Finance Assistant")
env = FinancialAssistantEnv()

# Global variable to store inference logs
inference_logs = {"status": "idle", "logs": "", "completed": False}
inference_lock = threading.Lock()

class StepRequest(BaseModel):
    action: Dict[str, Any]


class ResetRequest(BaseModel):
    task_id: str = "1"

@app.get("/")
def home():
    return {
        "message": "OpenEnv Financial Assistant API online!",
        "endpoints": {
            "/reset": "Reset environment",
            "/step": "Execute action",
            "/state": "Get current state",
            "/run-inference": "Run full inference pipeline",
            "/inference-status": "Check inference status and logs"
        }
    }

@app.get("/reset")
def reset_env(task_id: str = "1"):
    try:
        obs = env.reset(task_id)
        return {"observation": obs.dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/reset")
def reset_env_post(req: Optional[ResetRequest] = None):
    """OpenEnv-compatible reset route. Accepts optional JSON body: {"task_id": "1"}."""
    try:
        task_id = req.task_id if req else "1"
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

@app.get("/run-inference")
def run_inference_endpoint(background_tasks: BackgroundTasks):
    """Trigger inference pipeline in background"""
    # Check if API_KEY is set
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="API_KEY not configured. Add it to Repository secrets.")
    
    with inference_lock:
        inference_logs["status"] = "running"
        inference_logs["logs"] = "[INFERENCE] Starting...\n"
        inference_logs["completed"] = False
    
    # Run inference in background
    background_tasks.add_task(run_inference_background)
    
    return {
        "status": "inference started",
        "message": "Check /inference-status for logs and results"
    }

@app.get("/inference-status")
def inference_status_endpoint():
    """Get inference logs and status"""
    with inference_lock:
        return {
            "status": inference_logs["status"],
            "completed": inference_logs["completed"],
            "logs": inference_logs["logs"]
        }

def run_inference_background():
    """Execute inference pipeline in background thread"""
    try:
        with inference_lock:
            inference_logs["logs"] += "[INFERENCE] Loading inference module...\n"
        
        # Import and run inference
        import sys
        from io import StringIO
        
        # Capture stdout
        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        
        try:
            from inference import main
            main()
        finally:
            sys.stdout = old_stdout
        
        output = captured.getvalue()
        with inference_lock:
            inference_logs["logs"] += output
            inference_logs["status"] = "completed"
            inference_logs["completed"] = True
        
    except Exception as e:
        with inference_lock:
            inference_logs["logs"] += f"[ERROR] {str(e)}\n"
            inference_logs["status"] = "failed"
            inference_logs["completed"] = True
