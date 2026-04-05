from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv
from io import StringIO
import sys

from environment.env import FinancialAssistantEnv

# Load environment variables
load_dotenv()

app = FastAPI(title="OpenEnv Personal Finance Assistant")
env = FinancialAssistantEnv()

# Global variable to store inference logs
inference_logs = {"status": "idle", "logs": "", "completed": False}

class StepRequest(BaseModel):
    action: Dict[str, Any]

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
async def run_inference(background_tasks: BackgroundTasks):
    """Trigger inference pipeline in background"""
    # Check if API_KEY is set
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="API_KEY not configured. Add it to Repository secrets.")
    
    inference_logs["status"] = "running"
    inference_logs["logs"] = ""
    inference_logs["completed"] = False
    
    # Run inference in background
    background_tasks.add_task(run_inference_task)
    
    return {
        "status": "inference started",
        "message": "Check /inference-status for logs and results"
    }

@app.get("/inference-status")
def inference_status():
    """Get inference logs and status"""
    return {
        "status": inference_logs["status"],
        "completed": inference_logs["completed"],
        "logs": inference_logs["logs"]
    }

def run_inference_task():
    """Execute inference pipeline"""
    try:
        # Capture stdout
        captured_output = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output
        
        # Import and run inference
        from inference import main
        main()
        
        # Restore stdout
        sys.stdout = original_stdout
        
        # Store logs
        logs_output = captured_output.getvalue()
        inference_logs["logs"] = logs_output
        inference_logs["status"] = "completed"
        inference_logs["completed"] = True
        
    except Exception as e:
        sys.stdout = original_stdout
        error_log = f"[ERROR] Inference failed: {str(e)}\n{captured_output.getvalue()}"
        inference_logs["logs"] = error_log
        inference_logs["status"] = "failed"
        inference_logs["completed"] = True
