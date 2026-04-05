import os
import json
import asyncio
import textwrap
from typing import Optional, List
from dotenv import load_dotenv
from openai import OpenAI
from environment.env import FinancialAssistantEnv
from environment.tasks import TASKS

# Load .env file if it exists
load_dotenv()

# Environment variables
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")  # Optional
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Optional - used when using from_docker_image()

# Validate required variables
if not API_KEY:
    error_msg = "API_KEY environment variable is required"
    print(f"[ERROR] {error_msg}", flush=True)
    # Don't raise immediately - allow module to import for API usage
    # Only raise when main() is called

# Debug: Log loaded configuration (mask API key)
api_key_preview = f"{API_KEY[:10]}...{API_KEY[-4:]}" if API_KEY else "NOT SET"
print(f"[DEBUG] Configuration loaded:", flush=True)
print(f"[DEBUG]   API_KEY: {api_key_preview}", flush=True)
print(f"[DEBUG]   API_BASE_URL: {API_BASE_URL}", flush=True)
print(f"[DEBUG]   MODEL_NAME: {MODEL_NAME}", flush=True)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, task_desc: str, obs: dict, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Task: {task_desc}
        Current observation: {obs}
        Previous steps:
        {history_block}
        Send your next action.
        """
    ).strip()


def get_model_message(client: OpenAI, step: int, task_desc: str, obs: dict, history: List[str], system_prompt: str = "You are a Personal Finance Assistant.") -> str:
    user_prompt = build_user_prompt(step, task_desc, obs, history)
    try:
        print(f"[DEBUG] Sending API request with API_KEY to {API_BASE_URL}...", flush=True)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=512,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        print(f"[DEBUG] ✅ API request succeeded! Response length: {len(text)} chars", flush=True)
        return text if text else '{"analysis": "No response generated", "recommendation": "Retry"}'
    except Exception as exc:
        error_str = str(exc)
        if "invalid_api_key" in error_str.lower() or "unauthorized" in error_str.lower():
            print(f"[DEBUG] ❌ API KEY ERROR: {exc}", flush=True)
        elif "connection" in error_str.lower():
            print(f"[DEBUG] ❌ CONNECTION ERROR: {exc}", flush=True)
        else:
            print(f"[DEBUG] ❌ API request failed: {exc}", flush=True)
        return '{"analysis": "Error occurred", "recommendation": "Unable to process"}'


def main() -> None:
    if not API_KEY:
        raise ValueError("API_KEY environment variable is required to run inference")
    
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = FinancialAssistantEnv()
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    total_tasks = len(TASKS)
    
    log_start(task="financial_assistant", env="FinancialAssistantEnv", model=MODEL_NAME)
    
    try:
        for task_id, task_data in TASKS.items():
            task_description = task_data.get("description", "")
            error_msg = None
            
            try:
                obs = env.reset(task_id=task_id)
                obs_dict = obs.model_dump_json() if hasattr(obs, 'model_dump_json') else (obs.json() if hasattr(obs, 'json') else str(obs))
                
                # Step 1: Get model response
                step = 1
                message = get_model_message(client, step, task_description, obs_dict, history)
                history.append(message)
                
                # Step 2: Execute action
                try:
                    action_dict = json.loads(message) if isinstance(message, str) else message
                except json.JSONDecodeError:
                    action_dict = {"analysis": "Parse error", "recommendation": "Retry"}
                    error_msg = "json_parse_error"
                
                next_obs, reward, done, info = env.step(action_dict)
                reward_value = reward.score if hasattr(reward, 'score') else (reward if isinstance(reward, (int, float)) else 0.0)
                
                rewards.append(reward_value)
                steps_taken += 1
                score += reward_value
                
                log_step(
                    step=step,
                    action=str(action_dict).replace("\n", " "),
                    reward=reward_value,
                    done=done,
                    error=error_msg
                )
                
                if done:
                    success = True
                    
            except Exception as e:
                error_msg = str(e)
                log_step(
                    step=steps_taken + 1,
                    action="error",
                    reward=0.0,
                    done=True,
                    error=error_msg
                )
        
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards
        )
        
    except Exception as e:
        print(f"[DEBUG] Fatal error: {e}", flush=True)
        log_end(success=False, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
