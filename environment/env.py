from typing import Tuple, Dict, Any, Optional
from pydantic import ValidationError

from .models import Observation, Action, Reward, State
from .tasks import TASKS
from .graders import GRADERS

class FinancialAssistantEnv:
    def __init__(self):
        self._state = None
    
    def reset(self, task_id: str = "1") -> Observation:
        if str(task_id) not in TASKS:
            raise ValueError(f"Task ID {task_id} not found.")
            
        initial_obs = TASKS[str(task_id)]["observation"]
        self._state = State(
            current_step=0,
            history=[],
            current_observation=initial_obs,
            task_id=str(task_id)
        )
        return initial_obs
        
    def step(self, action_dict: dict) -> Tuple[Optional[Observation], Reward, bool, Dict[str, Any]]:
        if not self._state:
            raise RuntimeError("Environment must be reset before calling step.")
            
        task_id = self._state.task_id
        
        # 1. Parse JSON / Validate Action
        try:
            action = Action(**action_dict)
        except (ValidationError, TypeError) as e:
            # Penalty for invalid action format
            penalty_reward = Reward(score=0.1, feedback="Action parsing failed. Invalid format or missing fields.")
            self._state.current_step += 1
            self._state.history.append({"action": action_dict, "reward": penalty_reward.dict()})
            return (self._state.current_observation, penalty_reward, True, {"error": str(e)})

        # 2. Grade the valid action
        grader = GRADERS[task_id]
        reward = grader(action)
        
        # 3. Update State
        self._state.current_step += 1
        self._state.history.append({"action": action.dict(), "reward": reward.dict()})
        
        done = True # One-step environment per task
        info = {
            "task_description": TASKS[task_id]["description"],
            "feedback": reward.feedback
        }
        
        return (self._state.current_observation, reward, done, info)

    def state(self) -> dict:
        if not self._state:
            return {}
        return self._state.dict()
