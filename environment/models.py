from pydantic import BaseModel
from typing import List, Optional

class Expense(BaseModel):
    category: str
    amount: float

class Observation(BaseModel):
    expenses: List[Expense]
    budget: float
    total_spent: float

class Action(BaseModel):
    analysis: str
    recommendation: str

class Reward(BaseModel):
    score: float
    feedback: str

class State(BaseModel):
    current_step: int
    history: List[dict]
    current_observation: Optional[Observation] = None
    task_id: str
