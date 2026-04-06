---
title: OpenEnv Financial Assistant
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# OpenEnv Personal Finance Assistant

A real-world OpenEnv-compliant environment simulating a Personal Finance Assistant where an AI agent analyzes user expenses against a budget, providing actionable financial advice.

## Environment Description
This environment presents the AI Agent with a set of expenses and a budget. The agent's goal is to analyze the data and provide actionable recommendations. This is a practical, non-game scenario intended to evaluate an agent's proficiency at real-world financial data processing and advisory.

## Real-World Motivation
AI-driven personal finance is becoming an essential automation field. Being able to correctly digest tabular financial data (expenses vs budgets) and provide localized, actionable advice is highly relevant for banking apps, personal tools, and LLM-powered advisory bots.

## Observation Space
The environment emits a structured JSON object detailing:
- `expenses`: A list of objects with `category` and `amount`
- `budget`: The total allowed budget
- `total_spent`: A consolidated sum of all expenses

## Action Space
The agent responds with a structured JSON object containing:
- `analysis` (string): Text describing the financial situation based on the observation.
- `recommendation` (string): Text suggesting specific actions to improve the situation.

## Tasks
1. **Easy (Task 1):** Identify the highest expense category.
2. **Medium (Task 2):** Detect overspending against a defined budget and explicitly suggest a reduction category based on the provided dataset.
3. **Hard (Task 3):** Provide a full financial advisory analyzing multiple expense areas, summarizing patterns (e.g. tracking subscriptions or recurrent habits), and giving concrete, actionable recommendations.

## Setup Instructions

### Local Execution
1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure environment variables:
   ```bash
   API_KEY="your-key"
   API_BASE_URL="https://api.openai.com/v1"
   MODEL_NAME="gpt-3.5-turbo"
   ```
3. Run the baseline inference script:
   ```bash
   python inference.py
   ```

### Docker / Hugging Face Spaces
Build and run the OpenAI wrapper natively:
```bash
docker build -t openenv-finance .
docker run -p 7860:7860 openenv-finance
```

Once running, interact with the API endpoints:
- `GET /reset?task_id=1` -> returns initial Observation
- `POST /reset` -> accepts optional body `{"task_id": "1"}`
- `POST /step` -> uses `{"action": {"analysis": "...", "recommendation": "..."}}`
- `GET /state` -> returns internal State
- `GET /healthz` -> returns service health and API key presence flag
- `GET /run-inference` -> triggers background inference execution
- `GET /inference-status` -> returns live inference status and logs

## Multi-Mode Deployment Notes
This repo includes the required files for multi-mode validation:
- `pyproject.toml`
- `uv.lock`
- `server/app.py` with callable `main()` and `if __name__ == "__main__":` guard

## Baseline Scores
Using the default inference mocks in `inference.py`, the agent perfectly passes all three tasks achieving a score of `1.0` in each setup. Use actual LLM models directly by passing valid `OPENAI_API_KEY` to examine variance.
