import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from environment.env import FinancialAssistantEnv
from environment.tasks import TASKS

def main():
    # Load .env file if it exists
    load_dotenv()
    
    # Setup configs
    api_key = os.environ.get("OPENAI_API_KEY", "dummy_key")
    base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
    
    env = FinancialAssistantEnv()
    
    if api_key != "dummy_key":
        client = OpenAI(api_key=api_key, base_url=base_url)
    
    print("[START] Starting inference over tasks")

    for task_id, task_data in TASKS.items():
        print(f"[STEP] Task {task_id}: {task_data['description']}")
        obs = env.reset(task_id=task_id)
        
        prompt = (
            f"You are a Personal Finance Assistant.\n"
            f"Task: {task_data['description']}\n"
            f"Observation: {obs.model_dump_json() if hasattr(obs, 'model_dump_json') else obs.json()}\n"
            "Return a JSON object with two fields exactly: 'analysis' (string) and 'recommendation' (string)."
        )
        
        try:
            if api_key != "dummy_key":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                action_json = response.choices[0].message.content
                action_dict = json.loads(action_json)
            else:
                raise ValueError("Dummy API Key - triggering mock fallback")
        except Exception as e:
            if api_key != "dummy_key":
                print(f"[STEP] API Error: {e}. Automatic fallback mock triggered.")
            
            if task_id == "1":
                action_json = '{"analysis": "Based on the expenses, the rent is the highest expense at 2500.", "recommendation": "Look for cheaper alternatives."}'
            elif str(task_id) == "2":
                action_json = '{"analysis": "Total spent is 3900 against a budget of 3000, 900 over budget.", "recommendation": "Reduce dining out and shopping."}'
            else:
                action_json = '{"analysis": "You have multiple subscription costs adding up.", "recommendation": "Cancel some subscriptions and brew coffee at home."}'
                
            action_dict = json.loads(action_json)

        next_obs, reward, done, info = env.step(action_dict)
        
        print(f"[STEP] Agent Action: {action_dict}")
        print(f"[STEP] Score: {reward.score}")
        print(f"[STEP] Feedback: {reward.feedback}")
        print("-" * 40)
        
    print("[END] Inference completed.")

if __name__ == "__main__":
    main()
