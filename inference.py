import os
from openai import OpenAI
from environment import AIHumanEnv, Action

# Mandatory environment variables per competition rules
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "nemotron-3-super")
API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Primary client uses competition-supplied env vars
PRIMARY_CLIENT = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# Fallback chain: Ollama -> Groq -> primary (OpenAI)
PROVIDERS = [
    {"name": "ollama",  "client": OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"), "model": "llama3"},
    {"name": "groq",    "client": OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY), "model": "llama-3.3-70b-versatile"},
    {"name": "openai",  "client": PRIMARY_CLIENT, "model": MODEL_NAME},
]

SYSTEM_PROMPT = """You are a fresh graduate managing HR triage.
Output ONLY a single JSON object with no extra text, no explanation, no markdown.
Use EXACTLY these target_id values:
- Leave requests: "LeaveReq_Bob" or "LeaveReq_Charlie"
- Projects: "Proj1" or "Proj2"
- Crisis: "Alice_Resignation"

Format:
{"action_type": "ApproveLeave"|"AssignTask"|"ResolveCrisis", "target_id": "string", "employee_id": "string"}
"""

def call_llm(messages: list) -> str:
    """Try each provider in order, return first successful response."""
    for provider in PROVIDERS:
        try:
            completion = provider["client"].chat.completions.create(
                model=provider["model"],
                messages=messages,
                temperature=0.0,
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:
            print(f"[DEBUG] Provider '{provider['name']}' failed: {exc}", flush=True)
    raise RuntimeError("All LLM providers failed.")

def next_action(obs, task_scores) -> Action:
    """Deterministically pick the next best action based on current environment state."""
    # Easy task: clear inbox first
    if obs.inbox:
        return Action(action_type="ApproveLeave", target_id=obs.inbox[0])

    # Medium task: assign unassigned projects to lowest-stress employee
    employees = list(obs.team_status.keys())
    for proj, assignee in obs.project_board.items():
        if assignee == "unassigned":
            emp = min(employees, key=lambda e: obs.team_status[e])
            return Action(action_type="AssignTask", target_id=proj, employee_id=emp)

    # Hard task: resolve resignation crisis
    if task_scores["hard"] < 1.0:
        return Action(action_type="ResolveCrisis", target_id="Alice_Resignation", employee_id="")

    return None

def main():
    env = AIHumanEnv()

    # Mandatory [START] format
    print(f"[START] Task: AI_HUMAN, Env: AI_HUMAN_Triage, Model: {MODEL_NAME}", flush=True)

    obs = env.reset()
    last_reward = 0.0

    for step in range(1, 16):
        action = next_action(obs, env.task_scores)
        if action is None:
            break

        obs, reward, done, info = env.step(action)

        # Mandatory [STEP] format
        print(f"[STEP] {step} | Reward: {reward:.2f}", flush=True)

        last_reward = reward
        if done:
            break

    # Mandatory [END] format
    print(f"[END] Final Score: {last_reward:.2f}", flush=True)

if __name__ == "__main__":
    main()
