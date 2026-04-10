# AI HUMAN: Fresh Graduate Management Simulation

A real-world [OpenEnv](https://openenv.ai) simulation that evaluates an AI agent's ability to step into a new company as a "fresh graduate" and immediately manage human resources and project triage tasks.

---

## Environment Description & Motivation

Instead of a long-term simulation, the agent is injected with a complex "Day One" prompt containing the company's current state — including employee stress levels, project deadlines, and internal communications.

This environment models genuine organizational dynamics and provides immediate value for the AI community by testing an LLM's ability to balance productivity with team well-being under strict constraints.

---

## Action and Observation Spaces

The environment strictly complies with the OpenEnv specification using typed Pydantic models.

**Observation Space (`Observation`):**
- `company_context` — text block containing company policies and the initial "training data" for the fresh graduate
- `inbox` — list of incoming employee requests (e.g., leave requests, conflict reports)
- `project_board` — current tasks, assignees, and deadlines
- `team_status` — current stress levels (0–100) and skill sets of team members

**Action Space (`Action`):**
- `ApproveLeave(target_id: str)` — approve a leave request from the inbox
- `AssignTask(target_id: str, employee_id: str)` — assign a project to an employee
- `ResolveCrisis(target_id: str)` — handle a critical event such as a resignation

---

## Tasks and Graders

Three concrete tasks that scale in difficulty. Each features a deterministic, programmatic grader returning a score strictly between `0.0` and `1.0`.

**Task 1 — Email & Leave Triage (Easy)**
- Objective: Process a queue of employee leave requests against the company calendar and policy guidelines.
- Grader: `1.0` if all safe requests are approved and conflicting requests are rejected. `0.0` for failure to process the queue accurately.

**Task 2 — Workload Balancing (Medium)**
- Objective: Assign 2 pending projects to 2 available employees without pushing any individual's stress level above the critical threshold (80).
- Grader: Partial progress rewards (`+0.5` per task successfully assigned). Deduction of `0.2` applied if an employee is overloaded.

**Task 3 — Resignation Crisis Management (Hard)**
- Objective: A key employee (Alice) suddenly resigns. Reassign her overdue projects and stabilize the team within 15 steps.
- Grader: `1.0` if `ResolveCrisis(Alice_Resignation)` is executed after projects are reassigned and inbox is cleared. Scores the full trajectory.

---

## Reward Shaping & State Management

The environment provides meaningful, non-sparse signals over the full trajectory of the agent's actions.

- Positive rewards granted for partial progress (clearing an inbox item, moving a project forward)
- Negative penalties for destructive actions, infinite loops, missed deadlines, or stress spikes
- `reset()` guarantees a clean initial state
- `step(action)` reliably returns `observation, reward, done, info`

---

## Setup & Usage

### Prerequisites

Define the following environment variables before running the inference script:

```bash
API_BASE_URL=<your LLM API endpoint>
MODEL_NAME=<model identifier, e.g. nemotron-3-super>
HF_TOKEN=<your Hugging Face API key>
OPENAI_API_KEY=<your OpenAI API key>
```

### Installation & Validation

```bash
# Build the container
docker build -t aihuman .

# Run the container
docker run aihuman

# Validate OpenEnv spec
openenv validate

# Pre-validation: confirm /reset returns HTTP 200
curl -X POST http://localhost:8080/reset
```

---

## Baseline Inference Script

`inference.py` is located in the root directory and:

- Uses the OpenAI API client to evaluate a pre-trained LLM against all three tasks
- Completes well within the 20-minute limit on hardware constrained to 2 vCPUs / 8GB RAM
- Emits logs to stdout following mandatory `[START]`, `[STEP]`, and `[END]` formatting

Example output:

```
[START] Task: AI_HUMAN, Env: AI_HUMAN_Triage, Model: nemotron-3-super
[STEP] 1 | Reward: 0.17
[STEP] 2 | Reward: 0.33
[STEP] 3 | Reward: 0.50
[STEP] 4 | Reward: 0.67
[STEP] 5 | Reward: 1.00
[END] Final Score: 1.00
```

---

## Project Structure

```
.
├── environment.py   # OpenEnv-compliant environment with Pydantic models and graders
├── inference.py     # Baseline LLM inference script
├── server.py        # FastAPI server exposing the /reset endpoint
├── openenv.yaml     # OpenEnv spec manifest
├── Dockerfile       # Container definition
└── .env             # Local environment variables (do not commit)
```
