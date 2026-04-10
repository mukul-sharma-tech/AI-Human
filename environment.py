from pydantic import BaseModel
from typing import List, Dict, Optional

# 1. Typed Models for OpenEnv Spec Compliance
class Observation(BaseModel):
    company_context: str
    inbox: List[str]
    team_status: Dict[str, int]
    project_board: Dict[str, str]

class Action(BaseModel):
    action_type: str  # Expected: 'ApproveLeave', 'AssignTask', 'ResolveCrisis'
    target_id: str
    employee_id: Optional[str] = None

class Reward(BaseModel):
    score: float

# 2. Environment Class
class AIHumanEnv:
    def __init__(self):
        self.current_state = None
        self.step_count = 0
        self.max_steps = 15
        # 3 Task Graders (Easy, Medium, Hard)
        self.task_scores = {"easy": 0.0, "medium": 0.0, "hard": 0.0}

    def reset(self) -> Observation:
        self.step_count = 0
        self.task_scores = {"easy": 0.0, "medium": 0.0, "hard": 0.0}
        
        # Initial Injection Context
        self.current_state = Observation(
            company_context=(
                "Company: NovaTech Solutions — a 12-person software consultancy. "
                "You are a fresh graduate joining as Junior Project Coordinator on Day One. "
                "Your manager is out sick and you must handle urgent tasks immediately. "
                "\n\nTeam:\n"
                "- Bob (Senior Backend Dev, stress=50): 5 years experience, specializes in REST APIs and Python. "
                "Works methodically — prefers clear specs before starting. Communicates well but gets stressed under ambiguous deadlines. "
                "Currently mid-way through client API integration. Reliable for backend-heavy tasks. "
                "Has submitted LeaveReq_Bob for Friday — his wife's birthday. Policy allows if no critical deadline clashes.\n"
                "- Charlie (UI/UX Designer, stress=40): 2 years experience, strong in Figma and frontend prototyping. "
                "Creative and fast, but tends to over-iterate on designs without feedback. Works best with structured check-ins. "
                "Currently finishing UI mockups for the dashboard. Low stress, good capacity for additional work. "
                "Has submitted LeaveReq_Charlie for Monday — medical appointment. Policy auto-approves medical leave.\n"
                "- Alice (Lead Full-Stack Dev, stress=70): 8 years experience, knows the entire codebase. "
                "High performer but has been showing signs of burnout — missed two standups last week, terse in Slack. "
                "Was single-handedly managing Proj1 and Proj2. Sent resignation email this morning citing burnout. Effective immediately. "
                "Her departure is the biggest risk to both active projects.\n"
                "\n\nProjects:\n"
                "- Proj1 (Client Dashboard): Due in 3 days. Currently unassigned after Alice's resignation. High priority.\n"
                "- Proj2 (Internal Analytics Tool): Due in 7 days. Currently unassigned. Medium priority.\n"
                "\n\nYour tasks:\n"
                "1. Process the leave request inbox — approve or reject each request based on policy.\n"
                "2. Assign Proj1 and Proj2 to available team members without overloading anyone (stress must stay under 80).\n"
                "3. Resolve Alice's resignation — reassign her responsibilities and stabilize the team.\n"
                "\nUse actions: ApproveLeave, AssignTask, ResolveCrisis."
            ),
            inbox=["LeaveReq_Bob", "LeaveReq_Charlie"],
            team_status={"Bob": 50, "Charlie": 40},
            project_board={"Proj1": "unassigned", "Proj2": "unassigned"}
        )
        return self.current_state

    def state(self) -> Observation:
        return self.current_state

    def step(self, action: Action):
        self.step_count += 1
        done = False
        info = {}

        # Easy Task Grader: Triage Leave
        if action.action_type == "ApproveLeave":
            if action.target_id in self.current_state.inbox:
                self.current_state.inbox.remove(action.target_id)
                # 1.0 point if all inbox tasks are cleared
                self.task_scores["easy"] = 1.0 if len(self.current_state.inbox) == 0 else 0.5 

        # Medium Task Grader: Assign Tasks
        elif action.action_type == "AssignTask":
            if action.target_id in self.current_state.project_board and action.employee_id in self.current_state.team_status:
                self.current_state.project_board[action.target_id] = action.employee_id
                self.current_state.team_status[action.employee_id] += 20  # Stress increases
                
                # Partial progress: score based on how many tasks are assigned
                assigned = sum(1 for v in self.current_state.project_board.values() if v != "unassigned")
                score = assigned / len(self.current_state.project_board)
                
                # Penalty if the employee's stress exceeds threshold
                if self.current_state.team_status[action.employee_id] > 80:
                    score -= 0.2
                self.task_scores["medium"] = max(0.0, min(1.0, score))

        # Hard Task Grader: Resolve Crisis
        elif action.action_type == "ResolveCrisis":
            if action.target_id == "Alice_Resignation":
                self.task_scores["hard"] = 1.0

        # Calculate Meaningful Reward Function
        total_score = (self.task_scores["easy"] + self.task_scores["medium"] + self.task_scores["hard"]) / 3.0

        if self.step_count >= self.max_steps or total_score >= 0.99:
            done = True

        return self.current_state, total_score, done, info
