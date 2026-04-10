from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Tuple, Dict, Any

# ==========================================
# 1. Pydantic Models (OpenEnv Spec)
# ==========================================

class MaintainerObservation(BaseModel):
    ticket_id: str
    type: Literal["issue", "pull_request"]
    title: str
    body: str
    code_diff: Optional[str] = None
    available_labels: List[str] = ["bug", "enhancement", "duplicate", "help-wanted", "invalid"]

class MaintainerAction(BaseModel):
    decision: Literal["add_labels", "close_duplicate", "approve_pr", "request_changes"]
    labels_to_add: List[str] = []
    comment: str = ""

class MaintainerReward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)  # was gt/lt (exclusive), now ge/le (inclusive)
    feedback: str

# ==========================================
# 2. The Environment Class
# ==========================================

class OpenSourceMaintainerEnv:
    def __init__(self):
        # The 3 required tasks: Easy -> Medium -> Hard
        self.tasks = [
            {
                "id": "TASK_1_EASY",
                "obs": MaintainerObservation(
                    ticket_id="#101",
                    type="issue",
                    title="Navbar disappears on mobile screens",
                    body="When I shrink the browser window to mobile size, the hamburger menu doesn't show up and the links vanish.",
                ),
                "grader": self._grade_task_1
            },
            {
                "id": "TASK_2_MEDIUM",
                "obs": MaintainerObservation(
                    ticket_id="#102",
                    type="issue",
                    title="App crashes on startup",
                    body="Getting a NullReference exception when booting the app. Note: this looks exactly like issue #89 that was fixed yesterday.",
                ),
                "grader": self._grade_task_2
            },
            {
                "id": "TASK_3_HARD",
                "obs": MaintainerObservation(
                    ticket_id="#103",
                    type="pull_request",
                    title="Feat: Add linked list sorting utility",
                    body="Wrote a quick sorting algorithm for the linked list class.",
                    code_diff="+ def sort_list(head):\n+     # Using bubble sort for simplicity\n+     curr = head\n+     while curr:\n+         # ... nested loop implementation ...\n+     return head"
                ),
                "grader": self._grade_task_3
            }
        ]
        self.current_task_idx = 0
        self.is_done = False

    # 🚀 CRITICAL FIX: Allow the backend validator to request specific tasks!
    def reset(self, task_id: Optional[str] = None, options: Optional[Dict[str, Any]] = None, **kwargs) -> MaintainerObservation:
        """Resets the environment to the specified task, or the current one."""
        self.is_done = False
        
        # Sniff out the requested task_id no matter how the backend sends it
        target_task = task_id or kwargs.get("task_id")
        if not target_task and options and "task_id" in options:
            target_task = options["task_id"]
            
        if target_task:
            for i, task in enumerate(self.tasks):
                if task["id"] == target_task:
                    self.current_task_idx = i
                    break
                    
        return self.tasks[self.current_task_idx]["obs"]

    def step(self, action: MaintainerAction) -> Tuple[MaintainerObservation, float, bool, Dict[str, Any]]:
        """Takes an action, grades it, and progresses the environment."""
        if self.is_done:
            raise RuntimeError("Environment is done. Please call reset().")

        current_task = self.tasks[self.current_task_idx]
        reward_model = current_task["grader"](action)
        
        self.is_done = True 
        
        info = {
            "task_id": current_task["id"],
            "feedback": reward_model.feedback
        }

        return current_task["obs"], reward_model.score, self.is_done, info

    def state(self) -> Dict[str, Any]:
        """Returns the internal state of the environment."""
        return {
            "current_task_index": self.current_task_idx,
            "total_tasks": len(self.tasks),
            "is_done": self.is_done
        }
        
    def next_task(self) -> bool:
        """Advances to the next task. Returns True if there are more tasks."""
        if self.current_task_idx < len(self.tasks) - 1:
            self.current_task_idx += 1
            return True
        return False

    # ==========================================
    # 3. Deterministic Graders (Must be strictly between 0 and 1)
    # ==========================================

    # Outside the class, at the bottom of github_env.py
def grade_task_1(action: MaintainerAction) -> MaintainerReward:
    if action.decision == "add_labels" and "bug" in action.labels_to_add:
        return MaintainerReward(score=0.99, feedback="Perfect. Accurately labeled the frontend UI bug.")
    elif action.decision == "add_labels":
        return MaintainerReward(score=0.5, feedback="Labeled, but missed the 'bug' tag.")
    return MaintainerReward(score=0.01, feedback="Failed to label a clear bug report.")

def grade_task_2(action: MaintainerAction) -> MaintainerReward:
    if action.decision == "close_duplicate":
        return MaintainerReward(score=0.99, feedback="Correctly identified and closed the duplicate issue.")
    elif action.decision == "add_labels" and "duplicate" in action.labels_to_add:
        return MaintainerReward(score=0.8, feedback="Labeled as duplicate, but should have closed it.")
    return MaintainerReward(score=0.01, feedback="Failed to handle the duplicate issue.")

def grade_task_3(action: MaintainerAction) -> MaintainerReward:
    if action.decision == "request_changes":
        if any(kw in action.comment.lower() for kw in ["bubble sort", "inefficient", "time complexity"]):
            return MaintainerReward(score=0.99, feedback="Excellent review. Caught the O(n^2) flaw.")
        return MaintainerReward(score=0.7, feedback="Requested changes, but missed explaining time complexity.")
    elif action.decision == "approve_pr":
        return MaintainerReward(score=0.01, feedback="Critical failure: Approved a PR with an inefficient algorithm.")
    return MaintainerReward(score=0.01, feedback="Did not request changes on flawed code.")
