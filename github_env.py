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
    score: float = Field(gt=0.0, lt=1.0)
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

    def reset(self) -> MaintainerObservation:
        """Resets the environment to the current task."""
        self.is_done = False
        return self.tasks[self.current_task_idx]["obs"]

    def step(self, action: MaintainerAction) -> Tuple[MaintainerObservation, float, bool, Dict[str, Any]]:
        """Takes an action, grades it, and progresses the environment."""
        if self.is_done:
            raise RuntimeError("Environment is done. Please call reset().")

        # Grade the action using the current task's specific grader
        current_task = self.tasks[self.current_task_idx]
        reward_model = current_task["grader"](action)
        
        self.is_done = True # Each task is a single-step episode for now
        
        info = {
            "task_id": current_task["id"],
            "feedback": reward_model.feedback
        }

        # Return: observation, reward (float), done, info
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

    def _grade_task_1(self, action: MaintainerAction) -> MaintainerReward:
        # Easy: Must label as a bug.
        if action.decision == "add_labels" and "bug" in action.labels_to_add:
            return MaintainerReward(score=0.99, feedback="Perfect. Accurately labeled the frontend UI bug.")
        elif action.decision == "add_labels":
            return MaintainerReward(score=0.5, feedback="Labeled, but missed the 'bug' tag.")
        return MaintainerReward(score=0.01, feedback="Failed to label a clear bug report.")

    def _grade_task_2(self, action: MaintainerAction) -> MaintainerReward:
        # Medium: Must recognize it's a duplicate and close it.
        if action.decision == "close_duplicate":
            return MaintainerReward(score=0.99, feedback="Correctly identified and closed the duplicate issue.")
        elif action.decision == "add_labels" and "duplicate" in action.labels_to_add:
            return MaintainerReward(score=0.8, feedback="Labeled as duplicate, but should have closed it.")
        return MaintainerReward(score=0.01, feedback="Failed to handle the duplicate issue.")

    def _grade_task_3(self, action: MaintainerAction) -> MaintainerReward:
        # Hard: Must reject/request changes on a PR using an inefficient sorting algorithm (Bubble Sort).
        if action.decision == "request_changes":
            if "bubble sort" in action.comment.lower() or "inefficient" in action.comment.lower() or "time complexity" in action.comment.lower():
                return MaintainerReward(score=0.99, feedback="Excellent review. Caught the O(n^2) time complexity flaw in the linked list sort.")
            return MaintainerReward(score=0.7, feedback="Requested changes, but missed explaining the time complexity issue with Bubble Sort.")
        elif action.decision == "approve_pr":
            return MaintainerReward(score=0.01, feedback="Critical failure: Approved a PR with an inefficient algorithm for a core data structure.")
        return MaintainerReward(score=0.01, feedback="Did not request changes on flawed code.")

# Test it natively in Colab
if __name__ == "__main__":
    env = OpenSourceMaintainerEnv()
    obs = env.reset()
    print(f"Starting state: {env.state()}")
    print(f"First Observation: {obs.title}\n")
