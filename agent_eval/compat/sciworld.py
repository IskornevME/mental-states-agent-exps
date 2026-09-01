from scienceworld import ScienceWorldEnv


def _patched_step(self, input_str: str):
    """QLASS-compatible ScienceWorld step with explicit completion metadata."""
    observation = self.server.step(input_str)

    raw_score = self.server.getScore()
    score = int(round(100 * raw_score))
    task_completed = bool(self.server.getCompleted())
    is_completed = task_completed
    num_moves = self.getNumMoves()

    reward = score - self.lastStepScore
    self.lastStepScore = score

    terminated_by_step_limit = num_moves > self.envStepLimit
    terminated_by_negative_score = score < 0

    if terminated_by_step_limit or terminated_by_negative_score:
        is_completed = True

    infos = {
        "moves": num_moves,
        "raw_score": raw_score,
        "score": score,
        "reward": reward,
        "look": self.look(),
        "inv": self.inventory(),
        "taskDesc": self.taskdescription(),
        "valid": self.getValidActionObjectCombinations(),
        "variationIdx": self.variationIdx,
        "taskName": self.taskName,
        "simplificationStr": self.simplificationStr,
        "task_completed": task_completed,
        "terminated_by_step_limit": terminated_by_step_limit,
        "terminated_by_negative_score": terminated_by_negative_score,
    }
    return observation, reward, is_completed, infos


def apply_sciworld_step_patch() -> None:
    """Apply the ScienceWorld step behavior used by current QLASS experiments."""
    ScienceWorldEnv.step = _patched_step
