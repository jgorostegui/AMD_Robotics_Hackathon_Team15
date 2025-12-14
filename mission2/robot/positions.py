"""SmolVLA prompts for robot moves.

The real robot uses natural language prompts, not XYZ coordinates.
These prompts match the format in tasks_smolvla.json.
"""

# Column prompts - sent to SmolVLA
COLUMN_PROMPTS = {
    0: "Pick the orange ball and place it in column 0.",
    1: "Pick the orange ball and place it in column 1.",
    2: "Pick the orange ball and place it in column 2.",
    3: "Pick the orange ball and place it in column 3.",
    4: "Pick the orange ball and place it in column 4.",
}

HOME_PROMPT = "Move robot to home position."


def get_move_prompt(column: int) -> str:
    """Get the SmolVLA prompt for a column move."""
    return COLUMN_PROMPTS.get(column, f"Pick the orange ball and place it in column {column}.")
