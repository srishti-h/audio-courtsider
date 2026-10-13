"""Event classes the system detects and trades on, mapped from SoccerNet action-spotting labels."""

EVENT_TYPES = [
    "goal",
    "shot",
    "corner",
    "yellow_card",
    "red_card",
    "penalty",
    "offside",
    "substitution",
]

SOCCERNET_TO_EVENT = {
    "Goal": "goal",
    "Shots on target": "shot",
    "Shots off target": "shot",
    "Corner": "corner",
    "Yellow card": "yellow_card",
    "Red card": "red_card",
    "Yellow->red card": "red_card",
    "Penalty": "penalty",
    "Offside": "offside",
    "Substitution": "substitution",
}

# Events that move match-outcome prices and are therefore worth racing the feed for.
PRICE_MOVING = ("goal", "red_card", "penalty")

# Global timeline: second half is offset so (half, t) maps to a single monotone clock.
HALF_OFFSET = 3600.0


def global_time(half: int, t: float) -> float:
    return (half - 1) * HALF_OFFSET + t


def split_global(g: float) -> tuple[int, float]:
    half = 1 if g < HALF_OFFSET else 2
    return half, g - (half - 1) * HALF_OFFSET
