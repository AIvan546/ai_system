from typing import TypedDict, Dict


class UserState(TypedDict, total=False):
    query: str
    intent: str

    profile: Dict
    bmi: float
    daily_targets: dict
    macro_targets: dict
    preferences: dict

    week_menu: dict
    meals_per_day: int
    profile_ready: bool
    menu_ready: bool

    final_answer: str
