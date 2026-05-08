from typing import Dict, List

from tracing import trace_tool

Activities = {
    "low": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "high": 1.725,
    "very_high": 1.9
}

Goals = {
    "loss": 0.85,
    "gain": 1.1,
    "maintanance": 1
}

@trace_tool("calc_bmi")
def calc_bmi(weight_kg: float, height_cm: float) -> float:
    return round(weight_kg / ((height_cm / 100.0) ** 2), 2)

@trace_tool("calc_bmr")
def calc_bmr(sex: str, weight_kg: float, height_cm: float, age: int) -> float:
    if sex == "male":
        c = 5
    else:
        c = -161

    return 10 * weight_kg + 6.25 * height_cm + 5 * age + c


@trace_tool("calc_daily_target")
def calc_daily_target(sex: str, weight_kg: float, height_cm: float, age: int, activity: str, goal: str) -> Dict[str, float]:
    bmr = calc_bmr(sex, weight_kg, height_cm, age)
    tdee = bmr * Activities[activity]
    target = tdee * Goals[goal]

    return {
        "bmr": round(bmr, 0),
        "tdee": round(tdee, 0),
        "target_kcal": round(target, 0),
    }

@trace_tool("calc_macro_targets")
def calc_macro_targets(daily_target: float, protein_g_per_kg: float, fat_pct: float, weight_kg: float) -> Dict[str, float]:
    p_g = protein_g_per_kg * weight_kg
    p_kcal = 4 * p_g
    f_kcal = daily_target * fat_pct
    f_g = f_kcal / 9
    c_kcal = max(0.0, daily_target - p_kcal - f_kcal)
    c_g = c_kcal / 4
    return {
        "p": round(p_g, 0),
        "f": round(f_g, 0),
        "c": round(c_g, 0)
    }

@trace_tool("split_daily_meals")
def split_daily_meals(meals_per_day) -> List:
    patterns = {
        2: [0.45, 0.55],  # завтрак + ужин
        3: [0.25, 0.35, 0.40],  # завтрак / обед / ужин
        4: [0.25, 0.30, 0.20, 0.25],  # завтрак / обед / перекус / ужин
        5: [0.20, 0.25, 0.20, 0.20, 0.15],  # завтрак / перекус / обед / перекус / ужин
    }

    return patterns[meals_per_day]

@trace_tool("split_daily_names")
def split_daily_names(meals_per_day) -> List:
    patterns = {
        2: ["breakfast", "dinner"],  # завтрак + ужин
        3: ["breakfast", "lunch", "dinner"],  # завтрак / обед / ужин
        4: ["breakfast", "lunch", "snack", "dinner"],  # завтрак / обед / перекус / ужин
        5: ["breakfast", "snack", "lunch", "snack", "dinner"],  # завтрак / перекус / обед / перекус / ужин
    }

    return patterns[meals_per_day]

@trace_tool("split_weekly_targets")
def split_weekly_targets() -> List:
    return [1.05, 1.0, 0.95, 1.0, 1.05, 1.0, 0.95]
