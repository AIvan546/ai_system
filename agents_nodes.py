import tools
import prompts
import json
import re

from pathlib import Path
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import UserState
from tracing import trace_agent, trace_llm

def load_data():
    p = Path("memory.json")
    return json.loads(p.read_text(encoding="utf-8"))

def load_menu():
    p = Path("menu.json")
    return json.loads(p.read_text(encoding="utf-8"))


def save_data(data):
    p = Path("memory.json")
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

@trace_llm("ask_llm")
def ask_llm(llm: ChatOpenAI, system: str, human: str) -> str:
    out = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    content = out.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, ensure_ascii=False)


def safe_json_loads(s: str) -> Dict[str, Any]:
    """Достаём первый валидный JSON из ответа LLM"""
    if s is None:
        raise ValueError("LLM returned None")
    if not isinstance(s, str):
        s = json.dumps(s, ensure_ascii=False)

    s = s.strip()
    if not s:
        raise ValueError("Empty LLM response")

    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s, flags=re.IGNORECASE)
    s = s.strip()

    brace = s.find("{")
    bracket = s.find("[")
    candidates = [p for p in (brace, bracket) if p != -1]
    if not candidates:
        raise ValueError(f"No JSON found in LLM response: {s[:200]!r}")
    start_idx = min(candidates)
    s2 = s[start_idx:]

    dec = json.JSONDecoder()
    obj, _ = dec.raw_decode(s2)

    if isinstance(obj, list):
        return {"_list": obj}
    if not isinstance(obj, dict):
        return {"_value": obj}
    return obj


@trace_agent("router")
def router_node(state: UserState, llm: ChatOpenAI) -> UserState:
    q = state.get("query", "")
    out = safe_json_loads(ask_llm(llm, prompts.router_prompt, q))
    intent = out.get("intent", "unknown")
    # Fallback: короткие запросы про меню часто дают ambiguous intent у LLM.
    if intent == "unknown":
        q_low = q.lower()
        has_menu_word = "меню" in q_low or "menu" in q_low
        wants_detailed_menu = has_menu_word and any(
            marker in q_low
            for marker in (
                "ингредиент",
                "ингридиент",
                "грамм",
                "граммовк",
                "детальн",
                "подробн",
                "расклад",
                "состав",
                "вес продукт",
                "сколько грамм",
            )
        )
        if wants_detailed_menu:
            intent = "show_menu_detailed"
        elif has_menu_word:
            intent = "make_menu"
    state["intent"] = intent
    return state

def _valid_daily_targets(v) -> bool:
    return isinstance(v, dict) and v.get("target_kcal") is not None


def _valid_macro_targets(v) -> bool:
    return isinstance(v, dict) and v.get("p") is not None


_PROFILE_FIELD_KEYS = (
    "height_cm",
    "weight_kg",
    "age",
    "sex",
    "activity",
    "goal",
)


def _profile_metrics_complete(profile: Any) -> bool:
    if not isinstance(profile, dict):
        return False
    return all(
        k in profile and profile[k] not in (None, "")
        for k in _PROFILE_FIELD_KEYS
    )


def _profile_data_complete(data: dict) -> bool:
    """Достаточно данных в памяти, чтобы показать профиль (без profile_node / profile_ready в state)."""
    if not _profile_metrics_complete(data.get("profile")):
        return False
    return _valid_daily_targets(data.get("daily_targets"))


def _recalc_targets_from_stored_profile(data: dict) -> bool:
    """ИМТ, калории и БЖУ из memory.profile без LLM. False — не хватает полей или KeyError уровня activity/goal."""
    profile = data.get("profile") or {}
    if not _profile_metrics_complete(profile):
        return False
    try:
        w = float(profile["weight_kg"])
        h = float(profile["height_cm"])
        age = int(profile["age"])
        sex = str(profile["sex"])
        activity = str(profile["activity"])
        goal = str(profile["goal"])
    except (TypeError, ValueError, KeyError):
        return False
    try:
        data["bmi"] = tools.calc_bmi(w, h)
        data["daily_targets"] = tools.calc_daily_target(
            sex, w, h, age, activity, goal
        )
        data["macro_targets"] = tools.calc_macro_targets(
            float(data["daily_targets"]["target_kcal"]),
            protein_g_per_kg=1.8,
            fat_pct=0.25,
            weight_kg=w,
        )
    except KeyError:
        return False
    return True


_PROFILE_KEYS_FROM_ARGS = _PROFILE_FIELD_KEYS


def _merge_profile_from_args(mem: Dict[str, Any], args: Dict[str, Any]) -> None:
    """Пишем в память поля профиля из аргументов тулов (иначе остаётся только final от LLM)."""
    if not args:
        return
    profile = dict(mem.get("profile") or {})
    for key in _PROFILE_KEYS_FROM_ARGS:
        if key in args and args[key] is not None:
            profile[key] = args[key]
    mem["profile"] = profile


def _parse_ingredient_weight(ingredient: Dict[str, Any]) -> Any:
    """Нормализуем ключ веса и приводим к числу, если возможно."""
    raw = ingredient.get("weight (g)")
    if raw is None:
        raw = ingredient.get("weight(g)")
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        normalized = raw.strip().replace(",", ".")
        if not normalized:
            return None
        try:
            return float(normalized)
        except ValueError:
            return None
    return None


@trace_agent("profile")
def profile_node(state: UserState, llm: ChatOpenAI) -> UserState:
    q = state.get("query", "")

    if state.get("intent") == "calc_targets":
        data = load_data()
        if _recalc_targets_from_stored_profile(data):
            save_data(data)
            state["profile"] = dict(data.get("profile") or {})
            state["preferences"] = dict(data.get("preferences") or {})
            state["bmi"] = data["bmi"]
            state["daily_targets"] = data["daily_targets"]
            state["macro_targets"] = data["macro_targets"]
            state["final_answer"] = (
                "Характеристики пересчитаны по текущему профилю (ИМТ, калории, БЖУ)."
            )
            state["profile_ready"] = True
            return state

    for _ in range(6):
        data = load_data()
        profile = dict(data.get("profile") or {})
        preferences = dict(data.get("preferences") or {})
        bmi = data.get("bmi")
        daily_targets = data.get("daily_targets")
        macro_targets = data.get("macro_targets")
        if not _valid_daily_targets(daily_targets):
            daily_targets = None
        if not _valid_macro_targets(macro_targets):
            macro_targets = None

        tool_context = {
            "profile": profile,
            "preferences": preferences,
            "bmi": bmi,
            "daily_targets": daily_targets,
            "macro_targets": macro_targets,
        }

        human = (f"Текст пользователя: {q}"
                 f"Текущий контекст: {tool_context}")

        out = ask_llm(llm, prompts.profile_prompt, human)
        answer = safe_json_loads(out)

        if "final" in answer:
            new_profile = answer["final"].get("profile_update", {}) or {}
            if "preferences" in new_profile:
                preferences.update(new_profile.get("preferences"))
            profile.update(new_profile)

            profile_done = _profile_metrics_complete(profile)
            calories_done = _valid_daily_targets(daily_targets)

            if profile_done and calories_done:
                if macro_targets is None:
                    macro_targets = tools.calc_macro_targets(daily_targets["target_kcal"], protein_g_per_kg=1.8, fat_pct=0.25, weight_kg= profile["weight_kg"])

                profile_ready = True
            else:
                profile_ready = False

            data["profile"] = profile
            data["preferences"] = preferences
            if macro_targets is not None:
                data["macro_targets"] = macro_targets

            save_data(data)

            state["profile"] = profile
            state["preferences"] = preferences
            if bmi is not None:
                state["bmi"] = bmi
            if daily_targets is not None:
                state["daily_targets"] = daily_targets
            if macro_targets is not None:
                state["macro_targets"] = macro_targets

            state["final_answer"] = answer["final"].get("notes", "Профиль обновлён.")
            state["profile_ready"] = profile_ready
            return state

        if "tool" in answer:
            mem = load_data()
            args = answer.get("args")
            if not isinstance(args, dict):
                args = {}
            tool = answer["tool"]
            if tool == "calc_bmi":
                bmi = tools.calc_bmi(args["weight_kg"], args["height_cm"])
                _merge_profile_from_args(mem, args)
                mem["bmi"] = bmi
                save_data(mem)
                state["bmi"] = bmi
            elif tool == "calc_daily_target":
                daily_target = tools.calc_daily_target(args["sex"], args["weight_kg"], args["height_cm"], args["age"], args["activity"], args["goal"])
                _merge_profile_from_args(mem, args)
                mem["daily_targets"] = daily_target
                save_data(mem)
                state["daily_targets"] = daily_target
            elif tool == "calc_macro_targets":
                macro_targets = tools.calc_macro_targets(args["daily_target"], args["protein_g_per_kg"], args["fat_pct"], args["weight_kg"])
                _merge_profile_from_args(mem, args)
                mem["macro_targets"] = macro_targets
                save_data(mem)
                state["macro_targets"] = macro_targets
            else:
                state["final_answer"] = (
                    "Не понял вызов. Попробуй иначе "
                    "сформулировать параметры профиля."
                )
                state["profile_ready"] = False
                return state

    state["final_answer"] = (
        "Не удалось стабильно вычислить профиль. Укажи рост, вес, возраст, "
        "пол и активность одним сообщением."
    )
    state["profile_ready"] = False
    return state


@trace_agent("menu")
def menu_node(state: UserState, llm: ChatOpenAI) -> UserState:
    q = state["query"]
    data = load_data()

    daily_targets = data.get("daily_targets")
    macro_targets = data.get("macro_targets")

    if not daily_targets or not macro_targets:
        state["final_answer"] = (
            "Сначала рассчитай цели (ИМТ/калории/БЖУ). "
            "Напиши: 'Рассчитай цели' или обнови профиль."
        )
        return state

    preferences = data["preferences"]
    avoid = preferences["avoid"]
    meals_per_day = preferences.get("meals_per_day") or 3
    try:
        meals_per_day = int(meals_per_day)
    except (TypeError, ValueError):
        meals_per_day = 3

    menu = load_menu()
    valid_recipes = []

    for recipe in menu:
        flag = False
        tags = recipe["tags"]
        for tag in tags:
            if tag in avoid:
                flag = True
                break
        if not flag:
            valid_recipes.append(recipe)

    compact = [
        {
            "id": r["id"],
            "name": r["name"],
            "tags": r.get("tags", []),
            "portion": r["portion"],
        }
        for r in valid_recipes
    ]
    
    current_week_menu = state.get("week_menu")
    if not isinstance(current_week_menu, dict) or not current_week_menu:
        current_week_menu = data.get("week_menu")
    if not isinstance(current_week_menu, dict) or not current_week_menu:
        current_week_menu = None

    human = json.dumps(
        {
            "query": state["query"],
            "meals_per_day": meals_per_day,
            "avoid_list": avoid,
            "current_week_menu": current_week_menu,
            "menu": compact
        },
        ensure_ascii=False
    )

    out = ask_llm(llm, prompts.menu_prompt, human)
    answer = safe_json_loads(out)
    week_menu = answer["week_menu"]
    if not isinstance(week_menu, dict):
        state["final_answer"] = (
            "Не получилось составить меню. Попробуй уточнить ограничения "
            "(например, 'без рыбы', '3 приёма пищи')."
        )
        return state

    data["week_menu"] = week_menu
    state["week_menu"] = week_menu
    state["meals_per_day"] = meals_per_day

    return state


@trace_agent("macro_portion")
def macro_portion_node(state):
    data = load_data()
    week_menu = state.get("week_menu")
    if not week_menu:
        return state

    daily_targets = state.get("daily_targets") or data.get("daily_targets")
    macro_targets = state.get("macro_targets") or data.get("macro_targets")
    if not daily_targets or not macro_targets:
        return state

    daily_kcal = daily_targets["target_kcal"]
    meals_per_day = state.get("meals_per_day") or data.get("preferences", {}).get("meals_per_day") or 3
    try:
        meals_per_day = int(meals_per_day)
    except (TypeError, ValueError):
        meals_per_day = 3

    recipe_by_id = {r["id"]: r for r in load_menu()}
    macro_menu: Dict[str, Any] = {}
    week_pattern = tools.split_weekly_targets()
    keys = list(week_menu.keys())

    for i, day in enumerate(keys):
        daily_menu = week_menu[day]
        total_kcal = week_pattern[i] * daily_kcal
        meals = []
        names = tools.split_daily_names(meals_per_day)
        split = tools.split_daily_meals(meals_per_day)

        for j, meal in enumerate(daily_menu):
            rid = meal.get("id") or meal.get("recipe_id")
            recipe = recipe_by_id.get(rid)
            if not recipe:
                continue
            portion_kcal = recipe["portion"]["kcal"]
            c = (total_kcal * split[j]) / portion_kcal

            meals.append(
                {
                    "meal": names[j] if j < len(names) else f"meal_{j}",
                    "name": recipe["name"],
                    "portion_multiplier": round(c, 2),
                    "totals": {
                        "kcal": portion_kcal * c,
                        "p": recipe["portion"]["p"] * c,
                        "f": recipe["portion"]["f"] * c,
                        "c": recipe["portion"]["c"] * c,
                    },
                    "ingridients": [
                        {
                            "name": x["name"],
                            "weight (g)": (
                                round(weight * c, 1)
                                if (weight := _parse_ingredient_weight(x)) is not None
                                else None
                            ),
                        }
                        for x in recipe["ingridients"]
                    ],
                }
            )

        macro_menu[day] = meals

    state["week_menu"] = macro_menu
    data["week_menu"] = macro_menu
    save_data(data)
    state["final_answer"] = "Меню составлено"
    return state


@trace_agent("printer")
def printer_node(state):
    intent = state["intent"]
    data = load_data()
    final_answer = state.get("final_answer")
    profile_ready = state.get("profile_ready")
    profile_complete_in_memory = _profile_data_complete(data)
    if intent in ("update_profile", "show_profile"):
        if final_answer:
            return state
        if profile_ready or profile_complete_in_memory:
            profile = state.get("profile") or data.get("profile") or {}

            lines = []
            lines.append("Профиль:")
            lines.append(f"- Пол: {profile["sex"]}")
            lines.append(f"- Возраст: {profile["age"]} ")
            lines.append(f"- Рост: {profile["height_cm"]} см")
            lines.append(f"- Вес: {profile["weight_kg"]} кг")
            lines.append(f"- Активность: {profile["activity"]}")
            lines.append(f"- Цель: {profile["goal"]}")

            targets = state.get("daily_targets") or data.get("daily_targets") or {}
            bmi = state.get("bmi") if state.get("bmi") is not None else data.get("bmi")
            macro_targets = state.get("macro_targets") or data.get("macro_targets") or {}

            lines.append("ИМТ:")
            lines.append(f"- ИМТ: {bmi}")
            lines.append("Калории:")
            lines.append(f"- BMR: {targets.get('bmr', '—')} ккал")
            lines.append(f"- TDEE: {targets.get('tdee', '—')} ккал")
            lines.append(f"- Дневная цель: {targets.get('target_kcal', '—')} ккал")
            lines.append("Целевые БЖУ:")
            lines.append(f"- Белки: {macro_targets.get('p', '—')} г")
            lines.append(f"- Жиры: {macro_targets.get('f', '—')} г")
            lines.append(f"- Углеводы: {macro_targets.get('c', '—')} г")

            state["final_answer"] = "\n".join(lines)
            return state
        if intent == "show_profile":
            state["final_answer"] = (
                "Профиль ещё не заполнен. Сохрани данные: рост, вес, возраст, пол, активность, цель."
            )
        return state
    if intent == "calc_targets":
        if final_answer:
            return state
        elif profile_ready or profile_complete_in_memory:
            state["final_answer"] = "Характеристики успешно посчитаны"
            return state
    if intent in ("make_menu", "change_menu", "show_menu", "show_menu_detailed"):
        week_menu = state.get("week_menu") or data.get("week_menu") or {}
        detailed_menu = intent == "show_menu_detailed"
        if isinstance(week_menu, dict) and week_menu:
            lines = []
            keys = week_menu.keys()
            for day in keys:
                lines.append(f"### {day} ###")
                for meal in week_menu[day]:
                    lines.append(f"{meal["meal"]}: {meal["name"]}")
                    lines.append(f"- Порция х{meal["portion_multiplier"]}")
                    lines.append(f"- {round(meal["totals"]["kcal"], 0)} ккал")
                    lines.append(f"- БЖУ: {round(meal["totals"]["p"], 1)}/{round(meal["totals"]["f"], 1)}/{round(meal["totals"]["c"], 1)}")
                    if detailed_menu:
                        lines.append("  Ингредиенты:")
                        for ing in meal.get("ingridients") or []:
                            w = _parse_ingredient_weight(ing)
                            nm = ing.get("name", "—")
                            if isinstance(w, (int, float)):
                                lines.append(f"    - {nm}: {round(w, 1)} г")
                            else:
                                lines.append(f"    - {nm}: —")
            state["final_answer"] = "\n".join(lines)
        elif not state.get("final_answer"):
            state["final_answer"] = "Меню не составлено."
        return state
    if not state.get("final_answer"):
        state["final_answer"] = (
            "Не понял запрос. Уточни, что нужно: обновить профиль, "
            "пересчитать цели или составить меню."
        )
    return state

    
