from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from state import UserState
from agents_nodes import menu_node, macro_portion_node, router_node, printer_node, profile_node


def make_graph(state: UserState, llm: ChatOpenAI):
    graph = StateGraph(UserState)
    graph.add_node("router", lambda s: router_node(s, llm=llm))
    graph.add_node("profile", lambda s: profile_node(s, llm=llm))
    graph.add_node("menu", lambda s: menu_node(s, llm=llm))
    graph.add_node("macro_portion", macro_portion_node)
    graph.add_node("printer", printer_node)

    graph.set_entry_point("router")

    def edge_from_router(state: UserState):
        intent = state.get("intent", "unknown")
        if intent in ("update_profile", "calc_targets"):
            return "profile"
        if intent in ("make_menu", "change_menu"):
            return "menu"
        if intent in ("show_profile", "show_menu"):
            return "printer"
        return "printer"

    graph.add_conditional_edges(
        "router",
        edge_from_router,
        {
            "profile": "profile",
            "menu": "menu",
            "printer": "printer"
        }
    )

    def edge_after_menu(state: UserState):
        if state.get("week_menu"):
            return "macro_portion"
        return "printer"

    graph.add_conditional_edges(
        "menu",
        edge_after_menu,
        {"macro_portion": "macro_portion", "printer": "printer"},
    )
    graph.add_edge(["macro_portion", "profile"], "printer")
    graph.add_edge("printer", END)

    return graph.compile()

