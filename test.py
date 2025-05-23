from langgraph.graph import StateGraph
from typing import Literal, Optional

# 定义状态类型（包含分类结果）


class State(TypedDict):
    question: Optional[str]
    # 使用 Literal 限定分类范围
    category: Optional[Literal["complain", "consult", "other"]]
    answer: Optional[str]

# 分类节点


def classify_question(state: State):
    question = state["question"]
    if "投诉" in question:
        return {"category": "complain"}
    elif "咨询" in question:
        return {"category": "consult"}
    else:
        return {"category": "other"}

# 定义不同分类的处理节点


def handle_complain(state: State):
    question = state["question"]
    return {"answer": "您好！接下来为你解答投诉类问题。"}


def handle_consult(state: State):
    question = state["question"]
    # TODO: 处理question
    return {"answer": "您好！接下来为你解答咨询类问题。"}


def handle_other(state: State):
    question = state["question"]
    # TODO: 处理question
    return {"answer": "抱歉，我暂时无法回答这个问题。"}


# 构建图
builder = StateGraph(State)
builder.add_node("classify_node", classify_question)
builder.add_node("complain_node", handle_complain)
builder.add_node("consult_node", handle_consult)
builder.add_node("other_node", handle_other)

# 条件边：根据分类结果跳转到不同节点


def decide_next_node(state: State):
    return state["category"]  # 返回值必须匹配后续的节点名称映射


builder.add_conditional_edges(
    "classify",
    decide_next_node,
    {
        "greeting": "greeting_node",
        "technical": "tech_node",
        "other": "other_node",
    },
)

# 所有分支最终汇聚到 END
builder.add_edge("greeting_node", END)
builder.add_edge("tech_node", END)
builder.add_edge("other_node", END)

graph = builder.compile()
