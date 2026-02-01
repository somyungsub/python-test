import operator
from pprint import pprint
from typing import Annotated, Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, ConfigurableField, RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from common_openai import create_model


# state 입력 -> dict 반환 되도록
# def answering_node(state: State) -> dict[str, Any]:
#     query = state.query
#     role = state.current_role
#     generated_message = "#...생성처리"
#     return {"messages": [generated_message]}

# answering_node = (
#     RunnablePassthrough.assign(
#         query=lambda state: state.query,
#         role=lambda state: state.current_role
#     )
#     | prompt
#     | model
#     | StrOutputParser()
#     | RunnablePassthrough.assign(messages=lambda x: x[x])
# )

# def check_node(state: State) -> dict[str, Any]:
#     query=state.query
#     message = state.messages[-1]
#     judge = "판정결과"
#     reason = "이유생성"
#     return {"judgement_reason": reason, "current_judge": judge}

class State(BaseModel):
    query: str = Field(..., description="사용자의 질문")
    current_role: str = Field(default="", description="선정된 답변 역할")
    messages: Annotated[list[str], operator.add] = Field(default=[], description="답변 이력")
    current_judge: bool = Field(default=False, description="품질 검사 결과")
    judgement_reason: str = Field(default="", description="품질 검사 판정 이유")


class StateCheck(BaseModel):
    query: str = Field(..., description="사용자의 질문")
    messages: Annotated[list[Any], operator.add] = Field(default=[], description="메시지 이력")


class Judgement(BaseModel):
    judge: bool = Field(default=False, description="판정 결과")
    reason: str = Field(default="", description="판정 이유")


workflow = StateGraph(State)
model = create_model(model="gpt-4o")
model = model.configurable_fields(max_tokens=ConfigurableField(id="max_tokens"))

ROLES = {
    "1": {
        "name": "일반 지식 전문가",
        "description": "폭넓은 분야의 일반적인 질문에 답변",
        "details": "폭넓은 분야의 일반적인 질문에 대해 정확하고 이해하기 쉬운 답변을 제공하세요.",
    },
    "2": {
        "name": "생성형 AI 제품 전문가",
        "description": "생성형 AI와 관련 제품, 기술에 관한 전문적인 질문에 답변",
        "details": "생성형 AI와 관련 제품, 기술에 관한 전문적인 질문에 대해 최신 정보와 깊은 통찰력을 제공하세요.",
    },
    "3": {
        "name": "카운슬러",
        "description": "개인적인 고민이나 심리적인 문제에 대해 지원 제공",
        "details": "개인적인 고민이나 심리적인 문제에 대해 공감적이고 지원적인 답변을 제공하고, 가능하다면 적절한 조언도 해주세요.",
    }
}


def selection_node(state: State) -> dict[str, Any]:
    query = state.query
    role_options = "\n".join([f"{k}. {v['name']}: {v['description']}" for k, v in ROLES.items()])
    prompt = ChatPromptTemplate.from_template("""
    질문을 분석하고, 가장 적절한 답변 담당 역할을 선택하세요.
    선택지: {role_options}
    
    답변은 선택지의 번호 (1, 2, 또는 3)만 반환하세요.
    질문: {query} 
    """.strip())

    chain = prompt | model.with_config(configurable=dict(max_tokens=1)) | StrOutputParser()
    role_number = chain.invoke({"role_options": role_options, "query": query})
    selected_role = ROLES[role_number.strip()]["name"]

    return {"current_role": selected_role}


def answering_node(state: State) -> dict[str, Any]:
    query = state.query
    role = state.current_role
    role_details = "\n".join([f"- {v['name']}: {v['details']}" for v in ROLES.values()])

    prompt = ChatPromptTemplate.from_template("""
    당신은 {role} 로서 답변하세요. 다음 질문에 대해 당신의 역할에 기반한 적절한 답변을 제공하세요. 
    역할 상세: 
    {role_details}
    
    질문: {query}
    
    답변: """.strip())

    chain = prompt | model | StrOutputParser()
    answer = chain.invoke({"query": query, "role": role, "role_details": role_details})

    return {"messages": [answer]}


def check_node(state: State) -> dict[str, Any]:
    query = state.query
    answer = state.messages[-1]
    prompt = ChatPromptTemplate.from_template("""
    다음 다변의 품질을 체크하고, 문제가 있으면 'False', 문제가 없으면 'True'로 답변하세요.
    또한, 그 판정 이유도 설명하세요.
    
    사용자의 질문: {query}
    답변: {answer}
    """.strip())

    chain = prompt | model.with_structured_output(Judgement)
    result: Judgement = chain.invoke({"query": query, "answer": answer})

    return {"judgement_reason": result.reason, "current_judge": result.judge}

def init_workflow():
    # 노드 추가
    workflow.add_node("selection", selection_node)
    workflow.add_node("answering", answering_node)
    workflow.add_node("check", check_node)

    # 에지 정의
    workflow.set_entry_point("selection")
    # selection -> answering
    workflow.add_edge("selection", "answering")
    # answering -> check
    workflow.add_edge("answering", "check")

    workflow.add_conditional_edges(
        "check",
        lambda state: state.current_judge,
        {True: END, False: "selection"}
    )

    return workflow.compile()

def test1(compiled_workflow):
    initial_state = State(query="생성형 AI에 관해 알려주세요")
    result = compiled_workflow.invoke(initial_state)
    print(result)
    print(result['messages'][-1])

def test1_stream(compiled_workflow):
    initial_state = State(query="생성형 AI에 관해 알려주세요")
    for step in compiled_workflow.stream(initial_state):
        print(step)

def test2_checkpoint():
    def add_message(state: StateCheck) -> dict[str, Any]:
        additional_message = []
        if not state.messages:
            additional_message.append(SystemMessage(content="당신은 최소한의 응답을 하는 대화 에이전트 입니다."))
        additional_message.append(HumanMessage(content=state.query))
        return {"messages": additional_message}

    def llm_response(state: StateCheck) -> dict[str, Any]:
        llm = create_model(model="gpt-4o-mini", temperature=0.5)
        ai_message = llm.invoke(state.messages)
        return {"messages": [ai_message]}

    def print_checkpoint_dump(checkpointer: BaseCheckpointSaver, config: RunnableConfig):
        checkpoint_tuple = checkpointer.get_tuple(config)
        print("체크포인트 데이터:")
        pprint(checkpoint_tuple.checkpoint)
        print("\n메타데이터:")
        pprint(checkpoint_tuple.metadata)

    graph = StateGraph(StateCheck)
    graph.add_node("add_message", add_message)
    graph.add_node("llm_response", llm_response)
    graph.set_entry_point("add_message")
    graph.add_edge("add_message", "llm_response")
    graph.add_edge("llm_response", END)

    # 체크포인터 설정
    checkpointer = MemorySaver()

    # 그래프 컴파일
    compiled_graph = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "example-1"}}
    user_query = StateCheck(query="제가 좋아하는 것은 찹쌀떡입니다. 기억해주세요.")
    first_response = compiled_graph.invoke(user_query, config)

    print(first_response)
    for checkpoint in checkpointer.list(config):
        print(checkpoint)
    print_checkpoint_dump(checkpointer, config)

    user_query2 = StateCheck(query="제가 좋아하는 것이 뭔지 기억하세요??")
    second_response = compiled_graph.invoke(user_query2, config)

    print(second_response)
    for checkpoint in checkpointer.list(config):
        print(checkpoint)
    print_checkpoint_dump(checkpointer, config)

    config2 = {"configurable": {"thread_id": "example-2"}}
    user_query_ex2 = StateCheck(query="제가 좋아하는것은 무엇인가요?")
    ex2_response = compiled_graph.invoke(user_query_ex2, config2)

    print(ex2_response)
    for checkpoint in checkpointer.list(config2):
        print(checkpoint)
    print_checkpoint_dump(checkpointer, config2)


# test1(init_workflow())
test1_stream(init_workflow())
# test2_checkpoint()
