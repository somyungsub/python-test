import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

MODEL = "gpt-4o-mini"
PROMPT = "현재프로젝트에 대해서, 기술, 소스 등 분석해서 알려주세요."


def get_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Check your .env file.")
    return api_key


def create_client() -> ChatOpenAI:
    return ChatOpenAI(model=MODEL, api_key=get_api_key(), temperature=0)


def run_once(client: ChatOpenAI) -> None:
    response = client.invoke([HumanMessage(content=PROMPT)])
    print(response.content)


def run_stream(client: ChatOpenAI) -> None:
    for chunk in client.stream([HumanMessage(content=PROMPT)]):
        print(chunk.content, end="", flush=True)
    print()

def print_prompt() -> None:
    prompt = PromptTemplate.from_template("""다음요리의 레시피를 생각해주세요. 요리명: {dish}, 레시피: {recipe}""")
    prompt_invoke = prompt.invoke(
        {
            "dish": "짜장면",
            "recipe":"1. 춘창, 2. 면"
         }
    )
    print(prompt_invoke)

def print_chat_prompt() -> None:
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "사용자가 입력한 요리의 레시피를 생각해주세요."),
            ("human", "요리명: {dish}, 레시피: {recipe}")
        ]
    )
    chat_prompt_invoke = chat_prompt.invoke(
        {
            "dish": "짜장면",
            "recipe":"1. 춘창, 2. 면"
         }
    )
    print(chat_prompt_invoke)

def print_chat_history() -> None:
    prompt = ChatPromptTemplate.from_messages(
        [("system", "사용자가 입력한 요리의 레시피를 생각해주세요."), MessagesPlaceholder("chat_history", optional=True),
         ("human", "요리명: {dish}, 레시피: {recipe}")])

    prompt_value = prompt.invoke(
        {"chat_history": [HumanMessage(content="맛있게 만들어주세요."), AIMessage("네 성심성의껏 만들어드릴게요. 요리명과 레시피를 알려주세요")],
         "dish": "짜장면", "recipe": "1. 춘창, 2. 면"})

    print(prompt_value)


client = create_client()
# run_once(client)
# run_stream(client)
# print_prompt()
# print_chat_prompt()
print_chat_history()
