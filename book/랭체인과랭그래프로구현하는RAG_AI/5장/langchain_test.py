from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from common_openai import create_model

load_dotenv()

model = create_model()
parser = StrOutputParser()

## 1. 체인
def test1() -> None:
    # p1
    cot_prompt = ChatPromptTemplate.from_messages([("system", "사용자 질문에 단계적으로 답변하세요."), ("human", "{question}")])
    cot_chain = cot_prompt | model | parser

    # p2
    summerize_prompt = ChatPromptTemplate.from_messages([("system", "단계적으로 생각한 답변에서 결론만 추출하세요."), ("human", "{text}")])
    summerize_chain = summerize_prompt | model | parser

    # chain sum
    result_chain = cot_chain | summerize_chain
    output = result_chain.invoke({"question": "10+2*12"})
    print(output)


## 2. Lamda
def test2() -> None:
    def upper_text(text: str) -> str:
        return text.upper()

    prompt2 = ChatPromptTemplate.from_messages([
        ("system", "you are a helpful assistant."),
        ("human", "{input}")
    ])

    chain2 = prompt2 | model | parser | RunnableLambda(upper_text)
    invoke_result2 = chain2.invoke({"input": "hello~"})
    print(invoke_result2)


test2()



