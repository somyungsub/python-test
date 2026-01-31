import asyncio
from operator import itemgetter
from pprint import pprint
from typing import Iterator

from dotenv import load_dotenv
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, chain, RunnableParallel, RunnablePassthrough

from common_openai import create_model

load_dotenv()

model = create_model(temperature=0.7)
parser = StrOutputParser()

## 1. 체인
def test1_chain_sum() -> None:
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
def test2_chain_lambda() -> None:
    def upper_text(text: str) -> str:
        return text.upper()

    @chain
    def upper_text_decorator(text: str) -> str:
        return text.upper()

    prompt2 = ChatPromptTemplate.from_messages([
        ("system", "you are a helpful assistant."),
        ("human", "{input}")
    ])

    chain2 = prompt2 | model | parser | RunnableLambda(upper_text)
    invoke_result2 = chain2.invoke({"input": "i am hungry~"})
    print(invoke_result2)

    chain2_deco = prompt2 | model | parser | upper_text_decorator
    invoke_result2_decor = chain2_deco.invoke({"input": "hello world!"})
    print(invoke_result2_decor)

    # model_response = prompt2 | model
    # print(model_response)
    # print(type(model_response))
    # print(model_response.model_dump_json(indent=2))

    def upper_stream(input_stream: Iterator[str]) -> Iterator[str]:
        for text in input_stream:
            yield text.upper()

    stream_chain = prompt2 | model | parser | upper_stream
    for chunk in stream_chain.stream({"input": "hello world!. who are you?"}):
        print(chunk, end="", flush=True)


def test3_parallel_chain() -> None:
    optimistic_prompt = ChatPromptTemplate.from_messages(
        [("system", "당신은 낙관주의자 입니다. 사용자의 입력에 대해 낙관적인 의견을 제공하세요."), ("human", "{input}")])
    optimistic_chain = optimistic_prompt | model | parser

    pessimistic_prompt = ChatPromptTemplate.from_messages(
        [("system", "당신은 비관주의자 입니다. 사용자의 입력에 대해 비관적인 의견을 제공하세요."), ("human", "{input}")])
    pessimistic_chain = pessimistic_prompt | model | parser

    ## 1. 병렬처리 1
    parallel_chain = RunnableParallel({"optimistic_option": optimistic_chain, "pessimistic_option": pessimistic_chain})
    invoke_result = parallel_chain.invoke({"input": "생성 AI의 진화에 관해 "})
    print(invoke_result)
    print("===========optimistic_option")
    print(invoke_result["optimistic_option"])
    print("===========pessimistic_option")
    print(invoke_result["pessimistic_option"])

    ## 2. 병렬 체인 처리
    combined_opinion_prompt = ChatPromptTemplate.from_messages([("system", "당신은 객관적 AI 입니다. 두 가지 의견을 종합하세요"),
                                                                ("human",
                                                                 "낙관적의견: {optimistic_option}\n비관적의견: {pessimistic_option}")])

    combined_opinion_chain = (RunnableParallel({"optimistic_option": optimistic_chain,
                                 "pessimistic_option": pessimistic_chain}) | combined_opinion_prompt | model | parser)

    invoke_result2 = combined_opinion_chain.invoke({"input": "생성 AI의 진화에 관해 "})
    print(invoke_result2)

    ## 3. 체인처리 단축
    combined_opinion_chain = ({
                                  "optimistic_option": optimistic_chain,
                                  "pessimistic_option": pessimistic_chain
                              }
                              | combined_opinion_prompt
                              | model
                              | parser)

    invoke_result3 = combined_opinion_chain.invoke({"input": "생성 AI의 진화에 관해 "})
    print(invoke_result3)

    ## 4. itemgetter
    opinion_combination_prompt = ChatPromptTemplate.from_messages(
        [("system", "당신은 객관적 AI 입니다. {input}에 대해 두 가지 의견을 종합하세요"),
         ("human", "낙관적의견: {optimistic_option}\n비관적의견: {pessimistic_option}")])

    combined_opinion_chain4 = ({"optimistic_option": optimistic_chain, "pessimistic_option": pessimistic_chain, "input": itemgetter("input")}
                               | opinion_combination_prompt | model | parser
                               )
    invoke_result4 = combined_opinion_chain4.invoke({"input": "생성 AI의 진화에 관해 "})
    print(invoke_result4)

async def test4_passthrough_chain_rag():
    prompt_template = ChatPromptTemplate.from_template('''
        다음 문맥만을 고려하여 질문에 답하세요.
        문맥: """{context}"""
        질문: {question}
    ''')

    retriever = TavilySearchAPIRetriever(k=5)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | model
        | parser
    )

    output = chain.invoke("서울의 현재날씨는?")
    print(output)

    chain_assign = (
            {"context": retriever, "question": RunnablePassthrough()}
            | RunnablePassthrough.assign(answer=prompt_template | model | parser)
    )

    assign_invoke = chain_assign.invoke("서울의 현재날씨는?")
    pprint(assign_invoke)
    print("========\n", assign_invoke)
    print("========\n", chain_assign.pick(["context", "answer"]))

    chain_assign_pick = (
            {"context": retriever, "question": RunnablePassthrough()}
            | RunnablePassthrough.assign(answer=prompt_template | model | parser)
    ).pick(["context", "answer"])

    assign_pick_invoke = chain_assign_pick.invoke("서울의 현재날씨는?")
    print("========\n", assign_pick_invoke, "\n========\n")
    print("========\n", assign_pick_invoke["answer"], "\n========\n")

    async for event in chain.astream_events("서울의 현재날씨는?", version="v2"):
        print(event, flush=True)

    async for event in chain.astream_events("서울의 현재날씨는?", version="v2"):
        event_kind = event["event"]
        if event_kind == "on_retriever_end":
            print("=== 검색결과 ===")
            documents = event["data"]["output"]
            for doc in documents:
                print(doc.page_content)
        elif event_kind == "on_parser_start":
            print("=== 최종 출력 ===")
        elif event_kind == "on_parser_stream":
            print(event["data"]["chunk"], end="", flush=True)

# test1_chain_sum()
# test2_chain_lambda()
# test3_parallel_chain()
asyncio.run(test4_passthrough_chain_rag())
