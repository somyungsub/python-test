from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from common_class import Recipe
from common_openai import create_model

model = create_model()
prompt = ChatPromptTemplate.from_messages([SystemMessage(content="사용자가 입력한  레시피를 생각해주세요"), HumanMessage(content="{dish}")])

def test1() -> None:
    chain = prompt | model
    ai_message = chain.invoke({"dish": "짜장면"})
    print(ai_message.content)

def test2() -> None:
    chain = prompt | model | StrOutputParser()
    output = chain.invoke({"dish": "카레"})
    print(output)

def test3() -> None:
    output_parser = PydanticOutputParser(pydantic_object=Recipe)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "사용자가 입력한 요리의 레시피를 생각해주세요\n\n{format_instructions}"),
            ("human", "{dish}"),
        ]
    )
    prompt_partial = prompt.partial(
        format_instructions=output_parser.get_format_instructions()
    )
    bound_model = model.bind(response_format={"type": "json_object"})

    chain = prompt_partial | bound_model | output_parser
    output = chain.invoke({"dish": "카레"})
    print(type(output))
    print(output)

def test4() -> None:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "사용자가 입력한 요리의 레시피를 생각해주세요"),
            ("human", "{dish}"),
        ]
    )

    chain = prompt | model.with_structured_output(Recipe)
    output = chain.invoke({"dish": "카레"})
    print(type(output))
    print(output)


# test1()
# test2()
# test3()
test4()
