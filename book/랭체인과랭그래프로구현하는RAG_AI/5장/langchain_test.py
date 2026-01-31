from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common_openai import create_model

load_dotenv()

model = create_model()
parser = StrOutputParser()

#
cot_prompt = ChatPromptTemplate.from_messages([("system", "사용자 질문에 단계적으로 답변하세요."), ("human", "{question}")])
cot_chain = cot_prompt | model | parser

#
summerize_prompt = ChatPromptTemplate.from_messages([("system", "단계적으로 생각한 답변에서 결론만 추출하세요."), ("human", "{text}")])
summerize_chain = summerize_prompt | model | parser

#
result_chain = cot_chain | summerize_chain
output = result_chain.invoke({"question": "10+2*12"})
print(output)





