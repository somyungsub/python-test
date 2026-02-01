import os
from enum import Enum
from typing import Any

from langchain_chroma import Chroma
from langchain_cohere import CohereRerank
from langchain_community.document_loaders import GitLoader
from langchain_community.retrievers import TavilySearchAPIRetriever, BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

from common_openai import create_model

model = create_model()
parser = StrOutputParser()
PERSIST_DIR = "./chroma_db"


def file_filter(file_path):
    return (
        file_path.endswith((".mdx", ".md"))
    )


def vector_load():
    loader = GitLoader(clone_url="https://github.com/langchain-ai/langchain", repo_path="./langchain", branch="master",
                       file_filter=file_filter)

    documents = loader.load()
    return documents


def embedding_vector():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # DB가 이미 존재하면 로드, 없으면 새로 생성
    if os.path.exists(PERSIST_DIR):
        print("기존 DB 로드 중...")
        db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        print("새 DB 생성 중 (임베딩 진행)...")
        documents = vector_load()
        db = Chroma.from_documents(
            documents,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
    return db


def create_default_prompt():
    return ChatPromptTemplate.from_template('''
        다음 문맥만을 고려해 질문에 답하세요.

        문맥: """{context}"""
        질문: {question}

        ''')


def create_retriever():
    db = embedding_vector()
    return db.as_retriever()


def print_result(chain, input="LangChain의 개요를 알려줘"):
    output = chain.invoke(input)
    print(output)


def reciprocal_rank_fusion(retriever_outputs: list[list[Document]], k: int = 60) -> list[str]:
    content_score_mapping = {}

    for docs in retriever_outputs:
        for rank, doc in enumerate(docs):
            content = doc.page_content

            if content not in content_score_mapping:
                content_score_mapping[content] = 0
            else:
                content_score_mapping[content] += 1 / (rank + k)

    ranked = sorted(content_score_mapping.items(), key=lambda x: x[1], reverse=True)
    return [content for content, _ in ranked]


# ------ test
def test1_vector():
    prompt = create_default_prompt()
    chain = (
            {
                "context": create_retriever(),
                "question": RunnablePassthrough()
            }
            | prompt | model | parser)
    print(chain)


def test2_HyDE():
    prompt = create_default_prompt()
    hyde_prompt = ChatPromptTemplate.from_template('''
        다음 질문에 한문장으로 답하세요.
        질문: {question}
    ''')
    hyde_chain = hyde_prompt | model | parser

    retriever = create_retriever()

    hyde_rag_chain = {"context": hyde_chain | retriever, "question": RunnablePassthrough()} | prompt | model | parser
    print_result(hyde_rag_chain)


class QueryGenOutput(BaseModel):
    queries: list[str] = Field(..., description="list of queries")


def test3_querygen():
    prompt = ChatPromptTemplate.from_template("""
    질문에 대해 벡터 데이터베이스에서 관련 문서를 검색하기 위한 3개의 서로 다른 검색 쿼리를 생성하세요.
    거리 기반 유사성 검색의 한계를 극복하기 위해 사용자의 질문에 대해 여러 관점을 제공하는 것이 목표입니다.
    
    질문: {question}
    """)

    retriever = create_retriever()
    querygen_chain = prompt | model.with_structured_output(QueryGenOutput) | (lambda x: x.queries)
    multi_query_rag_chain = {"context": querygen_chain | retriever.map(),
                             "question": RunnablePassthrough()} | prompt | model | parser
    print_result(multi_query_rag_chain)


def test3_1_rerank_fusion():
    retriever = create_retriever()

    prompt = ChatPromptTemplate.from_template("""
        질문에 대해 벡터 데이터베이스에서 관련 문서를 검색하기 위한 3개의 서로 다른 검색 쿼리를 생성하세요.
        거리 기반 유사성 검색의 한계를 극복하기 위해 사용자의 질문에 대해 여러 관점을 제공하는 것이 목표입니다.

        질문: {question}
        """)

    querygen_chain = prompt | model.with_structured_output(QueryGenOutput) | (lambda x: x.queries)
    rag_fusion_chain = {
                           "question": RunnablePassthrough(),
                           "context": querygen_chain | retriever.map(),
                       } | prompt | model | parser

    print_result(rag_fusion_chain)


def test4_rerank():
    def rerank(inp: dict[str, Any], top_n: int = 3) -> list[Document]:
        question = inp["question"]
        documents = inp["documents"]
        cohere_rerank = CohereRerank(model="rerank-multilingual-v3.0", top_n=top_n)
        return cohere_rerank.compress_documents(documents, query=question)

    prompt = create_default_prompt()
    retriever = create_retriever()
    rerank_chain = ({"documents": retriever, "question": RunnablePassthrough()}
                    | RunnablePassthrough.assign(context=rerank)
                    | prompt | model | parser
                    )

    print_result(rerank_chain)


class Route(str, Enum):
    langchain_document = "langchain_document"
    web = "web"


class RouteOutput(BaseModel):
    route: Route


def test5_route_rag():
    langchain_document_retriever = create_retriever().with_config({"run_name": "langchain_document_retriever"})
    web_retriever = TavilySearchAPIRetriever(k=5).with_config({"run_name": "web_retriever"})

    prompt = ChatPromptTemplate.from_template("""
    질문에 답변하기 위해 적절한 retriever를 선택하세요.
    
    질문: {question}
    """)

    route_chain = prompt | model.with_structured_output(RouteOutput) | (lambda x: x.route)

    def route_retriever(inp: dict[str, Any]) -> list[Document]:
        route = inp["route"]
        question = inp["question"]

        if route == Route.langchain_document:
            return langchain_document_retriever.invoke(question)
        elif route == Route.web:
            return web_retriever.invoke(question)
        else:
            raise ValueError(f"Unknown route: {route}")

    default_prompt = create_default_prompt()
    route_rag_chain = ({"route": route_chain, "question": RunnablePassthrough()}
                       | RunnablePassthrough().assign(context=route_retriever)
                       | default_prompt | model | parser
                       )
    print_result(route_rag_chain)
    print_result(route_rag_chain, "서울날씨는?")


def test6_hybrid_rag():
    chroma_retriever = create_retriever().with_config({"run_name": "chroma_retriever"})
    bm25_retriever = BM25Retriever.from_documents(vector_load()).with_config({"run_name": "bm25_retriever"})
    hybrid_retriever = (
            RunnableParallel({
                "chroma_documents": chroma_retriever,
                "bm25_documents": bm25_retriever
            }) | (lambda x: [x["chroma_documents"], x["bm25_documents"]]) | reciprocal_rank_fusion
    )
    prompt = create_default_prompt()
    hybrid_rag_chain = {"context": hybrid_retriever, "question": RunnablePassthrough()} | prompt | model | parser
    print_result(hybrid_rag_chain)


# test1_vector()
# test2_HyDE()
# test3_querygen()
# test3_1_rerank_fusion()
# test4_rerank()
# test5_route_rag()
test6_hybrid_rag()
