import hashlib

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import GitLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from common_openai import create_model


def simple_embed(text: str, dim: int = 256) -> list[float]:
    # Deterministic, dependency-free embedding to avoid external API keys.
    vec = [0.0] * dim
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i in range(dim):
            vec[i] += digest[i % len(digest)] / 255.0
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]


def file_filter(file_path: str) -> bool:
    # LangChain repo docs are primarily .md; include .mdx if present.
    return file_path.endswith((".md", ".mdx"))

class LocalEmbeddings:
    # Minimal Embeddings-compatible class to avoid external API keys.
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [simple_embed(text, dim=self.dim) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return simple_embed(text, dim=self.dim)

loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter
)

raw_docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = splitter.split_documents(raw_docs)
query = "AWS S3에서 데이터를 읽어들이기 위한 Document loader가 있나?"
embed_query = simple_embed(query)

load_dotenv()

# OpenAIEmbeddings 사용 시 아래 한 줄만 사용하세요.
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# API 키가 없을 때는 로컬 임베딩으로 대체하세요.
# embeddings = LocalEmbeddings(dim=256)

db = Chroma.from_documents(documents, embedding=embedding)
retriever = db.as_retriever()
context_docs = retriever.invoke(query)
print(f"len={len(context_docs)}")
first_docs = context_docs[0]
print(f"metadata={first_docs.metadata}")
print(f"page_content={first_docs.page_content}")

print(len(raw_docs))
print(len(documents))
print(len(embed_query))
print(embed_query)

print("=========================")

prompt = ChatPromptTemplate.from_template('''
    다음 문맥만을 바탕으로 질문에 답변해주세요.

    문맥: """{context}"""

    질문: {question}
    ''')

model = create_model()
retriever = db.as_retriever()
chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
)

output = chain.invoke(query)
print(output)
