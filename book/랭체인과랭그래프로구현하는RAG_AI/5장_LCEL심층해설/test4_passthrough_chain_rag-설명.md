# test4_passthrough_chain_rag 함수 설명

## 코드 구조

```python
def test4_passthrough_chain_rag():
    prompt_template = ChatPromptTemplate.from_template('''
        다음 문맥만을 고려하여 질문에 답하세요.
        문맥: """{context}"""
        질문: {question}
    ''')

    retriever = TavilySearchAPIRetriever(k=5)

    # 기본 체인 - 최종 답변만 반환
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | model
        | parser
    )

    output = chain.invoke("서울의 현재날씨는?")
    print(output)

    # assign 체인 - 중간 데이터도 함께 반환
    chain_assign = {
        "question": RunnablePassthrough(),
        "context": retriever,
    } | RunnablePassthrough.assign(
        answer=prompt_template | model | parser
    )

    assign_invoke = chain_assign.invoke("서울의 현재날씨는?")
    pprint(assign_invoke)
```

---

## 입출력 흐름 (단계별)

```
입력: "서울의 현재날씨는?" (str)
         │
         ▼
┌────────────────────────────────────────────────┐
│  {"context": retriever, "question": RunnablePassthrough()}  │
│  (RunnableParallel로 자동 변환)                              │
│                                                │
│   ┌──────────────┐    ┌─────────────────────┐  │
│   │   retriever  │    │ RunnablePassthrough │  │
│   │              │    │                     │  │
│   │ 입력: "서울의"│    │ 입력: "서울의..."   │  │
│   │ 출력: [Doc...│    │ 출력: "서울의..."   │  │
│   └──────────────┘    └─────────────────────┘  │
└────────────────────────────────────────────────┘
         │
         ▼
    출력: {"context": [Document1, Document2...],
           "question": "서울의 현재날씨는?"}
         │
         ▼
┌─────────────────────────────┐
│      prompt_template        │
│                             │
│  문맥: """[검색결과들]"""    │
│  질문: 서울의 현재날씨는?   │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│          model              │
│   (GPT가 답변 생성)         │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│     StrOutputParser         │
│   (AIMessage → str 변환)    │
└─────────────────────────────┘
         │
         ▼
    출력: "서울의 현재 날씨는..."
```

---

## 핵심: retriever가 str 하나만 받는 이유

```python
{"context": retriever, "question": RunnablePassthrough()}
```

이 딕셔너리는 **RunnableParallel**로 자동 변환됩니다.

### 작동 방식

1. 입력 `"서울의 현재날씨는?"`이 들어오면
2. **같은 입력**이 `retriever`와 `RunnablePassthrough()` **둘 다**에게 전달됨
3. 각각 병렬로 실행:

| 키 | Runnable | 입력 | 출력 |
|----|----------|------|------|
| `context` | `retriever` | `"서울의 현재날씨는?"` | `[Document, Document, ...]` |
| `question` | `RunnablePassthrough()` | `"서울의 현재날씨는?"` | `"서울의 현재날씨는?"` |

### retriever가 str을 받을 수 있는 이유

- `TavilySearchAPIRetriever`는 `BaseRetriever`를 상속
- `BaseRetriever.invoke(query: str)` → 문자열을 검색 쿼리로 사용
- 내부적으로 `_get_relevant_documents(query)` 호출

```python
# BaseRetriever 내부 (간략화)
class BaseRetriever(Runnable[str, List[Document]]):
    def invoke(self, input: str) -> List[Document]:
        return self._get_relevant_documents(input)
```

---

## RunnablePassthrough의 역할

```python
RunnablePassthrough()
```

- 입력을 **그대로 통과**시킴
- 여기서는 원본 질문 `"서울의 현재날씨는?"`을 보존하기 위해 사용
- `context`는 검색 결과로 대체되므로, 원본 질문을 `question`에 따로 저장

---

## 타입 요약

| 컴포넌트 | 입력 타입 | 출력 타입 |
|----------|-----------|-----------|
| `chain.invoke()` | `str` | `str` |
| `retriever` | `str` | `List[Document]` |
| `RunnablePassthrough()` | `str` | `str` |
| `prompt_template` | `dict` | `ChatPromptValue` |
| `model` | `ChatPromptValue` | `AIMessage` |
| `parser` | `AIMessage` | `str` |

---

## 프롬프트 템플릿 문법

### `'''...'''` (바깥쪽) - Python 멀티라인 문자열

- Python에서 여러 줄 문자열을 작성할 때 사용
- 줄바꿈, 들여쓰기가 그대로 포함됨

### `"""{context}"""` (안쪽) - 프롬프트 내 구분자

- LLM에게 context 내용의 시작과 끝을 명확히 알려줌
- context에 특수문자, 줄바꿈 등이 있어도 혼동 방지
- Prompt Injection 방어에도 도움

---

## Runnable 표준 인터페이스

`invoke`는 LangChain의 **Runnable 표준 인터페이스**입니다.

모든 LangChain 컴포넌트는 `Runnable`을 상속하며, 다음 메서드들을 구현합니다:

| 메서드 | 설명 |
|--------|------|
| `invoke()` | 동기 실행 (단일 입력) |
| `ainvoke()` | 비동기 실행 |
| `stream()` | 스트리밍 실행 |
| `batch()` | 배치 실행 (여러 입력) |

### RunnablePassthrough 내부 구현

```python
# langchain_core/runnables/passthrough.py (간략화)
class RunnablePassthrough(Runnable[Input, Input]):

    def invoke(self, input: Input, config: Optional[...] = None) -> Input:
        return input  # 그대로 반환
```

### 파이프(`|`) 연산자의 작동 원리

```python
chain = retriever | prompt | model | parser
```

내부적으로:
```python
# chain.invoke("질문") 호출 시
result = "질문"
result = retriever.invoke(result)   # Runnable.invoke()
result = prompt.invoke(result)      # Runnable.invoke()
result = model.invoke(result)       # Runnable.invoke()
result = parser.invoke(result)      # Runnable.invoke()
return result
```

### Runnable을 구현하는 컴포넌트들

```
Runnable (base)
    ├── RunnablePassthrough
    ├── RunnableLambda
    ├── RunnableParallel
    ├── RunnableSequence (| 로 연결된 체인)
    ├── BaseRetriever
    ├── BaseChatModel
    ├── BaseOutputParser
    ├── ChatPromptTemplate
    └── ...
```

**결론**: `invoke()`는 LangChain LCEL(LangChain Expression Language)의 핵심 표준 메서드입니다.

---

## RunnablePassthrough.assign() 설명

### 코드

```python
chain_assign = {
    "question": RunnablePassthrough(),
    "context": retriever,
} | RunnablePassthrough.assign(
    answer=prompt_template | model | parser
)

assign_invoke = chain_assign.invoke("서울의 현재날씨는?")
pprint(assign_invoke)
```

### assign()이란?

`RunnablePassthrough.assign()`은 **기존 데이터를 유지하면서 새로운 키를 추가**하는 메서드입니다.

### 기본 체인 vs assign 체인 비교

| 체인 | 출력 |
|------|------|
| 기본 체인 | `"서울의 현재 날씨는 맑고..."` (str) |
| assign 체인 | `{"question": "...", "context": [...], "answer": "..."}` (dict) |

### 입출력 흐름

```
입력: "서울의 현재날씨는?" (str)
         │
         ▼
┌─────────────────────────────────────────┐
│  {"question": RunnablePassthrough(),    │
│   "context": retriever}                 │
└─────────────────────────────────────────┘
         │
         ▼
    중간 결과: {"question": "서울의 현재날씨는?",
               "context": [Document1, Document2...]}
         │
         ▼
┌─────────────────────────────────────────┐
│  RunnablePassthrough.assign(            │
│      answer=prompt_template | model | parser  │
│  )                                      │
│                                         │
│  기존 dict를 그대로 통과시키면서        │
│  "answer" 키를 새로 추가                │
└─────────────────────────────────────────┘
         │
         ▼
    최종 결과: {"question": "서울의 현재날씨는?",
               "context": [Document1, Document2...],
               "answer": "서울의 현재 날씨는 맑고..."}
```

### assign의 핵심 동작

```python
RunnablePassthrough.assign(answer=some_chain)
```

1. 입력 dict를 **그대로 통과**시킴 (passthrough)
2. `answer` 키에 `some_chain.invoke(입력 dict)` 결과를 **추가**
3. 기존 키들(`question`, `context`)은 **보존**됨

### 언제 사용하나?

| 상황 | 사용 |
|------|------|
| 최종 답변만 필요 | 기본 체인 |
| 중간 데이터도 필요 (디버깅, 로깅, UI 표시) | assign 체인 |

### 실제 출력 예시

```python
# 기본 체인
"서울의 현재 날씨는 맑고 기온은 5도입니다."

# assign 체인
{
    "question": "서울의 현재날씨는?",
    "context": [
        Document(page_content="서울 날씨 정보..."),
        Document(page_content="기상청 발표..."),
        ...
    ],
    "answer": "서울의 현재 날씨는 맑고 기온은 5도입니다."
}
```

### assign vs RunnableParallel 차이

```python
# RunnableParallel - 새로운 dict 생성
{"a": chain_a, "b": chain_b}

# assign - 기존 dict에 키 추가
RunnablePassthrough.assign(new_key=some_chain)
```

| 구분 | RunnableParallel | assign |
|------|------------------|--------|
| 입력 | 원본 입력 사용 | 이전 단계의 dict 사용 |
| 출력 | 새 dict 생성 | 기존 dict + 새 키 |
| 용도 | 병렬 처리 시작 | 체인 중간에 키 추가 |

---

## pick() - 특정 키만 선택

### 코드

```python
# pick 없이 - 모든 키 반환
chain_assign = (
    {"context": retriever, "question": RunnablePassthrough()}
    | RunnablePassthrough.assign(answer=prompt_template | model | parser)
)

assign_invoke = chain_assign.invoke("서울의 현재날씨는?")
# 결과: {"question": "...", "context": [...], "answer": "..."}

# pick 사용 - 특정 키만 선택
chain_assign_pick = (
    {"context": retriever, "question": RunnablePassthrough()}
    | RunnablePassthrough.assign(answer=prompt_template | model | parser)
).pick(["context", "answer"])  # question 제외

assign_pick_invoke = chain_assign_pick.invoke("서울의 현재날씨는?")
# 결과: {"context": [...], "answer": "..."}  ← question 없음
```

### pick()이란?

`pick()`은 dict 출력에서 **특정 키만 선택**하여 반환하는 메서드입니다.

- 불필요한 데이터 제거
- 필요한 필드만 다음 단계로 전달
- 출력 정리

### 입출력 흐름

```
입력: "서울의 현재날씨는?"
         │
         ▼
┌─────────────────────────────────────────┐
│  {"context": retriever,                 │
│   "question": RunnablePassthrough()}    │
└─────────────────────────────────────────┘
         │
         ▼
    {"question": "서울의 현재날씨는?",
     "context": [Document1, Document2...]}
         │
         ▼
┌─────────────────────────────────────────┐
│  RunnablePassthrough.assign(            │
│      answer=prompt_template | model | parser  │
│  )                                      │
└─────────────────────────────────────────┘
         │
         ▼
    {"question": "서울의 현재날씨는?",
     "context": [Document1, Document2...],
     "answer": "서울의 현재 날씨는..."}
         │
         ▼
┌─────────────────────────────────────────┐
│  .pick(["context", "answer"])           │
│                                         │
│  "question" 키 제거                     │
│  "context", "answer" 키만 유지          │
└─────────────────────────────────────────┘
         │
         ▼
    {"context": [Document1, Document2...],
     "answer": "서울의 현재 날씨는..."}
```

### 비교 표

| 메서드 | 출력 |
|--------|------|
| assign (pick 없이) | `{"question", "context", "answer"}` 전체 |
| assign + pick | `{"context", "answer"}` 선택된 키만 |

### 언제 사용하나?

| 상황 | 예시 |
|------|------|
| 중간 데이터 제거 | 최종 답변에 question 불필요 |
| API 응답 정리 | 클라이언트에 필요한 필드만 반환 |
| 다음 체인 입력 | 다음 단계에서 특정 키만 필요할 때 |

### pick vs itemgetter 차이

```python
# pick - Runnable 체인에서 사용
chain.pick(["key1", "key2"])

# itemgetter - dict에서 값 추출
itemgetter("key")(some_dict)  # 값만 반환 (dict 아님)
```

| 구분 | pick | itemgetter |
|------|------|------------|
| 반환 타입 | `dict` (선택된 키들) | 값 자체 |
| 용도 | 여러 키 선택 | 단일 값 추출 |
| 체인 사용 | Runnable 메서드 | 일반 Python 함수 |