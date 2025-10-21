from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def get_retriever():
    # --- Vector store / retriever 준비 ---
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'tax-markdown-index'
    database = Pinecone.from_existing_index(index_name=index_name, embedding=embedding)
    retriever = database.as_retriever(search_kwargs={'k': 4})
    
    return retriever


def get_llm(model='gpt-4o'): 
    # --- LLM ---
    llm = ChatOpenAI(model=model)
    return llm


def get_dictionary_chain():
    # --- 사용자 질문 정규화(사전 기반) 프리체인 ---
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    llm = get_llm()

    normalize_prompt = ChatPromptTemplate.from_template(
        "사전: {dictionary}\n\n"
        "아래 사용자의 질문을 필요시 사전을 참고해 한 줄로 바꿔 주세요.\n"
        "불필요하면 원문을 그대로 출력하세요.\n\n"
        "질문: {question}"
    ).partial(dictionary=str(dictionary))

    dictionary_chain = normalize_prompt | llm | StrOutputParser()
    return dictionary_chain


def get_rag_chain():
    llm = get_llm()
    retriever = get_retriever()
    # --- (선택) LangSmith Hub에서 프롬프트 pull 가능하지만,
    # create_stuff_documents_chain 은 {context} + {input}가 꼭 필요.
    # ⚠️ 'stuff' 체인은 prompt가 반드시 {context} 와 {input} 을 포함해야 함
    rag_prompt = ChatPromptTemplate.from_template(
        "아래 자료(context)를 참고해 질문에 간결하고 정확히 답하세요.\n\n"
        "자료:\n{context}\n\n"
        "질문: {input}\n"
        "답변:"
    )
    # --- 문서 결합 체인 (stuff) ---
    doc_chain = create_stuff_documents_chain(llm=llm, prompt=rag_prompt)
    # --- 검색 + 생성 체인 ---
    rag_chain = create_retrieval_chain(retriever, doc_chain)
    return rag_chain

def get_ai_message(user_message: str) -> str:
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    # rag_chain.invoke({"input": "..."})-> {'answer': '...', 'context': [...]}
    
    # --- 파이프라인 결합: (질문 정규화) → (RAG) ---
    pipeline = {"input": dictionary_chain} | rag_chain

    result = pipeline.invoke({"question": user_message})
    # create_retrieval_chain 의 표준 출력키는 'answer'
    return result["answer"]