import streamlit as st

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

st.set_page_config(page_title= "소득세 챗봇", page_icon="🤖")

st.title("🤖 소득세 챗봇")
st.caption("소득세에 관련된 모든 것들을 알려드립니다!")

if 'message_list' not in st.session_state:
    st.session_state.message_list = []
# := 의 역할을 알아야 할 것 같다
print(f'before == {st.session_state.message_list}')
for msg in st.session_state.message_list:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

def get_ai_message(user_message):

    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'tax-markdown-index'
    database = Pinecone.from_existing_index(index_name=index_name, embedding=embedding)

    llm = ChatOpenAI(model='gpt-4o')
    client = Client()
    hub_prompt = client.pull_prompt("rlm/rag-prompt")  # 공개 프롬프트면 owner 포함
    retriever = database.as_retriever(search_kwargs={'k':4})

    qa_chain = retrieval_qa.create_chain(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": hub_prompt},  # 이제 context를 받음
        return_source_documents=False,             # 원하면 출처도 반환
    )

    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전: {dictionary}
        
        질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()
    tax_chain = {"query": dictionary_chain} | qa_chain
    ai_message = tax_chain.invoke({"question" : user_message})

    return ai_message

if user_question := st.chat_input(placeholder='소득세 관련 궁금한 내용들을 말씀해주세요!'):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role":"user", "content":user_question})

    ai_message = get_ai_message(user_question)

    with st.chat_message("ai"):
        st.write(ai_message)
    st.session_state.message_list.append({"role":"ai", "content": ai_message})

print(f'after == {st.session_state.message_list}')