import streamlit as st
import tiktoken
from loguru import logger


from langchain_core.messages import ChatMessage
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import FAISS
from langchain.memory import StreamlitChatMessageHistory
from langchain_core.runnables import RunnablePassthrough

from langserve import RemoteRunnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def main():
    st.set_page_config(page_title="Streamlit_Rag", page_icon=":books:")
    st.title("_RAG_test4 :red[Q/A Chat]_ :books:")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "store" not in st.session_state:
        st.session_state["store"] = dict()

    def print_history():
        for msg in st.session_state.messages:
            st.chat_message(msg.role).write(msg.content)

    def add_history(role, content):
        st.session_state.messages.append(ChatMessage(role=role, content=content))

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
        process = st.button("Process")

    if process:
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)
        retriever = vectorstore.as_retriever(search_type='mmr', verbose=True)  # 'verbose' 수정
        st.session_state['retriever'] = retriever

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    RAG_PROMPT_TEMPLATE = """당신은 동서울대학교 컴퓨터소프트웨어과 안내 AI 입니다.
                             검색된 문맥을 사용하여 질문에 맞는 답변을 30단어 이내로 하세요
                             답을 모른다면 모른다고 답변하세요.
                             Question: {question}
                             Context: {context}
                             Answer: """

    print_history()

    if user_input := st.chat_input("메세지를 입력해 주세요"):
        add_history("user", user_input)
        st.chat_message("user").write(f"{user_input}")
        with st.chat_message("assistant"):

            llm = RemoteRunnable("https://subtle-hideously-malamute.ngrok-free.app/llm/")
            chat_container = st.empty()

            if st.session_state.processComplete:
                prompt1 = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

                retriever = st.session_state['retriever']

                rag_chain = (
                    {
                        "context": retriever | format_docs,
                        "question": RunnablePassthrough(),
                    }
                    | prompt1
                    | llm
                    | StrOutputParser()
                )

                # 스트리밍 응답 처리 (비동기 방식으로 처리해야 할 수도 있음)
                answer = rag_chain.stream(user_input)

                chunks = []
                try:
                    # 스트리밍을 기다리면서 하나씩 받아오기
                    for chunk in answer:
                        if chunk:  # chunk가 None이 아닌 경우에만 추가
                            chunks.append(chunk)
                            chat_container.markdown("".join(chunks))  # 중간 결과 출력
                    add_history("ai", "".join(chunks))
                except Exception as e:
                    logger.error(f"Error while streaming answer: {e}")
                    chat_container.markdown("응답을 처리하는 중 오류가 발생했습니다.")
            else:
                prompt2 = ChatPromptTemplate.from_template(
                    "다음의 질문에 간결하게 답변해주세요:\n{input}"
                )

                chain = prompt2 | llm | StrOutputParser()

                answer = chain.stream(user_input)
                chunks = []
                for chunk in answer:
                    chunks.append(chunk)
                    chat_container.markdown("".join(chunks))
                add_history("ai", "".join(chunks))

def get_text(docs):
    doc_list = []

    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")

        # 파일을 저장한 후 다시 읽어와서 처리
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb


if __name__ == '__main__':
    main()