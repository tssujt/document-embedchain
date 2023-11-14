import hashlib
import os
import tempfile
import time
import uuid
from contextlib import contextmanager
from typing import Dict, List

import streamlit as st
from langchain.chains import LLMChain, QAGenerationChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders import PyMuPDFLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from streamlit.runtime.uploaded_file_manager import UploadedFile

if openai_api_key := os.getenv("OPENAI_API_KEY"):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo-16k",
        temperature=0.1,
        max_tokens=1024,
        max_retries=1,
    )
elif os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_API_BASE"):
    openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai_api_base = os.getenv("AZURE_OPENAI_API_BASE")

    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        openai_api_type="azure",
        openai_api_version="2023-03-15-preview",
        deployment="text-embedding-ada-002",
    )
    llm = AzureChatOpenAI(
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        openai_api_type="azure",
        openai_api_version="2023-03-15-preview",
        deployment_name="gpt-35-turbo",
        temperature=0.1,
        max_tokens=1024,
        max_retries=1,
    )
else:
    raise ValueError("OPENAI_API_KEY environment variable must be set")


vector_store = ElasticsearchStore(
    index_name="embedding",
    es_url=os.getenv("ES_URL"),
    embedding=embeddings,
)


template_informed = """
我是一个乐于助人的人工智能，能回答问题。当我不知道答案时，我会说我不知道。
我了解的上下文: {context}
当被问到: {question}
无论问题用什么语言，回答都只能是简体中文。我的回答仅基于上下文中的信息:
"""

qa_prompt = PromptTemplate(
    template=template_informed, input_variables=["context", "question"]
)

template_process_question = """
I want you act as a language detector and translator.
I will provide a question sentence in any language and a context, you will translate the question sentence in which language the context I wrote is in you.
Do not write any explanations or other words, just translate the question sentence.

The context is:
```
{context}
```

The question sentence is:
```
{question}
```

My response for question translation using the language in the context is:
"""
question_process_prompt = PromptTemplate(
    template=template_process_question, input_variables=["context", "question"]
)


@contextmanager
def catchtime(event: str) -> float:
    start = time.time()
    yield
    st.write(f"{event}: {time.time() - start:.3f} seconds")


@st.cache_data(show_spinner=False)
def generate_qa_pairs(text: str) -> List[Dict[str, str]]:
    qa_generation_sys_template = """
You are an AI assistant tasked with generating question and answer pairs for the given context using the given format.
Only answer in the format with no other text.
You should create the following number of question/answer pairs: 3.
When coming up with this question/answer pair, you must respond in the following format:
```
[
{{
"question": "$YOUR_QUESTION_HERE",
"answer": "$THE_ANSWER_HERE"
}},
{{
"question": "$YOUR_QUESTION_HERE",
"answer": "$THE_ANSWER_HERE"
}},
]
```

Everything between the ``` must be valid JSON.
Do not provide additional commentary and do not wrap your response in Markdown formatting. Return one-line, RAW, VALID JSON.
"""  # noqa: E501
    qa_generation_human_template = """请提供 3 个使用简体中文输出的问题/答案对，最终以一行 JSON 格式输出，针对以下文本：
----------------
{text}"""  # noqa: E501
    qa_generation_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(qa_generation_sys_template),
            HumanMessagePromptTemplate.from_template(qa_generation_human_template),
        ]
    )
    chain = QAGenerationChain.from_llm(llm, prompt=qa_generation_prompt)
    qa = chain.run(text)
    return qa[0]


@st.cache_data(show_spinner=False)
def process_question(text: str, question: str, chunk=4000) -> str:
    chain = LLMChain(llm=llm, prompt=question_process_prompt)
    return chain.run(context=text[:chunk], question=question)


st.set_page_config(
    page_title="A streamlit app for embedding documents using langchain",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "mailto:github@tssujt.xyz",
        "Report a bug": "mailto:github@tssujt.xyz",
    },
)
st.title("A streamlit app for embedding documents using langchain")

if "docs" not in st.session_state:
    st.session_state.docs = set()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_pairs" not in st.session_state:
    st.session_state.qa_pairs = []
if "doc_content" not in st.session_state:
    st.session_state.doc_content = ""


def load_uploaded_document(document: UploadedFile, docid: str):
    def generate_id_from_content(s: str) -> str:
        return str(uuid.UUID(hashlib.sha256(s.encode()).hexdigest()[::2]))

    if docid in st.session_state.docs:
        return

    temp_dir = tempfile.TemporaryDirectory()
    temp_pdf = os.path.join(temp_dir.name, document.name)
    with open(temp_pdf, mode="wb") as f:
        f.write(document.getbuffer())
    file_path = str(temp_pdf)
    st.info(f"Uploaded filename: {document.name}", icon="ℹ️")

    start = time.time()
    with st.status("Loading documents..."):
        if file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_path.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
        else:
            st.error("Unsupported file type")
            st.stop()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )
        docs: List[Document] = loader.load_and_split(splitter)

        doc_content = ""
        st.write("Adding documents to vector store")
        with catchtime("Adding documents to vector store"):
            for doc in docs:
                doc_content += doc.page_content
                vector_store.add_texts(
                    texts=[doc.page_content],
                    ids=[generate_id_from_content(doc.page_content)],
                    metadatas=[{"doc_id": docid}],
                )
        st.session_state.doc_content = doc_content

        st.write("Generating QA pairs")
        with catchtime("Generating QA pairs"):
            st.session_state.qa_pairs = generate_qa_pairs(
                " ".join([doc.page_content for doc in docs[:3]])
            )
        st.session_state.chat_history = []
        st.write("Done")

    end = time.time()
    st.write(f"Loaded document in {end - start} seconds")

    st.session_state.docs.add(docid)


with st.sidebar:
    uploaded_document: UploadedFile | None = st.file_uploader(
        "Choose a document", type=["pdf", "txt"]
    )

    if uploaded_document:
        md5sum = hashlib.md5(uploaded_document.getbuffer())
        docid = md5sum.hexdigest()
        metadata_filter = {"term": {"metadata.doc_id": docid}}

        load_uploaded_document(uploaded_document, docid)

    for qa in st.session_state.qa_pairs:
        st.write("Q: " + qa["question"])

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


q = st.chat_input("Ask something related to the document")
if q:
    if not uploaded_document:
        st.error("Please upload a document first")
        st.stop()

    qq = process_question(st.session_state.doc_content, q)
    print(q, qq)

    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    docs = vector_store.similarity_search(qq, filter=metadata_filter)

    # https://python.langchain.com/docs/modules/chains/document/stuff
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=qa_prompt, verbose=True)
    answer = chain.run(input_documents=docs, question=qq, stream=False)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)