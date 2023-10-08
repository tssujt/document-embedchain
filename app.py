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
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain.vectorstores.milvus import Milvus
from streamlit.runtime.uploaded_file_manager import UploadedFile

if not (openai_api_key := os.getenv("OPENAI_API_KEY")):
    raise ValueError("OPENAI_API_KEY environment variable must be set")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-3.5-turbo-16k",
    temperature=0,
    max_tokens=1000,
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
I will provide a question sentence in any language and a context, you will translate the question sentence in which language the context I wrote is in you. Do not write any explanations or other words, just translate the question sentence.

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
def generate_qa_pairs(text: str, k: int = 1, chunk=1000) -> List[Dict[str, str]]:
    if k < 1:
        return []

    templ1 = (
        """您是一个智能助手，能帮助提供问题和答案。无论上下文用什么语言，问题和答案都只能是简体中文。
给定一段上下文文本，您必须提出"""
        + str(k)
        + """个问题和答案对。
在提出这个问题/答案对时，请按照以下格式回答：
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

```之间的所有内容必须是有效的 JSON。
"""
    )
    templ2 = (
        "请提供"
        + str(k)
        + """个问题/答案对，以指定的 JSON 格式，针对以下文本：
----------------
{text}"""
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(templ1),
            HumanMessagePromptTemplate.from_template(templ2),
        ]
    )

    chain = QAGenerationChain.from_llm(llm, prompt=prompt)

    qa = chain.run(text[:chunk])
    return qa[0]


@st.cache_data(show_spinner=False)
def process_question(text: str, question: str, chunk=1000) -> str:
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


def load_uploaded_document(vector_db: VectorStore, document: UploadedFile, docid: str):
    def generate_id_from_content(s: str) -> str:
        return uuid.UUID(hashlib.sha256(s.encode()).hexdigest()[::2])

    if vector_db.__class__.__name__ + docid in st.session_state.docs:
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
            loader = UnstructuredPDFLoader(file_path)
        else:
            st.error("Unsupported file type")
            st.stop()

        documents = loader.load()

        st.write("Splitting documents")
        with catchtime("Splitting documents"):
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs: List[Document] = text_splitter.split_documents(documents)

        doc_content = ""
        st.write("Adding documents to vector store")
        with catchtime("Adding documents to vector store"):
            for doc in docs:
                doc_content += doc.page_content
                vector_db.add_texts(
                    texts=[doc.page_content],
                    ids=[generate_id_from_content(doc.page_content)],
                    metadatas=[{"doc_id": docid}],
                )
        st.session_state.doc_content = doc_content

        st.write("Generating QA pairs")
        with catchtime("Generating QA pairs"):
            st.session_state.qa_pairs = generate_qa_pairs(doc_content, k=5)
        st.session_state.chat_history = []
        st.write("Done")

    end = time.time()
    st.write(f"Loaded document in {end - start} seconds")

    st.session_state.docs.add(vector_db.__class__.__name__ + docid)


with st.sidebar:
    st.sidebar.header("Vector store settings")
    db_type = st.sidebar.selectbox("DB Type", ("Elasticsearch", "Milvus", "Chroma"))
    uploaded_document: UploadedFile | None = st.file_uploader(
        "Choose a document", type=["pdf", "txt"]
    )

    if uploaded_document:
        md5sum = hashlib.md5(uploaded_document.getbuffer())
        docid = md5sum.hexdigest()

        metadata_filter = {"doc_id": docid}
        if db_type == "Elasticsearch":
            vector_store = ElasticsearchStore(
                index_name="embedding",
                es_url=os.getenv("ES_URL"),
                embedding=embeddings,
            )
            metadata_filter = {"term": {"metadata.doc_id": docid}}
        elif db_type == "Milvus":
            vector_store = Milvus(
                embeddings, connection_args={"address": os.getenv("MILVUS_ADDRESS")}
            )
        elif db_type == "Chroma":
            vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY"),
            )
        else:
            raise ValueError(f"Unknown db type {db_type}")

        load_uploaded_document(vector_store, uploaded_document, docid)

    for qa in st.session_state.qa_pairs:
        st.write("Q: " + qa["question"])
        st.write("A: " + qa["answer"])

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
