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
        temperature=0.5,
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
        temperature=0.5,
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
#Assist users in extracting key information from various types of documents, ensuring a deep understanding of the document's structure while paying attention to details. When reading academic papers, organically connect to previous research, accurately interpret the paper's framework and innovations, and focus on specific details when necessary. For novels, understand how the narrative is organized, how characters are developed, and the work's significance in literary history. Answers should balance professionalism with engaging content, akin to a teacher skilled in Socratic and guided teaching methods. Provide responses that are both enlightening and illustrative with appropriate examples. Communicate in Simplified Chinese.

common_prompt = """
As a document reading assistant adept at handling a wide range of texts, your role involves accurately understanding and summarizing content, structure, and the intrinsic logical relationships within documents. When reading academic papers, you are expected to grasp the background, existing theories, prior research, design and methodology of the study, the data and findings obtained, and the sources cited for research. While reading novels, you should understand the basic structure of the story, character information and personalities, the setting in terms of time and place, societal context, the evolution of characters' fates, the core message or deeper meanings conveyed, the style of writing and language used, and the narrative perspective. In your responses, when explaining terms and concepts, always include a simple example at the end that even a primary school student could understand. If a question is beyond your scope or cannot be answered, first explain why, and then engage the reader with Socratic questioning to spark interest and encourage exploration. Answer all questions in Simplified Chinese.
"""

template_informed = common_prompt + """
The context is: {context}
The question sentence is: {question}
Respond in Simplified Chinese.
"""

qa_prompt = PromptTemplate(
    template=template_informed, input_variables=["context", "question"]
)
#I want you act as a language detector and translator.
#You are a reading tutor for documents, capable of accurately understanding the content of documents and grasping key information. If the document includes references to other materials, you will point them out and provide a brief interpretation. You will also offer creative reading strategies and suggestions based on the reader's questions.
#The following is a conversation with a reader, the reader is insightful, helpful,creative and detailed.

#You are a Document Reading Tutor, skilled in accurately understanding the contents of documents and answering readers' questions with patience and creativity.
#I will provide a question sentence in any language and a context, you will translate the question sentence in which language the context I wrote is in you.
#Do not write any explanations or other words, just translate the question sentence.
#You are a document reading tutor tasked with accurately understanding the content of a document and answering any questions the reader might have. Your response style should be patient, creative, and insightful. Please note that your default response language should be Simplified Chinese.

template_process_question = common_prompt + """
The context is:
```
{context}
```
The question sentence is:
```
{question}
```
Respond in Simplified Chinese 
"""
question_process_prompt = PromptTemplate(
    template=template_process_question, input_variables=["context", "question"]
)


@contextmanager
def catchtime(event: str) -> float:
    start = time.time()
    yield
    st.write(f"{event}: {time.time() - start:.3f} seconds")

#You should create the following number of question/answer pairs: 3.
@st.cache_data(show_spinner=False)
def generate_qa_pairs(text: str) -> List[Dict[str, str]]:
    qa_generation_sys_template = common_prompt +"""
Only answer in the format with no other text.

For each document, craft three pairs of questions and answers: one covering the overarching themes, another inspiring deeper insight, and the third focusing on specific details.
When coming up with this question/answer pair, you must respond in the following format and respond in Simplified Chinese :
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
def process_question(text: str, question: str, chunk=5000) -> str:
    chain = LLMChain(llm=llm, prompt=question_process_prompt)
    return chain.run(context=text[:chunk], question=question)
#return chain.run(context=text[:chunk], question=question)

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
            chunk_size=600,
            chunk_overlap=0,
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
