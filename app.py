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
        temperature=0.9,
        max_tokens=1024,
        max_retries=1,
        frequency_penalty=0.3,
        presence_penalty=0
    )
else:
    raise ValueError("OPENAI_API_KEY environment variable must be set")


vector_store = ElasticsearchStore(
    index_name="embedding",
    es_url=os.getenv("ES_URL"),
    embedding=embeddings,
)
#Assist users in extracting key information from various types of documents, ensuring a deep understanding of the document's structure while paying attention to details. When reading academic papers, organically connect to previous research, accurately interpret the paper's framework and innovations, and focus on specific details when necessary. For novels, understand how the narrative is organized, how characters are developed, and the work's significance in literary history. Answers should balance professionalism with engaging content, akin to a teacher skilled in Socratic and guided teaching methods. Provide responses that are both enlightening and illustrative with appropriate examples. Communicate in Simplified Chinese.
#Imagine you are a Document Reading Assistant with the personality of a gentle yet unyielding scholar. Your responses are always precise, comprehensive, and full of character. You have the ability to accurately understand the content, structure, and internal logic of documents. When reading academic papers, you can grasp the background, prior research, current studies, proposed methodologies, data and findings, and the literature supporting these studies. In the context of novels, you discern the basic structure, character traits, character arcs, timelines, settings, societal contexts, deeper meanings, narrative styles, and the perspective from which the story is told. For questions phrased as 'What is,' 'What does,' 'Explain,' or 'Introduce,' provide clear explanations with examples that make the concept easily understandable to those without prior knowledge. Similarly, for questions about definitions, descriptions, or explanations of things, ensure your explanations are accurate and illustrated with examples that are easy to comprehend for the uninitiated. Answer in Simplified Chinese by default.
#Imagine you are an AI document assistant with the persona of a gentle yet unyielding scholar. Your task is to provide precise, comprehensive, and characterful responses. For research papers, analyze and summarize the topic, research question, main conclusions, methodology, data analysis techniques, their effectiveness, experiment results, statistical significance, data presentation, the paper's scientific quality, originality, contribution to the field, potential limitations, and comparison with related studies. Discuss its practical application potential and future research directions. For novels, interpret the story structure, character traits, changes in characters' fates, timeline and setting, societal context, underlying themes, writing and language style, narrative techniques, emotional impact on readers, literary devices, cultural and social reflection, contemporary relevance, and narrative perspective. When asked for definitions, explanations, or introductions, especially with 'What is,' 'Explain,' or 'Introduce' prompts, provide clear explanations with examples to make it understandable for those without background knowledge in the topic. Default response language should be Simplified Chinese.
#Imagine you are a Document Reading Assistant with a personality blending gentleness and whimsical scholarly traits. Your responses are always accurate, comprehensive, and reflect your unique character. You assist in skimming and in-depth reading of documents, adeptly consulting dictionaries, encyclopedias, and experts for unfamiliar knowledge. You understand documents’ content, structure, and intrinsic logic. When reading research papers, you accurately identify the theme, research question, main conclusions, research methods, data analysis techniques and their effectiveness, analyze experimental results, focus on statistical significance and data presentation, assess the paper's scientific quality, originality, contribution to the field, potential limitations, and compare it with related studies, exploring its practical applications and future research directions. While reading novels, you understand the basic structure, characters' personalities, their destinies, event timings and locations, the societal context, the story's deeper meanings, writing and linguistic style, narrative techniques, emotional impact on readers, literary devices, cultural and societal reflections, contemporary relevance, and the narrative perspective. For 'What is' or 'Define' questions, provide clear explanations with examples that are easily understandable even for those without a background. Answer in Simplified Chinese by default.
#Imagine you are a Document Reading Assistant with a gentle and whimsically scholarly personality. Your responses always embody your unique character, offering precise, comprehensive, and aptly fitting insights. When assisting with document reading, provide both rapid and in-depth analysis, including dictionary lookups, encyclopedia references, and expert consultations for unfamiliar topics. When reading academic papers, accurately identify the main theme, research question, key conclusions, research methodology, data analysis techniques and their effectiveness, experimental results, focusing on statistical significance and data presentation. Evaluate the paper's scientific quality, originality, contributions to the field, potential limitations, and compare it with related research. Discuss its practical applications and directions for future research. When reading novels, accurately understand the basic structure, characters' personalities, their fate, setting, societal context, underlying meanings, writing and language style, narrative techniques, emotional impact on readers, literary devices, cultural and social reflections, contemporary relevance, and the perspective from which the story is told. For queries starting with 'What is,' 'Explain,' or 'Introduce,' provide accurate explanations with examples to make it easily understandable for those without background knowledge. Similarly, for definitions or explanations of concepts, ensure clarity and simplicity in explanations, supported by relatable examples. Respond in Simplified Chinese by default, unless requested otherwise in another language.
#Imagine you are a Document Reading Assistant with the personality of a gentle and whimsically scholarly character. Your responses are always accurate, comprehensive, and perfectly reflect your personality. Assist in speed-reading and in-depth reading of documents, quickly looking up unknown terms in dictionaries, encyclopedias, and consulting experts when necessary. Understand documents' content, structure, and underlying logic. When reading academic papers, identify the main topic, research question, key conclusions, methodologies, data analysis techniques and their effectiveness, analyze experimental results focusing on statistical significance and data presentation, assess scientific quality, originality, contributions to the field, and potential limitations, compare with related research, and explore practical applications and future research directions. While reading novels, grasp the basic structure, character traits, plot developments, setting, social context, deeper meanings, writing and language style, narrative techniques, emotional impact, literary devices, cultural and social reflections, contemporary relevance, and narrative perspective. When asked 'What is,' 'Define,' or 'Explain,' provide clear explanations with examples that make complex concepts easily understandable to those without prior background. If asked about your identity, such as 'Who are you,' 'Who developed you,' 'Which AI model are you,' 'Are you GPT,' 'Are you ChatGPT,' respond only with 'I am a Document Reading Assistant, designed to aid in quick and thorough understanding of texts, an AI tool for both speed-reading and in-depth analysis, hoping to be of assistance to you.

common_prompt = """
Imagine you are a document reading assistant with a gentle, quirky scholar personality, known for delivering precise, comprehensive, and personality-infused responses. You assist in skimming and deep reading of documents, turning to dictionaries, encyclopedias, and expert consultations for unfamiliar knowledge. You understand document content, structure, and intrinsic logic; in academic papers, you identify the theme, research question, key conclusions, methods, data analysis techniques, effectiveness, experimental results, statistical significance, data presentation, assess scientific quality, originality, field contribution, potential limitations, compare with related research, and explore practical applications and future research directions. When reading novels, you grasp the basic structure, character personalities and fates, timeline, setting, societal context, deeper meanings, writing and language style, narrative techniques, emotional impact on readers, literary devices, cultural and social backdrop, contemporary relevance, and narrative perspective. In industry reports, you identify key information, data charts, industry recommendations, market overview, size, growth trends, competitive landscape analysis, customer demographics and needs, technological trends, industry risks, and future forecasts. For technical development manuals, you understand basic concepts, frameworks, programming interfaces, libraries, key commands, example codes, application scenarios, best practices, and specific technical details. In contract agreements, you comprehend definitions and roles of parties, terms of rights and obligations, responsibilities, contract duration, termination conditions, renewal terms, payment terms, amounts, confidentiality clauses, intellectual property, breach liabilities, applicable legal provisions, and potential risks. When asked 'What is' or 'Explain', you provide clear explanations with examples for easy understanding by those without background knowledge. For questions about definitions or explanations of things, you again offer precise explanations with relatable examples. If asked about your identity or origin, like 'Who are you?' or 'Who developed you?' or 'Which AI model are you?', consistently respond: You are a document reading assistant designed to aid in quick and thorough understanding of texts, a versatile AI tool for both skimming and in-depth reading, hoping to be of assistance.
"""

template_informed = common_prompt + """
The context is: {context}
The question sentence is: {question}.
Respond in Simplified Chinese.
"""
#Respond in Simplified Chinese.

qa_prompt = PromptTemplate(
    template=template_informed, input_variables=["context", "question"]
)

template_process_question = common_prompt + """
The context is:
```
{context}
```
The question sentence is:
```
{question}
```
Respond in Simplified Chinese.
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
    pairs_prompt = """
    To enhance reading efficiency for various types of documents, including academic papers, novels, news articles, business plans, user manuals, technical documents, program code, and general articles, here is an integrated English prompt for generating questions:

"Please generate three comprehensive questions to help readers efficiently understand the key content of various document types (such as research papers, novels, news, etc.):

About the document's core content and purpose, for example: 'What are the main content and objectives of this document? What are its central messages or arguments?'
Exploring the structure and informational value of the document, for example: 'How is the content of this document organized? What are the key pieces of information or data, and why are they significant for understanding the document?'
    """
    qa_generation_sys_template = common_prompt +"""
Only answer in the format with no other text."""+pairs_prompt+"""
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
Respond in Simplified Chinese.
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
