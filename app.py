import os
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv

from agents import CtrlPassAgent
from utils.document_processor import DocumentProcessor
from utils.exercise_generator import ExerciseGenerator
from utils.knowledge_mapper import KnowledgeMapper
from utils.qa_chain import QAChain
from utils.vector_store import VectorStoreManager

load_dotenv()

st.set_page_config(
    page_title="CtrlPass",
    page_icon="CP",
    layout="wide",
)


def ensure_session_state() -> None:
    defaults = {
        "vector_store_manager": None,
        "qa_chain": None,
        "exercise_generator": None,
        "knowledge_mapper": None,
        "agent": None,
        "processed_docs": False,
        "current_file_path": None,
        "chat_history": [],
        "agent_api_key": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_agent(api_key: str) -> None:
    if not (
        st.session_state.vector_store_manager
        and st.session_state.qa_chain
        and st.session_state.exercise_generator
        and st.session_state.knowledge_mapper
    ):
        st.session_state.agent = None
        return

    st.session_state.agent = CtrlPassAgent(
        qwen_api_key=api_key,
        vector_store_manager=st.session_state.vector_store_manager,
        qa_chain=st.session_state.qa_chain,
        exercise_generator=st.session_state.exercise_generator,
        knowledge_mapper=st.session_state.knowledge_mapper,
    )
    st.session_state.agent_api_key = api_key


def save_uploaded_file(uploaded_file) -> str:
    session_id = str(uuid.uuid4())
    os.makedirs("data", exist_ok=True)
    file_name, file_ext = os.path.splitext(uploaded_file.name)
    file_path = os.path.join("data", f"{file_name}_{session_id}{file_ext}")
    with open(file_path, "wb") as file_handle:
        file_handle.write(uploaded_file.getbuffer())
    return file_path


def process_document(uploaded_file, api_key: str, chunk_size: int, chunk_overlap: int, use_model_splitter: bool) -> None:
    with st.spinner("Processing document..."):
        file_path = save_uploaded_file(uploaded_file)
        doc_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_model_splitter=use_model_splitter,
        )
        documents = doc_processor.load_document(file_path)
        chunks = doc_processor.split_documents(documents)

        vector_store_manager = VectorStoreManager(api_key)
        vector_store_manager.create_vector_store(chunks, os.path.splitext(os.path.basename(file_path))[0].split("_")[-1])
        qa_chain = QAChain(api_key, vector_store_manager)
        exercise_generator = ExerciseGenerator(api_key)
        knowledge_mapper = KnowledgeMapper(api_key)

        st.session_state.vector_store_manager = vector_store_manager
        st.session_state.qa_chain = qa_chain
        st.session_state.exercise_generator = exercise_generator
        st.session_state.knowledge_mapper = knowledge_mapper
        st.session_state.processed_docs = True
        st.session_state.current_file_path = file_path
        st.session_state.chat_history = []

        build_agent(api_key)
        st.success(f"Document processed into {len(chunks)} chunks. Agent mode is ready.")


def render_mindmap(mindmap_data) -> None:
    net = st.session_state.knowledge_mapper.create_network(mindmap_data)
    html_path = os.path.join(tempfile.gettempdir(), "mindmap.html")
    net.save_graph(html_path)
    with open(html_path, "r", encoding="utf-8") as file_handle:
        html_content = file_handle.read()
    st.components.v1.html(html_content, height=600)
    try:
        os.remove(html_path)
    except OSError:
        pass


def render_agent_result(result: dict) -> None:
    if result.get("tool_name"):
        st.caption(f"Tool: `{result['tool_name']}`")
    if result.get("tool_reason"):
        st.caption(f"Reason: {result['tool_reason']}")

    display_type = result.get("display_type", "text")
    tool_result = result.get("tool_result")

    if display_type == "error":
        st.error(result.get("error") or tool_result or "Unknown error")
    elif display_type == "context":
        st.markdown("### Retrieved Context")
        st.write(tool_result)
    elif display_type == "exercises":
        st.markdown("### Exercises")
        for index, exercise in enumerate(tool_result or [], start=1):
            st.write(f"#### Exercise {index}")
            st.write(f"**Question:** {exercise['question']}")
            st.write(f"**Type:** {exercise['type']}")
            if exercise.get("options"):
                for option in exercise["options"]:
                    st.write(f"- {option}")
            with st.expander("Answer and Explanation"):
                st.write(f"**Answer:** {exercise['answer']}")
                st.write(f"**Explanation:** {exercise['explanation']}")
    elif display_type == "mindmap":
        st.markdown("### Knowledge Map")
        render_mindmap(tool_result)
    else:
        st.write(tool_result)

    messages = result.get("messages", [])
    if messages:
        with st.expander("Agent trace"):
            for message in messages:
                st.write(f"- {message}")


ensure_session_state()

with st.sidebar:
    st.title("Configuration")
    env_api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY") or ""
    api_key = st.text_input("Qwen API Key", type="password", value=env_api_key)
    uploaded_file = st.file_uploader("Upload document", type=["txt", "pdf", "docx"])
    st.subheader("Chunk settings")
    use_model_splitter = st.checkbox(
        "Use model-based splitter",
        value=False,
        help="This keeps semantic chunks more coherent but may take longer.",
    )
    chunk_size = st.slider("Chunk size", 100, 2000, 500)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 100)
    process_button = st.button("Process document")

    if process_button:
        if not uploaded_file:
            st.warning("Upload a document first.")
        elif not api_key:
            st.warning("Enter a valid API key first.")
        else:
            try:
                process_document(
                    uploaded_file,
                    api_key,
                    chunk_size,
                    chunk_overlap,
                    use_model_splitter,
                )
            except Exception as exc:
                st.error(f"Failed to process document: {exc}")

st.title("CtrlPass")
st.write("Chat with your document through a StateGraph-powered agent.")

if st.session_state.processed_docs and api_key and (
    st.session_state.agent is None or st.session_state.agent_api_key != api_key
):
    build_agent(api_key)

if st.session_state.current_file_path:
    st.caption(f"Current document: `{os.path.basename(st.session_state.current_file_path)}`")

for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["user"])
    with st.chat_message("assistant"):
        render_agent_result(entry["result"])

prompt = st.chat_input("Ask a question, request exercises, or generate a knowledge map...")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    if not st.session_state.agent:
        result = {
            "tool_name": "",
            "tool_reason": "",
            "tool_result": "Please upload and process a document before chatting with the agent.",
            "display_type": "error",
            "error": "Please upload and process a document before chatting with the agent.",
            "messages": ["Agent invocation blocked because no processed document is available."],
        }
    else:
        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                result = st.session_state.agent.run(
                    prompt,
                    st.session_state.current_file_path,
                )
                render_agent_result(result)

    if not st.session_state.agent:
        with st.chat_message("assistant"):
            render_agent_result(result)

    st.session_state.chat_history.append(
        {
            "user": prompt,
            "result": result,
        }
    )
