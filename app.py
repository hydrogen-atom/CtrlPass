import streamlit as st
import os
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStoreManager
from utils.qa_chain import QAChain
from utils.exercise_generator import ExerciseGenerator
import uuid
import time

# 设置页面配置
st.set_page_config(
    page_title="RAG问答系统 (通义千问)",
    page_icon="❓",
    layout="wide"
)

# 初始化session state
if "vector_store_manager" not in st.session_state:
    st.session_state.vector_store_manager = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = False
if "exercise_generator" not in st.session_state:
    st.session_state.exercise_generator = None
if "generated_exercises" not in st.session_state:
    st.session_state.generated_exercises = []
if "current_file_path" not in st.session_state:
    st.session_state.current_file_path = None

# 侧边栏配置
with st.sidebar:
    st.title("配置")
    qwen_api_key = st.text_input("通义千问 API Key", type="password")
    uploaded_files = st.file_uploader("上传文档", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    
    st.subheader("文档分块设置")
    question_type = st.selectbox(
        "问题类型",
        ["factual", "inferential", "summary", "comparison", "definition", "procedural", "opinion"],
        help="选择问题类型以优化文档分块策略"
    )
    chunk_size = st.slider("分块大小", 100, 2000, 500, help="每个文本块的最大字符数")
    chunk_overlap = st.slider("分块重叠", 0, 500, 100, help="相邻文本块之间的重叠字符数")
    process_button = st.button("处理文档")

# 主界面
st.title("RAG问答系统 (通义千问)")
st.write("上传文档并开始提问！")

# 处理上传的文档
if process_button and uploaded_files and qwen_api_key:
    try:
        with st.spinner("正在处理文档..."):
            # 创建会话ID
            session_id = str(uuid.uuid4())
            os.makedirs("data", exist_ok=True)
            
            # 处理所有上传的文件
            all_chunks = []
            processed_files = []
            
            for uploaded_file in uploaded_files:
                # 保存上传的文件
                file_path = os.path.join("data", f"{session_id}_{uploaded_file.name}")
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                processed_files.append(file_path)
                
                st.info(f"正在处理文件: {uploaded_file.name}")
                
                # 处理文档
                doc_processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                documents = doc_processor.load_document(file_path)
                chunks = doc_processor.split_documents(documents)
                all_chunks.extend(chunks)
                
                st.info(f"文件 {uploaded_file.name} 已分割成 {len(chunks)} 个块")
            
            st.info(f"所有文件共分割成 {len(all_chunks)} 个块，开始创建向量存储...")
            
            # 创建向量存储
            vector_store_manager = VectorStoreManager(qwen_api_key)
            vector_store_manager.create_vector_store(all_chunks, session_id)
            
            st.info("向量存储创建完成，正在初始化问答系统...")
            
            # 创建QA链
            qa_chain = QAChain(qwen_api_key, vector_store_manager.vector_store)
            
            # 创建练习生成器
            exercise_generator = ExerciseGenerator(qwen_api_key)
            
            # 更新session state
            st.session_state.vector_store_manager = vector_store_manager
            st.session_state.qa_chain = qa_chain
            st.session_state.exercise_generator = exercise_generator
            st.session_state.processed_docs = True
            st.session_state.processed_files = processed_files
            
            st.success(f"成功处理 {len(uploaded_files)} 个文件！现在可以开始提问或生成练习题。")
        
    except Exception as e:
        st.error(f"处理文档时出错: {str(e)}")
        st.error("请检查：")
        st.error("1. 通义千问 API Key 是否正确")
        st.error("2. 网络连接是否正常")
        st.error("3. 文档大小是否合适（建议每个文件小于10MB）")
        st.error("4. 文档格式是否正确（支持txt、pdf、docx）")

# 创建两列布局
col1, col2 = st.columns(2)

# 问答界面
with col1:
    st.subheader("问答功能")
    if st.session_state.processed_docs:
        question = st.text_input("输入你的问题:")
        question_type = st.selectbox(
            "问题类型",
            ["factual", "inferential", "summary", "comparison", "definition", "procedural", "opinion"],
            help="选择问题类型以获得更精准的回答"
        )
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if question and st.button("提交问题"):
                try:
                    with st.spinner("正在思考..."):
                        answer = st.session_state.qa_chain.answer_question(question, question_type)
                        if answer["status"] == "success":
                            st.write("回答:")
                            st.write(answer["answer"])
                            
                            # 显示相关文档片段
                            with st.expander("查看相关文档片段"):
                                for i, source in enumerate(answer["sources"], 1):
                                    st.write(f"片段 {i} (相关度: {source['relevance_score']:.2f}):")
                                    st.write(source["content"])
                                    st.write("---")
                        else:
                            st.error(answer["message"])
                except Exception as e:
                    st.error(f"获取答案时出错: {str(e)}")
                    st.error("请检查网络连接或稍后重试")
        with col_btn2:
            if st.button("清除历史记录"):
                st.session_state.qa_chain.clear_history()
                st.success("历史记录已清除！")
    else:
        st.info("请先上传并处理文档。")

# 练习生成界面
with col2:
    st.subheader("练习生成")
    if st.session_state.processed_docs:
        if st.button("生成练习题"):
            try:
                with st.spinner("正在生成练习题..."):
                    # 获取所有文档内容
                    doc_processor = DocumentProcessor()
                    all_content = []
                    
                    for file_path in st.session_state.processed_files:
                        documents = doc_processor.load_document(file_path)
                        if not documents:
                            st.warning(f"无法加载文件内容: {os.path.basename(file_path)}")
                            continue
                        content = "\n".join([doc.page_content for doc in documents])
                        if content.strip():
                            all_content.append(content)
                    
                    if not all_content:
                        st.error("无法加载任何文档内容")
                        st.stop()
                    
                    # 合并所有文档内容
                    combined_content = "\n\n".join(all_content)
                    
                    if not combined_content.strip():
                        st.error("文档内容为空")
                        st.stop()
                    
                    st.info(f"合并后的文档内容长度: {len(combined_content)} 字符")
                    
                    # 生成练习题
                    exercises = st.session_state.exercise_generator.generate_exercises(combined_content)
                    
                    if not exercises:
                        st.error("未能生成练习题，请检查文档内容或稍后重试")
                        st.stop()
                    else:
                        st.balloons()
                    
                    st.session_state.generated_exercises = exercises
                    
                    # 显示练习题
                    for i, exercise in enumerate(exercises, 1):
                        st.write(f"### 练习 {i}")
                        st.write(f"**题目：** {exercise['question']}")
                        st.write(f"**类型：** {exercise['type']}")
                        
                        if exercise['type'] == '选择题':
                            for option in exercise['options']:
                                st.write(f"- {option}")
                        
                        with st.expander("查看答案和解析"):
                            st.write(f"**正确答案：** {exercise['answer']}")
                            st.write(f"**解析：** {exercise['explanation']}")
                    
            except Exception as e:
                st.error(f"生成练习题时出错: {str(e)}")
                st.error("请检查网络连接或稍后重试")
    else:
        st.info("请先上传并处理文档。")
