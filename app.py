import os
import shutil
import tempfile
from typing import List

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
import pandas as pd
import gc
import time


load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "...")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "...")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "...")

CHROMA_DIR = "chroma_db"
st.set_page_config(
    page_title="多格式文档知识库问答助手",
    page_icon="📚",
    layout="wide"
)

st.title("📚 多格式文档知识库问答助手")
st.caption("支持 PDF / TXT / MD 的本地 RAG 知识库问答助手")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []

if "show_debug" not in st.session_state:
    st.session_state.show_debug = False

if "top_k" not in st.session_state:
    st.session_state.top_k = 4

if "table_df" not in st.session_state:
    st.session_state.table_df = None

if "table_analysis" not in st.session_state:
    st.session_state.table_analysis = ""

if st.session_state.table_df is not None:
    st.subheader("📊 实验表格预览")
    st.dataframe(st.session_state.table_df, use_container_width=True)

    if st.session_state.table_analysis:
        st.subheader("📈 实验结果分析")
        st.markdown(st.session_state.table_analysis)

def load_uploaded_files(uploaded_files) -> List[Document]: #输出是 Document 对象组成的列表

    all_docs = []   #用来装所有从文档里面读出来的内容

    for uploaded_file in uploaded_files:  #遍历用户上传的文件，一个个处理
        file_name = uploaded_file.name   #获取上传的文件名，戴后缀
        file_ext = os.path.splitext(file_name)[1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:

            tmp_file.write(uploaded_file.getvalue())  #uploaded_file.getvalue()获取文件的的原始二进制内容
            tmp_path = tmp_file.name # tmp_file.name = 临时文件的完整路径，比如：C:\Users\xxx\AppData\Local\Temp\tmp1234.pdf

        try:

            if file_ext == ".pdf":
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()


            elif file_ext in [".txt", ".md", ".markdown"]:
                try:
                    loader = TextLoader(tmp_path, encoding="utf-8")
                    docs = loader.load()
                except UnicodeDecodeError:

                    loader = TextLoader(tmp_path, encoding="gbk")
                    docs = loader.load()

            else:#弹出警告，返回空列表/不处理
                st.warning(f"暂不支持该文件类型：{file_name}")
                docs = []

            # 给每个文档加上来源信息，给每一段文字都打标签
            for doc in docs:  #将之前的一堆 Document,一个个处理
                doc.metadata["source"] = file_name  #给文档打上【来源文件名】
                doc.metadata["file_type"] = file_ext  #记录是 .pdf、.txt 还是 .md


                # TXT / MD 没有页码，这里统一标记
                if "page" not in doc.metadata:
                    doc.metadata["page"] = None

            all_docs.extend(docs)

        except Exception as e: #页面显示红色报错，告诉你哪个文件坏了
            st.error(f"读取文件失败：{file_name}，错误信息：{e}")

        finally: #finally:不管前面成功还是失败，都一定会执行
            # 删除临时文件，避免电脑堆满垃圾文件
            if os.path.exists(tmp_path):  #如果这个临时文件存在
                os.remove(tmp_path)  #删除

    return all_docs



def split_documents(docs: List[Document]) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        #Recursive：递归地、逐级地 # Character：按字符/分隔符处理  # TextSplitter：文本切分器
        chunk_size=900,   #每个文本块大约900个字符
        chunk_overlap=150, #相邻文本块之间重叠150个字符
        separators=[  #切分优先级：优先按照：
                      # 段落 → 换行 → 中文句号 → 感叹号 → 问号 → 英文句号 → 空格 → 单字符
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            ".",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(docs)

    for i ,chunk in enumerate(chunks):
        chunk.metadata["chunk_id"]=i
    return chunks


# =========================
# 6. 构建 Chroma 向量库
# =========================

def build_vectorstore(chunks: List[Document]) -> Chroma:
    """
    使用 Ollama 的 embedding 模型将文本块向量化，并存入 Chroma。
    """

    # 先释放旧的 Chroma 对象，避免 Windows 下 chroma_db 文件被占用
    if "vectorstore" in st.session_state:
        st.session_state.vectorstore = None

    gc.collect()
    time.sleep(0.5)

    # 删除旧的向量数据库，防止新旧文档混在一起
    if os.path.exists(CHROMA_DIR):
        try:
            if os.path.isdir(CHROMA_DIR):
                shutil.rmtree(CHROMA_DIR)
            else:
                os.remove(CHROMA_DIR)
        except PermissionError:
            st.error(
                "旧的 Chroma 向量库文件仍被占用。请停止 Streamlit 后手动删除 chroma_db 文件夹，"
                "然后重新运行 streamlit run app.py。"
            )
            raise

    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="document_knowledge_base"
    )

    return vectorstore


# =========================
# 7. 检索相关文档
# =========================

def retrieve_docs(question: str, k: int = 4):

    vectorstore = st.session_state.vectorstore

    if vectorstore is None:
        return []

    results = vectorstore.similarity_search_with_score(question, k=k)

    return results


# =========================
# 8. 格式化上下文
# =========================

def format_context(docs: List[Document]) -> str:

    context_parts = []  #空列表，用来装整理好的内容

    for i, doc in enumerate(docs, start=1):  #循环遍历每一段检索到的文档，编号1，2，3
        # enumerate()函数： 一边遍历列表里的元素，一边自动给每个元素编号。
        source = doc.metadata.get("source", "未知文件")  #去 metadata 里找 "source" 这个标签，
                                            # 如果找到了 → 就用真实文件名；如果没找到 → 就用 “未知文件” 代替
        file_type = doc.metadata.get("file_type", "未知类型")
        page = doc.metadata.get("page", None)

        if page is not None:
            page_text = f"第 {page + 1} 页" #PyPDFLoader读取pdf从0开始的
        else:
            page_text = "无页码"

        context_parts.append(  #append = 把新的一段内容加入之前创建的列表里面。向context_parts 这个列表里添加一个新元素。
            f"【来源{i}】文件：{source}，类型：{file_type}，位置：{page_text}\n"
            f"{doc.page_content}"  #f"..." :表示这是一个 f-string，也就是格式化字符串。并且Python 允许相邻的字符串自动拼接。
                #【来源1】文件：手册.pdf，类型：.pdf，位置：第 2 页
                #RAG 的全称是检索增强生成，通过先检索再生成提高回答准确性。
        )

    return "\n\n".join(context_parts)  #用两个换行符（空一行）连接起来，
                                        # "\n\n" = 两个换行（等于空一行）
                                        # .join() = 把列表里的所有字符串，用这个符号"\n\n"连起来

def format_chat_history(messages,max_rounds:int = 3) -> str:
    """
        整理最近几轮聊天记录，作为多轮对话上下文。
        只取最近 max_rounds 轮，避免 prompt 太长。
        """
    if not messages:
        return "无历史对话"

    recent_messages = messages[-max_rounds*2:]
    history_parts=[]

    for msg in recent_messages:
        role=msg.get("role","")
        content=msg.get("content","")

        if role=="user":
            history_parts.append(f"用户：{content}")
        elif role=="assistant":
            history_parts.append(f"助手：{content}")

    return "\n".join(history_parts)


def generate_answer(question: str, retrieved_docs: List[Document],chat_history: str = "") -> str:

    if not retrieved_docs:
        return "没有检索到相关资料，请先确认文档是否已成功构建知识库。"

    context = format_context(retrieved_docs)

    prompt = f"""
你是一个严谨的知识库问答助手。请只基于给定资料回答用户问题。

要求：
1. 只能根据下面的“资料”回答问题。
2. 如果资料中没有答案，请明确说明“根据当前资料无法确定”。
3. 如果检索到的资料与用户问题相关性不足，请说明“当前检索内容与问题相关性不足”。
4. 不要编造资料中没有的信息。
5. 回答尽量清晰，可以分点说明。
6. 如果用户要求总结文档，请优先总结主题、核心内容、关键概念和结论。
7. 如果使用了某段资料，请在句子后标注来源编号，例如【来源1】。

历史对话：
{chat_history}
资料：
{context}

用户问题：
{question}

请回答：
"""

    llm = ChatOllama(   #连接 Ollama 中的问答模型
        model=OLLAMA_MODEL,   #用哪个模型
        base_url=OLLAMA_BASE_URL,  #本地Ollama地址
        temperature=0.1            #严谨度，越低，回答越保守，稳定，可复现
    )
        # ChatOllama：生成回答
        # OllamaEmbeddings：文本转向量
        # Chroma：保存和检索向量

    response = llm.invoke(prompt)   #让AI生成回答，.invoke()：调用，让它干活，执行
    return response.content

def show_sources(sources, text_limit: int = 800) -> None:
    """
    显示检索来源信息。
    同时支持：
    1. 普通 Document 列表：[doc1, doc2, ...]
    2. 带 score 的结果列表：[(doc1, score1), (doc2, score2), ...]
    """
    for i, item in enumerate(sources, start=1):
        # 情况 1：item 是 (Document, score)
        if isinstance(item, tuple):
            doc, score = item
        # 情况 2：item 是普通 Document
        else:
            doc = item
            score = None

        source = doc.metadata.get("source", "未知文件")
        file_type = doc.metadata.get("file_type", "未知类型")
        page = doc.metadata.get("page", None)
        chunk_id = doc.metadata.get("chunk_id", "未知")

        if page is not None:
            page_text = f"第 {page + 1} 页"
        else:
            page_text = "无页码"

        if score is not None:
            st.markdown(
                f"**来源 {i}：{source}，类型：{file_type}，{page_text}，chunk_id：{chunk_id}，score：{score:.4f}**"
            )
        else:
            st.markdown(
                f"**来源 {i}：{source}，类型：{file_type}，{page_text}，chunk_id：{chunk_id}**"
            )

        st.write(doc.page_content[:text_limit])
        st.divider()

def load_table_file(uploaded_table):
    """
    读取用户上传的 CSV / Excel 实验结果表格。
    """
    file_name = uploaded_table.name
    file_ext = os.path.splitext(file_name)[1].lower()

    try:
        if file_ext == ".csv":
            try:
                df = pd.read_csv(uploaded_table, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_table, encoding="gbk")

        elif file_ext in [".xlsx", ".xls"]:
            df = pd.read_excel(uploaded_table)

        else:
            st.error("暂不支持该表格格式，请上传 CSV / XLSX / XLS 文件。")
            return None

        return df

    except Exception as e:
        st.error(f"读取表格失败：{e}")
        return None

def analyze_experiment_table(df: pd.DataFrame) -> str:
    """
    简单分析实验结果表格：
    - 对于 RMSE / MAE / MSE，数值越小越好
    - 对于 CSI / HSS / SSIM / POD / FAR 以外的大多数指标，默认数值越大越好
    """
    if df is None or df.empty:
        return "表格为空，无法分析。"

    if "Model" in df.columns:
        model_col = "Model"
    elif "model" in df.columns:
        model_col = "model"
    else:
        model_col = df.columns[0]

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        return "没有检测到数值型指标列，无法进行自动比较。"

    lines = []
    lines.append("### 实验结果自动分析")
    lines.append("")
    lines.append(f"检测到模型列：`{model_col}`")
    lines.append(f"检测到数值指标：{', '.join(numeric_cols)}")
    lines.append("")

    for col in numeric_cols:
        lower_col = col.lower()

        # 误差类指标：越小越好
        if any(key in lower_col for key in ["rmse", "mae", "mse", "loss", "error"]):
            best_idx = df[col].idxmin()
            direction = "最低"
        else:
            best_idx = df[col].idxmax()
            direction = "最高"

        best_model = df.loc[best_idx, model_col]
        best_value = df.loc[best_idx, col]

        lines.append(
            f"- 在 `{col}` 指标上，`{best_model}` 取得{direction}值：`{best_value}`。"
        )

    lines.append("")
    lines.append("### 简要结论")
    lines.append("整体来看，可以重点关注在多个 CSI、HSS、SSIM 等指标上表现靠前，同时在 RMSE 等误差指标上较低的模型。")
    lines.append("如果所提出模型在高阈值 CSI/HSS 上优势明显，通常可以说明其对强降水或高强度目标的识别能力更好。")

    return "\n".join(lines)

# =========================
# 10. 侧边栏：上传文档和构建知识库
# =========================

with st.sidebar:  #所有写在这个里面的内容，都出现在网页的左边
    st.header("📄 文档上传")   #显示大标题

    st.markdown("当前模型配置：") #显示普通文字
    st.code(   #显示灰色代码框（展示配置）
        f"回答模型：{OLLAMA_MODEL}\n"
        f"Embedding模型：{OLLAMA_EMBED_MODEL}\n"
        f"Ollama地址：{OLLAMA_BASE_URL}"
    )

    #文件上传组件
    uploaded_files = st.file_uploader(  # st.file_uploader：创建一个上传文件的框
        "上传一个或多个文档，支持 PDF / TXT / MD", #提示文字，原样输出
        type=["pdf", "txt", "md", "markdown"], #只允许上传 PDF / TXT / MD
        accept_multiple_files=True      #=True，一次可以上传多个文件
    )

    build_button = st.button("🚀 构建知识库", use_container_width=True)


    if build_button:   #按钮被点击了
        if not uploaded_files:
            st.warning("请先上传文档文件。")
        else:
            st.session_state.messages = []
            st.session_state.vectorstore = None

            with st.spinner("正在读取文档、切分文本并构建向量库，请稍等..."):
                try:
                    docs = load_uploaded_files(uploaded_files) #调用函数，把 PDF / TXT / MD 里的文字全部读出来。赋给docs

                    if not docs:
                        st.error("没有成功读取到任何文档内容，请检查文件格式或文件编码。")
                    else:
                        chunks = split_documents(docs) #调用切片函数，切成一段一段小文本

                        if not chunks:
                            st.error("文档切分失败，没有生成有效文本块。")
                        else:
                            vectorstore = build_vectorstore(chunks)
                                    #调用构建向量库函数，把小块文本变成向量并存到向量库里面

                            st.session_state.vectorstore = vectorstore  # 保存向量库
                            st.session_state.doc_count = len(docs)  # # # 记录读取出的 Document 数量：PDF 通常按页计数，TXT/MD 通常按文件计数
                            st.session_state.chunk_count = len(chunks)  #保存文本块数量
                            st.session_state.messages = []              #清空聊天
                            st.session_state.uploaded_file_names = [
                                file.name for file in uploaded_files
                            ]

                            st.success("知识库构建完成！")

                except Exception as e:
                    st.error(f"构建知识库失败：{e}")

    st.divider()
    st.header("📊 实验表格分析")

    uploaded_table = st.file_uploader(
        "上传实验结果表格，支持 CSV / XLSX / XLS",
        type=["csv", "xlsx", "xls"],
        key="table_uploader"
    )

    analyze_button = st.button("📈 分析实验表格", use_container_width=True)

    if analyze_button:
        if uploaded_table is None:
            st.warning("请先上传 CSV / Excel 表格。")
        else:
            df = load_table_file(uploaded_table)

            if df is not None:
                st.session_state.table_df = df
                st.session_state.table_analysis = analyze_experiment_table(df)
                st.success("表格分析完成！")

    st.divider()  #st.divider()画一条横线（分割线）

    st.subheader("📊 当前知识库状态")  #显示一个小标题
    st.write(f"文档页数 / 文件条目数：{st.session_state.doc_count}")  #显示读取了多少个文档
    st.write(f"文本块数量：{st.session_state.chunk_count}")  #显示文档被切成了多少个小块

    st.session_state.show_debug=st.checkbox(
        "显示检索调试信息",
         value=st.session_state.show_debug
    )

    st.session_state.top_k=st.slider(
        "检索文本块数量 top_k",
        min_value=1,
        max_value=10,
        value=st.session_state.top_k,
        step=1
    )

    if st.session_state.uploaded_file_names:
        st.markdown("已加载文件：")
        for file_name in st.session_state.uploaded_file_names:
            st.write(f"- {file_name}")

    if st.button("🧹 清空对话", use_container_width=True):
                        #use_container_width=True 让按钮宽度 = 所在区域的最大宽度，现在就是侧边栏宽度
        st.session_state.messages = []
        st.rerun()  #刷新页面，让清空立刻生效



for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("查看来源"):
                show_sources(message["sources"])


question = st.chat_input("请输入你想问文档的问题...")


if question:
    if st.session_state.vectorstore is None:
        st.warning("请先在左侧上传文档，并点击“构建知识库”。")
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("正在检索资料并生成回答..."):
                try:
                    retrieved_results = retrieve_docs(question,k=st.session_state.top_k)
                    retrieved_docs =[ doc for doc,score in retrieved_results]
                    chat_history =format_chat_history(st.session_state.messages,max_rounds=3)
                    answer = generate_answer(question, retrieved_docs,chat_history)

                    st.markdown(answer)

                    with st.expander("查看来源"):
                        show_sources(retrieved_results)

                    # 2. 检索调试信息
                    if st.session_state.show_debug:
                        with st.expander("检索调试信息", expanded=True):
                            st.write(f"用户问题：{question}")
                            st.write(f"检索数量：{len(retrieved_results)}")
                            st.write(f"当前top_k：{st.session_state.top_k}")
                            show_sources(retrieved_results, text_limit=300)



                    # 保存助手消息
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieved_results
                    })

                except Exception as e:  #如果出错，显示错误，不崩溃
                    error_msg = f"回答生成失败：{e}"  #组合一段错误提示文字
                    st.error(error_msg) #在网页界面上，显示一个红色的错误提示框！显示给用户

                    st.session_state.messages.append({  #往聊天记录里添加一条：【AI 助手发送的 → 错误提示消息 → 不带任何参考资料】
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })
