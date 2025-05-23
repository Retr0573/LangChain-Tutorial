# Tutorial of LangChain

> ref: https://python.langchain.com/docs/tutorials/
> 前置知识：LLM，Prompt，Embedding，RAG等基础概念

## 1. Chat Model 和 Prompt Template
首先我们熟悉下Langchain中封装的LLM模型和Prompt模板的使用，这是后面构建LLM应用的基础中的基础，即如何向一个LLM提供基础的上下文输入。

### 1.1 定义一个chat模型（init_chat_model）


在这里，我们使用 langchain 提供的封装方法 init_chat_model 来构建一个可用的聊天模型实例。通过加载 .env 文件中的环境变量（如 OPENAI_API_KEY 和 OPENAI_API_BASE），实现了与模型服务的动态连接。本示例中采用的是兼容 OpenAI 接口格式的星火大模型（x1），便于直接复用 OpenAI 的 API 调用逻辑，从而实现无缝接入国产大模型服务。


```python
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
# 读取 .env 文件中的环境变量, 我们可以在 .env 文件中设置 OPENAI_API_KEY，文件内容类似于： OPENAI_API_KEY=your_api_key
load_dotenv() 
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv('OPENAI_API_BASE')

# 调用langchain的封装init_chat_model方法，初始化一个chat模型
# 这里我用的是星火x1模型，x1兼容openai的调用格式，所以可以直接调用openai的接口
# base_url和api_key是从.env 文件中读取的, 来自星火开放平台
model = init_chat_model(model = "x1", model_provider="openai",
                        base_url=base_url, api_key=api_key)
```

### 1.2 用Message构建输入prompt
Langchain中的一个常见的输入prompt的方式是构建一个messages列表（里面一般是一些Message对象），然后将messages作为参数传入invoke

其实直接在invoke中传入字符串或者openai格式的json也是可以的


```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="你是一个翻译家，根据输入的中文翻译成英文"),
    HumanMessage(content="你好，现在是周五的晚上，我在学习langchain。"),
]

AIMessage = model.invoke(messages)
print(AIMessage)
```

    content="Hello, it's currently Friday night, and I'm studying LangChain." additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 529, 'prompt_tokens': 23, 'total_tokens': 552, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': None, 'system_fingerprint': None, 'id': None, 'service_tier': None, 'finish_reason': None, 'logprobs': None} id='run--d75838c7-e3ab-4974-9c77-cc6b440e5c30-0' usage_metadata={'input_tokens': 23, 'output_tokens': 529, 'total_tokens': 552, 'input_token_details': {}, 'output_token_details': {}}


### 1.3 用Prompt templates管理对话上下文

虽然可以直接用Message对象来构建prompt输入，但一般会使用Prompt Templates来做Message格式的预定义。

在Langchain中，Prompt Templates 用于将用户输入和参数转化为对语言模型的指令。这可以用来引导模型的响应，帮助其理解上下文，并生成相关且连贯的语言输出。

- Prompt Templates 的输入是一个字典，其中每个键代表提示模板中需要填充的一个变量。
- Prompt Templates 的输出是一个 PromptValue。这个 PromptValue 可以传递给 LLM 或 ChatModel，也可以转换为字符串或消息列表。
- PromptValue 的设计目的是为了方便在字符串和消息之间进行切换。

#### 1.3.1 PromptTemplate

Prompt templates有很多不同的类，PromptTemplate是最简单的一个：


```python
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")
# prompt_template = PromptTemplate.from_file("prompt_template.txt")  # 也可以从文件加载
prompt_value = prompt_template.invoke({"topic": "cats"})

print(type(prompt_value))
print("-------------")
print(prompt_value)
print("-------------")
print(prompt_value.to_string())
print("-------------")
print(prompt_value.to_messages())
```

    <class 'langchain_core.prompt_values.StringPromptValue'>
    -------------
    text='Tell me a joke about cats'
    -------------
    Tell me a joke about cats
    -------------
    [HumanMessage(content='Tell me a joke about cats', additional_kwargs={}, response_metadata={})]



```python
model.invoke(prompt_value)
```




    AIMessage(content='Why did the cat become a detective?  \nBecause it was good at **pawing** through clues! 😸', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 741, 'prompt_tokens': 6, 'total_tokens': 747, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': None, 'system_fingerprint': None, 'id': None, 'service_tier': None, 'finish_reason': None, 'logprobs': None}, id='run--f2105269-f4c7-4331-a63f-c309e19c187b-0', usage_metadata={'input_tokens': 6, 'output_tokens': 741, 'total_tokens': 747, 'input_token_details': {}, 'output_token_details': {}})



#### 1.3.2 ChatPromptTemplates

ChatPromptTemplates 用于格式化一组消息（messages）。这些“模板”本身由多个子模板组成。构建和使用 ChatPromptTemplate 的常见方式如下：


```python
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

prompt_template = ChatPromptTemplate([
    SystemMessagePromptTemplate.from_template("You are a helpful assistant that translates {input_language} to {output_language}."),
    HumanMessagePromptTemplate.from_template("{sentence}")
])

prompt_value = prompt_template.invoke({
    "input_language": "English", 
    "output_language": "French", 
    "sentence": "I love programming."
    })

print(type(prompt_value))
print("-------------")
print(prompt_value)
print("-------------")
print(prompt_value.to_string())
print("-------------")
print(prompt_value.to_messages())
```

    <class 'langchain_core.prompt_values.ChatPromptValue'>
    -------------
    messages=[SystemMessage(content='You are a helpful assistant that translates English to French.', additional_kwargs={}, response_metadata={}), HumanMessage(content='I love programming.', additional_kwargs={}, response_metadata={})]
    -------------
    System: You are a helpful assistant that translates English to French.
    Human: I love programming.
    -------------
    [SystemMessage(content='You are a helpful assistant that translates English to French.', additional_kwargs={}, response_metadata={}), HumanMessage(content='I love programming.', additional_kwargs={}, response_metadata={})]



```python
ai_message = model.invoke(prompt_value)
print(ai_message)
print(ai_message.content)
```

    content='**French Translation:**  \n**"J\'aime programmer."**  \n\nThis translates directly while preserving the active sense of enjoying the activity. If emphasizing enthusiasm, you could also say:  \n**"J\'adore programmer!"** (*"I adore programming!"*)  \n\nBoth are natural and context-appropriate. 😊' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 703, 'prompt_tokens': 16, 'total_tokens': 719, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': None, 'system_fingerprint': None, 'id': None, 'service_tier': None, 'finish_reason': None, 'logprobs': None} id='run--6c6f0e1f-b0c1-4f96-a867-59d2d1f20ad8-0' usage_metadata={'input_tokens': 16, 'output_tokens': 703, 'total_tokens': 719, 'input_token_details': {}, 'output_token_details': {}}
    **French Translation:**  
    **"J'aime programmer."**  
    
    This translates directly while preserving the active sense of enjoying the activity. If emphasizing enthusiasm, you could also say:  
    **"J'adore programmer!"** (*"I adore programming!"*)  
    
    Both are natural and context-appropriate. 😊


#### 1.3.3 MessagesPlaceholder
这个提示模板的作用是在特定位置插入一组消息。在上面的 ChatPromptTemplate 中，我们看到如何格式化两条消息，其中每个消息都是一个MessagePromptTemplate对象，可以让用户传入特定的字符串参数比如{input_language}。但如果我们希望用户直接传入一组消息，并将它们插入到提示中的某个位置，该怎么办呢？这时就可以使用 MessagesPlaceholder。


```python
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage,SystemMessage
prompt_template = ChatPromptTemplate([
    MessagesPlaceholder(variable_name="history"),
])
history=[
    SystemMessage(content="你是一个翻译机器人，将英文翻译成中文"),
    HumanMessage(content="你好")
]
prompt_value = prompt_template.invoke(
    {"history":history}
)

print(type(prompt_value))
print("-------------")
print(prompt_value)
print("-------------")
print(prompt_value.to_string())
print("-------------")
print(prompt_value.to_messages())
```

    <class 'langchain_core.prompt_values.ChatPromptValue'>
    -------------
    messages=[SystemMessage(content='你是一个翻译机器人，将英文翻译成中文', additional_kwargs={}, response_metadata={}), HumanMessage(content='你好', additional_kwargs={}, response_metadata={})]
    -------------
    System: 你是一个翻译机器人，将英文翻译成中文
    Human: 你好
    -------------
    [SystemMessage(content='你是一个翻译机器人，将英文翻译成中文', additional_kwargs={}, response_metadata={}), HumanMessage(content='你好', additional_kwargs={}, response_metadata={})]


## 2. 外部数据连接与检索
这一章介绍 LangChain 中的文档加载器（[document loader](https://python.langchain.com/docs/concepts/document_loaders/)）、嵌入（[embedding](https://python.langchain.com/docs/concepts/embedding_models/)）和向量存储（vector store）这几个抽象模块。这些模块的设计目的是支持从（向量）数据库或其他数据源中检索数据，并将其整合到 LLM 的工作流程中。它们对于需要在模型推理过程中获取并利用外部数据的应用非常重要，比如检索增强生成（RAG）就是一个典型案例

接下来，我们将基于一个 PDF 文档构建一个简单的本地文档内容检索引擎，使我们能够根据输入的查询，在 PDF 中检索出相似的内容片段。

```bash
pip install langchain-community pypdf
```

### 2.1 文档（Documents）
[Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html)是LangChain中的一个用于存储文本段及其元数据的类，一个 Document 对象通常代表的是一个较大文档中的一个片段（chunk）。
Document类的主要属性包括：
- `page_content`：文本段的内容。
- `metadata`：与文本段相关的元数据，通常是一个字典，可以用于记录文档的来源、与其他文档之间的关系，以及其他相关信息。。
- `id(可选)`：文本段的唯一标识符。


```python

from langchain_core.documents import Document

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]
```

### 2.2 加载文档（document_loaders）
LangChain的社区生态中，集成了很多文档加载方式，包括PDF、Word、CSV等，这里我们以PDF为例，介绍下document_loaders中PyPDFLoader的使用。
其他类型可参考：https://python.langchain.com/docs/integrations/document_loaders/

PyPDFLoader 会将每一页 PDF 加载为一个 Document 对象。对于每个对象，我们都可以轻松访问以下内容：
- 该页的字符串内容；
- 包含文件名和页码的元数据。


```python
from langchain_community.document_loaders import PyPDFLoader
file_path = "data/论文送审与答辩规定.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load() # docs 是一个列表，每个元素是一个 Document 对象

print(len(docs))
for doc in docs:
    print(doc.page_content[:100],"...........")
    print(doc.metadata)
    print("----------------")
```

    5
    计算机学院关于硕士研究生学位论文送审与答辩的若干规定
    （试行）
    为保证研究生学位授予质量，根据《杭州电子科技大学博士、硕士学位授予工作细则》（杭
    电研[2021]165 号）和《杭州电子科技大学关于研 ...........
    {'producer': '', 'creator': 'WPS 文字', 'creationdate': '2022-04-11T14:27:44+06:27', 'author': 'Admin', 'comments': '', 'company': '', 'keywords': '', 'moddate': '2022-04-11T14:27:44+06:27', 'sourcemodified': "D:20220411142744+06'27'", 'subject': '', 'title': '', 'trapped': '/False', 'source': 'data/论文送审与答辩规定.pdf', 'total_pages': 5, 'page': 0, 'page_label': '1'}
    ----------------
    （二）上一年度，其导师所指导的硕士研究生盲审结果平均分为 2.5 及以下的；
    （三）截至本年度，其导师尚未有指导硕士研究生毕业的。
    （四）本人学位论文初次申请送审时因院内评审结果不佳而被终止送审程序的 ...........
    {'producer': '', 'creator': 'WPS 文字', 'creationdate': '2022-04-11T14:27:44+06:27', 'author': 'Admin', 'comments': '', 'company': '', 'keywords': '', 'moddate': '2022-04-11T14:27:44+06:27', 'sourcemodified': "D:20220411142744+06'27'", 'subject': '', 'title': '', 'trapped': '/False', 'source': 'data/论文送审与答辩规定.pdf', 'total_pages': 5, 'page': 1, 'page_label': '2'}
    ----------------
    硕士学位论文送审程序终止后，申请人须根据评阅意见书对学位论文进行实质性修改，3
    个月后 1 年内按程序重新申请送审。
    第十一条 涉及国家秘密（军事安全信息）的论文，严格按照《杭州电子科技大学研究生
    从 ...........
    {'producer': '', 'creator': 'WPS 文字', 'creationdate': '2022-04-11T14:27:44+06:27', 'author': 'Admin', 'comments': '', 'company': '', 'keywords': '', 'moddate': '2022-04-11T14:27:44+06:27', 'sourcemodified': "D:20220411142744+06'27'", 'subject': '', 'title': '', 'trapped': '/False', 'source': 'data/论文送审与答辩规定.pdf', 'total_pages': 5, 'page': 2, 'page_label': '3'}
    ----------------
    个月后 1 年内按程序重新送审。第二次盲审仍然不通过，不再受理其答辩申请。
    第十七条 硕士研究生学位论文首次盲审评阅意见含 C 或 D 的，须列入独立答辩组答辩。
    第十八条 独立答辩组的评审工作由学院 ...........
    {'producer': '', 'creator': 'WPS 文字', 'creationdate': '2022-04-11T14:27:44+06:27', 'author': 'Admin', 'comments': '', 'company': '', 'keywords': '', 'moddate': '2022-04-11T14:27:44+06:27', 'sourcemodified': "D:20220411142744+06'27'", 'subject': '', 'title': '', 'trapped': '/False', 'source': 'data/论文送审与答辩规定.pdf', 'total_pages': 5, 'page': 3, 'page_label': '4'}
    ----------------
    内外学术期刊投稿并发表论文；在 CCF 推荐会议列表的 C 类会议上发表论文等效于向 EI 收
    录的国内外学术期刊投稿并发表论文。
    第二十五条 本规定从 2019 级硕士研究生开始执行。
    第二十六条  ...........
    {'producer': '', 'creator': 'WPS 文字', 'creationdate': '2022-04-11T14:27:44+06:27', 'author': 'Admin', 'comments': '', 'company': '', 'keywords': '', 'moddate': '2022-04-11T14:27:44+06:27', 'sourcemodified': "D:20220411142744+06'27'", 'subject': '', 'title': '', 'trapped': '/False', 'source': 'data/论文送审与答辩规定.pdf', 'total_pages': 5, 'page': 4, 'page_label': '5'}
    ----------------


### 2.3 文档的自定义分割（Text Splitting）
直接使用PyPDFLoader可以获得一个简单的文档分割，即每一页都是一个Document。但是对于信息检索和下游问答任务来说，直接以页面为单位可能过于粗略。因为我们的目标是根据输入查询检索到能回答问题的 Document 对象，所以有必要将 PDF 进一步拆分，避免文中相关内容被上下文稀释，从而提升检索效果。

Langchain封装了一系列文本分割器（[text splitter](https://python.langchain.com/docs/concepts/text_splitters/#document-structured-based)），可以根据不同的需求选择合适的分割器。以下是一些常见的文本分割器：

1. **CharacterTextSplitter**：基于字符的分割器，将文本拆分为每段固定长度的字符。
2. **TokenTextSplitter**：基于标记（token）的分割器，将文本拆分为每段固定数量的标记。
3. **RecursiveCharacterTextSplitter**：递归分割器，根据常见的分隔符（如换行符）将文本拆分为段落。
4. **SpacyTextSplitter**：基于 Spacy 的分割器，使用 Spacy 进行文本分割。
5. **NLTKTextSplitter**：基于 NLTK 的分割器，使用 NLTK 进行文本分割。
6. **MarkdownHeaderTextSplitter**：基于 Markdown 标题的分割器，将文本拆分为每个标题作为一个段落。

这里我们使用RecursiveCharacterTextSplitter作为分割器，RecursiveCharacterTextSplitter 是 LangChain 中常用的文本分割工具，主要用于将较长的文本切分成更小的片段，以便后续处理（如向量化、检索等）。它的核心思想是递归地按照指定的分隔符（如换行符、句号、逗号等）进行分割，优先使用较大的分隔符，如果分割后片段仍然过长，则继续用更小的分隔符递归分割，直到每个片段都不超过设定的最大长度。

主要参数包括：

- chunk_size ：每个分片的最大字符数。
- chunk_overlap ：相邻分片之间的重叠字符数，保证上下文连续性。
- separators ：分割时优先使用的分隔符列表（如["\n\n", "\n", "。", "，", " "]）。

此外，我们设置 add_start_index=True，这样每个分段在原始文档中的起始字符位置将会作为 start_index 添加到该段的元数据中。


```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))
print(type(all_splits[0]))
print(all_splits[0].page_content)
print("------------")
print(all_splits[1].page_content)
```

    12
    <class 'langchain_core.documents.base.Document'>
    计算机学院关于硕士研究生学位论文送审与答辩的若干规定
    （试行）
    为保证研究生学位授予质量，根据《杭州电子科技大学博士、硕士学位授予工作细则》（杭
    电研[2021]165 号）和《杭州电子科技大学关于研究生学位论文盲审工作的规定（试行）》（杭
    电研[2021]164 号），结合我院具体情况，特制定本规定。
    第一章 学位论文的送审申请
    第一条 凡攻读本院硕士学位的研究生，在规定的学习期限内，修完本人培养计划中的全
    部课程，成绩合格，达到所规定的总学分，取得相应的科技成果，完成硕士学位论文的研究
    和撰写工作，经导师审阅通过后，方可向学院申请硕士学位论文盲审送审。
    第二条 学位论文实行双盲评审，送审的研究生学位论文须隐去作者和导师的相关信息，
    同时反馈的评阅结果须隐去评阅人的信息，以保证论文评阅的客观公正。
    第三条 论文提交要求：
    （一）学生申请学位论文盲审送审时须填写《研究生学位论文送审资格审查表》，导师同
    意送审并写出评阅意见。
    （二）盲审学位论文应为 PDF 格式文档，符合盲审学位论文格式要求，命名规则：“学生
    学号_学生姓名_学位论文题目”。
    ------------
    意送审并写出评阅意见。
    （二）盲审学位论文应为 PDF 格式文档，符合盲审学位论文格式要求，命名规则：“学生
    学号_学生姓名_学位论文题目”。
    （三）盲审论文格式：学位论文主体部分（中英文摘要、目录、正文、附录、参考文献）
    必须完整，不能故意删除或隐瞒，附录中如有涉及本人、导师的敏感文字可以用“*”替换处
    理。封面、原创声明、授权声明、致谢等非主体部分，学术不端检测容易出现重复，学生自
    行删除，然后统一采用盲审专用论文封面。
    第四条 计算机学院要求本院硕士研究生将撰写完整的学位论文递交导师第一次审阅的
    时间最晚截止至送审日之前30 日，论文修改后递交导师最后一次审阅的时间最晚截止至送审
    日之前 10 日，超过此规定时间仍未将论文递交导师审阅者，导师有权在《论文送审资格审查
    表》中批复不同意送审。
    第五条 计算机学院提倡所有硕士学位论文实行预答辩制。对于出现下列情况之一的硕士
    研究生必须参加由学院统一组织的预答辩。
    （一）上一年度，其导师所指导的硕士研究生盲审结果出现“C、C”、“C、D”或“D、
    D”的；


关于处理 PDF 的详细内容（例如提取特定章节或图片中的文本），可以参考 [How to load PDFs](https://python.langchain.com/docs/how_to/document_loader_pdf/)

### 2.4 Embeddings


上一节中我们读取了文档并划分成了若干文本chunk，但是当我们要构建大型知识库时，往往需要将这些文本数据转化为数值向量（通过embedding model），方便后续应用中做对知识库做向量搜索。

**向量搜索**是一种常见的方式，用于存储和检索非结构化数据（如非结构化文本）。其核心思想是将文本表示为数值向量进行存储。查询时，我们可以将输入也转化为同维度的向量，并通过向量相似度度量（如余弦相似度）来找到相关文本。

LangChain 封装了很多不同的embedding模型，可查询：https://python.langchain.com/docs/integrations/text_embedding/。或者也可以仿照这些封装去自定义一些embedding模型。这里我们使用一个知名的[bge](https://huggingface.co/BAAI/bge-m3) embedding模型，可以从huggingface上下载，更多的模型可以参考[embedding leaderboard](https://huggingface.co/spaces/mteb/leaderboard)。


```python
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_name = "models/bge-m3"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding_model_bge = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vector_1 = embedding_model_bge.embed_query(all_splits[0].page_content)
vector_2 = embedding_model_bge.embed_query(all_splits[1].page_content)
assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])

```

    Generated vectors of length 1024
    
    [-0.09491164982318878, -0.0020208219066262245, 0.024046452715992928, -0.019713258370757103, -0.008063230663537979, -0.03716067597270012, -0.006405872758477926, -0.030813977122306824, 0.010731630958616734, 0.01573730818927288]


### 2.5 构建向量数据库（VectorStore）
向量数据库是一种专门设计用于高效存储、索引和查询高维向量数据的数据库系统。与传统数据库不同，它不依赖于精确匹配或预定义标准，而是基于向量距离或相似性进行搜索和检索。
主流向量数据库类型：
1. Chroma：轻量级开源向量数据库，简单易用，适合本地开发和小型项目
2. FAISS：Facebook开发的高性能相似性搜索库，适合大规模数据集
3. Pinecone：完全托管的向量数据库服务，适合生产环境
4. Weaviate：开源向量搜索引擎，支持向量搜索和知识图谱
5. Qdrant：云原生向量数据库服务，提供高效API接口

之前我们介绍了怎么将文本或者文档转换成向量，接下来我们以FAISS为例介绍怎么用向量数据库来存储并检索这些向量。

LangChain 的 VectorStore 供了统一的API来与不同的底层向量数据库交互，具体来说提供了用于添加文本和 Document 对象的方法，并支持使用多种相似度度量方式进行查询。这类对象通常需要与embeddings模型配合使用，以确定如何将文本数据转换为数值向量。





```python
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

embedding_dim = len(embedding_model_bge.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

# 实例化一个vecotr store
vector_store = FAISS(
    embedding_function=embedding_model_bge, # 指定embedding模型
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

```


```python
# 为向量数据库添加文档数据
ids = vector_store.add_documents(documents=all_splits)
print(ids)

```

    ['b188bfed-4b5c-4196-bdc1-c4d7453807fb', 'd3e63b1a-adb3-4bfb-abe0-ae782f304488', '529cef49-26bf-40ba-b4ff-3ad3860ffbe5', '776c864a-222a-4d0b-a487-6af9b30dea98', '36835fe8-4f93-4414-ab54-3971d98e0217', 'c94c5015-3fb5-4da1-adab-2ca6e713111a', '1f842869-28e7-44e2-8f86-fb18d6a49bcd', '907bc740-ac92-4301-bc8a-a9cc21b96c0f', 'c36d8245-46ea-4fc4-8dfb-d7453127bf7c', '00120957-e3a9-43c3-b460-a9701585d6c6', 'b4872df8-5c91-48c8-901b-5acd3b0786d4', 'b30a635d-fb6a-483f-9930-65303c2eb4aa']



```python
results = vector_store.similarity_search(
    "专家评审意见有哪几个等级"
)
print(len(results))
print("-------------")
print(results[0].page_content)
print("-------------")
print(results[0].metadata)

```

    4
    -------------
    上院内评审专家审阅，对论文写作规范性是否已达到相关学位的学术水平，能否参加送审等
    给出明确评阅意见。
    专家评阅意见分为：A．同意送审；B．同意经过小的修改后送审（不再进行院内评审）；
    C．需要进行较大的修改，暂缓送审（3 日内修改后送原专家再审）；D．未达到学位论文要求，
    不同意送审（视为“存在异议”，自动加送 1 位专家院内评审）。
    第十条 根据院内评审结果的具体情况，将对送审申请作如下处理：
    （一）评阅意见为“C、C”、“C、D”或“D、D”时，本次送审程序终止。
    （二）评阅意见为“D、A”或“D、B”时，申请人及导师须签署《计算机学院学位论文
    质量承诺书》后方能送审，否则本次送审程序终止。
    （三）其余情况皆视为无异议，同意送审。
    -------------
    {'producer': '', 'creator': 'WPS 文字', 'creationdate': '2022-04-11T14:27:44+06:27', 'author': 'Admin', 'comments': '', 'company': '', 'keywords': '', 'moddate': '2022-04-11T14:27:44+06:27', 'sourcemodified': "D:20220411142744+06'27'", 'subject': '', 'title': '', 'trapped': '/False', 'source': 'data/论文送审与答辩规定.pdf', 'total_pages': 5, 'page': 1, 'page_label': '2', 'start_index': 794}


## 3. 结构化输出（Structured outputs）

在某些应用场景里，我们需要让LLM输出结构化的数据，比如JSON或者XML。LangChain提供了一些工具来帮助我们实现这个目标。
   

### 3.1 with_structured_output()
比如，LangChain 提供了一个名为 `with_structured_output()` 的方法，用于自动完成将模式（schema）绑定到模型以及解析输出的整个过程。对于所有支持结构化输出的模型提供商，这个辅助函数都是可用的，具体可参考：https://python.langchain.com/docs/concepts/structured_outputs/，大致工作流程如下：


```python
# Define schema
schema = ...
# Bind schema to model
model_with_structure = model.with_structured_output(schema)
# Invoke the model to produce structured output that matches the schema
structured_output = model_with_structure.invoke(user_input)
```

### 3.2  prompting+输出解析器（Output Parser）
然而，并非所有模型都支持 `.with_structured_output()`，因为并不是所有模型都具备tool calling 功能或 JSON 模式支持。对于这类模型，需要直接通过提示（prompt）引导模型以特定格式输出，然后使用输出解析器（output parser）从模型原始输出中提取结构化的结果。

以下示例使用内置的 PydanticOutputParser 来解析一个被提示生成符合指定 Pydantic 模式（schema）的聊天模型的输出。
在一般的模型问答基础上，主要做了两件事：
1. 使用parse的get_format_instructions()方法，将 format_instructions 直接添加到prompt中
2. 然后在模型生成回答后，让parser进一步解析模型的回答


```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field

# 定义一个pydantic类
class Person(BaseModel):
    """个人信息"""

    name: str = Field(description="姓名")
    height_in_meters: float = Field( description="身高")

# 根据pydantic类定义一个输出解析器
parser = PydanticOutputParser(pydantic_object=Person)

# 用parse对应的format_instructions构建prompt
prompt_template = ChatPromptTemplate([
    SystemMessagePromptTemplate.from_template("回答用户, 用json格式输出\n{format_instructions}"),
    HumanMessagePromptTemplate.from_template("{sentence}")
]).partial(format_instructions=parser.get_format_instructions())
sentence = "Anna is 23 years old and she is 6 feet tall"
prompt = prompt_template.invoke({"sentence": sentence})
print(prompt)
print("--------------------------")

# 调用llm模型
output = model.invoke(prompt)
print(output)
print("--------------------------")

# 解析输出为一个Person对象
result = parser.parse(output.content)
print(type(result))
print(result)
print("--------------------------")
```

    messages=[SystemMessage(content='回答用户, 用json格式输出\nThe output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"description": "个人信息", "properties": {"name": {"description": "姓名", "title": "Name", "type": "string"}, "height_in_meters": {"description": "身高", "title": "Height In Meters", "type": "number"}}, "required": ["name", "height_in_meters"]}\n```', additional_kwargs={}, response_metadata={}), HumanMessage(content='Anna is 23 years old and she is 6 feet tall', additional_kwargs={}, response_metadata={})]
    --------------------------
    content='```json\n{\n  "name": "Anna",\n  "height_in_meters": 1.8288\n}\n```' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 604, 'prompt_tokens': 244, 'total_tokens': 848, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': None, 'system_fingerprint': None, 'id': None, 'service_tier': None, 'finish_reason': None, 'logprobs': None} id='run--f0479681-60cf-4f15-b4f1-6fd8137ae49d-0' usage_metadata={'input_tokens': 244, 'output_tokens': 604, 'total_tokens': 848, 'input_token_details': {}, 'output_token_details': {}}
    --------------------------
    <class '__main__.Person'>
    name='Anna' height_in_meters=1.8288
    --------------------------


JsonOutputParser也是一个常用的parser：


```python
from langchain_core.output_parsers import JsonOutputParser
parser = JsonOutputParser(pydantic_object=Person)
prompt_template = ChatPromptTemplate([
    SystemMessagePromptTemplate.from_template("回答用户, 用json格式输出\n{format_instructions}"),
    HumanMessagePromptTemplate.from_template("{sentence}")
]).partial(format_instructions=parser.get_format_instructions())
sentence = "Anna is 23 years old and she is 6 feet tall"
prompt = prompt_template.invoke({"sentence": sentence})
print(prompt)
print("--------------------------")

# 调用llm模型
output = model.invoke(prompt)
print(output)
print("--------------------------")

# 解析输出为一个dict字典对象
result = parser.parse(output.content)
print(type(result))
print(result)
print("--------------------------")
```

    messages=[SystemMessage(content='回答用户, 用json格式输出\nThe output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"description": "个人信息", "properties": {"name": {"description": "姓名", "title": "Name", "type": "string"}, "height_in_meters": {"description": "身高", "title": "Height In Meters", "type": "number"}}, "required": ["name", "height_in_meters"]}\n```', additional_kwargs={}, response_metadata={}), HumanMessage(content='Anna is 23 years old and she is 6 feet tall', additional_kwargs={}, response_metadata={})]
    --------------------------
    content='```json\n{\n  "name": "Anna",\n  "height_in_meters": 1.8288\n}\n```' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 567, 'prompt_tokens': 244, 'total_tokens': 811, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': None, 'system_fingerprint': None, 'id': None, 'service_tier': None, 'finish_reason': None, 'logprobs': None} id='run--1a179c73-fe9d-46e6-a3c4-aa660c9a7ee5-0' usage_metadata={'input_tokens': 244, 'output_tokens': 567, 'total_tokens': 811, 'input_token_details': {}, 'output_token_details': {}}
    --------------------------
    <class 'dict'>
    {'name': 'Anna', 'height_in_meters': 1.8288}
    --------------------------


除此之外，为了保持chain定义格式的一致性和可组合性，即使不需要结构化输出，也可以用StrOutputParser来作为chain的结尾组件。

## 4. LangChain Expression Language (LCEL)
> **LCEL** (**L**ang**C**hain **E**xpression **L**anguage) 是 LangChain 推出的一个“表达式语言”，专为构建、组合和运行链式组件而设计。它用一种声明式、可组合的方式来描述链式任务，让你更容易构建复杂的 LLM 应用。

### 4.1 LCEL 的核心：Runnable
Runnable 是所有可被“执行”的组件的基础接口。可以把它想象成“可运行的函数”，它接收一个输入，返回一个输出，并且可以被组合、调用、批量执行、异步调用、或流式输出。有了 Runnable，就可以把各种功能组件像“乐高积木”一样组合起来，构建非常复杂的语言模型应用，而不需要写很多控制逻辑。在 LCEL 中，一切组件都实现了 Runnable 接口。

任何实现了 Runnable 的对象都具备以下能力：
- Invoked（调用）：单个输入被转换为一个输出。
- Batched（批处理）：多个输入被高效地转换为多个输出。
- Streamed（流式输出）：输出会在生成的同时被流式传输出来。
- Inspected（可检查）：可以访问关于 Runnable 的输入、输出和配置的结构信息。
- Composed（可组合）：多个 Runnable 可以使用LCEL组合在一起，构建复杂的处理流程。




```python
from langchain_core.runnables import RunnableLambda

runnable_to_str = RunnableLambda(lambda x: str(x))

runnable_to_str.invoke(547) # 常规调用
runnable_to_str.batch([7, 8, 9]) # 批量调用
runnable_to_str.stream([10, 11, 12]) # 流式调用
```

LCEL允许对多个runnable做组合（comosition），事实上，LCEL chain 就是通过组合现有的 Runnable 构建的。其两个主要的组合原语是：
- RunnableSequence：表示按顺序执行的一系列 Runnable，每个步骤的输出会作为下一个步骤的输入。
- RunnableParallel：表示并行执行的多个 Runnable，所有步骤同时接收相同的输入，并分别生成各自的输出。

一个RunnableSequence简单例子：


```python
# RunnableSequence example
from langchain_core.runnables import RunnableLambda, RunnableSequence
runnable1 = RunnableLambda(lambda x: x + 1)
runnable2 = RunnableLambda(lambda x: x + 2)
chain = RunnableSequence(runnable1, runnable2)
chain.invoke(0)
```




    3



接下来看一个RunnableParallel的简单例子。需要注意的是，RunnableParallel的输入是一个dict，key代表某个runnable的名字，value则是runnable本身。调用RunnableParallel后，输出的还是一个dict，并且key和输入的key是一一对应的，只是value变成了对应runnable的输出。


```python
# RunnableParallel example
from langchain_core.runnables import RunnableLambda, RunnableParallel
runnable1 = RunnableLambda(lambda x: x + 1)
runnable2 = RunnableLambda(lambda x: x + 2)
chain = RunnableParallel({"a": runnable1, "b": runnable2})
chain.invoke(0)

```




    {'a': 1, 'b': 2}



两者一起用的例子


```python
# RunnableSequence & RunnableParallel example:
from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnableParallel
runnable1 = RunnableLambda(lambda x: x + 1)
runnable2 = RunnableLambda(lambda x: x + 2)
runnable3 = RunnableLambda(lambda x: x["a"] * x["b"])
chain = RunnableSequence(RunnableParallel({"a": runnable1, "b": runnable2}), runnable3)
chain.invoke(1)
# output: (1+1) * (1+2) = 6

```




    6



### 4.2 LCEL 的语法糖：| 操作符

`RunnableSequence`和`RunnableParallel`的使用非常普遍，因此LangChain为它们创建了一种速记语法。这有助于使代码更具可读性和简洁性。
简单来说，LCEL 通过操作符重载，实现了组件之间的组合，最常见的就是用｜来代替`RunnableParallel`，即｜之间的组件会被组合成一个`RunnableParallel`。


```python
# RunnableSequence example
from langchain_core.runnables import RunnableLambda, RunnableSequence
runnable1 = RunnableLambda(lambda x: x + 1)
runnable2 = RunnableLambda(lambda x: x + 2)
# chain = RunnableSequence(runnable1, runnable2)
chain = runnable1 | runnable2  # | 相当于RunnableSequence
chain.invoke(0)
```




    3



在LCEL表达式内部，字典dict会自动转换为RunnableParallel。但是需要注意的是，dict只是在LCEL的chain中被转换了，dict本身并不是RunnableParallel，所以不能单独对一个dict直接做invoke，比如{"a": runnable1, "b": runnable2}.invoke()是不合法的。


```python
# RunnableSequence & RunnableParallel example:
from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnableParallel
runnable1 = RunnableLambda(lambda x: x + 1)
runnable2 = RunnableLambda(lambda x: x + 2)
runnable3 = RunnableLambda(lambda x: x["a"] * x["b"])
chain = {"a": runnable1, "b": runnable2} | runnable3

chain.invoke(1)
# output: (1+1) * (1+2) = 6

```




    6



最后关于LCEL需要提的一点是，虽然确实有用户在生产环境中运行包含数百个步骤的链，但通常建议将 LCEL 用于较为简单的编排任务。而如果应用场景涉及复杂的状态管理、分支、循环或多个智能体（agent），LangChain推荐用户使用 [LangGraph](https://python.langchain.com/docs/concepts/architecture/#langgraph)，它更适合处理此类复杂流程。

## 5. LangGraph
LangGraph 是 LangChain 生态系统中的扩展库，专为构建多智能体系统或者复杂工作流设计。它可以通过图形化的工作流管理多个 LLM 代理的协作，支持复杂任务的动态决策和状态管理。核心优势包括：

- 循环图结构：允许代理根据新信息调整流程，支持动态决策。
- 状态持久化：自动保存执行状态，便于错误恢复和断点续传。
- 人工干预：在关键节点引入人工审核，确保系统可控性。

### 5.1 核心概念

1. 图（Graph）
LangGraph 的核心是**状态图（StateGraph）**，**由节点（Nodes）** 和 **边（Edges）** 构成：
- 节点：代表执行单元（如调用 LLM 或工具），是 Python 函数，接收并更新状态。
- 边：定义节点间的执行路径，分为：
    - 普通边：无条件跳转。
    - 条件边：基于状态动态选择路径（类似 if-else 逻辑）。

2. **状态（State）**
状态是一个共享的字典对象，存储工作流的上下文信息（如对话历史、临时变量）。所有节点通过读取和修改状态实现协作。

3. 持久化（Persistence）
LangGraph 自动保存每个步骤的状态为检查点（Checkpoint），支持从中断处恢复执行，适用于长时间任务。



### 5.2 LangGraph基础Demo（简单问答系统构建）
利用State、Node、Edge构建一个StateGraph，实现一个简单的问答系统。

1. 定义状态（State）

使用 TypedDict 定义状态结构，比如，包含用户问题和答案：


```python
from typing import TypedDict, Optional
from langchain_core.messages import HumanMessage,AIMessage

class State(TypedDict):
    question: Optional[HumanMessage]
    category: Optional[str]
    answer: Optional[AIMessage]
```

2. 创建节点（Node）执行函数

假设我们要用到问题分类节点和回答生成节点，则先分别创建一个节点函数，这个节点函数定义了节点的执行逻辑。


```python

def classify_question(state: State):
    print("state:", state)
    question = state["question"].content
    # 假设分类逻辑：判断是否为问候
    if "你好" in question:
        return {"category": "greeting"}
    else:
        return {"category": "general"}

def generate_answer(state: State):
    print("state:", state)
    question = state["question"].content
    category = state.get("category")
    if category == "greeting":
        return {"answer": AIMessage(content = "你好！我是你的智能伙伴小Q，有什么可以帮助您的？")}
    elif category == "general":
        response = model.invoke(question)
        return {"answer": response}
    else:
        return {"answer": "对不起，我暂时无法回答您的问题。"}
```

3. 构建图

创建一个状态图（StateGraph）对象，将其状态设置为我们的自定义状态（state），并在图内添加点（node）和边（edge）。


```python
from langgraph.graph import StateGraph, END, START
# 创建图，并设置状态
builder = StateGraph(State)

# 添加节点
builder.add_node("node_classify", classify_question)
builder.add_node("node_answer", generate_answer)

# 设置边
# START -> classify -> answer -> END
builder.add_edge(START, "node_classify")
builder.add_edge("node_classify", "node_answer")
builder.add_edge("node_answer", END)  # 终止节点
# builder.set_entry_point("classify")  # 入口节点

# 编译图
graph = builder.compile()
```

4. 运行图


```python
start_state = {"question": HumanMessage(content="你好")}
end_state = graph.invoke(start_state)
print(end_state)  


```

    state: {'question': HumanMessage(content='你好', additional_kwargs={}, response_metadata={})}
    state: {'question': HumanMessage(content='你好', additional_kwargs={}, response_metadata={}), 'category': 'greeting'}
    {'question': HumanMessage(content='你好', additional_kwargs={}, response_metadata={}), 'category': 'greeting', 'answer': AIMessage(content='你好！我是你的智能伙伴小Q，有什么可以帮助您的？', additional_kwargs={}, response_metadata={})}



```python
start_state = {"question": HumanMessage(content="你是谁")}
end_state = graph.invoke(start_state)
print(end_state)  
```

    state: {'question': HumanMessage(content='你是谁', additional_kwargs={}, response_metadata={})}
    state: {'question': HumanMessage(content='你是谁', additional_kwargs={}, response_metadata={}), 'category': 'general'}
    {'question': HumanMessage(content='你是谁', additional_kwargs={}, response_metadata={}), 'category': 'general', 'answer': AIMessage(content='您好！我是科大讯飞自主研发的认知智能大模型——深度推理模型X1（iFLYTEK Spark X1），专注于通过自然语言交互提供精准的语言理解、复杂推理及多领域知识服务。我的核心能力包括逻辑分析、跨学科知识整合以及开放域问题解答。作为中国人工智能"国家队"成员，我致力于用安全可控的技术帮助用户高效解决认知智能需求。请问有什么可以帮您？', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 92, 'prompt_tokens': 2, 'total_tokens': 94, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': None, 'system_fingerprint': None, 'id': None, 'service_tier': None, 'finish_reason': None, 'logprobs': None}, id='run--0c52040f-8c31-4966-9d1e-0c96dbf555c1-0', usage_metadata={'input_tokens': 2, 'output_tokens': 92, 'total_tokens': 94, 'input_token_details': {}, 'output_token_details': {}})}


### 5.3 LangGraph进阶：条件边实现动态路由 



在5.2中，通过if-else的逻辑判断，在单个节点中处理了不同类别的问题，现在我们考虑是否可以直接在节点外判断，然后根据判断决定路由至某个节点，这样可以减少节点间的耦合。

条件边允许工作流根据当前状态值动态选择下一个节点（类似编程中的 if-else 或 switch-case 逻辑）。适用于需要分支判断的场景，例如：

根据用户问题类型选择不同的处理节点（如客服系统中的“投诉” vs “咨询”）。
根据任务复杂度决定是否调用外部工具。




```python
from langgraph.graph import StateGraph
from typing import Literal, Optional

# 定义状态类型（包含分类结果）
class State(TypedDict):
    question: Optional[str]
    category: Optional[Literal["complain", "consult", "other"]]  # 使用 Literal 限定分类范围
    answer: Optional[str]

# 分类节点
def classify_question(state: State):
    question = state["question"]
    if "投诉" in question:
        return {"category": "complain"}
    elif "咨询" in question:
        return {"category": "consult"}
    else:
        return {"category": "other"}

# 定义不同分类的处理节点
def handle_complain(state: State):
    question = state["question"]
    return {"answer": "您好！接下来为你解答投诉类问题。"}

def handle_consult(state: State):
    question = state["question"]
    # TODO: 处理question
    return {"answer": "您好！接下来为你解答咨询类问题。"}

def handle_other(state: State):
    question = state["question"]
    # TODO: 处理question
    return {"answer": "抱歉，我暂时无法回答这个问题。"}

# 构建图
builder = StateGraph(State)
builder.add_node("classify_node", classify_question)
builder.add_node("complain_node", handle_complain)
builder.add_node("consult_node", handle_consult)
builder.add_node("other_node", handle_other)

# 条件边：根据分类结果跳转到不同节点
def decide_next_node(state: State):
    return state["category"]  # 返回值必须匹配后续的节点名称映射

builder.add_conditional_edges(
    "classify_node",
    decide_next_node,
    {
        "complain": "complain_node",
        "consult": "consult_node",
        "other": "other_node",
    },
)

# 所有分支最终汇聚到 END
builder.add_edge(START, "classify_node")
builder.add_edge("complain_node", END)
builder.add_edge("consult_node", END)
builder.add_edge("other_node", END)

graph = builder.compile()
```


```python
start_state = {"question": "我想咨询下关于XX的问题"}
end_state = graph.invoke(start_state)
print(end_state)
```

    {'question': '我想咨询下关于XX的问题', 'category': 'consult', 'answer': '您好！接下来为你解答咨询类问题。'}



```python
start_state = {"question": "我想投诉下关于XX的问题"}
end_state = graph.invoke(start_state)
print(end_state)
```

    {'question': '我想投诉下关于XX的问题', 'category': 'complain', 'answer': '您好！接下来为你解答投诉类问题。'}


### 5.4 LangGraph进阶：状态持久化

LangGraph 内置了一个持久化层，通过 checkpointer 实现。当我们在编译图时使用了 checkpointer，系统会在每个 super-step 后保存一次图的状态快照（checkpoint）。这些快照会被保存到一个 thread（线程） 中，该线程在图执行完毕后依然可以访问。
<img src="https://langchain-ai.github.io/langgraph/concepts/img/persistence/checkpoints.jpg" width="80%"/>

由于线程提供了对图执行后状态的访问能力，一系列强大的功能就成为可能，包括：人类参与（human-in-the-loop），记忆（memory），时间回溯（time travel），容错（fault-tolerance）。
具体内容可参考 [LangGraph Persistence Docs](https://langchain-ai.github.io/langgraph/concepts/persistence/)。

一个典型的持久化使用方式是：
1. 定义一个 StateGraph，并用某类checkpointer编译该图，其中，所有checkpointer需遵循 [BaseCheckpointSaver](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.base.BaseCheckpointSaver) 接口，确保统一行为。LangGraph提供不同级别的Checkpointer库，适应不同场景：

    | 库名称                         | 实现类               | 存储方式       | 适用场景                          | 安装要求          |
    |-------------------------------|----------------------|--------------|---------------------------------|------------------|
    | `langgraph-checkpoint`        | `InMemorySaver`      | 内存存储       | 实验/快速测试(数据易失)           | LangGraph内置     |
    | `langgraph-checkpoint-sqlite` | `SqliteSaver`        | SQLite数据库   | 本地开发/轻量级生产环境            | 需要单独安装      |
    | `langgraph-checkpoint-postgres`| `PostgresSaver`      | PostgreSQL    | 分布式/高可用生产环境             | 需要单独安装      |

2. 定义输入状态，包括一个包含thread_id的config字典，这将用于区分会话线程，该会话期间保存下来的所有checkpoint都将唯一对应到该thread_id。
3. 运行图。
4. 访问checkpoint，并作进一步操作。


#### 5.4.1 Thread Id

当调用带有 checkpointer 的图（graph）时，必须填入一个config，并在config中指定一个 thread_id。这个 thread_id 用于标识和跟踪该图执行过程中的所有状态快照，便于后续访问、恢复或扩展图的执行。config示例：


```python
config = {"configurable": {"thread_id": "user1234_2025_05_20"}}
```

#### 5.4.2 Checkpoint与StateSnapshot

Checkpoint（检查点） 是在每个 super-step（超级步骤） 保存的一份图状态快照，实际上是由 `StateSnapshot` 对象表示的，具有以下关键属性：
- config：与该检查点关联的配置。
- metadata：与该检查点关联的元数据。
- values：此时状态中的值。
- next：一个元组，表示图中下一步将要执行的节点名称。
- tasks：一个 PregelTask 对象的元组，包含有关接下来要执行任务的信息。如果该步骤之前尝试过，还会包含错误信息。如果图是在某个节点内被动态中断，tasks 中也会包含与中断相关的附加数据。

#### 5.4.3 状态持久化的简单示例

接下来我们将通过一个简单图的调用示例，看看系统会保存哪些checkpoint。

(1) 首先，我们定义一个简单的图，然后用InMemorySaver作为checkpointer来编译该图，这将在内存中保存checkpoint。


```python

from tracemalloc import start
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

# 定义状态类型
class State(TypedDict):
    query: str
    result: str

# 定义节点函数
def node_a(state: State):
    query = state["query"]
    return {"result": f"answer from node a (Echo: {query})"}

def node_b(state: State):
    query = state["query"]
    return {"result": f"answer from node b (Echo: {query})"}


# 定义工作流（配置状态，节点，边）
workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

# 这里使用内存作为保存器
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# 定义config，设置一个线程id
config = {"configurable": {"thread_id": "user1234_2025_05_20"}}

# 运行
start_state = {"query": "hello!", "result": ""}
end_state = graph.invoke(start_state, config)
print(end_state)
```

    {'query': 'hello!', 'result': 'answer from node b (Echo: hello!)'}


(2) 接下来，我们访问保存下来的checkpoints
   
之前说到，当graph执行完后，依旧可以访问某个线程对应的checkpoints（其实就是一些保存下来的StateSnapshot对象）。

比如，通过调用 `.get_state(config)` ，可以查看保存下来的的最新状态。这个方法会返回一个 StateSnapshot 对象，表示与提供的 thread ID（或如果指定了具体的 checkpoint ID，则为该检查点）关联的最新检查点。


```python
last_checkpoint = graph.get_state(config)
print(last_checkpoint)
last_state = last_checkpoint.values
print(last_state)
```

    StateSnapshot(values={'query': 'hello!', 'result': 'answer from node b (Echo: hello!)'}, next=(), config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-0895-67ce-8002-fa52ad44755d'}}, metadata={'source': 'loop', 'writes': {'node_b': {'result': 'answer from node b (Echo: hello!)'}}, 'step': 2, 'parents': {}, 'thread_id': 'user1234_2025_05_20'}, created_at='2025-05-23T07:23:25.124591+00:00', parent_config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-0892-6c2c-8001-97d98d544891'}}, tasks=(), interrupts=())
    {'query': 'hello!', 'result': 'answer from node b (Echo: hello!)'}


或者，调用`.get_state_history(config)`来获取该graph被记录下来的所有state历史。


```python
checkpoint_history = list(graph.get_state_history(config))
for i, checkpoint in enumerate(reversed(checkpoint_history)):
    print(f"checkpoint {i}, {checkpoint}")
```

    checkpoint 0, StateSnapshot(values={}, next=('__start__',), config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-088f-6c5c-bfff-80df9467380c'}}, metadata={'source': 'input', 'writes': {'__start__': {'query': 'hello!', 'result': ''}}, 'step': -1, 'parents': {}, 'thread_id': 'user1234_2025_05_20'}, created_at='2025-05-23T07:23:25.122255+00:00', parent_config=None, tasks=(PregelTask(id='5cc88ce0-0564-9269-9795-a683f71564ec', name='__start__', path=('__pregel_pull', '__start__'), error=None, interrupts=(), state=None, result={'query': 'hello!', 'result': ''}),), interrupts=())
    checkpoint 1, StateSnapshot(values={'query': 'hello!', 'result': ''}, next=('node_a',), config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-0891-6408-8000-b1f4680552d3'}}, metadata={'source': 'loop', 'writes': None, 'step': 0, 'parents': {}, 'thread_id': 'user1234_2025_05_20'}, created_at='2025-05-23T07:23:25.122860+00:00', parent_config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-088f-6c5c-bfff-80df9467380c'}}, tasks=(PregelTask(id='365757e1-9013-4ab7-d318-7c6c048e96c0', name='node_a', path=('__pregel_pull', 'node_a'), error=None, interrupts=(), state=None, result={'result': 'answer from node a (Echo: hello!)'}),), interrupts=())
    checkpoint 2, StateSnapshot(values={'query': 'hello!', 'result': 'answer from node a (Echo: hello!)'}, next=('node_b',), config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-0892-6c2c-8001-97d98d544891'}}, metadata={'source': 'loop', 'writes': {'node_a': {'result': 'answer from node a (Echo: hello!)'}}, 'step': 1, 'parents': {}, 'thread_id': 'user1234_2025_05_20'}, created_at='2025-05-23T07:23:25.123475+00:00', parent_config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-0891-6408-8000-b1f4680552d3'}}, tasks=(PregelTask(id='9593c484-ff01-ba11-a492-3a80a3b58a41', name='node_b', path=('__pregel_pull', 'node_b'), error=None, interrupts=(), state=None, result={'result': 'answer from node b (Echo: hello!)'}),), interrupts=())
    checkpoint 3, StateSnapshot(values={'query': 'hello!', 'result': 'answer from node b (Echo: hello!)'}, next=(), config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-0895-67ce-8002-fa52ad44755d'}}, metadata={'source': 'loop', 'writes': {'node_b': {'result': 'answer from node b (Echo: hello!)'}}, 'step': 2, 'parents': {}, 'thread_id': 'user1234_2025_05_20'}, created_at='2025-05-23T07:23:25.124591+00:00', parent_config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-0892-6c2c-8001-97d98d544891'}}, tasks=(), interrupts=())


从这里可以看出，在START前，有一份state是空的StateSnapshot，这是因为我们没有运行过该graph；然后接受用户输入后，新的StateSnapshot中的state就有了query；然后再经过每个node的处理后，都会保存一份更新了结果的StateSnapshot。

（3）Replay

可以在调用图（graph）时传入了 thread_id 和 checkpoint_id，这将会放到之前的对应检查点，具体来说，指定checkpoint_id的检查点之前的所有步骤都会被回放（re-play）但不会实际执行，然后从该检查点开始正式执行，



```python
config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_id': '1f037a6d-0892-6c2c-8001-97d98d544891'}}
start_state = {"query": "Lohha", "answer": ""}
graph.invoke(start_state, config)

config = {"configurable": {"thread_id": "user1234_2025_05_20"}}
checkpoint_history = list(graph.get_state_history(config))
for i, checkpoint in enumerate(reversed(checkpoint_history)):
    print(f"checkpoint {i}, {checkpoint}")
```

    checkpoint 0, StateSnapshot(values={}, next=('__start__',), config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-088f-6c5c-bfff-80df9467380c'}}, metadata={'source': 'input', 'writes': {'__start__': {'query': 'hello!', 'result': ''}}, 'step': -1, 'parents': {}, 'thread_id': 'user1234_2025_05_20'}, created_at='2025-05-23T07:23:25.122255+00:00', parent_config=None, tasks=(PregelTask(id='5cc88ce0-0564-9269-9795-a683f71564ec', name='__start__', path=('__pregel_pull', '__start__'), error=None, interrupts=(), state=None, result={'query': 'hello!', 'result': ''}),), interrupts=())
    checkpoint 1, StateSnapshot(values={'query': 'hello!', 'result': ''}, next=('node_a',), config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-0891-6408-8000-b1f4680552d3'}}, metadata={'source': 'loop', 'writes': None, 'step': 0, 'parents': {}, 'thread_id': 'user1234_2025_05_20'}, created_at='2025-05-23T07:23:25.122860+00:00', parent_config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-088f-6c5c-bfff-80df9467380c'}}, tasks=(PregelTask(id='365757e1-9013-4ab7-d318-7c6c048e96c0', name='node_a', path=('__pregel_pull', 'node_a'), error=None, interrupts=(), state=None, result={'result': 'answer from node a (Echo: hello!)'}),), interrupts=())
    checkpoint 2, StateSnapshot(values={'query': 'hello!', 'result': 'answer from node a (Echo: hello!)'}, next=('node_b',), config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-0892-6c2c-8001-97d98d544891'}}, metadata={'source': 'loop', 'writes': {'node_a': {'result': 'answer from node a (Echo: hello!)'}}, 'step': 1, 'parents': {}, 'thread_id': 'user1234_2025_05_20'}, created_at='2025-05-23T07:23:25.123475+00:00', parent_config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-0891-6408-8000-b1f4680552d3'}}, tasks=(PregelTask(id='9593c484-ff01-ba11-a492-3a80a3b58a41', name='node_b', path=('__pregel_pull', 'node_b'), error=None, interrupts=(), state=None, result={'result': 'answer from node b (Echo: hello!)'}),), interrupts=())
    checkpoint 3, StateSnapshot(values={'query': 'hello!', 'result': 'answer from node b (Echo: hello!)'}, next=(), config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-0895-67ce-8002-fa52ad44755d'}}, metadata={'source': 'loop', 'writes': {'node_b': {'result': 'answer from node b (Echo: hello!)'}}, 'step': 2, 'parents': {}, 'thread_id': 'user1234_2025_05_20'}, created_at='2025-05-23T07:23:25.124591+00:00', parent_config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-0892-6c2c-8001-97d98d544891'}}, tasks=(), interrupts=())
    checkpoint 4, StateSnapshot(values={'query': 'hello!', 'result': 'answer from node a (Echo: hello!)'}, next=('__start__',), config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-f153-668c-8002-54044dc785b5'}}, metadata={'source': 'input', 'writes': {'__start__': {'query': 'Lohha', 'answer': ''}}, 'step': 2, 'parents': {}, 'thread_id': 'user1234_2025_05_20'}, created_at='2025-05-23T07:23:49.529336+00:00', parent_config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-0892-6c2c-8001-97d98d544891'}}, tasks=(PregelTask(id='6567d18f-f3da-ada9-9456-90c8f0e1046d', name='__start__', path=('__pregel_pull', '__start__'), error=None, interrupts=(), state=None, result={'query': 'Lohha'}),), interrupts=())
    checkpoint 5, StateSnapshot(values={'query': 'Lohha', 'result': 'answer from node a (Echo: hello!)'}, next=('node_a',), config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-f155-6f90-8003-157f0e9b746f'}}, metadata={'source': 'loop', 'writes': None, 'step': 3, 'parents': {}, 'thread_id': 'user1234_2025_05_20'}, created_at='2025-05-23T07:23:49.530398+00:00', parent_config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-f153-668c-8002-54044dc785b5'}}, tasks=(PregelTask(id='0e146e1a-a321-00a6-cdec-b851c82a79d2', name='node_a', path=('__pregel_pull', 'node_a'), error=None, interrupts=(), state=None, result={'result': 'answer from node a (Echo: Lohha)'}),), interrupts=())
    checkpoint 6, StateSnapshot(values={'query': 'Lohha', 'result': 'answer from node a (Echo: Lohha)'}, next=('node_b',), config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-f157-68f4-8004-6b881e3b8f01'}}, metadata={'source': 'loop', 'writes': {'node_a': {'result': 'answer from node a (Echo: Lohha)'}}, 'step': 4, 'parents': {}, 'thread_id': 'user1234_2025_05_20'}, created_at='2025-05-23T07:23:49.531052+00:00', parent_config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-f155-6f90-8003-157f0e9b746f'}}, tasks=(PregelTask(id='256e857f-6793-8973-9a65-009c6a1792ce', name='node_b', path=('__pregel_pull', 'node_b'), error=None, interrupts=(), state=None, result={'result': 'answer from node b (Echo: Lohha)'}),), interrupts=())
    checkpoint 7, StateSnapshot(values={'query': 'Lohha', 'result': 'answer from node b (Echo: Lohha)'}, next=(), config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-f158-6a24-8005-1372280ecdce'}}, metadata={'source': 'loop', 'writes': {'node_b': {'result': 'answer from node b (Echo: Lohha)'}}, 'step': 5, 'parents': {}, 'thread_id': 'user1234_2025_05_20'}, created_at='2025-05-23T07:23:49.531492+00:00', parent_config={'configurable': {'thread_id': 'user1234_2025_05_20', 'checkpoint_ns': '', 'checkpoint_id': '1f037a6d-f157-68f4-8004-6b881e3b8f01'}}, tasks=(), interrupts=())

