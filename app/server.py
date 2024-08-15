from langchain_core.runnables import chain
from fastapi.responses import RedirectResponse
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langserve import add_routes
from fastapi import FastAPI, Depends
from fastapi.security import OAuth2PasswordBearer
from app.classes.file_processing_req import PDFInput
from app.utils.auth import verify_token
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import OpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from app.utils.document_helper import load_pdf, combine_docs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(
    title="LangChain Server",
    version="1.0",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

llm = ChatOpenAI(model="gpt-4")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Example function using dependency parameter if user info IS needed beyond authentication
@app.get("/api/protected")
async def protected_route(user_info: dict = Depends(verify_token)):
    return {"message": user_info["name"]}


@app.get("/")
async def redirect_root_to_docs(user_info: dict = Depends(verify_token)):
    return RedirectResponse("/docs")


add_routes(
    app,
    llm,
    path="/openai",
    dependencies=[Depends(verify_token)],
)


@chain
def custom_chain(text):
    prompt = PromptTemplate.from_template("tell me a short joke about {topic}")
    output_parser = StrOutputParser()

    return prompt | llm | output_parser


add_routes(
    app,
    custom_chain,
    path="/tell-joke",
    dependencies=[Depends(verify_token)],
)


def map_reduce_doc(file: PDFInput) -> str:
    # initialize llm and load documents
    documents = load_pdf(file.pdf_source)

    # create summary chain and summary
    summary_chain = load_summarize_chain(llm=ChatNVIDIA(model="meta/llama3-70b-instruct"),
                                         chain_type='map_reduce', verbose=True)
    output = summary_chain.run(documents)

    return output


add_routes(
    app,
    RunnableLambda(map_reduce_doc).with_types(input_type=PDFInput),
    path="/mapreduce",
    dependencies=[Depends(verify_token)],
)


def summarize_chain(file: PDFInput) -> str:
    # initialize llm and load documents
    model_openai = OpenAI(temperature=0)
    documents = load_pdf(file.pdf_source)

    # create summary chain and summary
    summary_chain = load_summarize_chain(llm=model_openai, chain_type='map_reduce', verbose=True)
    output = summary_chain.run(documents)

    return output


add_routes(
    app,
    RunnableLambda(summarize_chain).with_types(input_type=PDFInput),
    path="/summarize-chain",
    dependencies=[Depends(verify_token)],
)


def vector_search(request: PDFInput) -> str:
    documents = load_pdf(request.pdf_source)

    # create a vector store using documents and embeddings
    db = FAISS.from_documents(documents, OpenAIEmbeddings())

    # perform similarity search using passed prompt
    docs = db.similarity_search(request.query)
    response_text = docs[0].page_content
    print(response_text)

    return response_text


add_routes(
    app,
    RunnableLambda(vector_search).with_types(input_type=PDFInput),
    path="/vector_search",
    dependencies=[Depends(verify_token)],
)


def retrieval_agent(request: PDFInput):
    documents = load_pdf(request.pdf_source)
    embeddings = OpenAIEmbeddings()

    # create a vector store using documents and embeddings
    text_splitter = RecursiveCharacterTextSplitter()
    docs = text_splitter.split_documents(documents)
    vector = FAISS.from_documents(docs, embeddings)

    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    
    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    chat_history = [HumanMessage(content="Can you summarize resumes?"), AIMessage(content="Yes!")]
    return retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": request.query
    })


add_routes(
    app,
    RunnableLambda(retrieval_agent).with_types(input_type=PDFInput),
    path="/retrieval_agent",
    dependencies=[Depends(verify_token)],
)


# define runnable chain that takes in PDF document as a parameter
pdf_qa_chain = (
    RunnablePassthrough()
    | {
        "docs": lambda x: load_pdf(x["pdf_source"]),
        "query": lambda x: x["query"]
    }
    | {
        "text": lambda x: combine_docs(x["docs"]),
        "query": lambda x: x["query"]
    }
    | ChatPromptTemplate.from_template("""Using the following text:
        {text}
        
        Answer the following question: {query}""")
    | llm
)

add_routes(
    app,
    pdf_qa_chain,
    path="/pdf_qa",
    input_type=PDFInput,
    dependencies=[Depends(verify_token)]
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
