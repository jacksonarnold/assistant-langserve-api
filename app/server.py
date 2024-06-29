from langchain_core.runnables import chain
from fastapi.responses import RedirectResponse
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langserve import add_routes
from fastapi import FastAPI, Depends
from fastapi.security import OAuth2PasswordBearer
from app.classes.file_processing_req import FileProcessingRequest
from app.utils.auth import verify_token
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import OpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from app.utils.document_helper import load_pdf_documents
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
    return {"message": "Hello, {user}!".format(**user_info)}


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


add_routes(
    app,
    ChatOpenAI(),
    path="/openai",
)


@chain
def custom_chain(text):
    prompt = PromptTemplate.from_template("tell me a short joke about {topic}")
    output_parser = StrOutputParser()

    return prompt | llm | output_parser


add_routes(
    app,
    custom_chain,
    path="/tell-joke"
)


def map_reduce_doc(file: FileProcessingRequest) -> str:
    # initialize llm and load documents
    llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
    documents = load_pdf_documents(file)

    # create summary chain and summary
    summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce', verbose=True)
    output = summary_chain.run(documents)

    return output


add_routes(
    app,
    RunnableLambda(map_reduce_doc).with_types(input_type=FileProcessingRequest),
    path="/mapreduce",
)


def summarize_chain(file: FileProcessingRequest) -> str:
    # initialize llm and load documents
    model_openai = OpenAI(temperature=0)
    documents = load_pdf_documents(file)

    # create summary chain and summary
    summary_chain = load_summarize_chain(llm=model_openai, chain_type='map_reduce', verbose=True)
    output = summary_chain.run(documents)

    return output


add_routes(
    app,
    RunnableLambda(summarize_chain).with_types(input_type=FileProcessingRequest),
    path="/summarize-chain",
)


def vector_search(request: FileProcessingRequest) -> str:
    documents = load_pdf_documents(request)

    # create a vector store using documents and embeddings
    db = FAISS.from_documents(documents, OpenAIEmbeddings())

    # perform similarity search using passed prompt
    docs = db.similarity_search(request.prompt)
    response_text = docs[0].page_content
    print(response_text)

    return response_text


add_routes(
    app,
    RunnableLambda(vector_search).with_types(input_type=FileProcessingRequest),
    path="/vector_search",
)


def retrieval_agent(request: FileProcessingRequest):
    documents = load_pdf_documents(request)
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
        "input": request.prompt
    })


add_routes(
    app,
    RunnableLambda(retrieval_agent).with_types(input_type=FileProcessingRequest),
    path="/retrieval_agent",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
