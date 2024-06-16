import base64

from langchain_core.runnables import chain
from tempfile import NamedTemporaryFile
from fastapi.responses import RedirectResponse
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
from langserve import add_routes, CustomUserType
from fastapi import FastAPI, Depends
from fastapi.security import OAuth2PasswordBearer
from .auth import verify_token
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.output_parsers import StrOutputParser
from langchain.pydantic_v1 import Field
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import OpenAI

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

model_gpt4 = ChatOpenAI(model="gpt-4")
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

    return prompt | model_gpt4 | output_parser


add_routes(
    app,
    custom_chain,
    path="/tell-joke"
)


class FileProcessingRequest(CustomUserType):
    """Request including a base64 encoded file."""

    # The extra field is used to specify a widget for the playground UI.
    file: str = Field(..., extra={"widget": {"type": "base64file"}})
    prompt: str


def map_reduce_doc(file: FileProcessingRequest) -> str:
    # initialize llm
    llm = OpenAI(temperature=0)

    # decode file and save to temporary location
    decoded_data = base64.b64decode(file.file)
    with NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(decoded_data)
        tmp_file_path = tmp_file.name

    # load PDF documents
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    # create summary chain and summary
    summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce', verbose=True)
    output = summary_chain.run(documents)

    return output


add_routes(
    app,
    RunnableLambda(map_reduce_doc).with_types(input_type=FileProcessingRequest),
    path="/mapreduce",
)


def vector_search(request: FileProcessingRequest) -> str:
    # decode file and save to temporary location
    decoded_data = base64.b64decode(request.file)
    with NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(decoded_data)
        tmp_file_path = tmp_file.name

    # load PDF documents
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
