from tempfile import NamedTemporaryFile
from fastapi.responses import RedirectResponse
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langserve import add_routes
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from starlette import status
from starlette.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from google.oauth2 import id_token
from google.auth.transport import requests
from . import config  # Relative import

app = FastAPI(
    title="LangChain Server",
    version="1.0",
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

CLIENT_ID = config.CLIENT_ID


def verify_token(token: str):
    try:
        id_info = id_token.verify_oauth2_token(token, requests.Request(), CLIENT_ID)
        if id_info['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('Wrong issuer.')
        return id_info
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


@app.get("/api/protected")
async def protected_route(token: str = Depends(oauth2_scheme)):
    user_info = verify_token(token)
    return {"message": "Hello, " + user_info["name"]}


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


add_routes(
    app,
    ChatOpenAI(),
    path="/openai",
)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary location
    with NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(await file.read())
        tmp_file_path = tmp_file.name

    if file.content_type == 'application/pdf':
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(tmp_file_path)
    elif file.content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(tmp_file_path)
    elif file.content_type == 'text/plain':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(tmp_file_path)
    else:
        print('Document format is not supported!')
        return HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)

    documents = loader.load()

    # Chunk the text
    # Extract the text content from the document
    text = "\n".join([doc.page_content for doc in documents])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    chunks = text_splitter.split_text(text)

    # Generate embeddings
    embeddings = OpenAIEmbeddings()
    vectors = embeddings.embed_documents(chunks)

    # TODO: add FAISS vector store
    # Store embeddings in a vector store
    # vector_store = FAISS()
    # vector_store.add_documents(chunks, vectors)

    # Create a summarization chain
    prompt_template = PromptTemplate.from_template("Summarize the following text: {text}")
    summarization_chain = LLMChain(
        llm=OpenAI(),
        prompt=prompt_template
    )

    summaries = []
    for chunk in chunks:
        summary = summarization_chain.run({"text": chunk})
        summaries.append(summary)

    return JSONResponse(content={"summaries": summaries})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
