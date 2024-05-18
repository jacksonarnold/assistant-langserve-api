from tempfile import NamedTemporaryFile
from fastapi.responses import RedirectResponse
from langchain.chains.llm import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langserve import add_routes
from fastapi import FastAPI, UploadFile, File
from starlette.responses import JSONResponse

app = FastAPI()


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
