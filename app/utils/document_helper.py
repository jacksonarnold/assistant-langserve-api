import base64
from tempfile import NamedTemporaryFile
from langchain_community.document_loaders import PyPDFLoader
from app.classes.file_processing_req import FileProcessingRequest


def load_pdf_documents(file: FileProcessingRequest):

    # decode file and save to temporary location
    decoded_data = base64.b64decode(file.file)
    with NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(decoded_data)
        tmp_file_path = tmp_file.name

    # load PDF documents
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    return documents

