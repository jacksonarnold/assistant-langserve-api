import base64
from tempfile import NamedTemporaryFile
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file: str) -> List[Document]:
    # decode file and save to temporary location
    decoded_data = base64.b64decode(file)
    with NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(decoded_data)
        tmp_file_path = tmp_file.name

    # load PDF documents
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    return documents


def combine_docs(docs: List[Document]) -> str:
    return " ".join([doc.page_content for doc in docs])
