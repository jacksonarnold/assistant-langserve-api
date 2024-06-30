from pydantic import BaseModel, Field


class PDFInput(BaseModel):
    pdf_source: str = Field(..., description="URL or file path to the PDF",
                            extra={"widget": {"type": "base64file"}})
    query: str = Field(..., description="Query to run on the PDF content")
