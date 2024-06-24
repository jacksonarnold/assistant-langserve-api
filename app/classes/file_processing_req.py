from langserve import CustomUserType
from langchain.pydantic_v1 import Field


class FileProcessingRequest(CustomUserType):
    """Request including a base64 encoded file."""

    # The extra field is used to specify a widget for the playground UI.
    file: str = Field(..., extra={"widget": {"type": "base64file"}})
    prompt: str
