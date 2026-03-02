"""
tools.py — LangChain tools available to the multi-modal answer-generation LLM.

Tools
-----
fetch_image_details(image_id)
    Look up the SQLite registry and return caption, file path, and page number
    for a given image_id.  The LLM calls this when it encounters "[IMAGE ID: …]"
    in the context and wants more detail.
"""

from langchain_core.tools import tool

from src.image_captioner import get_image_record


@tool
def fetch_image_details(image_id: str) -> str:
    """
    Retrieve caption, file path and page number for a visual (image / table /
    diagram) identified by its image_id.

    Use this tool whenever you see an [IMAGE ID: <id>] reference in the context
    and need to inspect the visual in detail.

    Parameters
    ----------
    image_id : str
        The UUID string identifying the image, as it appears in [IMAGE ID: …].

    Returns
    -------
    str
        A formatted string with caption, source page, and file path.
    """
    record = get_image_record(image_id)
    if record is None:
        return f"No image found with ID {image_id}."

    return (
        f"Image ID   : {record['image_id']}\n"
        f"Page       : {record['page_num']}\n"
        f"Source     : {record['source_doc']}\n"
        f"Caption    : {record['caption']}\n"
        f"File Path  : {record['file_path']}"
    )


# Collect all tools for binding to the LLM
AGENT_TOOLS = [fetch_image_details]
