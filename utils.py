"""Utility functions for the agentic RAG tutorial."""

import copy
import os
import time
import urllib

from flytekit import current_context


def set_openai_api_key():
    os.environ["OPENAI_API_KEY"] = current_context().secrets.get(key="openai_api_key")


def split_text_into_lines(text: str, chars_per_line: int) -> str:
    assert chars_per_line > 0
    assert len(text) > 0

    segments = []
    while len(text) > 0:

        if len(text) <= chars_per_line:
            segments.append(text)
            break

        # Each line is a maximum of `chars_per_line` characters.
        # If the ith character is not a space, walk backwards until
        # a space is found.
        i = chars_per_line
        while i > 0:
            if text[i] == " ":
                segments.append(text[:i])
                text = text[i:]
                break
            else:
                i -= 1

    return "\n".join(segments)


def generate_data_card(docs: list, head: int = 5, chars_per_line: int = 80) -> str:
    _docs = docs[:head]
    document_preview_str = ""
    for i, doc in enumerate(_docs):
        page_content = split_text_into_lines(doc.page_content.replace("```", ""), chars_per_line)
        document_preview_str += f"""\n\n---

### 📖 Chunk {i}

**Page metadata:**

{doc.metadata}

**Content:**

```
{page_content}
```
"""

    return f"""# 📚 Vector store knowledge base.

This artifact is a vector store of {len(_docs)} document chunks.

## Preview

{document_preview_str}
"""


def get_pubmed_loader(*args, **kwargs):
    from langchain_community.document_loaders import PubMedLoader as _PubMedLoader
    from langchain_community.utilities.pubmed import PubMedAPIWrapper as _PubMedAPIWrapper


    class PubMedAPIWrapper(_PubMedAPIWrapper):

        def retrieve_article(self, uid: str, webenv: str) -> dict:
            _sleep_time = copy.copy(self.sleep_time)
            for _ in range(self.max_retry):
                try:
                    article = super().retrieve_article(uid, webenv)
                    # reset sleep time
                    self.sleep_time = _sleep_time
                    return article
                except urllib.error.HTTPError:
                    time.sleep(self.sleep_time)

    class PubMedLoader(_PubMedLoader):

        def __init__(self, *args, max_retry: int = 100, sleep_time: float = 0.5, **kwargs):
            super().__init__(*args, **kwargs)
            self._client = PubMedAPIWrapper(  # type: ignore[call-arg]
                top_k_results=kwargs["load_max_docs"],  # type: ignore[arg-type]
                max_retry=max_retry,
                sleep_time=sleep_time,
            )

    return PubMedLoader(*args, **kwargs)


def parse_doc(doc):
    # make sure the title is a string
    title = doc.metadata["Title"]
    if isinstance(title, dict):
        title = " ".join(title.values())
    doc.metadata["Title"] = title
    doc.metadata["source"] = doc.metadata["uid"]
    return doc


def get_vector_store_retriever(path: str):
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain.tools.retriever import create_retriever_tool

    retriever = Chroma(
        collection_name="rag-chroma",
        persist_directory=path,
        embedding_function=OpenAIEmbeddings(),
    ).as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_pubmed_research",
        "Search and return information about pubmed research papers relating "
        "to the user query.",
    )
    return retriever_tool


def set_openai_api_key():
    os.environ["OPENAI_API_KEY"] = current_context().secrets.get(key="openai_api_key")