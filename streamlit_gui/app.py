from __future__ import annotations

import streamlit as st

from streamlit_gui.api_client import ApiClientError, api_is_available, ask_question, process_repository
from streamlit_gui.config import api_base_url
from streamlit_gui.state import (
    add_chat_message,
    chat_messages,
    initialize_state,
    processed_repository,
    reset_repository,
    store_processed_repository,
)


def main() -> None:
    st.set_page_config(page_title="Repository Chat", page_icon=":material/code:", layout="centered")
    initialize_state()

    repository = processed_repository()
    if repository is None:
        _render_repository_input()
        return

    _render_chat(repository)


def _render_repository_input() -> None:
    st.title("Repository Chat")
    current_api_base_url = api_base_url()
    if not api_is_available(current_api_base_url):
        st.warning(
            f"FastAPI is not reachable at {current_api_base_url}. "
            "Start both services with: python run_app.py"
        )

    github_url = st.text_input("GitHub repository URL", placeholder="https://github.com/owner/repo")

    if not github_url:
        return

    with st.spinner("Processing repository..."):
        try:
            repository = process_repository(current_api_base_url, github_url)
        except ApiClientError as exc:
            st.error(str(exc))
            return

    store_processed_repository(repository)
    st.rerun()


def _render_chat(repository) -> None:
    st.title("Repository Chat")
    st.text_input("Processed repository", value=repository.github_url, disabled=True)

    # Paths are intentionally kept in session state rather than rendered as visible fields.
    chroma_path = repository.chroma_path
    networkx_path = repository.networkx_path

    for message in chat_messages():
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask a question about this repository")
    if question:
        add_chat_message("user", question)
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_question(
                        api_base_url=api_base_url(),
                        question=question,
                        chroma_path=chroma_path,
                        networkx_path=networkx_path,
                    )
                except ApiClientError as exc:
                    answer = str(exc)
                    st.error(answer)
                else:
                    st.write(answer)

        add_chat_message("assistant", answer)

    if st.button("Process another repository"):
        reset_repository()
        st.rerun()


if __name__ == "__main__":
    main()
