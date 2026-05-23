from __future__ import annotations

import streamlit as st

from streamlit_gui.models import ProcessedRepository


PROCESSED_REPOSITORY_KEY = "processed_repository"
CHAT_MESSAGES_KEY = "chat_messages"


def initialize_state() -> None:
    st.session_state.setdefault(PROCESSED_REPOSITORY_KEY, None)
    st.session_state.setdefault(CHAT_MESSAGES_KEY, [])


def processed_repository() -> ProcessedRepository | None:
    return st.session_state.get(PROCESSED_REPOSITORY_KEY)


def store_processed_repository(repository: ProcessedRepository) -> None:
    st.session_state[PROCESSED_REPOSITORY_KEY] = repository


def add_chat_message(role: str, content: str) -> None:
    st.session_state[CHAT_MESSAGES_KEY].append({"role": role, "content": content})


def chat_messages() -> list[dict]:
    return st.session_state[CHAT_MESSAGES_KEY]


def reset_repository() -> None:
    st.session_state[PROCESSED_REPOSITORY_KEY] = None
    st.session_state[CHAT_MESSAGES_KEY] = []
