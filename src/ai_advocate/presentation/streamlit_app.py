from __future__ import annotations

from datetime import date

import streamlit as st

from ai_advocate.application.use_cases.ask_question import AskParams
from ai_advocate.domain.exceptions import ConfigurationError, IngestionError
from ai_advocate.infrastructure.config import Settings
from ai_advocate.infrastructure.container import Container

DISCLAIMER = (
    "This tool is for learning and legal research assistance only. "
    "It is not legal advice. Consult a qualified advocate for decisions."
)


def _domain_override_widget() -> str | None:
    choice = st.sidebar.selectbox(
        "Domain",
        options=["Auto", "Finance", "Land"],
        index=0,
        help="Auto lets the agent route queries; pick a domain to override.",
    )
    if choice == "Auto":
        return None
    return choice


def main() -> None:
    st.set_page_config(page_title="AI Advocate", layout="wide")
    st.title("AI Advocate")
    st.info(DISCLAIMER)

    settings = Settings()
    container = Container(settings)

    st.sidebar.header("Settings")
    domain_override = _domain_override_widget()
    ref_date = st.sidebar.date_input(
        "Reference date",
        value=settings.reference_date,
        help="Used to interpret phrases like 'this year' for 2026 vs legacy routing.",
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("Re-ingest PDFs into Chroma"):
        try:
            result = container.ingest_use_case().run()
            st.sidebar.success(
                f"Indexed {result.documents_indexed} documents / {result.chunks_indexed} chunks."
            )
        except (ConfigurationError, IngestionError) as exc:
            st.sidebar.error(str(exc))
            st.stop()

    try:
        vector_count = container.chroma.count()
    except Exception:
        vector_count = 0

    if vector_count == 0:
        st.warning(
            "Your vector database is empty. Click 'Re-ingest PDFs into Chroma' in the sidebar "
            "(requires `OPENAI_API_KEY`)."
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about the statutes in data/")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            answer = container.ask_use_case().run(
                AskParams(
                    question=prompt,
                    domain_filter_override=domain_override,
                    reference_date=ref_date if isinstance(ref_date, date) else None,
                )
            )
        except ConfigurationError as exc:
            st.error(str(exc))
            st.stop()

        if answer.routing_reasoning:
            with st.expander("Routing", expanded=False):
                st.markdown(answer.routing_reasoning)

        st.markdown(answer.text)

        if answer.citations:
            with st.expander("Sources", expanded=False):
                for c in answer.citations:
                    section = c.section_label or "n/a"
                    st.markdown(f"- {c.act_title} ({c.act_version}) — {section} — pages {c.pages}")
                    st.code(c.source_path)

    st.session_state.messages.append({"role": "assistant", "content": answer.text})


if __name__ == "__main__":
    main()
