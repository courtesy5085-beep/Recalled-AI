from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from container import AppContainer
from core.config import get_settings


st.set_page_config(
    page_title="Recalled AI",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_container() -> AppContainer:
    return AppContainer.bootstrap(get_settings())


def init_session() -> None:
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("user_id", None)
    st.session_state.setdefault("username", None)


def login_page(container: AppContainer) -> None:
    st.title("🧠 Recalled AI")

    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

            if submitted:
                user_id = container.auth_service.authenticate(username, password)
                if user_id is None:
                    st.error("Invalid credentials")
                else:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user_id
                    st.session_state.username = username.strip()
                    st.success("Logged in successfully")
                    st.rerun()

    with register_tab:
        with st.form("register_form"):
            username = st.text_input("New Username")
            email = st.text_input("Email")
            password = st.text_input("New Password", type="password")
            submitted = st.form_submit_button("Register")

            if submitted:
                try:
                    container.auth_service.register(username, password, email)
                    st.success("Registration successful. Please log in.")
                except ValueError as exc:
                    st.error(str(exc))


def home_page(container: AppContainer) -> None:
    st.title("🏠 Memory Timeline")
    memories = container.memory_repository.list_by_user(st.session_state.user_id)

    if not memories:
        st.info("No memories yet")
        return

    for memory in memories:
        st.subheader(memory.title)
        st.write(memory.summary)
        st.caption(
            f"Emotion: {memory.emotion} | "
            f"Tags: {', '.join(memory.tags) or 'None'} | "
            f"Indexed: {memory.index_status}"
        )
        st.markdown("---")


def add_memory_page(container: AppContainer) -> None:
    st.title("➕ Add Memory")

    memory_type = st.selectbox(
        "Memory Type",
        ["Note", "Journal", "Idea", "Quote"],
    )
    text = st.text_area("Write memory", height=220)
    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, width=240)

    if st.button("Save Memory"):
        combined_text = text.strip()
        source_type = "text"

        if uploaded_image:
            combined_text = (
                f"{combined_text}\n\n"
                f"[Attached image: {uploaded_image.name}]"
            ).strip()
            source_type = "text_image"

        try:
            container.memory_service.save_memory(
                user_id=st.session_state.user_id,
                text=combined_text,
                memory_type=memory_type,
                source_type=source_type,
            )
            st.success("Memory saved successfully")
        except ValueError as exc:
            st.warning(str(exc))
        except Exception:
            st.error("Unexpected error while saving memory")


def search_page(container: AppContainer) -> None:
    st.title("🔍 Search Memories")
    query = st.text_input("Search")

    if not query.strip():
        return

    results = container.search_service.search(
        query=query,
        user_id=st.session_state.user_id,
    )

    if not results:
        st.warning("No results found")
        return

    for result in results:
        st.markdown("---")
        st.subheader(result.title)
        st.write(result.excerpt)
        st.caption(
            f"Emotion: {result.emotion} | "
            f"Tags: {', '.join(result.tags) or 'None'} | "
            f"Score: {result.score:.3f}"
        )


def insights_page(container: AppContainer) -> None:
    st.title("📊 Insights")
    memories = container.memory_repository.list_by_user(st.session_state.user_id)

    if not memories:
        st.info("No data available")
        return

    df = pd.DataFrame([asdict(item) for item in memories])

    emotion_counts = (
        df["emotion"]
        .value_counts()
        .reset_index()
    )
    emotion_counts.columns = ["Emotion", "Count"]

    fig = px.bar(
        emotion_counts,
        x="Emotion",
        y="Count",
        title="Mood Trends",
    )
    st.plotly_chart(fig, use_container_width=True)


def settings_page(container: AppContainer) -> None:
    st.title("⚙️ Settings")
    memories = container.memory_repository.list_by_user(st.session_state.user_id)

    if not memories:
        st.info("No memories available for export")
        return

    records = [asdict(item) for item in memories]
    df = pd.DataFrame(records)

    st.subheader("Export Memories")

    st.download_button(
        "Download JSON",
        data=json.dumps(records, ensure_ascii=False, indent=2),
        file_name="memories.json",
        mime="application/json",
    )

    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False),
        file_name="memories.csv",
        mime="text/csv",
    )


def main_app(container: AppContainer) -> None:
    st.sidebar.title("🧠 Recalled AI")

    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Home",
            "➕ Add Memory",
            "🔍 Search",
            "📊 Insights",
            "⚙️ Settings",
        ],
    )

    st.sidebar.write(f"Logged in as: {st.session_state.username}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.rerun()

    if page == "🏠 Home":
        home_page(container)
    elif page == "➕ Add Memory":
        add_memory_page(container)
    elif page == "🔍 Search":
        search_page(container)
    elif page == "📊 Insights":
        insights_page(container)
    elif page == "⚙️ Settings":
        settings_page(container)


def main() -> None:
    init_session()
    container = get_container()

    if st.session_state.logged_in:
        main_app(container)
    else:
        login_page(container)


if __name__ == "__main__":
    main()
  
