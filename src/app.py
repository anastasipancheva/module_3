import sys
import time

import streamlit as st

sys.path.insert(0, "src")

from agent import MindlyAgent

st.set_page_config(page_title="Mindly AI Coach", layout="wide", page_icon="🧠")

if "agent" not in st.session_state:
    with st.spinner("Loading Mindly…"):
        st.session_state.agent = MindlyAgent()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_ttft_ms" not in st.session_state:
    st.session_state.last_ttft_ms = None

agent: MindlyAgent = st.session_state.agent

with st.sidebar:
    st.header("⚙️ Settings")

    user_id = st.text_input("User ID", value="user_1", help="Tenant identifier — each ID is fully isolated")
    persona = st.selectbox(
        "Persona",
        ["wellness-friend", "tough-love"],
        format_func=lambda p: {
            "wellness-friend": "🌿 Wellness Friend",
            "tough-love": "💪 Tough Love",
        }[p],
    )

    st.divider()
    st.subheader("🗑️ Memory Control")

    if st.button("Delete ALL my memory", type="primary", use_container_width=True):
        agent.forget(user_id, "all")
        st.session_state.messages = []
        st.success(f"All memory for **{user_id}** deleted.")
        st.rerun()

    forget_query = st.text_input("Forget specific topic…", placeholder="e.g. my weight goal")
    if st.button("Forget this topic", use_container_width=True):
        if forget_query.strip():
            agent.forget(user_id, forget_query.strip())
            st.success(f"Memories matching «{forget_query}» removed.")
        else:
            st.warning("Enter a topic first.")

    st.divider()
    st.subheader("🧩 Current Memories")
    if st.button("Refresh memory list", use_container_width=True):
        st.session_state["show_memories"] = True

    if st.session_state.get("show_memories"):
        mems = agent.get_all_memories(user_id)
        if mems:
            for m in mems:
                st.markdown(f"- {m.get('memory', '—')}")
        else:
            st.info("No memories stored yet.")

    if st.session_state.last_ttft_ms is not None:
        st.divider()
        st.metric("Last TTFT", f"{st.session_state.last_ttft_ms:.0f} ms")

st.title("🧠 Mindly — Memory-Powered AI Coach")
st.caption(f"User: **{user_id}** · Persona: **{persona}**")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("What's on your mind?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        ttft_ms: float | None = None
        t0 = time.perf_counter()

        for token in agent.chat(user_id, persona, user_input, stream=True):
            if ttft_ms is None and token:
                ttft_ms = (time.perf_counter() - t0) * 1000
            full_response += token
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)
        st.session_state.last_ttft_ms = ttft_ms

    st.session_state.messages.append({"role": "assistant", "content": full_response})