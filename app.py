"""
OpenAI API + Streamlit 챗봇
로컬: .streamlit/secrets.toml 에 OPENAI_API_KEY 설정
Streamlit Cloud: Settings → Secrets 에 TOML 형식으로 동일 키 입력
"""

import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="OpenAI 챗봇",
    page_icon="💬",
    layout="centered",
)

# --- Secrets 검증 ---
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error(
        "API 키가 설정되지 않았습니다.\n\n"
        "**로컬:** `.streamlit/secrets.toml` 을 만들고 `OPENAI_API_KEY` 를 넣으세요. "
        "(`secrets.toml.example` 참고)\n\n"
        "**Streamlit Cloud:** App → Settings → Secrets 에 `OPENAI_API_KEY` 를 추가하세요."
    )
    st.stop()

model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer concisely in Korean when the user writes in Korean."}
    ]

client = OpenAI(api_key=api_key)


def stream_chat(messages: list[dict]):
    """OpenAI 스트리밍 응답을 문자 단위로 yield."""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=0.7,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


st.title("💬 OpenAI 챗봇")
st.caption(f"모델: `{model}` · Streamlit + OpenAI API")

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("메시지를 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        assistant_text = st.write_stream(
            stream_chat(st.session_state.messages)
        )

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_text or ""}
    )

with st.sidebar:
    st.subheader("설정")
    if st.button("대화 초기화"):
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely in Korean when the user writes in Korean."}
        ]
        st.rerun()
    st.markdown("---")
    st.markdown(
        "**배포 (Streamlit Cloud)**  \n"
        "Secrets 예시:\n```toml\nOPENAI_API_KEY = \"sk-...\"\nOPENAI_MODEL = \"gpt-4o-mini\"\n```"
    )
