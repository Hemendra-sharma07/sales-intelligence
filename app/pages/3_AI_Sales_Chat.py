from pathlib import Path
import os
import sys

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.db_utils import DATABASE_PATH, get_conn, load_data_to_sqlite


st.title("AI Sales Chat")

if not DATABASE_PATH.exists():
    load_data_to_sqlite()


@st.cache_data
def build_context() -> str:
    with get_conn() as conn:
        summary = pd.read_sql(
            """
            SELECT
                COUNT(*) AS records,
                SUM(Weekly_Sales) AS total_sales,
                COUNT(DISTINCT Store) AS stores,
                COUNT(DISTINCT Dept) AS departments,
                AVG(Weekly_Sales) AS avg_sales
            FROM sales
            """,
            conn,
        )
        top_store = pd.read_sql(
            """
            SELECT Store, SUM(Weekly_Sales) AS total_sales
            FROM sales
            GROUP BY Store
            ORDER BY total_sales DESC
            LIMIT 1
            """,
            conn,
        )
    row = summary.iloc[0]
    top = top_store.iloc[0]
    return (
        f"Walmart sales dataset with {int(row.records):,} records, "
        f"${row.total_sales / 1e9:.2f}B total sales, {int(row.stores)} stores, "
        f"{int(row.departments)} departments, average weekly sales ${row.avg_sales:,.2f}. "
        f"Top store by total sales is Store {int(top.Store)}."
    )


def ask_groq(question: str, context: str) -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            "GROQ_API_KEY is not configured yet. Add it to your environment or `.env` "
            "when you are ready to enable the live AI chat feature."
        )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model="llama-3.1-70b-versatile",
            api_key=api_key,
            temperature=0.2,
        )
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a senior retail business analyst. Answer using only the "
                        f"available project context unless clearly asked for general advice. Context: {context}"
                    )
                ),
                HumanMessage(content=question),
            ]
        )
        return response.content
    except Exception as exc:
        return f"AI chat is not available yet: {exc}"


context = build_context()
st.caption(context)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Ask a sales question...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            answer = ask_groq(prompt, context)
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
