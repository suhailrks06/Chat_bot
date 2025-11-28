


from dotenv import load_dotenv
import os

load_dotenv()  # load environment variables from .env


import os
from dotenv import load_dotenv

load_dotenv()  # loads .env automatGRically
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # fetch the key from .env

# backend.py
# Full working backend with SqliteSaver + persistent UI history + MULTI-USER SUPPORT

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import uuid
import time
import anyio
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from threading import Lock
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DB_PATH = "chat_memory.db"

# -------------------------------
# SQLite Setup
# -------------------------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.row_factory = sqlite3.Row


def init_ui_tables(connection: sqlite3.Connection):
    cur = connection.cursor()

    # Add user_id column if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ui_threads (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ui_messages (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            sender TEXT NOT NULL,
            text TEXT NOT NULL,
            ts INTEGER NOT NULL,
            FOREIGN KEY(thread_id) REFERENCES ui_threads(id)
        )
    """)

    connection.commit()


init_ui_tables(conn)

# -------------------------------
# LLM + LangGraph Setup
# -------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=GROQ_API_KEY
)


class HistoryState(TypedDict):
    history: Annotated[list[BaseMessage], add_messages]


def chat_bot(state: HistoryState):
    messages = state["history"]
    response = llm.invoke(messages)
    return {"history": [response]}


graph = StateGraph(HistoryState)
graph.add_node("CHATBOT", chat_bot)
graph.add_edge(START, "CHATBOT")
graph.add_edge("CHATBOT", END)

checkpointer = SqliteSaver(conn)
workflow = graph.compile(checkpointer=checkpointer)

# -------------------------------
# DB Helper Functions (USER-ISOLATED)
# -------------------------------

db_lock = Lock()


def create_thread_in_db(user_id: str, name: str = "New Chat"):
    tid = str(uuid.uuid4())
    ts = int(time.time())
    with db_lock:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO ui_threads(id, user_id, name, created_at) VALUES (?, ?, ?, ?)",
            (tid, user_id, name, ts)
        )
        conn.commit()
    return tid, name


def list_threads_from_db(user_id: str):
    with db_lock:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name FROM ui_threads WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        )
        rows = cur.fetchall()
    return {r["id"]: r["name"] for r in rows}


def get_history_from_db(thread_id: str, user_id: str):
    with db_lock:
        cur = conn.cursor()
        cur.execute(
            "SELECT sender, text, ts FROM ui_messages WHERE thread_id = ? AND user_id = ? ORDER BY ts ASC",
            (thread_id, user_id),
        )
        rows = cur.fetchall()
    return [{"sender": r["sender"], "text": r["text"], "ts": r["ts"]} for r in rows]


def append_message_in_db(thread_id: str, user_id: str, sender: str, text: str):
    mid = str(uuid.uuid4())
    ts = int(time.time())
    with db_lock:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO ui_messages(id, thread_id, user_id, sender, text, ts) VALUES (?, ?, ?, ?, ?, ?)",
            (mid, thread_id, user_id, sender, text, ts),
        )
        conn.commit()
    return mid


def update_thread_name_in_db(thread_id: str, user_id: str, name: str):
    with db_lock:
        cur = conn.cursor()
        cur.execute(
            "UPDATE ui_threads SET name = ? WHERE id = ? AND user_id = ?",
            (name, thread_id, user_id)
        )
        conn.commit()


# -------------------------------
# FastAPI App
# -------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def ui():
    return FileResponse("index.html")


# Create thread (now requires user_id)
@app.post("/thread/new")
async def new_thread(request: Request):
    data = await request.json()
    user_id = data.get("user_id")

    tid, name = create_thread_in_db(user_id, "New Chat")
    return {"thread_id": tid, "name": name}


# Get ALL threads for specific user
@app.get("/thread/all")
async def all_threads(user_id: str):
    return list_threads_from_db(user_id)


# Get chat history for specific user + thread
@app.get("/thread/history")
async def history(thread_id: str, user_id: str):
    hist = get_history_from_db(thread_id, user_id)
    return [{"sender": h["sender"], "text": h["text"]} for h in hist]


# Chat endpoint
@app.post("/chat")
async def chat_api(request: Request):
    data = await request.json()

    msg = data.get("message", "")
    tid = data.get("thread_id")
    user_id = data.get("user_id")

    if not tid:
        return {"response": "No thread_id provided"}, 400

    append_message_in_db(tid, user_id, "user", msg)

    threads = list_threads_from_db(user_id)
    current_name = threads.get(tid)

    if current_name == "New Chat":
        title = " ".join(msg.strip().split(" ")[0:5]).title() or "Chat"
        update_thread_name_in_db(tid, user_id, title)

    invoke_input = {"history": [HumanMessage(content=msg)]}

    response = await anyio.to_thread.run_sync(
        workflow.invoke,
        invoke_input,
        {"configurable": {"thread_id": tid}}
    )

    bot_reply = ""
    try:
        bot_reply = response["history"][-1].content
    except Exception:
        bot_reply = str(response)

    append_message_in_db(tid, user_id, "bot", bot_reply)

    return {"response": bot_reply}


@app.on_event("startup")
def startup():
    init_ui_tables(conn)
