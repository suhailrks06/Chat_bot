# backend.py
# Full working backend with SqliteSaver + persistent UI history

from fastapi import FastAPI
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


# -------------------------------
# Config
# -------------------------------
DB_PATH = "chat_memory.db"   # single DB used for LangGraph saver + UI persistence
LLM_API_KEY = "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"

# -------------------------------
# Prepare sqlite connection
# -------------------------------
# check_same_thread=False allows access from worker threads (we also protect critical sections below)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.row_factory = sqlite3.Row

# ensure tables for UI persistency
def init_ui_tables(connection: sqlite3.Connection):
    cur = connection.cursor()
    # threads: id, name, created_at
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ui_threads (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
        """
    )
    # messages: id, thread_id, sender ('user'|'bot'), text, ts
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ui_messages (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            sender TEXT NOT NULL,
            text TEXT NOT NULL,
            ts INTEGER NOT NULL,
            FOREIGN KEY(thread_id) REFERENCES ui_threads(id)
        )
        """
    )
    connection.commit()

init_ui_tables(conn)

# -------------------------------
# LLM / LangGraph Setup
# -------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=LLM_API_KEY
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

# Use a real sqlite3.Connection for SqliteSaver (documentation pattern)
# NOTE: SqliteSaver expects a sqlite3.Connection object.
checkpointer = SqliteSaver(conn)
workflow = graph.compile(checkpointer=checkpointer)

# -------------------------------
# Helper DB functions (UI)
# -------------------------------
from threading import Lock
db_lock = Lock()

def create_thread_in_db(name: str = "New Chat"):
    tid = str(uuid.uuid4())
    ts = int(time.time())
    with db_lock:
        cur = conn.cursor()
        cur.execute("INSERT INTO ui_threads(id, name, created_at) VALUES (?, ?, ?)", (tid, name, ts))
        conn.commit()
    return tid, name

def list_threads_from_db():
    with db_lock:
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM ui_threads ORDER BY created_at DESC")
        rows = cur.fetchall()
    # return dict {id: name}
    return {r["id"]: r["name"] for r in rows}

def get_history_from_db(thread_id: str):
    with db_lock:
        cur = conn.cursor()
        cur.execute(
            "SELECT sender, text, ts FROM ui_messages WHERE thread_id = ? ORDER BY ts ASC",
            (thread_id,),
        )
        rows = cur.fetchall()
    # return list of {sender, text}
    return [{"sender": r["sender"], "text": r["text"], "ts": r["ts"]} for r in rows]

def append_message_in_db(thread_id: str, sender: str, text: str):
    mid = str(uuid.uuid4())
    ts = int(time.time())
    with db_lock:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO ui_messages(id, thread_id, sender, text, ts) VALUES (?, ?, ?, ?, ?)",
            (mid, thread_id, sender, text, ts),
        )
        conn.commit()
    return mid

def update_thread_name_in_db(thread_id: str, name: str):
    with db_lock:
        cur = conn.cursor()
        cur.execute("UPDATE ui_threads SET name = ? WHERE id = ?", (name, thread_id))
        conn.commit()

# -------------------------------
# FastAPI app & endpoints
# -------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html
@app.get("/")
def ui():
    return FileResponse("index.html")

# Create thread
@app.post("/thread/new")
def new_thread():
    tid, name = create_thread_in_db("New Chat")
    return {"thread_id": tid, "name": name}

# Get all threads
@app.get("/thread/all")
def all_threads():
    return list_threads_from_db()

# Get chat history for a thread
@app.get("/thread/history")
def history(thread_id: str):
    # return only sender and text to keep front-end compatible
    hist = get_history_from_db(thread_id)
    # convert structure to match what front-end expects: list of {sender, text}
    return [{"sender": h["sender"], "text": h["text"]} for h in hist]

# Chat endpoint (async wrapper around blocking LangGraph invoke)
@app.post("/chat")
async def chat_api(data: dict):
    msg = data.get("message", "")
    tid = data.get("thread_id")

    if not tid:
        return {"response": "No thread_id provided"}, 400

    # record user message in UI DB
    append_message_in_db(tid, "user", msg)

    # auto-name thread if still "New Chat"
    threads = list_threads_from_db()
    current_name = threads.get(tid)
    if current_name == "New Chat":
        title_words = msg.strip().split(" ")[0:5]
        title = " ".join(title_words).title() or "Chat"
        update_thread_name_in_db(tid, title)

    # Prepare invoke input for LangGraph
    invoke_input = {"history": [HumanMessage(content=msg)]}
    config = {"configurable": {"thread_id": tid}}

    # Run blocking workflow.invoke in a worker thread so FastAPI event loop is not blocked
    response = await anyio.to_thread.run_sync(workflow.invoke, invoke_input, {"configurable": {"thread_id": tid}})

    # Extract bot reply
    bot_reply = ""
    try:
        bot_reply = response["history"][-1].content
    except Exception:
        # fallback
        bot_reply = str(response)

    # store bot message in UI DB
    append_message_in_db(tid, "bot", bot_reply)

    return {"response": bot_reply}

# Optional: startup hook to ensure DB/tables exist (already done above)
@app.on_event("startup")
def startup():
    init_ui_tables(conn)
