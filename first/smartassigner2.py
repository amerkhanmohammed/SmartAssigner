# langgraph_pipeline.py
import os
import re
import requests
import json
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict
from IPython.display import Image, display
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ----------------------------
# Load env
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JIRA_BASE_URL = "https://ca-il-jira-test.il.cyber-ark.com"
BEARER_TOKEN = "aWTqpMSa8F67iyFonBF6ln6FvNXua4JmQiVKOk"
SPRINT_ID = "34108"

# ----------------------------
# Your State
# ----------------------------
class State(TypedDict):
    issues: List[Dict[str, Any]]
    developers: Dict[str, int]
    history: List[Dict[str, Any]]
    index: Any
    search_results: Dict[str, List[Dict[str, Any]]]
    assignments: List[Tuple[str, str]]

model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------------------
# Tool 1: Fetch Open Issues
# ----------------------------
def fetch_open_issues(_: State) -> List[Dict[str, Any]]:
    print("started fetch_open_issues")
    JQL_QUERY = (
        'project = "Cross RND Ticket" AND type = Ticket  AND component = Identity-Integrations '
        'And status = "In Progress" and (assignee is EMPTY or assignee = "Ratan sharma") '
        'ORDER BY Severity'
    )
    API_ENDPOINT = f"{JIRA_BASE_URL}/rest/api/2/search"
    params = {
        "jql": JQL_QUERY,
        "fields": "summary,description,customfield_19321,customfield_16122",
        "maxResults": 50
    }
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Accept": "application/json"
    }
    resp = requests.get(API_ENDPOINT, headers=headers, params=params)
    issues = []
    if resp.status_code == 200:
        for issue in resp.json().get("issues", []):
            key = issue.get("key")
            fields = issue.get("fields", {})
            title = fields.get("customfield_19321", {}).get("value", "No Title")
            desc = fields.get("summary", "No Description")
            issues.append({"key": key, "title": title, "description": desc})
    print("completed fetch_open_issues")
    return issues

# ----------------------------
# Tool 2: Fetch Sprint Developers
# ----------------------------
def fetch_sprint_developers(_: State) -> Dict[str, int]:
    url = f"{JIRA_BASE_URL}/rest/agile/1.0/sprint/{SPRINT_ID}/issue"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}", "Accept": "application/json"}
    resp = requests.get(url, headers=headers)
    developer_points: Dict[str, int] = {}
    if resp.status_code == 200:
        for issue in resp.json().get("issues", []):
            fields = issue.get("fields", {})
            status = fields.get("status", {}).get("name", "").lower()
            if status in ["done", "completed", "closed"]:
                continue
            assignee = fields.get("assignee", {}) or {}
            name = assignee.get("name", "Unassigned")
            sp = fields.get("aggregatetimeestimate", 0) or 0
            developer_points[name] = developer_points.get(name, 0) + sp

    return developer_points

# ----------------------------
# Tool 3: Fetch History from JIRA
# ----------------------------
def fetch_history(_: State) -> List[Dict[str, Any]]:
    JQL_QUERY = (
        'project = "Cross RND Ticket" AND type = Ticket '
        'AND status in ("11018", "10019") '
        'AND NOT (updated <= -1w OR statusCategory != Done) '
        'ORDER BY Severity'
    )

    API_ENDPOINT = f"{JIRA_BASE_URL}/rest/api/2/search"
    params = {
        "jql": JQL_QUERY,
        "fields": "summary,description,assignee",
        "maxResults": 50
    }
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Accept": "application/json"
    }

    resp = requests.get(API_ENDPOINT, headers=headers, params=params)
    history: List[Dict[str, Any]] = []

    if resp.status_code == 200:
        data = resp.json()
        for issue in data.get("issues", []):
            key = issue.get("key")
            fields = issue.get("fields", {})
            desc = fields.get("summary", "No Description")
            assignee = fields.get("assignee") or {}
            developer = assignee.get("name", "Unknown")

            history.append({
                "ticket_id": key,
                "desc": desc,
                "developer": developer
            })
    else:
        print("Failed to fetch history:", resp.status_code, resp.text)

    return history

# ----------------------------
# Tool 4: Build FAISS index
# ----------------------------
def build_index(history: List[Dict[str, Any]]):
    descriptions = [h["desc"] for h in history]
    embeddings = model.encode(descriptions, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index

# ----------------------------
# Tool 5: Search FAISS index
# ----------------------------
def search_index(state: State) -> Dict[str, List[Dict[str, Any]]]:
    index = state["index"]
    history = state["history"]
    issues = state["issues"]

    search_results: Dict[str, List[Dict[str, Any]]] = {}

    for issue in issues:
        query_emb = model.encode([issue["description"]], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        D, I = index.search(query_emb, k=3)
        results = []
        for rank, idx in enumerate(I[0]):
            ticket = history[idx]
            results.append({
                "rank": rank + 1,
                "ticket_id": ticket["ticket_id"],
                "desc": ticket["desc"],
                "developer": ticket["developer"],
                "similarity": float(D[0][rank])
            })
        search_results[issue["key"]] = results

    return search_results

# ----------------------------
# Tool 6: LLM Assignment
# ----------------------------
def llm_assign(state: State) -> List[Tuple[str, str]]:
    developers = state["developers"]
    issues = state["issues"]
    search_results = state.get("search_results", {})

    state_prompt = "".join(f"Developer: {d}, Points: {p}\n" for d, p in developers.items())
    issue_prompt = "".join(f"{i['key']}: {i['title']} | {i['description']}\n" for i in issues)

    history_prompt = ""
    for issue_key, res_list in search_results.items():
        history_prompt += f"\nSimilar Past Tickets for {issue_key}:\n"
        for r in res_list:
            history_prompt += (
                f"  Ticket: {r['ticket_id']}, Dev: {r['developer']}, "
                f"Similarity: {r['similarity']:.2f}, Desc: {r['desc']}\n"
            )

    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        model="qwen/qwq-32b:free"
    )

    prompt = (
        state_prompt + "\n" +
        issue_prompt + "\n" +
        history_prompt + "\n" +
        "Assign each CRT number to a developer.\n"
        "- Prefer developers with more available time, but do not assign everything to just one developer.\n"
        "- Distribute the issues fairly so that workload is balanced.\n"
        "- If multiple developers have similar availability, assign randomly to keep distribution even.\n"
        "- No developer should get more than ~60% of the total CRTs if others still have time remaining. and consider history similarity like which developer worked on which story\n"
        "Format strictly:\nCRT-xxxx: developer_name"
    )

    resp = llm.invoke(prompt)
    resp_text = resp.content.strip()
    assignments = []
    for line in resp_text.splitlines():
        m = re.match(r"^(CRT-\d+):\s+([\w\d_-]+)$", line.strip())
        if m:
            assignments.append((m.group(1), m.group(2)))
    return assignments


# ----------------------------
# Tool 7: Assign JIRA Issue
# ----------------------------
def assign_jira_issue(assignments: List[Tuple[str, str]]) -> None:
    for crt, dev in assignments:
        url = f"{JIRA_BASE_URL}/rest/api/2/issue/{crt}/assignee"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {BEARER_TOKEN}"}
        payload = {"name": dev}
        r = requests.put(url, headers=headers, data=json.dumps(payload))
        if r.status_code == 204:
            print(f"Assigned {crt} -> {dev}")
        else:
            print(f"Failed {crt}: {r.status_code} {r.text}")

# ----------------------------
# Build LangGraph
# ----------------------------
graph = StateGraph(State)

graph.add_node("fetch_open_issues", lambda s: {"issues": fetch_open_issues(s)})
graph.add_node("fetch_sprint_developers", lambda s: {"developers": fetch_sprint_developers(s)})
graph.add_node("fetch_history", lambda s: {"history": fetch_history(s)})
graph.add_node("build_index", lambda s: {"index": build_index(s["history"])})
graph.add_node("search_index", lambda s: {"search_results": search_index(s)})
graph.add_node("llm_assign", lambda s: {"assignments": llm_assign(s)})
graph.add_node("assign_jira_issue", lambda s: assign_jira_issue(s["assignments"]) or {})

graph.set_entry_point("fetch_open_issues")
graph.add_edge("fetch_open_issues", "fetch_sprint_developers")
graph.add_edge("fetch_sprint_developers", "fetch_history")
graph.add_edge("fetch_history", "build_index")
graph.add_edge("build_index", "search_index")
graph.add_edge("search_index", "llm_assign")
graph.add_edge("llm_assign", "assign_jira_issue")
graph.add_edge("assign_jira_issue", END)

app = graph.compile()
'''display(Image(app.get_graph().draw_mermaid_png()))'''

if __name__ == "__main__":
    final_state = app.invoke({})
    print("Pipeline finished:", final_state)
