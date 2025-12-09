from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import os

from openai import OpenAI
import requests
from urllib.parse import quote
from typing import Dict, List, Optional


app = FastAPI()

# /static 경로로 정적 파일 제공
app.mount("/static", StaticFiles(directory="static"), name="static")

# OpenAI 클라이언트
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== 여러 대화방 메모리 관리용 전역 딕셔너리 =====
# conversation_state = {
#   "conversation_id_1": {
#       "history": [...],
#       "summary": "...",
#       "messages_since_last_summary": 0,
#       "model": "gpt-5-mini"
#   },
# }
conversation_state: Dict[str, Dict] = {}

NOT_FOUND_MSG = "위키백과에서 해당 주제에 대한 요약을 찾지 못했어."


# ==== Pydantic 모델들 ====

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = "default"
    model: Optional[str] = None  # ★ 모델 선택 가능하도록 추가


class ChatResponse(BaseModel):
    reply: str


class SearchRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = "default"


class SearchResponse(BaseModel):
    answer: str


# ==== 루트에서 바로 UI로 리다이렉트 ====

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/static/index.html")


# ===== 대화방 상태 가져오기 / 초기화 =====

def get_conversation_state(conversation_id: str) -> Dict:
    if conversation_id is None:
        conversation_id = "default"

    if conversation_id not in conversation_state:
        conversation_state[conversation_id] = {
            "history": [],
            "summary": "",
            "messages_since_last_summary": 0,
            "model": "gpt-5-mini",  # ★ 새 채팅 기본 모델
        }
    return conversation_state[conversation_id]


# ===== 메모리 관리용 헬퍼 함수들 =====

def add_to_history(conversation_id: str, role: str, content: str):
    state = get_conversation_state(conversation_id)
    state["history"].append({"role": role, "content": content})
    state["messages_since_last_summary"] += 1


def summarize_conversation_if_needed(conversation_id: str):
    state = get_conversation_state(conversation_id)

    SUMMARY_TRIGGER = 10
    KEEP_RECENT = 5

    if state["messages_since_last_summary"] < SUMMARY_TRIGGER:
        return
    if not state["history"]:
        return

    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in state["history"]
    )

    system_prompt = (
        "너는 대화를 간단하게 요약하는 도우미야.\n"
        "- 사용자와 AI의 대화를 핵심 정보 위주로 한두 단락으로 정리해.\n"
        "- 사용자의 목표, 선호, 진행 중인 작업 등을 중심으로 작성해.\n"
        "- 불필요한 잡담은 줄여."
    )
    user_content = (
        f"이전까지의 요약:\n{state['summary'] or '(없음)'}\n\n"
        f"새로운 대화 기록:\n{history_text}\n\n"
        "위 내용을 합쳐서 앞으로 참고할 수 있도록 짧게 요약해줘."
    )

    # 요약 모델은 고정 (mini면 충분)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    new_summary = completion.choices[0].message.content.strip()
    state["summary"] = new_summary

    if len(state["history"]) > KEEP_RECENT:
        state["history"] = state["history"][-KEEP_RECENT:]

    state["messages_since_last_summary"] = 0


def build_llm_messages(
    conversation_id: str,
    user_msg: str,
    system_prompt: str,
    extra_context: Optional[str] = None,
):
    state = get_conversation_state(conversation_id)
    summary = state["summary"]
    history = state["history"]

    messages = []

    full_system = system_prompt
    if summary:
        full_system += "\n\n[요약된 이전 대화]\n" + summary
    messages.append({"role": "system", "content": full_system})

    if history:
        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in history
        )
        messages.append({
            "role": "system",
            "content": "[최근 대화 일부]\n" + history_text,
        })

    user_content_parts = []
    if extra_context:
        user_content_parts.append(f"추가 컨텍스트:\n{extra_context}")
    user_content_parts.append(f"사용자 질문: {user_msg}")

    messages.append({
        "role": "user",
        "content": "\n\n".join(user_content_parts),
    })

    return messages


# ==== 검색 함수 ====

def search_wikipedia_ko(query: str) -> str:
    base_url = "https://ko.wikipedia.org/api/rest_v1/page/summary/"
    url = base_url + quote(query)

    try:
        resp = requests.get(url, headers={"User-Agent": "mini-wrtn/0.1"})
    except Exception as e:
        return f"검색 중 오류가 발생했어: {e}"

    if resp.status_code == 200:
        data = resp.json()
        extract = data.get("extract")
        if extract:
            return extract

    if "한국 대통령" in query or "대한민국 대통령" in query:
        return search_wikipedia_ko("대한민국의 대통령")
    if "미국 대통령" in query:
        return search_wikipedia_ko("미국 대통령")

    return NOT_FOUND_MSG


# ==== /search 엔드포인트 ====

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    conversation_id = req.conversation_id or "default"
    user_query = req.query

    add_to_history(conversation_id, "user", f"[검색모드 질문] {user_query}")
    summarize_conversation_if_needed(conversation_id)

    snippet = search_wikipedia_ko(user_query)

    system_prompt = (
        "너는 검색 결과 + 일반 지식 + 대화 문맥을 조합해서 답변하는 정보 비서야.\n"
        "- 추가 컨텍스트가 있으면 우선 참고해.\n"
        "- 정확하지 않을 수 있으면 말해줘.\n"
        "- 딱 한 문단의 자연스러운 한국어 답변만 출력해."
    )

    messages = build_llm_messages(
        conversation_id=conversation_id,
        user_msg=user_query,
        system_prompt=system_prompt,
        extra_context=snippet or "검색 결과 없음",
    )

    # ★ 검색 모델 고정
    completion = client.chat.completions.create(
        model="gpt-4o-mini-search-preview",
        messages=messages,
    )
    ai_answer = completion.choices[0].message.content.strip()

    add_to_history(conversation_id, "assistant", f"[검색모드 답변] {ai_answer}")
    summarize_conversation_if_needed(conversation_id)

    return SearchResponse(answer=ai_answer)


# ==== /chat 엔드포인트 ====

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    conversation_id = req.conversation_id or "default"
    user_msg = req.message

    # 대화 저장
    add_to_history(conversation_id, "user", user_msg)
    summarize_conversation_if_needed(conversation_id)

    # 현재 방 상태
    state = get_conversation_state(conversation_id)

    # ★ 프론트에서 모델이 전달되면 그걸로 업데이트 (새 채팅 생성 시)
    if req.model is not None:
        state["model"] = req.model

    model_name = state["model"]  # ★ 이번 대화에 사용할 모델

    system_prompt = (
        "너는 친절한 한국어 AI 비서야.\n"
        "이전 대화의 요약과 최근 맥락을 활용해서 자연스럽게 이어서 답변해."
    )

    messages = build_llm_messages(
        conversation_id=conversation_id,
        user_msg=user_msg,
        system_prompt=system_prompt,
    )

    completion = client.chat.completions.create(
        model=model_name,   # ★ 여기서 방 모델로 호출됨
        messages=messages,
    )
    ai_reply = completion.choices[0].message.content.strip()

    add_to_history(conversation_id, "assistant", ai_reply)
    summarize_conversation_if_needed(conversation_id)

    return ChatResponse(reply=ai_reply)

