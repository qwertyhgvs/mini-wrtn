from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import os

from openai import OpenAI
import requests
from urllib.parse import quote
from typing import Dict, Optional

app = FastAPI()

# /static 경로로 정적 파일 제공
app.mount("/static", StaticFiles(directory="static"), name="static")

# OpenAI 클라이언트
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== 여러 대화방 메모리 관리용 전역 딕셔너리 =====
conversation_state: Dict[str, Dict] = {}

NOT_FOUND_MSG = "위키백과에서 해당 주제에 대한 요약을 찾지 못했어."


# ==== Pydantic 모델들 ====

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = "default"
    model: Optional[str] = None  # ★ 방별 모델 설정


class ChatResponse(BaseModel):
    reply: str


class SearchRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = "default"


class SearchResponse(BaseModel):
    answer: str


# ==== 루트: UI로 리다이렉트 ====

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/static/index.html")


# ===== 대화방 상태 가져오기 =====

def get_conversation_state(conversation_id: str) -> Dict:
    if conversation_id is None:
        conversation_id = "default"

    if conversation_id not in conversation_state:
        conversation_state[conversation_id] = {
            "history": [],
            "summary": "",
            "messages_since_last_summary": 0,
            "model": "gpt-5-mini",  # ★ 기본 모델
        }
    return conversation_state[conversation_id]


# ===== 히스토리 관리 =====

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

    # 안전한 요약 모델
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        state["summary"] = completion.choices[0].message.content.strip()
    except Exception as e:
        state["summary"] = f"(요약 실패: {e})"

    # 최근 KEEP_RECENT만 유지
    state["history"] = state["history"][-KEEP_RECENT:]
    state["messages_since_last_summary"] = 0


# ===== messages 생성 =====

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
        messages.append(
            {
                "role": "system",
                "content": "[최근 대화 일부]\n" + history_text,
            }
        )

    user_parts = []
    if extra_context:
        user_parts.append(f"추가 컨텍스트:\n{extra_context}")
    user_parts.append(f"사용자 질문: {user_msg}")

    messages.append(
        {
            "role": "user",
            "content": "\n\n".join(user_parts),
        }
    )

    return messages


# ===== 검색 기능 =====

def search_wikipedia_ko(query: str) -> str:
    base_url = "https://ko.wikipedia.org/api/rest_v1/page/summary/"
    url = base_url + quote(query)

    try:
        resp = requests.get(url, headers={"User-Agent": "mini-wrtn/0.1"})
        if resp.status_code == 200:
            extract = resp.json().get("extract")
            if extract:
                return extract
    except Exception as e:
        return f"검색 오류: {e}"

    if "대한민국 대통령" in query or "한국 대통령" in query:
        return search_wikipedia_ko("대한민국의 대통령")
    if "미국 대통령" in query:
        return search_wikipedia_ko("미국 대통령")

    return NOT_FOUND_MSG


# ===== /search 엔드포인트 =====

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    conversation_id = req.conversation_id or "default"
    user_query = req.query

    add_to_history(conversation_id, "user", f"[검색 질문] {user_query}")
    summarize_conversation_if_needed(conversation_id)

    snippet = search_wikipedia_ko(user_query)

    system_prompt = (
        "너는 검색 결과 + 대화 맥락을 활용하는 정보 비서야.\n"
        "항상 한국어로 자연스럽게 한 문단으로만 답해."
    )

    messages = build_llm_messages(
        conversation_id,
        user_query,
        system_prompt,
        extra_context=snippet,
    )

    # 검색 모델 (chat.completions 사용 가능 모델)
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini-search-preview",
            messages=messages,
        )
        ai_answer = completion.choices[0].message.content.strip()
    except Exception as e:
        ai_answer = f"검색 중 오류 발생: {e}"

    add_to_history(conversation_id, "assistant", f"[검색 답변] {ai_answer}")
    summarize_conversation_if_needed(conversation_id)

    return SearchResponse(answer=ai_answer)


# ===== /chat 엔드포인트 =====

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    conversation_id = req.conversation_id or "default"
    user_msg = req.message

    add_to_history(conversation_id, "user", user_msg)
    summarize_conversation_if_needed(conversation_id)

    state = get_conversation_state(conversation_id)

    # ★ 방 모델 업데이트 (첫 요청에서만 넘어오게 프론트 설계하면 됨)
    if req.model is not None:
        state["model"] = req.model

    model_name = state["model"]

    system_prompt = (
        "너는 친절한 한국어 AI 비서야.\n"
        "이전 대화 요약과 최근 맥락을 활용해 자연스럽게 이어서 답해."
    )

    messages = build_llm_messages(
        conversation_id,
        user_msg,
        system_prompt,
    )

    # === 핵심: 모델에 따라 chat.completions vs responses 분기 ===
    try:
        if model_name == "gpt-5-pro":
            # gpt-5-pro는 v1/responses 전용
            response = client.responses.create(
                model=model_name,
                # responses API는 messages 스타일도 input으로 받을 수 있음
                input=messages,
                max_output_tokens=1024,
            )

            # 안전하게 텍스트 뽑기
            ai_reply = ""
            try:
                ai_reply = response.output[0].content[0].text.strip()
            except Exception:
                # 혹시 구조가 다르면 fallback
                ai_reply = (getattr(response, "output_text", "") or "").strip()
                if not ai_reply:
                    ai_reply = "응답을 해석하는 중 오류가 발생했어."
        else:
            # 일반 chat.completions 모델들
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            ai_reply = completion.choices[0].message.content.strip()

    except Exception as e:
        ai_reply = f"서버 또는 모델 오류 발생: {e}"

    add_to_history(conversation_id, "assistant", ai_reply)
    summarize_conversation_if_needed(conversation_id)

    return ChatResponse(reply=ai_reply)

