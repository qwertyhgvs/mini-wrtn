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
#       "history": [ {"role": "user", "content": "..."}, {"role": "assistant", ...}, ... ],
#       "summary": "지금까지 대화 요약 텍스트",
#       "messages_since_last_summary": 7,
#   },
#   ...
# }
conversation_state: Dict[str, Dict] = {}

NOT_FOUND_MSG = "위키백과에서 해당 주제에 대한 요약을 찾지 못했어."


# ==== Pydantic 모델들 ====

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    reply: str


class SearchRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = "default"


class SearchResponse(BaseModel):
    # 이제 최종 답변만 프론트로 보내기 (요약은 LLM에만 사용)
    answer: str


# ==== 루트에서 바로 UI로 리다이렉트 ====

@app.get("/", include_in_schema=False)
def root():
    # Render에서 https://mini-wrtn.onrender.com/ 로 들어오면 바로 index.html 띄우기
    return RedirectResponse(url="/static/index.html")


# ===== 대화방 상태 가져오기 / 초기화 =====

def get_conversation_state(conversation_id: str) -> Dict:
    """
    해당 conversation_id에 대한 상태 딕셔너리를 가져오고,
    없으면 새로 초기화해서 만든 후 반환.
    """
    if conversation_id is None:
        conversation_id = "default"

    if conversation_id not in conversation_state:
        conversation_state[conversation_id] = {
            "history": [],                  # 최근 대화들
            "summary": "",                  # 지금까지 요약
            "messages_since_last_summary": 0
        }
    return conversation_state[conversation_id]


# ===== 메모리 관리용 헬퍼 함수들 =====

def add_to_history(conversation_id: str, role: str, content: str):
    """
    해당 conversation_id의 대화 히스토리에 한 줄 추가
    (role: 'user' 또는 'assistant')
    """
    state = get_conversation_state(conversation_id)
    state["history"].append({"role": role, "content": content})
    state["messages_since_last_summary"] += 1


def summarize_conversation_if_needed(conversation_id: str):
    """
    대화가 어느 정도 쌓이면(예: 10줄 이상) 요약을 한 번 돌리고,
    요약본을 업데이트한 뒤 히스토리는 최근 몇 줄만 남겨둔다.
    """
    state = get_conversation_state(conversation_id)

    SUMMARY_TRIGGER = 10  # 이 개수 이상 쌓이면 요약 시도
    KEEP_RECENT = 5       # 요약 후 히스토리에서 유지할 최근 메시지 수

    if state["messages_since_last_summary"] < SUMMARY_TRIGGER:
        return
    if not state["history"]:
        return

    # 지금까지 쌓인 히스토리 텍스트로 만들기
    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in state["history"]
    )

    # 요약용 프롬프트
    system_prompt = (
        "너는 대화를 간단하게 요약하는 도우미야.\n"
        "- 사용자와 AI의 대화를 핵심 정보 위주로 한두 단락으로 정리해.\n"
        "- 사용자의 목표, 선호, 진행 중인 작업(예: 프로젝트, 글쓰기 주제 등)을 중심으로 작성해.\n"
        "- 불필요한 인사말이나 사소한 잡담은 최대한 줄여."
    )
    user_content = (
        f"이전까지의 요약:\n{state['summary'] or '(없음)'}\n\n"
        f"새로운 대화 기록:\n{history_text}\n\n"
        "위 내용을 합쳐서, 앞으로 참고할 수 있도록 짧게 요약해줘."
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    new_summary = completion.choices[0].message.content.strip()
    state["summary"] = new_summary

    # 히스토리는 최근 KEEP_RECENT개만 남기기
    if len(state["history"]) > KEEP_RECENT:
        state["history"] = state["history"][-KEEP_RECENT:]

    # 요약 이후 카운터 초기화
    state["messages_since_last_summary"] = 0


def build_llm_messages(
    conversation_id: str,
    user_msg: str,
    system_prompt: str,
    extra_context: Optional[str] = None,
):
    """
    - system_prompt: 기본 역할/스타일 지시
    - (대화방별) summary: 이전까지의 요약
    - (대화방별) history: 최근 몇 줄
    - extra_context: 검색 결과 등 추가 컨텍스트
    를 한 번에 합쳐서 OpenAI로 보낼 messages 리스트 생성
    """
    state = get_conversation_state(conversation_id)
    summary = state["summary"]
    history = state["history"]

    messages = []

    # 1) 기본 시스템 프롬프트 + 요약된 과거 대화
    full_system = system_prompt
    if summary:
        full_system += "\n\n[요약된 이전 대화]\n" + summary
    messages.append({"role": "system", "content": full_system})

    # 2) 최근 히스토리를 한 덩어리로 제공
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

    # 3) 이번 사용자 질문 + (옵션) 검색 결과 등을 묶어서 user 메시지로
    user_content_parts = []
    if extra_context:
        user_content_parts.append(f"추가 컨텍스트(검색 등):\n{extra_context}")
    user_content_parts.append(f"사용자 질문: {user_msg}")

    messages.append(
        {
            "role": "user",
            "content": "\n\n".join(user_content_parts),
        }
    )

    return messages


# ====  검색용 헬퍼 함수  ====

def search_wikipedia_ko(query: str) -> str:
    """
    한글 위키백과 요약을 가져오는 간단한 함수.
    질문형 문장이어도 일단 그대로 시도하고,
    안 되면 '대한민국의 대통령' / '미국의 대통령' 같은 예외 처리도 해볼 수 있음.
    """
    base_url = "https://ko.wikipedia.org/api/rest_v1/page/summary/"
    url = base_url + quote(query)

    try:
        resp = requests.get(url, headers={"User-Agent": "mini-wrtn/0.1"})
    except Exception as e:
        # 네트워크 오류 등
        return f"검색 중 오류가 발생했어: {e}"

    if resp.status_code == 200:
        data = resp.json()
        extract = data.get("extract")
        if extract:
            return extract

    # 간단 예외 처리: 대통령 관련 질문일 때는 정식 문서 제목으로 한 번 더 시도
    if "한국 대통령" in query or "대한민국 대통령" in query:
        return search_wikipedia_ko("대한민국의 대통령")
    if "미국 대통령" in query:
        return search_wikipedia_ko("미국 대통령")

    # 그래도 못 찾으면 고정 문구 반환
    return NOT_FOUND_MSG


# ====  /search 엔드포인트  ====

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    conversation_id = req.conversation_id or "default"
    user_query = req.query

    # 1) 유저의 검색 모드 질문도 히스토리에 기록
    add_to_history(conversation_id, "user", f"[검색모드 질문] {user_query}")
    summarize_conversation_if_needed(conversation_id)

    # 2) 위키 검색
    snippet = search_wikipedia_ko(user_query)

    # 3) LLM에게 넘길 시스템 프롬프트
    system_prompt = (
        "너는 '검색 결과 + 일반 지식 + 대화 문맥'을 함께 사용하는 한국어 정보 비서야.\n"
        "- 아래에 제공되는 '추가 컨텍스트(검색 등)'이 있으면 우선 참고해.\n"
        "- 그 내용이 부족하거나 없으면, 네가 알고 있는 일반 지식과 이전 대화 내용을 활용해서 최대한 정확하게 답해.\n"
        "- 최신 정보는 아닐 수 있으니, 헷갈리면 '정확하지 않을 수 있어' 정도로만 덧붙여.\n"
        "- 최종적으로는 사용자 질문에 대한 한 문단짜리 답변 하나만 한국어로 자연스럽게 출력해."
    )

    extra_context = snippet if snippet else "검색 결과가 없었어."

    messages = build_llm_messages(
        conversation_id=conversation_id,
        user_msg=user_query,
        system_prompt=system_prompt,
        extra_context=extra_context,
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    ai_answer = completion.choices[0].message.content.strip()

    # 4) AI 답변도 히스토리에 추가
    add_to_history(conversation_id, "assistant", f"[검색모드 답변] {ai_answer}")
    summarize_conversation_if_needed(conversation_id)

    return SearchResponse(
        answer=ai_answer,
    )


# ====  /chat 엔드포인트 (LLM 단독 + 메모리)  ====

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    conversation_id = req.conversation_id or "default"
    user_msg = req.message

    # 1) 대화 히스토리에 유저 메시지 저장
    add_to_history(conversation_id, "user", user_msg)
    summarize_conversation_if_needed(conversation_id)

    # 2) 이번 요청에 사용할 messages 생성
    system_prompt = (
        "너는 친절한 한국어 AI 비서야. 답변은 짧고 명확하게 해.\n"
        "가능하면 이전 대화의 맥락(요약 + 최근 메시지)을 고려해서 자연스럽게 이어서 대답해."
    )

    messages = build_llm_messages(
        conversation_id=conversation_id,
        user_msg=user_msg,
        system_prompt=system_prompt,
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    ai_reply = completion.choices[0].message.content.strip()

    # 3) AI 답변도 히스토리에 저장
    add_to_history(conversation_id, "assistant", ai_reply)
    summarize_conversation_if_needed(conversation_id)

    return ChatResponse(reply=ai_reply)
