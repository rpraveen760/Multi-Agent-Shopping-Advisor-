"""In-memory transcript chat session storage for the YouTube review agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from uuid import uuid4

from common.a2a_models import ConversationTurn, ExtractedProductDetails, VideoMetadata


@dataclass(slots=True)
class VideoChatSession:
    session_id: str
    video: VideoMetadata
    transcript_hash: str
    extracted_product: ExtractedProductDetails | None = None
    history: list[ConversationTurn] = field(default_factory=list)


class VideoSessionStore:
    """Thread-safe in-memory store for transcript chat sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, VideoChatSession] = {}
        self._lock = Lock()

    def create(
        self,
        *,
        video: VideoMetadata,
        transcript_hash: str,
        extracted_product: ExtractedProductDetails | None = None,
        session_id: str | None = None,
    ) -> VideoChatSession:
        with self._lock:
            resolved_session_id = session_id or str(uuid4())
            session = VideoChatSession(
                session_id=resolved_session_id,
                video=video,
                transcript_hash=transcript_hash,
                extracted_product=extracted_product,
            )
            self._sessions[resolved_session_id] = session
            return session

    def get(self, session_id: str) -> VideoChatSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def pop(self, session_id: str) -> VideoChatSession | None:
        with self._lock:
            return self._sessions.pop(session_id, None)

    def append_turn(self, session_id: str, *, role: str, content: str) -> VideoChatSession | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.history.append(ConversationTurn(role=role, content=content))
            return session

    def replace_history(self, session_id: str, history: list[ConversationTurn]) -> VideoChatSession | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.history = list(history)
            return session


SESSION_STORE = VideoSessionStore()
