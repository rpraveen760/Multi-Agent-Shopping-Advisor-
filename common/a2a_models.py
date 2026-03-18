"""Pydantic v2 models for the A2A v0.3 protocol and orchestrator response contract.

Covers:
- Agent Card (discovery)
- Message / TextPart / Artifact (content)
- Task / TaskStatus (lifecycle)
- SendMessageRequest / SendMessageResponse (operations)
- JsonRpcRequest / JsonRpcResponse (transport)
- UnifiedResponse / RankedRecommendation / UnifiedSourceLink (orchestrator output)
- Helper factory functions
- JSON-RPC error code constants
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════
# JSON-RPC Error Codes
# ═══════════════════════════════════════════════════════════════════

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


# ═══════════════════════════════════════════════════════════════════
# A2A Content Models
# ═══════════════════════════════════════════════════════════════════

class TextPart(BaseModel):
    type: str = "text"
    text: str


class Message(BaseModel):
    role: str  # "user" or "agent"
    parts: list[TextPart]
    messageId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] | None = None


class Artifact(BaseModel):
    name: str
    parts: list[TextPart]
    metadata: dict[str, Any] | None = None


# ═══════════════════════════════════════════════════════════════════
# A2A Task Lifecycle
# ═══════════════════════════════════════════════════════════════════

class TaskStatus(BaseModel):
    state: str  # submitted | working | completed | failed | input-required | ...
    message: Message | None = None


class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    contextId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus
    artifacts: list[Artifact] | None = None
    history: list[TaskStatus] | None = None


# ═══════════════════════════════════════════════════════════════════
# A2A Agent Card (Discovery)
# ═══════════════════════════════════════════════════════════════════

class AgentCardSkill(BaseModel):
    id: str
    name: str
    description: str
    inputModes: list[str] = ["text/plain"]
    outputModes: list[str] = ["application/json"]


class AgentCardCapabilities(BaseModel):
    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = False


class AgentCardInterface(BaseModel):
    url: str
    transport: str = "JSONRPC"
    metadata: dict[str, Any] | None = None


class AgentCard(BaseModel):
    protocolVersion: str = "0.3.0"
    name: str
    description: str
    url: str  # Full JSON-RPC endpoint URL
    preferredTransport: str = "JSONRPC"
    additionalInterfaces: list[AgentCardInterface] | None = None
    version: str = "1.0.0"
    capabilities: AgentCardCapabilities = AgentCardCapabilities()
    defaultInputModes: list[str] = ["text/plain"]
    defaultOutputModes: list[str] = ["application/json"]
    skills: list[AgentCardSkill] = []


# ═══════════════════════════════════════════════════════════════════
# A2A Operations
# ═══════════════════════════════════════════════════════════════════

class SendMessageRequest(BaseModel):
    message: Message
    configuration: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class SendMessageResponse(BaseModel):
    task: Task | None = None
    message: Message | None = None


# ═══════════════════════════════════════════════════════════════════
# JSON-RPC 2.0 Transport
# ═══════════════════════════════════════════════════════════════════

class JsonRpcRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int
    method: str  # "SendMessage" or "GetTask"
    params: dict[str, Any] = {}


class JsonRpcError(BaseModel):
    code: int
    message: str
    data: Any | None = None


class JsonRpcResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int
    result: dict[str, Any] | None = None
    error: JsonRpcError | None = None


# ═══════════════════════════════════════════════════════════════════
# Orchestrator Response Models (used in Phase 5)
# ═══════════════════════════════════════════════════════════════════

class UnifiedSourceLink(BaseModel):
    """A single source link in the final response — product page or video URL."""
    type: str  # "product" or "video"
    title: str
    url: str
    agent: str  # which agent provided this link


class RankedRecommendation(BaseModel):
    """A single product recommendation in the final ranked output."""
    rank: int
    product_name: str
    price: str | None = None
    rating: str | None = None
    sentiment: str | None = None
    score: float | None = None
    rationale: str  # why this product was ranked here
    pros: list[str] = []
    cons: list[str] = []
    confidence: float | None = None


class UnifiedResponse(BaseModel):
    """Structured final output from the orchestrator before rendering."""
    query: str
    recommendations: list[RankedRecommendation]
    sources: list[UnifiedSourceLink]
    mode: Literal["shopping_query", "youtube_video"] = "shopping_query"
    partial: bool = False  # True if some agent data was unavailable
    notes: str | None = None  # e.g. "YouTube agent was unreachable"
    youtube_url: str | None = None
    session_id: str | None = None
    video: "VideoMetadata | None" = None
    transcript_status: str | None = None
    indexing_status: str | None = None
    extracted_product: "ExtractedProductDetails | None" = None
    chat_response: "VideoChatResponse | None" = None
    similar_products: "SimilarProductsResponse | None" = None


# ═══════════════════════════════════════════════════════════════════
# Agent Domain Models (shared across agents)
# ═══════════════════════════════════════════════════════════════════

class VideoSource(BaseModel):
    title: str
    channel: str
    url: str
    published_at: str
    view_count: str | None = None


class ReviewSummary(BaseModel):
    product_name: str
    overall_sentiment: str  # "positive" | "mixed" | "negative"
    score: float  # 1.0 - 10.0
    pros: list[str] = []  # max 5
    cons: list[str] = []  # max 5
    key_quotes: list[str] = []  # evidence snippets
    recommendation: str  # 2-3 sentence summary
    confidence: float = 0.0  # 0.0 - 1.0
    sources: list[VideoSource] = []


class Product(BaseModel):
    name: str
    price: str | None = None
    rating: str | None = None
    url: str
    key_features: list[str] = []  # max 3
    source: str  # "Amazon", "Best Buy", etc.
    evidence_urls: list[str] = []
    confidence: float = 0.0  # 0.0 - 1.0


class ProductDiscoveryResult(BaseModel):
    query: str
    products: list[Product] = []
    summary: str = ""


class TranscriptChunk(BaseModel):
    chunk_index: int
    text: str
    start_seconds: float | None = None
    end_seconds: float | None = None


class VideoMetadata(BaseModel):
    video_id: str
    video_url: str
    title: str
    channel: str
    published_at: str | None = None
    view_count: str | None = None


class ExtractedProductDetails(BaseModel):
    product_name: str
    category: str | None = None
    features: list[str] = []
    price: str | None = None
    summary: str = ""
    evidence_quotes: list[str] = []
    confidence: float = 0.0


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class YouTubeVideoRequest(BaseModel):
    youtube_url: str
    chat_message: str | None = None
    find_similar_products: bool = False
    product_mcp_url: str | None = None
    legacy_query: str | None = None
    session_id: str | None = None


class VideoChatRequest(BaseModel):
    session_id: str
    chat_message: str


class VideoChatResponse(BaseModel):
    youtube_url: str
    answer: str
    citations: list[str] = []
    confidence: float = 0.0
    session_id: str | None = None
    history: list[ConversationTurn] = []


class SimilarProductsRequest(BaseModel):
    product_name: str
    category: str | None = None
    features: list[str] = []
    price: str | None = None
    source_video_url: str
    constraints: dict[str, Any] | None = None


class SimilarProductsResponse(BaseModel):
    request: SimilarProductsRequest
    products: list[Product] = []
    summary: str = ""


class UnifiedVideoAnalysisResponse(BaseModel):
    youtube_url: str
    video: VideoMetadata | None = None
    session_id: str | None = None
    transcript_status: str = "accepted"
    indexing_status: str = "pending"
    transcript_chunks: list[TranscriptChunk] = []
    extracted_product: ExtractedProductDetails | None = None
    chat_response: VideoChatResponse | None = None
    similar_products: SimilarProductsResponse | None = None
    summary: str = ""
    partial: bool = False
    notes: str | None = None


# ═══════════════════════════════════════════════════════════════════
# Helper Factory Functions
# ═══════════════════════════════════════════════════════════════════

def make_user_message(text: str) -> Message:
    """Create a user-role Message with a single TextPart."""
    return Message(
        role="user",
        parts=[TextPart(text=text)],
    )


def make_agent_message(text: str) -> Message:
    """Create an agent-role Message with a single TextPart."""
    return Message(
        role="agent",
        parts=[TextPart(text=text)],
    )


def make_task(message: Message, state: str = "submitted") -> Task:
    """Create a Task in the given state with the message attached to status."""
    return Task(
        status=TaskStatus(state=state, message=message),
    )


def make_artifact(name: str, text: str, metadata: dict[str, Any] | None = None) -> Artifact:
    """Create an Artifact with a single TextPart."""
    return Artifact(
        name=name,
        parts=[TextPart(text=text)],
        metadata=metadata,
    )


UnifiedResponse.model_rebuild()
