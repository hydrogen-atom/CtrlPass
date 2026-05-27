from typing import Any, Dict, List, Optional, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except ModuleNotFoundError:
    START = "__start__"
    END = "__end__"

    class _CompiledStateGraph:
        def __init__(self, nodes, edges, conditional_edges):
            self.nodes = nodes
            self.edges = edges
            self.conditional_edges = conditional_edges

        def invoke(self, initial_state):
            state = dict(initial_state)
            current = self.edges[START]

            while current != END:
                node_result = self.nodes[current](state) or {}
                state.update(node_result)

                if current in self.conditional_edges:
                    router, mapping = self.conditional_edges[current]
                    current = mapping[router(state)]
                else:
                    current = self.edges[current]

            return state

    class StateGraph:
        def __init__(self, _state_type):
            self.nodes = {}
            self.edges = {}
            self.conditional_edges = {}

        def add_node(self, name, func):
            self.nodes[name] = func

        def add_edge(self, source, target):
            self.edges[source] = target

        def add_conditional_edges(self, source, router, mapping):
            self.conditional_edges[source] = (router, mapping)

        def compile(self):
            return _CompiledStateGraph(self.nodes, self.edges, self.conditional_edges)

from utils.document_processor import DocumentProcessor
from utils.qwen_client import QwenClient
from utils.vector_store import RetrievalStrategy


SUPPORTED_TOOLS = {
    "retrieve_context",
    "answer_question",
    "generate_exercises",
    "generate_knowledge_map",
}

DOCUMENT_REQUIRED_TOOLS = SUPPORTED_TOOLS


class AgentState(TypedDict, total=False):
    user_input: str
    current_file_path: str
    has_document: bool
    intent: str
    tool_name: str
    tool_input: str
    tool_reason: str
    retrieved_context: str
    document_content: str
    tool_result: Any
    display_type: str
    messages: List[str]
    error: str


class CtrlPassAgent:
    def __init__(
        self,
        qwen_api_key: str,
        vector_store_manager: Any,
        qa_chain: Any,
        exercise_generator: Any,
        knowledge_mapper: Any,
    ):
        self.vector_store_manager = vector_store_manager
        self.qa_chain = qa_chain
        self.exercise_generator = exercise_generator
        self.knowledge_mapper = knowledge_mapper
        self.document_processor = DocumentProcessor()
        self.qwen_client = QwenClient(qwen_api_key)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("initialize", self._initialize)
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("validate_requirements", self._validate_requirements)
        workflow.add_node("prepare_inputs", self._prepare_inputs)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("execute_tool", self._execute_tool)
        workflow.add_node("format_response", self._format_response)
        workflow.add_node("handle_error", self._handle_error)
        workflow.add_node("finalize", self._finalize)

        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "classify_intent")
        workflow.add_edge("classify_intent", "validate_requirements")
        workflow.add_conditional_edges(
            "validate_requirements",
            self._route_after_validation,
            {
                "handle_error": "handle_error",
                "prepare_inputs": "prepare_inputs",
            },
        )
        workflow.add_conditional_edges(
            "prepare_inputs",
            self._route_after_prepare,
            {
                "retrieve_context": "retrieve_context",
                "execute_tool": "execute_tool",
            },
        )
        workflow.add_edge("retrieve_context", "execute_tool")
        workflow.add_edge("execute_tool", "format_response")
        workflow.add_edge("format_response", "finalize")
        workflow.add_edge("handle_error", "finalize")
        workflow.add_edge("finalize", END)
        return workflow.compile()

    def run(self, user_input: str, current_file_path: Optional[str]) -> Dict[str, Any]:
        result = self.graph.invoke(
            {
                "user_input": user_input,
                "current_file_path": current_file_path or "",
            },
        )
        return {
            "tool_name": result.get("tool_name", ""),
            "tool_reason": result.get("tool_reason", ""),
            "tool_result": result.get("tool_result"),
            "display_type": result.get("display_type", "text"),
            "error": result.get("error", ""),
            "messages": result.get("messages", []),
        }

    def _initialize(self, state: AgentState) -> AgentState:
        current_file_path = state.get("current_file_path", "")
        return {
            "has_document": bool(current_file_path),
            "messages": [
                "Initialized agent workflow.",
                "Uploaded document detected." if current_file_path else "No uploaded document detected.",
            ],
        }

    def _classify_intent(self, state: AgentState) -> AgentState:
        user_input = state["user_input"]
        messages = list(state.get("messages", []))
        prompt = f"""
You are the CtrlPass planner. Select exactly one tool for the user's request.

Available tools:
1. retrieve_context: Return the most relevant excerpts from the uploaded document.
2. answer_question: Answer a question using the uploaded document as study material.
3. generate_exercises: Generate practice exercises from the uploaded document.
4. generate_knowledge_map: Generate a knowledge map or mind map from the uploaded document.

Return JSON only:
{{
  "intent": "retrieve_context|answer_question|generate_exercises|generate_knowledge_map",
  "tool_name": "retrieve_context|answer_question|generate_exercises|generate_knowledge_map",
  "tool_input": "input for the selected tool",
  "reason": "one-sentence reason for the choice"
}}

User request: {user_input}
""".strip()

        try:
            plan_text = self.qwen_client.generate(
                prompt,
                temperature=0.1,
                max_tokens=300,
            )
            plan = QwenClient.extract_json_block(plan_text)
            tool_name = plan.get("tool_name", "").strip()
            intent = plan.get("intent", tool_name).strip() or tool_name
            tool_input = plan.get("tool_input", user_input).strip() or user_input
            reason = plan.get("reason", "").strip()

            if tool_name not in SUPPORTED_TOOLS:
                raise ValueError(f"Unsupported tool from planner: {tool_name}")

            messages.append(f"Intent classified as {intent}.")
            messages.append(f"Selected tool: {tool_name}.")
            if reason:
                messages.append(f"Planner reason: {reason}")

            return {
                "intent": intent,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_reason": reason or "Planner selected the most relevant tool.",
                "messages": messages,
            }
        except Exception:
            tool_name = self._fallback_tool_selection(user_input)
            fallback_reason = "Planner fallback selected the closest matching tool."
            messages.append(f"Planner fallback selected tool: {tool_name}.")
            messages.append(fallback_reason)
            return {
                "intent": tool_name,
                "tool_name": tool_name,
                "tool_input": user_input,
                "tool_reason": fallback_reason,
                "messages": messages,
            }

    def _validate_requirements(self, state: AgentState) -> AgentState:
        tool_name = state.get("tool_name", "")
        messages = list(state.get("messages", []))

        if tool_name not in SUPPORTED_TOOLS:
            messages.append("Validation failed: unsupported tool selection.")
            return {
                "error": f"Unsupported tool selected: {tool_name}",
                "messages": messages,
            }

        if tool_name in DOCUMENT_REQUIRED_TOOLS and not state.get("has_document", False):
            messages.append("Validation failed: this request needs an uploaded document.")
            return {
                "error": "Please upload and process a document before using this capability.",
                "messages": messages,
            }

        messages.append("Validation passed.")
        return {"messages": messages}

    def _prepare_inputs(self, state: AgentState) -> AgentState:
        tool_name = state["tool_name"]
        tool_input = state.get("tool_input", state["user_input"])
        messages = list(state.get("messages", []))

        if tool_name in {"generate_exercises", "generate_knowledge_map"}:
            content = self._load_document_content(state.get("current_file_path", ""))
            messages.append("Loaded full document content for downstream tool execution.")
            return {
                "tool_input": tool_input,
                "document_content": content,
                "messages": messages,
            }

        messages.append("Prepared query input for retrieval-aware execution.")
        return {
            "tool_input": tool_input,
            "messages": messages,
        }

    def _retrieve_context(self, state: AgentState) -> AgentState:
        tool_input = state.get("tool_input", state["user_input"])
        tool_name = state.get("tool_name", "")
        messages = list(state.get("messages", []))

        # Strategy routing: precise for raw retrieval, balanced for Q&A
        if tool_name == "retrieve_context":
            strategy = RetrievalStrategy.PRECISE
        elif tool_name == "answer_question":
            strategy = RetrievalStrategy.BALANCED
        else:
            strategy = RetrievalStrategy.BALANCED

        context = self.vector_store_manager.build_context(
            tool_input,
            use_enhanced=True,
            max_tokens=1500,
            strategy=strategy,
        )
        messages.append(f"Retrieved relevant context from the vector store (strategy={strategy.value}).")
        return {
            "retrieved_context": context or "",
            "messages": messages,
        }

    def _execute_tool(self, state: AgentState) -> AgentState:
        tool_name = state["tool_name"]
        tool_input = state.get("tool_input", state["user_input"])
        messages = list(state.get("messages", []))

        try:
            if tool_name == "retrieve_context":
                messages.append("Returned retrieved context directly to the user.")
                return {
                    "tool_result": state.get("retrieved_context", ""),
                    "display_type": "context",
                    "messages": messages,
                }

            if tool_name == "answer_question":
                answer = self.qa_chain.get_answer(
                    tool_input,
                    context_override=state.get("retrieved_context", ""),
                )
                messages.append("Generated answer using the retrieved context.")
                return {
                    "tool_result": answer,
                    "display_type": "text",
                    "messages": messages,
                }

            if tool_name == "generate_exercises":
                exercises = self.exercise_generator.generate_exercises(
                    state.get("document_content", ""),
                )
                messages.append("Generated exercises from the uploaded document.")
                return {
                    "tool_result": exercises,
                    "display_type": "exercises",
                    "messages": messages,
                }

            if tool_name == "generate_knowledge_map":
                mindmap_data = self.knowledge_mapper.generate_mindmap(
                    state.get("document_content", ""),
                )
                messages.append("Generated knowledge map data from the uploaded document.")
                return {
                    "tool_result": mindmap_data,
                    "display_type": "mindmap",
                    "messages": messages,
                }

            raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as exc:
            messages.append(f"Tool execution failed: {exc}")
            return {
                "tool_result": None,
                "display_type": "error",
                "error": str(exc),
                "messages": messages,
            }

    def _format_response(self, state: AgentState) -> AgentState:
        tool_result = state.get("tool_result")
        display_type = state.get("display_type", "text")
        messages = list(state.get("messages", []))

        if state.get("error"):
            return state

        if display_type == "exercises" and not tool_result:
            messages.append("Formatting detected empty exercise output.")
            return {
                "display_type": "error",
                "error": "Exercise generation returned no results.",
                "messages": messages,
            }

        if display_type == "mindmap" and not tool_result:
            messages.append("Formatting detected empty knowledge map output.")
            return {
                "display_type": "error",
                "error": "Knowledge map generation returned no results.",
                "messages": messages,
            }

        if display_type in {"text", "context"} and not tool_result:
            messages.append("Formatting detected an empty text response.")
            return {
                "display_type": "error",
                "error": "The selected tool returned an empty response.",
                "messages": messages,
            }

        if display_type in {"text", "context"}:
            messages.append("Response formatted as text output.")
        elif display_type == "exercises":
            messages.append("Response formatted as exercise content.")
        elif display_type == "mindmap":
            messages.append("Response formatted as mind map data.")

        return {"messages": messages}

    def _handle_error(self, state: AgentState) -> AgentState:
        messages = list(state.get("messages", []))
        error_message = state.get("error", "An unexpected error occurred.")
        messages.append(f"Workflow stopped with error: {error_message}")
        return {
            "tool_result": error_message,
            "display_type": "error",
            "messages": messages,
        }

    def _finalize(self, state: AgentState) -> AgentState:
        return {
            "tool_name": state.get("tool_name", ""),
            "tool_reason": state.get("tool_reason", ""),
            "tool_result": state.get("tool_result"),
            "display_type": state.get("display_type", "text"),
            "error": state.get("error", ""),
            "messages": state.get("messages", []),
        }

    def _load_document_content(self, file_path: str) -> str:
        if not file_path:
            raise ValueError("No uploaded document is available for this tool.")

        documents = self.document_processor.load_document(file_path)
        if not documents:
            raise ValueError("Failed to load the uploaded document.")

        content = "\n".join(doc.page_content for doc in documents)
        if not content.strip():
            raise ValueError("The uploaded document is empty.")
        return content

    @staticmethod
    def _route_after_validation(state: AgentState) -> str:
        if state.get("error"):
            return "handle_error"
        return "prepare_inputs"

    @staticmethod
    def _route_after_prepare(state: AgentState) -> str:
        if state.get("tool_name") in {"retrieve_context", "answer_question"}:
            return "retrieve_context"
        return "execute_tool"

    @staticmethod
    def _fallback_tool_selection(user_input: str) -> str:
        lowered = user_input.lower()
        if any(keyword in lowered for keyword in ["知识图谱", "思维导图", "脑图", "mindmap"]):
            return "generate_knowledge_map"
        if any(keyword in lowered for keyword in ["练习", "出题", "测验", "quiz", "习题"]):
            return "generate_exercises"
        if any(keyword in lowered for keyword in ["检索", "找原文", "相关片段", "上下文", "context"]):
            return "retrieve_context"
        return "answer_question"
