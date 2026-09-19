"""Bounded LangGraph loop for the autonomous study workflow."""
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph


class AgentLoopError(RuntimeError):
    """Raised when the model proposes an invalid or non-terminating plan."""


class AgentState(TypedDict, total=False):
    goal: str
    observation: dict[str, Any]
    last_tool: str
    last_tool_result: dict[str, Any]
    decision: dict[str, Any]
    trace: list[dict[str, Any]]
    tool_steps: int
    result: dict[str, Any]
    final_message: str


Observer = Callable[[AgentState], dict[str, Any]]
Decider = Callable[[AgentState], dict[str, Any]]
Tool = Callable[[dict[str, Any]], dict[str, Any]]


def _observation_summary(state: AgentState, observation: dict[str, Any]) -> str:
    if not state.get('last_tool'):
        return (
            f"发现 {len(observation.get('knowledge_points', []))} 个知识点，"
            f"读取 {len(observation.get('recent_history', []))} 条近期完成记录。"
        )
    if state['last_tool'] == 'inspect_learning_history':
        return f"学习历史分析完成，共比较 {len(observation.get('point_stats', []))} 个知识点。"
    if state['last_tool'] == 'generate_practice_question':
        question = observation.get('question', {})
        return f"练习题已生成，题目编号 {question.get('id')}。"
    if state['last_tool'] == 'answer_from_material':
        return f"已根据 {len(observation.get('source_refs', []))} 处资料内容生成回答。"
    if state['last_tool'] == 'generate_knowledge_graph':
        graph = observation.get('graph', {})
        return (
            f"知识图谱已生成，包含 {len(graph.get('nodes', []))} 个节点和"
            f" {len(graph.get('edges', []))} 条关系。"
        )
    return f"工具 {state['last_tool']} 已返回结果。"


def run_study_agent(
    goal: str,
    observe: Observer,
    decide: Decider,
    tools: dict[str, Tool],
    result_actions: set[str] | None = None,
    max_tool_steps: int = 4,
) -> AgentState:
    """Run observe -> decide -> act -> observe until the model finishes.

    The model may only select a registered tool. A hard step limit prevents an
    accidental infinite loop, while the graph state keeps every observation
    available to the next decision.
    """
    allowed_actions = {*tools, 'finish'}
    result_actions = result_actions or {'generate_practice_question'}
    if not result_actions <= tools.keys():
        raise ValueError('Every result action must be a registered tool')

    def observe_node(state: AgentState) -> dict[str, Any]:
        observation = observe(state)
        if not isinstance(observation, dict):
            raise AgentLoopError('Agent observer returned an invalid result')
        event = {
            'phase': 'observe',
            'step': state.get('tool_steps', 0),
            'summary': _observation_summary(state, observation),
        }
        return {'observation': observation, 'trace': [*state.get('trace', []), event]}

    def decide_node(state: AgentState) -> dict[str, Any]:
        decision = decide(state)
        if not isinstance(decision, dict):
            raise AgentLoopError('Agent decision must be an object')
        action = decision.get('action')
        arguments = decision.get('arguments', {})
        reason = decision.get('reason', '')
        if action not in allowed_actions:
            raise AgentLoopError(f'Agent selected an unknown action: {action}')
        if not isinstance(arguments, dict) or not isinstance(reason, str):
            raise AgentLoopError('Agent decision has invalid arguments or reason')
        if len(reason) > 500:
            raise AgentLoopError('Agent decision reason is too long')
        if action == 'finish' and 'result' not in state:
            raise AgentLoopError('Agent tried to finish before producing a result')
        if action != 'finish' and state.get('tool_steps', 0) >= max_tool_steps:
            raise AgentLoopError('Agent exceeded the tool step limit')
        normalized = {
            'action': action,
            'arguments': arguments,
            'reason': reason,
            'final_message': decision.get('final_message', ''),
        }
        event = {
            'phase': 'decide',
            'step': state.get('tool_steps', 0),
            'action': action,
            'summary': reason or f'选择动作 {action}。',
        }
        return {'decision': normalized, 'trace': [*state.get('trace', []), event]}

    def route_decision(state: AgentState) -> str:
        return 'finish' if state['decision']['action'] == 'finish' else 'act'

    def act_node(state: AgentState) -> dict[str, Any]:
        action = state['decision']['action']
        result = tools[action](state['decision']['arguments'])
        if not isinstance(result, dict):
            raise AgentLoopError(f'Tool {action} returned an invalid result')
        step = state.get('tool_steps', 0) + 1
        update: dict[str, Any] = {
            'last_tool': action,
            'last_tool_result': result,
            'tool_steps': step,
            'trace': [*state.get('trace', []), {
                'phase': 'act',
                'step': step,
                'action': action,
                'summary': f'已调用工具 {action}。',
            }],
        }
        if action in result_actions:
            update['result'] = result
        return update

    def finish_node(state: AgentState) -> dict[str, Any]:
        message = state['decision'].get('final_message')
        if not isinstance(message, str) or not message.strip():
            message = '已根据资料完成任务。'
        return {'final_message': message.strip()[:500]}

    graph = StateGraph(AgentState)
    graph.add_node('observe', observe_node)
    graph.add_node('decide', decide_node)
    graph.add_node('act', act_node)
    graph.add_node('finish', finish_node)
    graph.add_edge(START, 'observe')
    graph.add_edge('observe', 'decide')
    graph.add_conditional_edges('decide', route_decision, {'act': 'act', 'finish': 'finish'})
    graph.add_edge('act', 'observe')
    graph.add_edge('finish', END)

    return graph.compile().invoke(
        {'goal': goal, 'trace': [], 'tool_steps': 0},
        config={'recursion_limit': max_tool_steps * 3 + 6},
    )
