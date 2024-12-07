"""
File containing the state class implementation for the stateful graph
"""

from typing import Annotated, TypedDict, Literal
from langgraph.graph.message import AnyMessage, add_messages

from agentic.utils import update_dialog_stack

# State class for the workflow
class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    dialog_state: Annotated[
        list[
            Literal[
                "primary_assistant",
                "architect_assistant",
                "coder_assistant",
                "tester_assistant"
            ]
        ],
        update_dialog_stack,
    ]