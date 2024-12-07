"""
File containing assistant class implementations for the agent nodes of the graph
"""

from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.prebuilt import tools_condition
from langgraph.graph import END

from agentic.prompts import PRIMARY_ASSISTANT_PROMPT
from agentic.states import GraphState
from agentic.tools import *

# base assistant class for agent nodes
class Assistant:
    """
    Base class implementation for different assistants
    """
    def __init__(self, runnable: Runnable, name: str | None=None, tools: list | None=None):
        self.runnable = runnable
        self.name = "assistant" if name is None else name
        self.tools = tools if tools is not None else []

    def __call__(self, state: GraphState, config: RunnableConfig) -> dict:
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break

        return {"messages": result}

    def route_assistant(self, state: GraphState) -> str:
        pass


class PrimaryAssistant(Assistant):
    """
    Class implementation for the primary assistant.

    To create a new primary assistant, you only need to specify llm and name:
    ```python
    primary_assistant = PrimaryAssistant(llm=ChatOllama(...), name='primary_assistant')
    ```
    """
    def __init__(self, llm: Runnable, name: str | None=None):
        name = "primary_assistant" if name is None else name
        self.basic_tools = []
        self.routing_tools = [ToArchitectAssistant, ToCoderAssistant, ToTesterAssistant]
        runnable = PRIMARY_ASSISTANT_PROMPT | llm.bind_tools(
            self.basic_tools + self.routing_tools
        )
        super().__init__(runnable=runnable, name=name)     

    def route_assistant(self, state: GraphState) -> str:
        """
        Method that handles the tool routing of the primary assistant.
        """
        route = tools_condition(state)
        if route == END:
            return END
        
        tool_calls = state["messages"][-1].tool_calls
        if tool_calls:
            if tool_calls[0]["name"] == ToArchitectAssistant.__name__:
                return "enter_architect_assistant"
            elif tool_calls[0]["name"] == ToCoderAssistant.__name__:
                return "enter_coder_assistant"
            elif tool_calls[0]["name"] == ToTesterAssistant.__name__:
                return "enter_tester_assistant"
            
            # if no specialized assistant was chosen next, continue to tool node with basic tools
            return f"{self.name}_tools"
        
        raise ValueError("Invalid route")


class SecondaryAssistant(Assistant):
    """
    Class implementation for secondary assistants.
    To create a new assistant, you can use the following template:
    ```python
        name = 'coder_assistant'
        llm = ChatOllama(...)
        tools = [...]
        c_assistant = SecondaryAssistant(llm, name, tools)
    ```

    --------------------------------------------------------------------------------
    Otherwise, define parameters directly in the object creation:
    
    ```python
    assistant = SecondaryAssistant(name='coder_assistant', llm=..., tools=[...])
    ```

    --------------------------------------------------------------------------------
    """
    def route_assistant(self, state: GraphState) -> str:
        """
        Method that handles the tool routing of the secondary assistants.
        """
        route = tools_condition(state)
        if route == END:
            return END
        
        tool_calls = state["messages"][-1].tool_calls
        did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
        if did_cancel:
            return "leave_skill"

        return f"{self.name}_tools"