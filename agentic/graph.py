
import os
import logging
import argparse

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama, OllamaLLM

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agentic.assistants import PrimaryAssistant, SecondaryAssistant
from agentic.states import GraphState
from agentic.utils import (
    create_tool_node_with_fallback,
    create_entry_node, format_name,
    pop_dialog_state,
    route_to_workflow
)

from rich.console import Console
console = Console()

# create logging directory and configure logger behaviour
if not os.path.exists("logs"):
    os.makedirs("logs")
logging.basicConfig(filename="logs/agent_graph.log", level=logging.INFO, filemode="a",
                     format="%(asctime)s - %(levelname)s - %(message)s")


from dotenv import load_dotenv
load_dotenv()


class AgentGraph:
    def __init__(
        self,
        primary_model_name: str="",
        architect_model_name: str="",
        coder_model_name: str="",
        tester_model_name: str="",
        thread_id: str = None,
    ):
        # define graph configuration
        trd_id = "0" if (thread_id is None or thread_id == "") else str(thread_id)
        self.config = {"configurable": {"thread_id": trd_id}}

        # initialize primary llm model
        ollama_url = os.getenv("OLLAMA_URL")
        self.primary_llm = ChatOllama(
            base_url=ollama_url,
            model=primary_model_name,
            temperature=0.1,
            num_predict=-2,
        )

        # initialize secondary llm models
        self.architect_llm = OllamaLLM(base_url=ollama_url, model=architect_model_name)
        self.coder_llm = OllamaLLM(base_url=ollama_url, model=coder_model_name)
        self.tester_llm = OllamaLLM(base_url=ollama_url, model=tester_model_name)

    def define_workflow(self, draw_image: bool=False):
        """
        Defines and compiles the agentic graph workflow. If you want to also get an image of the compiled graph,
        set parameter `draw_image = True`
        """
        # initialize assistants and tools
        primary_assistant = PrimaryAssistant(self.primary_llm)
        architect_tools = []
        coder_tools = []
        tester_tools = []

        secondary_assistants = [
            SecondaryAssistant(self.architect_llm, "architect_assistant", architect_tools),
            SecondaryAssistant(self.coder_llm, "coder_assistant", coder_tools),
            SecondaryAssistant(self.tester_llm, "tester_assistant", tester_tools)
        ]

        # define graph builder
        builder = StateGraph(GraphState)

        # define secondary assistant nodes and edges
        for assistant in secondary_assistants:
            builder.add_node(
                f"enter_{assistant.name}",
                create_entry_node(format_name(assistant.name), assistant.name),
            )
            builder.add_node(assistant.name, assistant)
            builder.add_node(
                f"{assistant.name}_tools",
                create_tool_node_with_fallback(assistant.tools),
            )

            builder.add_edge(f"enter_{assistant.name}", assistant.name)
            builder.add_edge(f"{assistant.name}_tools", assistant.name)
            builder.add_conditional_edges(
                assistant.name,
                assistant.route_assistant,
                [
                    f"{assistant.name}_tools",
                    "leave_skill",
                    END,
                ]
            )

        # Primary assistant
        builder.add_node("primary_assistant", primary_assistant)
        builder.add_node(
            "primary_assistant_tools",
            create_tool_node_with_fallback(primary_assistant.basic_tools),
        )
        # add leave_skill node and edge to escalate from secondary assistants back to the primary one
        builder.add_node("leave_skill", pop_dialog_state)

        builder.add_conditional_edges(
            "primary_assistant",
            primary_assistant.route_assistant,
            [
                "enter_device_assistant",
                "enter_telco_contract_assistant",
                "primary_assistant_tools",
                END,
            ],
        )
        builder.add_edge("primary_assistant_tools", "primary_assistant")
        builder.add_edge("leave_skill", "primary_assistant")

        # add option to go directly to last assistant from start, ignoring the primary assistant
        builder.add_conditional_edges(START, route_to_workflow)

        # Compile graph
        self.graph = builder.compile(checkpointer=MemorySaver())

        if draw_image:
            self.graph.get_graph().draw_mermaid_png(output_file_path="agent_graph.png")

    def run_graph_flow(self, user_input: str, silent: bool=False) -> str:
        """
        Method that runs a user query through the graph and returns the response
        as a string and the checkpoint of the graph.

        ------------------------------------------------------------------------

        To ignore logging, set parameter `silent` to True.
        """
        for s in self.graph.stream(
            {"messages": [HumanMessage(content=user_input, name="user")]},
            config=self.config,
        ):
            if "__end__" not in s and not silent:
                logging.info(s)

        # get the message that is intended for the user
        to_user_message = (
            self.graph.get_state(config=self.config).values["messages"][-1].content
        )

        return to_user_message


def parse_input():
    parser = argparse.ArgumentParser(prog="AIEngineer")
    parser.add_argument("-p", "--primary-model-name", action="store", dest="primary_model_name", default="llama3.1:8b",
                        required=False, help="Ollama LLM name for primary agent(default: llama3.1:8b)")
    parser.add_argument("-a", "--architect-model-name", action="store", dest="architect_model_name", default="llama3.1:8b",
                        required=False, help="Ollama LLM name for architect agent (default: llama3.1:8b)")
    parser.add_argument("-c", "--coder-model-name", action="store", dest="coder_model_name", default="llama3.1:8b",
                        required=False, help="Ollama LLM name for coding agent (default: llama3.1:8b)")
    parser.add_argument("--tester-model-name", action="store", dest="tester_model_name", default="llama3.1:8b",
                        required=False, help="Ollama LLM name for testing agent (default: llama3.1:8b)")
    parser.add_argument("-t", "--thread-id", dest="thread_id", default="", required=False,
                         help="The conversation thread id (default empty to create a new conversation thread)")
    parser.add_argument("-s", "--silent", action="store_true", default=False, dest="silent",
                        help="Whether to write logs into log file during execution (default: False)")
    
    return parser.parse_args()


def main(args):
    # initialize the agent and compile the graph
    ag = AgentGraph(
        primary_model_name=args.primary_model_name, 
        architect_model_name=args.architect_model_name,
        coder_model_name=args.coder_model_name,
        tester_model_name=args.tester_model_name,
        thread_id=args.thread_id  
    )
    ag.define_workflow()

    # user interaction
    while True:
        user_input = console.input("[blue bold]User: ")

        with console.status("[cyan]Generating response"):
            agent_response = ag.run_graph_flow(user_input=user_input, silent=args.silent)

        console.print("[green bold]" + agent_response)

        # end condition
        if agent_response == "Assistant: Goodbye!":
            break


if __name__ == "__main__":
    ARGS = parse_input()
    main(ARGS)