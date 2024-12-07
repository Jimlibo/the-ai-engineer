from typing import Literal
from langchain_core.tools import tool 
from pydantic import BaseModel, Field

##### Primary Tools for delegating tasks to specialized assistants #####
class ToArchitectAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle project structure and directory creation"""

    request: str = Field(
        description="Any necessary followup questions the architect assistant should clarify before proceeding."
    )


class ToCoderAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle code writing into files previously created from Architect Assistant"""

    request: str = Field(
        description="Any necessary followup questions the coder assistant should clarify before proceeding."
    )


class ToTesterAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle unittest writing for the code previously written from Coder Assistant"""

    request: str = Field(
        description="Any necessary followup questions the tester assistant should clarify before proceeding."
    )


#### Tool to cancel the task and return back to primary assistant ####
class CompleteOrEscalate(BaseModel):
    """
    A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs.
    """

    cancel: bool = True
    reason: str

    class Config:
        json_schema_extra = {
            "example 1": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I do not have the expertise to match the user's needs.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I can try using a different tool",
            },
        }