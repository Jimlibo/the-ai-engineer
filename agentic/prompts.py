from langchain_core.prompts import ChatPromptTemplate

PRIMARY_ASSISTANT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            """
        ),
        ("placeholder", "{messages}")
    ]
)