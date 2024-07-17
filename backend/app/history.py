from typing import Dict, Any, List
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseChatMessageHistory, BaseMessage
from langchain.schema import HumanMessage, AIMessage, SystemMessage

class CustomChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, max_messages=10):
        self.messages: List[BaseMessage] = []
        self.max_messages = max_messages

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def clear(self) -> None:
        self.messages = []

class CustomConversationBufferMemory(ConversationBufferMemory):
    def __init__(self, *args, max_messages=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat_memory = CustomChatMessageHistory(max_messages=max_messages)

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"history": self.chat_memory.messages}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        input_str = inputs.get("input", "")
        output_str = outputs.get("output", "")
        
        self.chat_memory.add_message(HumanMessage(content=input_str))
        self.chat_memory.add_message(AIMessage(content=output_str))

    def clear(self) -> None:
        self.chat_memory.clear()