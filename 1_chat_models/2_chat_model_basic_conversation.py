from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek  # include the deepseek api

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")
model_dps = ChatDeepSeek(model="deepseek-chat")

# SystemMessage:
#   Message for priming AI behavior, usually passed in as the first of a sequenc of input messages.


# HumanMessagse:
#   Message from a human to the AI model.
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
]

# AIMessage:
#   Message from an AI.
messages2 = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="What is 10 times 5?"),
]


# Invoke the model with messages (Other models not DeepSeek)
#result = model.invoke(messages)
#print(f"Answer from AI: {result.content}")

# Invoke the model but try out with the AI system messages included  (DeepSeek)
# result = model.invoke(messages2)
# print(f"Answer from AI: {result.content}")



# Invoke the model with messages (DeepSeek)
# result = model_dps.invoke(messages)
# print(f"Answer from AI: {result.content}")

# Invoke the model but try out with the AI system messages included  (DeepSeek)
resultz = model_dps.invoke(messages2)
print(f"Answer from AI: {resultz.content}")
