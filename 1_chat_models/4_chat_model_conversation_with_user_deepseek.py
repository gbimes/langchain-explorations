from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek  # include the deepseek api
import json

# Load environment variables from .env
load_dotenv()

# Create a ChatDeepSeek model
model = ChatDeepSeek(model="deepseek-chat")


chat_history = []  # Use a list to store messages

# Set an initial system message (optional)
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)  # Add system message to chat history

# Chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))  # Add user message

    # Get AI response using history
    result = {}
    response = {}
    try:
        result = model.invoke(chat_history)
        response = result.content

        # Check if the response is a valid string before trying to parse to JSON
        if isinstance(response, str):
          try:
              json_response = json.loads(response)
              print("This is a valid JSON response")
              # If the output of the API is meant to be JSON then process it here
          except json.JSONDecodeError:
              # If the output is not a JSON, just output the text
             pass
        else:
           print(f"the response received was not a string: {response}")
           pass # if it is not a string don't process it.

        chat_history.append(AIMessage(content=response))  # Add AI message
        print(f"AI: {response}")
    except Exception as e:
      print(f"An error occurred: {e}")

print("---- Message History ----")
print(chat_history)