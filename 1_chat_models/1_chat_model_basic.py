# Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/
# OpenAI Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/openai/
# Youtube Tutorial Video : https://www.youtube.com/watch?v=yF9kGESAi3M&ab_channel=aiwithbrandon 

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek  # include the deepseek api

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
# model = ChatOpenAI(model="gpt-4o-mini")

#create a ChatDeepSeek model 
#model_dps = ChatDeepSeek(model="deepseek-chat")
model_dps = ChatDeepSeek(model="deepseek-reasoner")

# Invoke the chatopenai model with a message
# result = model.invoke("What is 81 divided by 9?")
# print("Full result:")
# print(result)
# print("Content only:")
# print(result.content)



# Invoke the ChatDeepSeek model with a message
result_dps = model_dps.invoke("What is 81 divided by 9?")
print("Full result:")
print(result_dps)
print("Content only:")
print(result_dps.content)


