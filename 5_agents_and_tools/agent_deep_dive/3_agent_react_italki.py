import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Load the existing Chroma vector store
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "..", "..", "4_rag", "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_zendesk")

# Check if the Chroma vector store already exists
if os.path.exists(persistent_directory):
    print("Loading existing vector store...")
    db = Chroma(persist_directory=persistent_directory,
                embedding_function=None)
else:
    raise FileNotFoundError(
        f"The directory {persistent_directory} does not exist. Please check the path."
    )

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Create a retriever for querying the vector store
# `search_type` specifies the type of search (e.g., similarity)
# `search_kwargs` contains additional arguments for the search (e.g., number of results to return)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Create a ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o")

# Contextualize question prompt
# This system prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
# This uses the LLM to help reformulate the question based on chat history
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question prompt
# This system prompt helps the AI understand that it should provide concise answers
# based on the retrieved context and indicates what to do if the answer is unknown
qa_system_prompt = (
    "You are a user learning assistant on italki for question-answering tasks. "
    "Refer to yourself as Beni and alway respond as though you are part of the italki brand."
    "Use the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
# `create_stuff_documents_chain` feeds all retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain)


# Set Up ReAct Agent with Document Store Retriever
# Load the ReAct Docstore Prompt
react_docstore_prompt = hub.pull("hwchase17/react")

# Define Tools
def view_lesson_details(*args, **kwargs):
    return {
        "lesson_id": "L12345",
        "title": "Introduction to Spanish Grammar",
        "teacher": "Maria Gomez",
        "date": "2024-06-21",
        "duration_minutes": 60,
        "status": "Scheduled",
        "language": "Spanish",
        "topics": ["Basics", "Grammar", "Pronunciation"]
    }

def view_lesson_insights(*args, **kwargs):
    return {
        "lesson_id": "L12345",
        "completion_rate": 95,
        "engagement_score": 4.7,
        "top_topics": ["Grammar", "Pronunciation"],
        "teacher_feedback": "Excellent participation!"
    }

def nudge_take_lesson_action(*args, **kwargs):
    return {
        "nudge_type": "action_reminder",
        "preferred_action": "Do your lesson homework!",
        "action_link": "https://platform.com/lessons/L12345/homework"
    }

def nudge_explore_1on1_lessons(*args, **kwargs):
    return {
        "nudge_type": "explore_1on1",
        "recommended_teachers": [
            {"teacher_id": "T1001", "name": "Luis Fernandez", "language": "Spanish"},
            {"teacher_id": "T1002", "name": "Anna Schmidt", "language": "German"}
        ],
        "explore_link": "https://platform.com/1on1-lessons"
    }

def nudge_upcoming_1on1_lessons(*args, **kwargs):
    return {
        "nudge_type": "reminder_upcoming_lessons",
        "upcoming_lessons": [
            {
                "lesson_id": "L12346",
                "teacher": "Anna Schmidt",
                "date": "2024-06-22T15:00:00Z",
                "language": "German"
            },
            {
                "lesson_id": "L12347",
                "teacher": "Luis Fernandez",
                "date": "2024-06-24T09:30:00Z",
                "language": "Spanish"
            }
        ]
    }

def nudge_lesson_experience_sentiment(*args, **kwargs):
    return {
        "nudge_type": "feedback_request",
        "question": "How was your last lesson?",
        "response": ["🙁 Not good"],
        "feedback_link": "https://platform.com/lessons/L12345/feedback"
    }

def nudge_rebook_next_lesson_previous_teacher(*args, **kwargs):
    return {
        "nudge_type": "rebook_previous_teacher",
        "teacher": {"teacher_id": "T1001", "name": "Maria Gomez"},
        "next_available_date": "2024-06-28",
        "rebook_link": "https://platform.com/rebook?teacher_id=T1001"
    }

def nudge_book_next_lesson_new_teacher(*args, **kwargs):
    return {
        "nudge_type": "book_new_teacher",
        "new_teachers": [
            {"teacher_id": "T2002", "name": "Carlos Silva", "language": "Portuguese"},
            {"teacher_id": "T3003", "name": "Jean Rousseau", "language": "French"}
        ],
        "book_link": "https://platform.com/book-new-teacher"
    }

def nudge_book_next_lesson_new_language(*args, **kwargs):
    return {
        "nudge_type": "book_new_language",
        "language_options": [
            {"language": "Italian", "teachers_available": 3},
            {"language": "Japanese", "teachers_available": 2},
            {"language": "Russian", "teachers_available": 4}
        ],
        "discover_languages_link": "https://platform.com/discover-languages"
    }

def nudge_teacher_invite_student(*args, **kwargs):
    return {
        "nudge_type": "invite_student",
        "teacher_id": "T4001",
        "potential_students": [
            {"student_id": "S101", "name": "John Doe"},
            {"student_id": "S102", "name": "Emily Chan"}
        ],
        "invite_link": "https://platform.com/teacher/invite"
    }

def view_my_student_calendar(*args, **kwargs):
    return {
        "user_type": "student",
        "scheduled_lessons": [
            {
                "lesson_id": "L2001",
                "teacher": "Maria Gomez",
                "subject": "Spanish",
                "date": "2024-06-22T14:00:00Z",
                "duration_minutes": 60
            },
            {
                "lesson_id": "L2002",
                "teacher": "Anna Schmidt",
                "subject": "German",
                "date": "2024-06-23T10:00:00Z",
                "duration_minutes": 45
            }
        ],
        "calendar_link": "https://platform.com/my-calendar"
    }

def nudge_sync_my_calendar(*args, **kwargs):
    return {
        "nudge_type": "calendar_sync_prompt",
        "message": "Sync your lesson schedule with your personal calendar for easy reminders.",
        "supported_calendars": ["Google Calendar", "Outlook", "Apple Calendar"],
        "sync_link": "https://platform.com/sync-calendar"
    }

def view_check_my_teacher_calendar(*args, **kwargs):
    return {
        "user_type": "teacher",
        "scheduled_lessons": [
            {
                "lesson_id": "L3001",
                "student": "John Doe",
                "subject": "French",
                "date": "2024-06-22T09:00:00Z",
                "duration_minutes": 60
            },
            {
                "lesson_id": "L3002",
                "student": "Emily Chan",
                "subject": "English",
                "date": "2024-06-23T15:00:00Z",
                "duration_minutes": 30
            }
        ],
        "calendar_link": "https://platform.com/teacher-calendar"
    }

def action_sync_my_calendar(*args, **kwargs):
    return {
        "action": "sync_calendar",
        "status": "success",
        "synced_with": "Google Calendar",
        "last_sync": "2024-06-20T18:00:00Z",
        "details": "All lessons up to date with your Google Calendar."
    }

def nudge_plan_my_calendar(*args, **kwargs):
    return {
        "nudge_type": "calendar_planning",
        "message": "Plan your upcoming lessons or open time slots to optimize your schedule.",
        "plan_link": "https://platform.com/plan-calendar",
        "suggested_time_slots": [
            {"date": "2024-06-26", "time": "15:00", "duration_minutes": 60},
            {"date": "2024-06-29", "time": "10:00", "duration_minutes": 45}
        ]
    }


tools = [
    Tool(
        name="Answer Question",
        func=lambda input, **kwargs: rag_chain.invoke(
            {"input": input, "chat_history": kwargs.get("chat_history", [])}
        ),
        description="useful for when you need to answer questions about the context",
    ),
    Tool(
        name="lesson_details",
        func=view_lesson_details,
        description="useful to show learners detailed information about one of their lesson.",
    ),
    Tool(
        name="lesson_insights",
        func=view_lesson_insights,
        description="useful to show learners deduced insights from one of their lessons",
    ),
    Tool(
        name="nudge_take_lesson_action",
        func=nudge_take_lesson_action,
        description="useful to push learners to take an action on a lesson they took example to confirm that the lesson happened or request refund.",
    ),
    Tool(
        name="nudge_explore_1on1_lessons",
        func=nudge_explore_1on1_lessons,
        description="useful to encourage a non learner to try out 1on1 lessons with teachers.",
    ),
    Tool(
        name="nudge_upcoming_1on1_lessons",
        func=nudge_upcoming_1on1_lessons,
        description="useful to enocurage a learner who has an upcoming lesson to review the details of their upcoming lesson and prepare ahead.",
    ),
    Tool(
        name="nudge_lesson_experience_sentiment",
        func=nudge_lesson_experience_sentiment,
        description="useful to get information form the learner about their feeling about the just completed lesson with a teacher.",
    ),
    Tool(
        name="nudge_rebook_next_lesson_previous_teacher",
        func=nudge_rebook_next_lesson_previous_teacher,
        description="useful to encourage a learner to rebook another lesson with the previous teacher they had.",
    ),
    Tool(
        name="nudge_book_next_lesson_new_teacher",
        func=nudge_book_next_lesson_new_teacher,
        description="useful to encourage a learner to rebook another lesson with a new teacher different from the previous teacher.",
    ),
    Tool(
        name="nudge_book_next_lesson_new_language",
        func=nudge_book_next_lesson_new_language,
        description="useful to encourage a learner to start learning a new language differnet form the previous language being learnt",
    ),
        Tool(
        name="nudge_teacher_invite_student",
        func=nudge_teacher_invite_student,
        description="useful to encourage a teacher to invite a learner to take lessons with them",
    ),
    Tool(
        name="view_my_student_calendar",
        func=view_my_student_calendar,
        description="useful to show the learner's availability based on thier calendar",
    ),
    Tool(
        name="nudge_sync_my_calendar",
        func=nudge_sync_my_calendar,
        description="useful to encourage learners to sync their italki calendar to their personal calendar",
    ),
        Tool(
        name="view_check_my_teacher_calendar",
        func=view_check_my_teacher_calendar,
        description="useful to show teachers their availbility by showing the occupied availabiltiy slots on calendar ",
    ),
        Tool(
        name="action_sync_my_calendar",
        func=action_sync_my_calendar,
        description="useful to encourage a teacher to sync their personal calendar to their italki calendar",
    ),
        Tool(
        name="nudge_plan_my_calendar",
        func=nudge_plan_my_calendar,
        description="useful to encourage a teacher to plan their weekly availability",
    ),

]

# Create the ReAct Agent with document store retriever
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_docstore_prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=True,
)

chat_history = []
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    response = agent_executor.invoke(
        {"input": query, "chat_history": chat_history})
    print(f"AI: {response['output']}")

    # Update history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["output"]))
