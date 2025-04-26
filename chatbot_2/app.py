import openai
import json
import os
from tools.rag_search import setup_faiss_index, search_knowledge_base
from tools.course_actions import drop_course, add_course, excuse_course, manipulate_course
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Setup FAISS if not exist
if not os.path.exists("faiss_index/faiss.index"):
    setup_faiss_index()

# === Tools
TOOLS = {
    "search_knowledge_base": search_knowledge_base,
    "drop_course": drop_course,
    "add_course": add_course,
    "excuse_course": excuse_course,
    "manipulate_course": manipulate_course
}

TOOL_DESCRIPTIONS = """
Available tools:

1. search_knowledge_base: Search university policies and info.
2. drop_course: Drop a course (only course_code needed).
3. add_course: Add a course (only course_code needed).
4. excuse_course: Excuse a course (only course_code needed).
5. manipulate_course: Change a course (needs old_code and new_code).

Student ID is always 443102109 automatically.
"""

# === Main Chat Loop
print("👩‍🎓 KSU Smart Assistant جاهز! (اكتب 'exit' للخروج)\n")

while True:
    user_input = input("👤 You: ")

    if user_input.lower() == "exit":
        break

    # Step 1: Ask OpenAI which tool to use
    plan_response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.3,
        messages=[
            {"role": "system", "content": f"You are a planning model. Decide which tool to use. Assume student_id = 443102109 automatically. Available tools:\n{TOOL_DESCRIPTIONS}\n\nReturn only JSON like {{'tool': '...', 'params': {{...}}}}."},
            {"role": "user", "content": user_input}
        ]
    )

    plan = plan_response['choices'][0]['message']['content']

    try:
        plan = json.loads(plan.replace("'", '"'))
        tool_name = plan["tool"]
        params = plan["params"]

        # Step 2: Inject student ID if action tool
        if tool_name in ["drop_course", "add_course", "excuse_course", "manipulate_course"]:
            params["student_id"] = "443102109"

        # Step 3: Execute the tool
        if tool_name == "search_knowledge_base":
            docs, refs = TOOLS[tool_name](**params)

            context = "\n".join(docs)
            references = "\n".join(refs)

            final_response = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=0.5,
                messages=[
                    {"role": "system", "content": "أنت مساعد ذكي لجامعة الملك سعود. استخدم المستندات والمراجع للإجابة فقط."},
                    {"role": "user", "content": f"السؤال: {user_input}\n\nالمستندات:\n{context}\n\nمراجع:\n{references}\n\nأجب بدقة."}
                ]
            )

            print(f"\n🤖 Assistant:\n{final_response['choices'][0]['message']['content']}\n")
        
        elif tool_name in TOOLS:
            result = TOOLS[tool_name](**params)
            print(f"\n🤖 Assistant:\n{result}\n")
        
        else:
            print("\n⚠️ Tool not found.\n")

    except Exception as e:
        print("\n⚠️ Error:", e)
