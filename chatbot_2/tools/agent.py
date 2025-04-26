import json
from typing import Any
import openai
from pydantic import BaseModel

from chatbot_2.tools.course_actions import add_course, drop_course, excuse_course, manipulate_course
from chatbot_2.tools.rag_search import search_knowledge_base

MODEL_NAME = "gpt-4o"



class ConversationMemory:
    def __init__(self, max_history_length=10000):
        self.memory = []
        self.max_history_length = max_history_length
        self.length = 0

    def add_interaction(self, user_query: str, bot_response: str):
        self.memory.append({"user": user_query, "bot": bot_response})
        self.length += len(str({"user": user_query, "bot": bot_response}))
        if self.length > self.max_history_length:
            self.length -= len(str(self.memory[0]))
            self.memory = self.memory[1:]

    def get_conversation_context(self) -> str:
        context = "Conversation History:\n"
        for interaction in self.memory:
            context += f"User: {interaction['user']}\n"
            context += f"Bot: {interaction['bot']}\n"
        return context


TOOLS = {
    "search_knowledge_base": search_knowledge_base,
    "drop_course": drop_course,
    "add_course": add_course,
    "excuse_course": excuse_course,
    "manipulate_course": manipulate_course
}

TOOL_DESCRIPTIONS = """
1. general: use this when you want dont know what to use.
2. search_knowledge_base: Search FAQs, university policies, procedures, and info, use this when you need content to answer the user query (only "query" needed in params).
3. drop_course: Drop a course (only "course_code" needed in params).
4. add_course: Add a course (only "course_code" needed in params).
5. excuse_course: Excuse a course (only "course_code" needed in params).
6. manipulate_course: Change a course ("old_code" and "new_code" needed in params).
"""

class Plan(BaseModel):
    tool: str
    params: dict[str, Any]

class KSUAgent:
    def __init__(self):
        self.memory = ConversationMemory()

    def get_plan(self, user_input: str) -> Plan:
        try:
            plan_response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": f"""You are a planning model. Decide which tool to use. Assume student_id = 443102109 automatically. Available tools:\n{TOOL_DESCRIPTIONS}\n\nReturn only JSON like {{"tool": "...", "params": {{...}}}}."""},
                    {"role": "system", "content": self.memory.get_conversation_context()},
                    {"role": "user", "content": user_input}]
            )

            res = plan_response['choices'][0]['message']['content']
            print(res)
            plan = Plan.model_validate_json(res)
        
            return plan
        except Exception as e:
            print(f'ERROR: {e}')
            return f'ERROR: {e}'
    
    def execute_plan(self, user_input: str, plan: Plan) -> str:
        print(f"executing {plan} with query={user_input}")
        try:
            if plan.tool in {"drop_course", "add_course", "excuse_course", "manipulate_course"}:
                plan.params["student_id"] = "443102109"
            
            if plan.tool == "general" or plan.tool not in TOOLS:
                if plan.tool not in TOOLS:
                    print(f'wrong tool: {plan}')
                response = self.respond(user_input)
                self.memory.add_interaction(user_input, response)
                return response


            if plan.tool == "search_knowledge_base":
                docs, refs = TOOLS[plan.tool](**plan.params)
                context = ""
                for d, r in zip(docs, refs):
                    context += "مرجع" + ":" + r + "\n" + "المستند" + ":" + d + "\n\n"
                
                print(f"RAG CONTENT:\n\n{context}")

                final_response = openai.ChatCompletion.create(
                    model=MODEL_NAME,
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": """أنت مساعد ذكي لجامعة الملك سعود. استخدم المستندات والمراجع للإجابة فقط.
الرجاء تضمين المرجع مع الرد
المرجع يكتب بالطريقة الاتية:
المرجع: "https://www.x.y.z" """},
                        {"role": "system", "content": self.memory.get_conversation_context()},
                        {"role": "user", "content": f"السؤال: {user_input}\n\nالمستندات:\n\n{context}\n\nأجب بدقة."}
                    ]
                )
                result = final_response['choices'][0]['message']['content']
                self.memory.add_interaction(user_input, result)
                
                return result
            
            result = TOOLS[plan.tool](**plan.params)
            self.memory.add_interaction(user_input, result)
            return result
        except Exception as e:
            print(f'ERROR: {e}')
            return f'ERROR: {e}'
    
    def respond(self, user_input: str, memory: list[dict] = list()) -> str:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            temperature=0.3,
            messages=[
                {"role": "system", "content": f"You are a helpful assistan for king saud university."},
                {"role": "system", "content": self.memory.get_conversation_context()},
                {"role": "user", "content": user_input}
            ]
        )
        return response['choices'][0]['message']['content']
