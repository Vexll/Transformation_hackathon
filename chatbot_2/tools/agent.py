import json
from typing import Any
import openai
from pydantic import BaseModel

from tools.course_actions import add_course, drop_course, excuse_course, manipulate_course
from tools.rag_search import search_knowledge_base


TOOLS = {
    "search_knowledge_base": search_knowledge_base,
    "drop_course": drop_course,
    "add_course": add_course,
    "excuse_course": excuse_course,
    "manipulate_course": manipulate_course
}

TOOL_DESCRIPTIONS = """
1. general: use this when you want dont know what to use.
2. search_knowledge_base: Search university policies and info.
3. drop_course: Drop a course (only course_code needed).
4. add_course: Add a course (only course_code needed).
5. excuse_course: Excuse a course (only course_code needed).
6. manipulate_course: Change a course (needs old_code and new_code).
"""

class Plan(BaseModel):
    tool: str
    params: dict[str, Any]

class KSUAgent:
    def get_plan(self, user_input: str, memory: list[dict] = list()) -> Plan:
        try:
            plan_response = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=0.3,
                messages=[
                    {"role": "system", "content": f"""You are a planning model. Decide which tool to use. Assume student_id = 443102109 automatically. Available tools:\n{TOOL_DESCRIPTIONS}\n\nReturn only JSON like {{"tool": "...", "params": {{...}}}}."""},
                ]
                + memory + [{"role": "user", "content": user_input}]
            )

            res = plan_response['choices'][0]['message']['content']
            print(res)
            plan = Plan.model_validate_json(res)
        
            return plan
        except Exception as e:
            print(f'ERROR: {e}')
            return f'ERROR: {e}'
    
    def execute_plan(self, user_input: str, plan: Plan, memory: list[dict] = list()) -> str:
        try:
            if plan.tool in {"drop_course", "add_course", "excuse_course", "manipulate_course"}:
                plan.params["student_id"] = "443102109"
            
            if plan.tool == "general" or plan.tool not in TOOLS:
                return self.respond(user_input, memory)


            if plan.tool == "search_knowledge_base":
                docs, refs = TOOLS[plan.tool](**plan.params)
                context = ""
                for d, r in zip(docs, refs):
                    context += "المستند" + ":" + d + "\n" + "مرجع" + ":" + r + "\n\n"
                
                print(f"RAG CONTENT:\n\n{context}")

                final_response = openai.ChatCompletion.create(
                    model="gpt-4",
                    temperature=0.5,
                    messages=[
                        {"role": "system", "content": "أنت مساعد ذكي لجامعة الملك سعود. استخدم المستندات والمراجع للإجابة فقط."},
                    ] + memory +[
                        {"role": "user", "content": f"السؤال: {user_input}\n\nالمستندات:\n\n{context}\n\nأجب بدقة."}
                    ]
                )
                
                return final_response['choices'][0]['message']['content']
            
            result = TOOLS[plan.tool](**plan.params)
            return result
        except Exception as e:
            print(f'ERROR: {e}')
            return f'ERROR: {e}'
    
    def respond(self, user_input: str, memory: list[dict] = list()) -> str:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.3,
            messages=[
                {"role": "system", "content": f"You are a helpful assistan for king saud university."},
            ] + memory + [
                {"role": "user", "content": user_input}
            ]
        )
        return response['choices'][0]['message']['content']
