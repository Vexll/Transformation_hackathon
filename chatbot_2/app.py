from tools.agent import KSUAgent
from tools.utils import init

# Initialize
init()

# === Main Chat Loop

agent = KSUAgent()

print("👩‍🎓 KSU Smart Assistant جاهز! (اكتب 'exit' للخروج)\n")
while True:
    user_input = input("👤 You: ")

    if user_input.lower() == "exit":
        break

    plan = agent.get_plan(user_input)
    response = agent.execute_plan(user_input, plan)
    
    print(f"\n🤖 Assistant:\n{response}\n")