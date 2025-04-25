from main import PreorderAgent
import openai

# 1. Replace with your actual API key
openai.api_key = "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"
# 2. Import your full chatbot code (if in another file, like chatbot.py, use: from chatbot import PreorderAgent)
# Assuming all your classes are already in this same script as shown in your code above


def run_chatbot():
    bot = PreorderAgent()

    print("🤖 Welcome to the Multi-Agent Chatbot!")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        result = bot.process_order(user_input, memory_input=[])
        print(f"\n📦 Category: {result['category']}")
        print(f"🤖 Bot: {result['response']}\n")


# Run it
if __name__ == "__main__":
    run_chatbot()
