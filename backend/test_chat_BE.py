import requests
import json

API_URL = "http://127.0.0.1:8000/chat"  # Adjust if needed

# Keep track of the conversation memory
current_memory = []

while True:
    try:
        user_input = input("user_input msg: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        payload = {
            "query": user_input,
            "memory_input": current_memory,  # Send the current memory
        }

        print("\nSending request...")
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        response_data = response.json()

        print(f"Bot: {response_data.get('response')}")
        print(
            f"(Category: {response_data.get('category')})"
        )  # Optional: print category

        # --- IMPORTANT: Update memory for the next turn ---
        current_memory = response_data.get("memory", [])
        # print(f"Updated Memory: {current_memory}\n") # Optional: view memory

    except requests.exceptions.RequestException as e:
        print(f"\nError connecting to API: {e}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

print("\nExiting chat.")
