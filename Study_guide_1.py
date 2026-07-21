"""
Then the code defines three small agents with their own specific prompts:

planner_agent() creates a 3-part outline for the topic.

teacher_agent() turns that outline into short beginner-friendly notes.

quiz_agent() creates 3 review questions from the notes.

The build_study_guide() function runs those three agents in sequence, passing each output into the next step.
"""

import time
from langchain_ollama import ChatOllama

Model = ChatOllama(model="qwen3.5:9b", temperature = 0)

def ask(system: str, user: str) -> str:
    # Run one LLM call with system prompt and user input
    response = Model.invoke(
        [
            {"role":"system", "content":system},
            {"role":"user","content":user},
        ]
    )
    return str(response.content)

def run_agent(name:str, system:str, user:str) -> str:
    # Helper that logs how long each agent takes
    print(f"Calling Agent {name}...")
    start = time.time()
    result = ask(system, user)
    print(f"Finished {name} in {time.time() - start:.1f}s")
    return result

# Agent 1: Create a short outline

def planner_agent(topic:str) -> str:
    return run_agent("Planner Agent", "Break this topic into 3 short study sections", topic)


