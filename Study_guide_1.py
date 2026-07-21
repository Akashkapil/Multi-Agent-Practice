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


# Agent 2: Turn the outline into notes
def teacher_agent(topic:str, outline:str) -> str:
    return run_agent("Teacher Agent", "Write short beginner-friendly notes using the outline. Keep it concise.",
        f"Topic: {topic}\n\nOutline:\n{outline}")


# Agent 3: Write review questions from the notes
def quiz_agent(topic:str, notes:str) -> str:
    return run_agent(
        "quiz_agent",
        "Write 3 short review questions based on the notes.",
        f"Topic: {topic}\n\nNotes:\n{notes}",
    )


def build_study_guide(topic:str) -> str:
    # Run all 3 agents in sequence and combine their outputs
    outline = planner_agent(topic)
    notes = teacher_agent(topic, outline)
    quiz = quiz_agent(topic, notes)

    return(
        f"# Study Guide: {topic}\n\n"
        f"## Outline\n{outline}\n\n"
        f"## Notes\n{notes}\n\n"
        f"## Review Questions\n{quiz}\n"
    )


if __name__ == "__main__":
    print("Warming up model...")
    Model.invoke("Say ready.")
    print("Model ready.\n")

    topic = input("Enter a study topic: ").strip()
    print("\n" + build_study_guide(topic))
