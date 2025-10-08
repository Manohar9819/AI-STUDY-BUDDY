import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Get API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in a .env file.")

# Initialize Groq LLM
# You can choose different models like 'llama3-8b-8192' or 'mixtral-8x7b-32768'
llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="meta-llama/llama-4-maverick-17b-128e-instruct")

# --- Prompt Templates ---

# 1. Subject Context Prompt
# This sets the persona for the AI.
subject_context_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI study buddy specialized in {subject}."),
        ("user", "{question}"),
    ]
)

# 2. Quiz Generation Prompt
quiz_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI study buddy specialized in {subject}. Your task is to generate {num_questions} multiple-choice quiz questions based on the provided topic. Each question should have 4 options and clearly indicate the correct answer."),
        ("user", "Generate quiz questions on the topic: {topic}"),
    ]
)

# --- Chains for specific functionalities ---

def get_explanation_or_answer(subject: str, question: str):
    """Generates an explanation or answers a study question."""
    chain = subject_context_template | llm
    response = chain.invoke({"subject": subject, "question": question})
    return response.content

def generate_quiz(subject: str, topic: str, num_questions: int = 3):
    """Generates multiple-choice quiz questions."""
    chain = quiz_template | llm
    response = chain.invoke({"subject": subject, "topic": topic, "num_questions": num_questions})
    return response.content

# --- User Interface (CLI) ---

def main():
    print("Welcome to your AI Study Buddy!")
    print("Type 'exit' to quit.")
    print("---")

    while True:
        mode = input("Choose mode (explain/quiz): ").strip().lower()

        if mode == 'exit':
            print("Goodbye!")
            break
        elif mode == 'explain':
            subject = input("Enter the subject (e.g., Math, History, Physics): ").strip()
            if not subject:
                print("Subject cannot be empty. Please try again.")
                continue
            question = input(f"What concept in {subject} do you need help with or what's your question? ").strip()
            if not question:
                print("Question cannot be empty. Please try again.")
                continue

            print("\nThinking...")
            explanation = get_explanation_or_answer(subject, question)
            print("\n--- Explanation/Answer ---")
            print(explanation)
            print("--------------------------\n")

        elif mode == 'quiz':
            subject = input("Enter the subject for the quiz: ").strip()
            if not subject:
                print("Subject cannot be empty. Please try again.")
                continue
            topic = input(f"What specific topic in {subject} do you want a quiz on? ").strip()
            if not topic:
                print("Topic cannot be empty. Please try again.")
                continue
            try:
                num_questions_str = input("How many questions (default 3)? ")
                num_questions = int(num_questions_str) if num_questions_str else 3
            except ValueError:
                print("Invalid number of questions. Defaulting to 3.")
                num_questions = 3

            print("\nGenerating quiz...")
            quiz = generate_quiz(subject, topic, num_questions)
            print("\n--- Quiz Questions ---")
            print(quiz)
            print("----------------------\n")

        else:
            print("Invalid mode. Please choose 'explain' or 'quiz'.")

if __name__ == "__main__":
    main()