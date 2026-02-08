import google.generativeai as genai
import rag_engine as rag  
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY not found in environment variables. Please set it before running the application.")
else:
    pass

genai.configure(api_key=API_KEY)


model = genai.GenerativeModel('gemini-2.0-flash')


chat = model.start_chat(history=[])

SYSTEM_INSTRUCTIONS = """
You are "EduBot," a warm, empathetic, and professional school counselor assistant. 
Your goal is to help parents track their child's progress and understand school policies.

CORE BEHAVIORS:
1. **Be Warm & Encouraging:** Never just state a bad grade. If a student scored low, add a constructive remark like "This area needs a little more focus" or "He is showing potential to improve."
2. **Context First:** Answer strictly using the provided Context. If the answer isn't there, say, "I'm sorry, I don't have that specific record right now. Please check with the administration."
3. **Privacy:** Only discuss the student named in the query.
4. **Language:** Adapt to the language of the user. If they ask in Hindi, reply in Hindi. If they ask in Hinglish, reply in Hinglish.

TONE EXAMPLES:
- Bad Grade: "Aarav scored 45/100 in Math. While this is below the passing mark, his teacher noted he is polite. A bit of extra practice in Algebra could really help him bounce back!"
- Good Grade: "Great news! Vivaan scored 92/100. He is really excelling in this subject."
"""

def extract_potential_student_names(query):
    """
    Simple helper to check if any known student name exists in the query.
    In a real app, we would query the DB for a list of all names first.
    For this demo, we check if the query contains names we know exist in our dummy DB.
    """
    # This is a simplified check. In production, fetch this list dynamically from MongoDB.
    known_students = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", 
                      "Ayaan", "Krishna", "Ishaan", "Diya", "Saanvi", "Ananya", "Aadhya", 
                      "Pari", "Kiara", "Myra", "Riya", "Anvi", "Fatima"]
    
    found_names = [name for name in known_students if name.lower() in query.lower()]
    return found_names[0] if found_names else None


def run_chat():
    print("-----------------------------------------------------------")
    print("🎓 PARENT-TEACHER ASSISTANT IS LIVE")
    print("Ask about: Grades, Fees, Syllabus, Bus Routes, or specific students.")
    print("Type 'exit' to quit.")
    print("-----------------------------------------------------------")

    while True:
        try:
            user_input = input("\nParent: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Assistant: Goodbye! Have a great day.")
                break

            
            context_pieces = []
            
            
            general_info = rag.search_general_knowledge(user_input)
            if general_info:
                context_pieces.append(f"📚 SCHOOL KNOWLEDGE BASE:\n{general_info}")

            
            student_name = extract_potential_student_names(user_input)
            if student_name:
                print(f"(System: Fetching detailed records for {student_name}...)")
                student_record = rag.get_student_info(student_name)
                if student_record:
                    context_pieces.append(f"👤 STUDENT RECORD:\n{student_record}")

            # Combine Context
            full_context = "\n\n".join(context_pieces)

            prompt = f"""
            {SYSTEM_INSTRUCTIONS}

            CONTEXT FOR THIS QUERY:
            {full_context}

            USER QUERY:
            {user_input}
            """

            # Send to Gemini
            response = chat.send_message(prompt)
            
            print(f"Assistant: {response.text}")

        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_chat()