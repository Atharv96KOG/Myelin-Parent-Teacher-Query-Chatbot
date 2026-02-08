import pymongo
import chromadb
from chromadb.utils import embedding_functions
import json

# SETUP DATABASES
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["school_rag_db"]

chroma_client = chromadb.PersistentClient(path="./chroma_db") 
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

vector_collection = chroma_client.get_or_create_collection(
    name="school_knowledge",
    embedding_function=embed_fn
)

def get_student_info(student_name_query):
    """
    Structured Retrieval with SAFETY CHECKS.
    - If 0 matches: Returns None.
    - If >1 match: Returns a list of candidates (Disambiguation).
    - If 1 match: Returns the detailed record (Grades + Full Syllabus).
    """
    
    # case-insensitive regex search on the 'name' field
    query_regex = {"name": {"$regex": student_name_query, "$options": "i"}}
    
    # Fetch ALL matching students to check for duplicates
    matches = list(mongo_db.students.find(query_regex))
    
    if len(matches) == 0:
        return "SYSTEM_MESSAGE: No student found with that name. Please verify the spelling."

    if len(matches) > 1:
        candidate_names = [s['name'] for s in matches]
        return f"SYSTEM_MESSAGE: Multiple students found matching '{student_name_query}': {', '.join(candidate_names)}. Please ask the user to specify the full name."

    student = matches[0]
    s_id = student["_id"]
    grade = student["grade"]
    
    # Fetch Academic Record
    academics = mongo_db.academic_records.find_one({"student_id": s_id})
    
    # Fetch Curriculum (Detailed)
    curriculum = mongo_db.curriculum.find_one({"grade": grade})
    
    info = {
        "Student Profile": {
            "Name": student["name"],
            "Grade": student["grade"],
            "Section": student["section"],
            "Emergency Contact": student["parent_details"]["emergency_contact"],
            "Bus Details": student["logistics"]
        },
        "Academic Performance": {
            "Attendance %": academics["attendance_summary"]["percentage"],
            "Latest Report Card": academics["grade_card"], 
            "Pending Homework": academics["pending_assignments"]
        },

        "Class Syllabus & Timetable": {
            "Complete Syllabus": curriculum["syllabus"], 
            "Weekly Timetable": curriculum["timetable"]
        }
    }
    return json.dumps(info, indent=2)

def search_general_knowledge(query):
    """
    Vector Retrieval: Increased limit to catch more context.
    """
    results = vector_collection.query(
        query_texts=[query],
        n_results=10  # Increased to capture broader context like full subject lists
    )
    
    if not results['documents']:
        return ""

    # Extract just the text
    retrieved_texts = results['documents'][0]
    return "\n---\n".join(retrieved_texts)