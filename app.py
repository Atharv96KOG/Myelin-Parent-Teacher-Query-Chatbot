# import os
# import sys
# import json
# import time
# import random
# import argparse
# from io import BytesIO
# from datetime import datetime, timedelta

# import streamlit as st
# from langdetect import detect

# import pymongo
# import chromadb
# from chromadb.utils import embedding_functions

# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# import ollama
# from gtts import gTTS
# from streamlit_mic_recorder import speech_to_text


# # ============================================================
# # CONFIG
# # ============================================================
# MONGO_URI = "mongodb://localhost:27017/"
# DB_NAME = "school_rag_db"
# CHROMA_PATH = "./chroma_db"
# CHROMA_COLLECTION = "school_knowledge"
# EMBED_MODEL = "all-MiniLM-L6-v2"

# OLLAMA_MODEL = "llama3:8b"

# SYSTEM_INSTRUCTIONS = """
# You are EduBot, a warm and professional school assistant.

# Rules:
# 1. Reply in the SAME language as the user.
# 2. Be encouraging if academic performance is low.
# 3. Answer strictly using ONLY the given context.
# 4. If information is missing, politely say it is unavailable.
# """

# SUPPORTED_LANGS = ["en", "hi", "mr"]
# LANG_MAP = {"en": "eng_Latn", "hi": "hin_Deva", "mr": "mar_Deva"}
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# # ============================================================
# # DB + CHROMA CLIENTS
# # ============================================================
# def get_mongo():
#     client = pymongo.MongoClient(MONGO_URI)
#     return client[DB_NAME]

# def get_chroma_collection():
#     chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
#     embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
#     col = chroma_client.get_or_create_collection(
#         name=CHROMA_COLLECTION,
#         embedding_function=embed_fn
#     )
#     return col


# # ============================================================
# # RAG ENGINE (INLINE)
# # ============================================================
# def get_student_info(db, student_name_query: str):
#     """
#     Structured Retrieval with safety checks.
#     Returns:
#       - "SYSTEM_MESSAGE: ..." for not found / multiple / missing
#       - JSON string for valid single student
#     """
#     if not student_name_query or not student_name_query.strip():
#         return "SYSTEM_MESSAGE: Student name was not provided."

#     query_regex = {"name": {"$regex": student_name_query, "$options": "i"}}
#     matches = list(db.students.find(query_regex))

#     if len(matches) == 0:
#         return "SYSTEM_MESSAGE: No student found with that name. Please verify the spelling."

#     if len(matches) > 1:
#         candidate_names = [s["name"] for s in matches]
#         return (
#             f"SYSTEM_MESSAGE: Multiple students found matching '{student_name_query}': "
#             f"{', '.join(candidate_names)}. Please specify the full name."
#         )

#     student = matches[0]
#     s_id = student["_id"]
#     grade = student["grade"]

#     academics = db.academic_records.find_one({"student_id": s_id})
#     curriculum = db.curriculum.find_one({"grade": grade})

#     if academics is None or curriculum is None:
#         return "SYSTEM_MESSAGE: Student record exists but academic/curriculum data is missing."

#     info = {
#         "Student Profile": {
#             "Name": student["name"],
#             "Grade": student["grade"],
#             "Section": student["section"],
#             "Roll No": student.get("roll_no"),
#             "Emergency Contact": student["parent_details"]["emergency_contact"],
#             "Bus Details": student["logistics"],
#         },
#         "Academic Performance": {
#             "Attendance %": academics["attendance_summary"]["percentage"],
#             "Attendance Breakdown": academics["attendance_summary"]["monthly_breakdown"],
#             "Latest Report Card": academics["grade_card"],
#             "Pending Homework": academics["pending_assignments"],
#         },
#         "Class Syllabus & Timetable": {
#             "Complete Syllabus": curriculum["syllabus"],
#             "Weekly Timetable": curriculum["timetable"],
#             "Exam Datesheet": curriculum.get("exam_datesheet", {}),
#         },
#     }
#     return json.dumps(info, indent=2)


# def search_general_knowledge(chroma_col, query: str, n_results: int = 8) -> str:
#     """
#     Vector retrieval from Chroma. Returns text or "".
#     """
#     if not query or not query.strip():
#         return ""

#     results = chroma_col.query(query_texts=[query], n_results=n_results)
#     docs = results.get("documents", [])
#     if not docs or not docs[0]:
#         return ""

#     return "\n---\n".join(docs[0])


# def seed_chroma_from_mongo_school_info(db, chroma_col):
#     """
#     Seeds Chroma from MongoDB school_info.
#     Upserts:
#       - policies (title + content)
#       - transport routes
#       - calendar events
#     """
#     school_docs = list(db.school_info.find({}))
#     if not school_docs:
#         print("No school_info docs found in MongoDB to seed Chroma.")
#         return

#     ids, documents, metadatas = [], [], []

#     for doc in school_docs:
#         category = doc.get("category", "unknown")

#         if category == "policies":
#             title = doc.get("title", "Untitled Policy")
#             content = doc.get("content", "")
#             text = f"[POLICY] {title}\n{content}".strip()

#             ids.append(f"policy::{title}".lower().replace(" ", "_"))
#             documents.append(text)
#             metadatas.append({"category": "policies", "title": title})

#         elif category == "transport":
#             routes = doc.get("routes", [])
#             for r in routes:
#                 route_id = r.get("route_id", "unknown")
#                 driver = r.get("driver_name", "")
#                 contact = r.get("driver_contact", "")
#                 stops = r.get("stops", [])
#                 timings = r.get("timings", {})

#                 text = (
#                     f"[TRANSPORT] {route_id}\n"
#                     f"Driver: {driver} ({contact})\n"
#                     f"Stops: {', '.join(stops)}\n"
#                     f"Timings: {json.dumps(timings, ensure_ascii=False)}"
#                 )

#                 ids.append(f"route::{route_id}".lower())
#                 documents.append(text)
#                 metadatas.append({"category": "transport", "route_id": route_id})

#         elif category == "calendar":
#             events = doc.get("events", [])
#             for e in events:
#                 key = e.get("date") or (e.get("start_date", "") + "_" + e.get("end_date", ""))
#                 event_name = e.get("event", "event")
#                 text = f"[CALENDAR] {json.dumps(e, ensure_ascii=False)}"

#                 ids.append(f"calendar::{key}::{event_name}".lower().replace(" ", "_"))
#                 documents.append(text)
#                 metadatas.append({"category": "calendar"})

#         else:
#             text = f"[SCHOOL_INFO] {json.dumps(doc, default=str, ensure_ascii=False)}"
#             ids.append(f"schoolinfo::{str(doc.get('_id'))}")
#             documents.append(text)
#             metadatas.append({"category": category})

#     chroma_col.upsert(ids=ids, documents=documents, metadatas=metadatas)
#     print(f"Seeded/Upserted {len(ids)} docs into Chroma '{CHROMA_COLLECTION}'.")


# # ============================================================
# # SEED DATABASE (INLINE) - Your generator + Chroma seed
# # ============================================================
# def seed_database():
#     db = get_mongo()

#     # CLEAN SLATE
#     db.students.drop()
#     db.academic_records.drop()
#     db.curriculum.drop()
#     db.school_info.drop()

#     print("Cleaning complete. Generating complex real-world data...")

#     subjects_list = ["Mathematics", "Science", "English", "Social Studies", "Hindi", "Computer Science"]

#     chapter_pool = {
#         "Mathematics": ["Real Numbers", "Polynomials", "Linear Equations", "Quadratic Equations", "Arithmetic Progressions", "Triangles", "Coordinate Geometry", "Trigonometry", "Circles", "Constructions", "Areas Related to Circles", "Surface Areas and Volumes", "Statistics", "Probability"],
#         "Science": ["Chemical Reactions", "Acids Bases Salts", "Metals and Non-metals", "Carbon Compounds", "Periodic Classification", "Life Processes", "Control and Coordination", "How Organisms Reproduce", "Heredity", "Light Reflection", "Human Eye", "Electricity", "Magnetic Effects", "Sources of Energy"],
#         "English": ["A Letter to God", "Nelson Mandela", "Two Stories about Flying", "From the Diary of Anne Frank", "The Hundred Dresses I", "The Hundred Dresses II", "Glimpses of India", "Mijbil the Otter", "Madam Rides the Bus", "The Sermon at Benares", "The Proposal", "Dust of Snow"],
#         "Social Studies": ["Rise of Nationalism in Europe", "Nationalism in India", "Making of Global World", "Age of Industrialization", "Resources and Development", "Forest and Wildlife", "Water Resources", "Agriculture", "Minerals and Energy", "Manufacturing Industries", "Lifelines of Economy", "Power Sharing"],
#         "Hindi": ["Pad", "Ram-Lakshman", "Savaiya", "Aatmakathya", "Utsah", "Att Nahi Rahi", "Yah Danturit Muskan", "Chaya Mat Chuna", "Kanyadan", "Sangatkar", "Netaji ka Chasma", "Balgovin Bhagat"],
#         "Computer Science": ["Networking Concepts", "HTML and CSS", "Cyber Ethics", "Scratch Programming", "Python Basics", "Conditional Loops", "Lists and Dictionaries", "Database Management", "SQL Commands", "AI Introduction", "Emerging Trends", "Data Visualization"]
#     }

#     # ===== School info
#     bus_routes = []
#     areas = ["Green Valley", "Highland Park", "Sector 15", "Civil Lines", "Model Town", "Railway Colony", "Airport Road", "Tech Park", "River View", "Old City"]
#     for i in range(1, 11):
#         route_id = f"Route_{i:02d}"
#         area = areas[i - 1]
#         stops = [f"{area} Main Gate", f"{area} Market", f"{area} Phase 1", f"{area} Phase 2", "School Drop Point"]
#         bus_routes.append({
#             "route_id": route_id,
#             "driver_name": random.choice(["Ramesh Singh", "Suresh Yadav", "Dalip Kumar", "Rajesh Gill"]),
#             "driver_contact": f"98765432{i:02d}",
#             "stops": stops,
#             "timings": {"pickup_start": "06:45 AM", "school_reach": "07:50 AM", "drop_start": "02:10 PM"}
#         })

#     policies = [
#         {
#             "category": "policies", "title": "Fee Structure 2024-25",
#             "content": "Admission Fee: $500 (One time). Annual Charges: $300. Tuition Fee (Monthly): Grade 1-5: $150, Grade 6-10: $200. Lab Charges: $50/month (Gr 9-10). Transport Fee: varies by route ($80-$120). Late Fee: $10 per day after the 10th of the month."
#         },
#         {
#             "category": "policies", "title": "Uniform Code",
#             "content": "Summer (Mon/Tue/Thu/Fri): White shirt with school logo, Grey trousers/skirt, Black shoes, Grey socks. Winter: Navy Blue Blazer mandatory. Sports (Wed/Sat): House colored T-shirt, White track pants, White canvas shoes."
#         },
#         {
#             "category": "policies", "title": "Assessment & Promotion",
#             "content": "Student must secure 40% in aggregate and 35% in each subject to pass. Attendance requirement is 75% minimum. Medical certificates must be submitted within 3 days of leave."
#         }
#     ]

#     calendar_events = []
#     holidays = {
#         "2024-08-15": "Independence Day",
#         "2024-10-02": "Gandhi Jayanti",
#         "2024-11-01": "Diwali Break Start",
#         "2024-11-05": "Diwali Break End",
#         "2024-12-25": "Christmas"
#     }
#     for date, event in holidays.items():
#         calendar_events.append({"date": date, "event": event, "type": "Holiday", "school_closed": True})

#     calendar_events.append({"start_date": "2024-09-15", "end_date": "2024-09-25", "event": "Half-Yearly Examinations", "type": "Exam"})
#     calendar_events.append({"start_date": "2025-03-01", "end_date": "2025-03-15", "event": "Final Examinations", "type": "Exam"})
#     calendar_events.append({"date": "2024-11-14", "event": "Children's Day Fete", "type": "Celebration", "school_closed": False})
#     calendar_events.append({"date": "2024-12-10", "event": "Annual Sports Day", "type": "Sports", "school_closed": False})

#     db.school_info.insert_many(
#         [{"category": "transport", "routes": bus_routes}] + policies + [{"category": "calendar", "events": calendar_events}]
#     )

#     # ===== Curriculum
#     for g in range(6, 11):
#         timetable = {}
#         days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
#         periods = ["08:00-08:45", "08:45-09:30", "09:30-10:15", "10:45-11:30", "11:30-12:15", "12:15-01:00"]

#         for day in days:
#             daily_schedule = []
#             daily_subjects = random.sample(subjects_list, 6)
#             for i, period in enumerate(periods):
#                 daily_schedule.append({"time": period, "subject": daily_subjects[i]})
#             timetable[day] = daily_schedule

#         grade_syllabus = {}
#         for sub in subjects_list:
#             num_chapters = random.randint(10, 14)
#             selected_chaps = random.sample(chapter_pool[sub], min(num_chapters, len(chapter_pool[sub])))
#             grade_syllabus[sub] = [f"Ch {idx+1}: {name}" for idx, name in enumerate(selected_chaps)]

#         db.curriculum.insert_one({
#             "grade": g,
#             "section": "General",
#             "syllabus": grade_syllabus,
#             "timetable": timetable,
#             "exam_datesheet": {"Half-Yearly": {sub: f"2024-09-{random.randint(15,25)}" for sub in subjects_list}}
#         })

#     # ===== Students + academics
#     names = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan",
#              "Diya", "Saanvi", "Ananya", "Aadhya", "Pari", "Kiara", "Myra", "Riya", "Anvi", "Fatima"]
#     surnames = ["Sharma", "Verma", "Gupta", "Malhotra", "Iyer", "Khan", "Patel", "Singh", "Das", "Nair"]

#     student_docs = []
#     academic_docs = []

#     student_counter = 0
#     for grade in range(6, 11):
#         for _ in range(4):
#             student_counter += 1
#             s_id = f"STU_{student_counter:03d}"
#             fname = names[student_counter - 1]
#             lname = random.choice(surnames)

#             stu_doc = {
#                 "_id": s_id,
#                 "name": f"{fname} {lname}",
#                 "grade": grade,
#                 "section": random.choice(["A", "B", "C"]),
#                 "roll_no": random.randint(1, 40),
#                 "dob": "2010-05-20",
#                 "parent_details": {
#                     "father_name": f"Mr. {lname}",
#                     "mother_name": f"Mrs. {lname}",
#                     "primary_email": f"parent.{fname.lower()}@example.com",
#                     "emergency_contact": f"98765{random.randint(10000, 99999)}"
#                 },
#                 "logistics": {
#                     "mode": "School Bus",
#                     "route_id": f"Route_{random.randint(1,10):02d}",
#                     "stop_name": "Market Stop"
#                 }
#             }
#             student_docs.append(stu_doc)

#             months = ["June", "July", "August", "September", "October"]
#             attendance_log = {}
#             total_present = 0
#             total_working = 0
#             for m in months:
#                 working_days = 24
#                 present = random.randint(18, 24)
#                 attendance_log[m] = {"working_days": working_days, "present": present}
#                 total_present += present
#                 total_working += working_days

#             grade_card = []
#             for sub in subjects_list:
#                 grade_card.append({
#                     "subject": sub,
#                     "unit_test_1": random.randint(15, 25),
#                     "half_yearly": random.randint(60, 100),
#                     "project_score": random.randint(15, 20),
#                     "remarks": random.choice([
#                         "Participates well",
#                         "Needs to submit homework on time",
#                         "Excellent concept clarity",
#                         "Distracted in class"
#                     ])
#                 })

#             acad_doc = {
#                 "student_id": s_id,
#                 "academic_year": "2024-25",
#                 "class_teacher": "Mrs. Anderson",
#                 "attendance_summary": {
#                     "total_working_days": total_working,
#                     "total_present": total_present,
#                     "percentage": round((total_present / total_working) * 100, 1),
#                     "monthly_breakdown": attendance_log
#                 },
#                 "grade_card": grade_card,
#                 "pending_assignments": [
#                     {"subject": "Science", "title": "Model of Atom", "due_date": "2024-11-20", "status": "Pending"},
#                     {"subject": "English", "title": "Essay on Pollution", "due_date": "2024-11-18", "status": "Pending"}
#                 ]
#             }
#             academic_docs.append(acad_doc)

#     db.students.insert_many(student_docs)
#     db.academic_records.insert_many(academic_docs)

#     print("----------------------------------------------------------------")
#     print("DATABASE GENERATION SUCCESSFUL")
#     print(f"1. Students: {len(student_docs)} records (Grades 6-10)")
#     print(f"2. Academic Records: {len(academic_docs)} records")
#     print("----------------------------------------------------------------")

#     # Seed Chroma from school_info
#     chroma_col = get_chroma_collection()
#     seed_chroma_from_mongo_school_info(db, chroma_col)
#     print("✅ Chroma seeding complete.")


# # ============================================================
# # TRANSLATION (NLLB OFFLINE)
# # ============================================================
# @st.cache_resource
# def load_translation_model():
#     model_name = "facebook/nllb-200-distilled-600M"
#     tok = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
#     return tok, model

# def translate_text(tok, model, text, src, tgt):
#     if src not in LANG_MAP or tgt not in LANG_MAP or src == tgt:
#         return text
#     tok.src_lang = LANG_MAP[src]
#     inputs = tok(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             forced_bos_token_id=tok.lang_code_to_id[LANG_MAP[tgt]],
#             max_length=512
#         )
#     return tok.decode(output[0], skip_special_tokens=True)

# def detect_language(text):
#     try:
#         lang = detect(text)
#         return lang if lang in SUPPORTED_LANGS else "en"
#     except:
#         return "en"


# # ============================================================
# # LLM + TTS
# # ============================================================
# def generate_llm_response(prompt: str) -> str:
#     resp = ollama.chat(
#         model=OLLAMA_MODEL,
#         messages=[
#             {"role": "system", "content": SYSTEM_INSTRUCTIONS},
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return resp["message"]["content"]

# def text_to_audio(text, lang):
#     audio = BytesIO()
#     gTTS(text=text, lang=lang).write_to_fp(audio)
#     audio.seek(0)
#     return audio


# # ============================================================
# # STUDENT NAME EXTRACTION (simple)
# # ============================================================
# KNOWN_FIRST_NAMES = [
#     "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun",
#     "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan",
#     "Diya", "Saanvi", "Ananya", "Aadhya", "Pari",
#     "Kiara", "Myra", "Riya", "Anvi", "Fatima"
# ]

# def extract_student_name(text):
#     # Works if user says first name; Mongo regex matches full "First Last"
#     t = text.lower()
#     for s in KNOWN_FIRST_NAMES:
#         if s.lower() in t:
#             return s
#     return None


# # ============================================================
# # STREAMLIT APP
# # ============================================================
# def run_app():
#     st.set_page_config(page_title="EduBot - School Assistant", page_icon="🎓", layout="centered")
#     st.title("🎓 EduBot: Parent Assistant")

#     tok, trans_model = load_translation_model()
#     db = get_mongo()
#     chroma_col = get_chroma_collection()

#     if "messages" not in st.session_state:
#         st.session_state.messages = [
#             {"role": "assistant", "content": "Namaste! I am EduBot. How can I help you today?"}
#         ]

#     st.markdown("### 🎙️ Voice Input")
#     voice_text = speech_to_text(
#         language="en-IN",
#         start_prompt="Click to Speak",
#         stop_prompt="Stop Recording",
#         just_once=True,
#         key="STT"
#     )

#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])
#             if "audio" in msg:
#                 st.audio(msg["audio"], format="audio/mp3")

#     chat_input = st.chat_input("Ask about grades, attendance, syllabus, transport...")

#     prompt = voice_text if voice_text else chat_input
#     if not prompt:
#         return

#     # show user prompt
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     user_lang = detect_language(prompt)
#     prompt_en = translate_text(tok, trans_model, prompt, user_lang, "en")

#     context_parts = []
#     student_system_message = None

#     with st.spinner("Searching school records..."):
#         # Chroma (school knowledge)
#         general_info = search_general_knowledge(chroma_col, prompt_en)
#         if general_info:
#             context_parts.append(f"SCHOOL KNOWLEDGE:\n{general_info}")

#         # Mongo (student)
#         student_name = extract_student_name(prompt_en)
#         if student_name:
#             student_info = get_student_info(db, student_name)
#             if isinstance(student_info, str) and student_info.startswith("SYSTEM_MESSAGE:"):
#                 student_system_message = student_info.replace("SYSTEM_MESSAGE:", "").strip()
#             else:
#                 context_parts.append(f"STUDENT RECORD:\n{student_info}")

#     # If we got a system message, reply directly (no LLM)
#     if student_system_message:
#         reply_en = student_system_message
#         reply = translate_text(tok, trans_model, reply_en, "en", user_lang)
#         audio = text_to_audio(reply, user_lang)
#         with st.chat_message("assistant"):
#             st.markdown(reply)
#             st.audio(audio, format="audio/mp3")
#         st.session_state.messages.append({"role": "assistant", "content": reply, "audio": audio})
#         return

#     context = "\n\n".join(context_parts).strip()

#     # If no context at all → data unavailable (no LLM)
#     if not context:
#         msg_en = "Sorry, I don’t have that information in the school records right now."
#         msg = translate_text(tok, trans_model, msg_en, "en", user_lang)
#         audio = text_to_audio(msg, user_lang)
#         with st.chat_message("assistant"):
#             st.markdown(msg)
#             st.audio(audio, format="audio/mp3")
#         st.session_state.messages.append({"role": "assistant", "content": msg, "audio": audio})
#         return

#     # LLM with strict context
#     final_prompt = f"""
# Answer ONLY using the CONTEXT below.
# If the answer is not present, say politely that the data is unavailable.

# CONTEXT:
# {context}

# QUESTION:
# {prompt_en}
# """.strip()

#     with st.chat_message("assistant"):
#         placeholder = st.empty()
#         response_en = generate_llm_response(final_prompt)
#         final_response = translate_text(tok, trans_model, response_en, "en", user_lang)

#         animated = ""
#         for word in final_response.split():
#             animated += word + " "
#             time.sleep(0.02)
#             placeholder.markdown(animated + "▌")
#         placeholder.markdown(animated)

#         audio = text_to_audio(final_response, user_lang)
#         st.audio(audio, format="audio/mp3")

#     st.session_state.messages.append({"role": "assistant", "content": final_response, "audio": audio})


# # ============================================================
# # ENTRYPOINT
# # ============================================================
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--seed", action="store_true", help="Seed MongoDB + ChromaDB and exit.")
#     args, _ = parser.parse_known_args()

#     if args.seed:
#         seed_database()
#         return

#     # If streamlit is running, it will call this file without --seed
#     run_app()

# if __name__ == "__main__":
#     main()

# import os
# import sys
# import json
# import time
# import random
# import argparse
# from io import BytesIO
# from datetime import datetime

# import streamlit as st
# from langdetect import detect

# import pymongo
# import chromadb
# from chromadb.utils import embedding_functions

# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# import ollama
# from gtts import gTTS
# from streamlit_mic_recorder import speech_to_text

# import pandas as pd
# import matplotlib.pyplot as plt

# from reportlab.lib.pagesizes import A4
# from reportlab.pdfgen import canvas
# from reportlab.lib.utils import ImageReader


# # ============================================================
# # CONFIG
# # ============================================================
# MONGO_URI = "mongodb://localhost:27017/"
# DB_NAME = "school_rag_db"
# CHROMA_PATH = "./chroma_db"
# CHROMA_COLLECTION = "school_knowledge"
# EMBED_MODEL = "all-MiniLM-L6-v2"

# OLLAMA_MODEL = "llama3:8b"

# SYSTEM_INSTRUCTIONS = """
# You are EduBot, a warm and professional school assistant.

# Rules:
# 1. Reply in the SAME language as the user.
# 2. Be encouraging if academic performance is low.
# 3. Answer strictly using ONLY the given context.
# 4. If information is missing, politely say it is unavailable.
# """

# SUPPORTED_LANGS = ["en", "hi", "mr"]
# LANG_MAP = {"en": "eng_Latn", "hi": "hin_Deva", "mr": "mar_Deva"}
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# # ============================================================
# # DB + CHROMA CLIENTS
# # ============================================================
# def get_mongo():
#     client = pymongo.MongoClient(MONGO_URI)
#     return client[DB_NAME]


# def get_chroma_collection():
#     chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
#     embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
#     col = chroma_client.get_or_create_collection(
#         name=CHROMA_COLLECTION,
#         embedding_function=embed_fn
#     )
#     return col


# # ============================================================
# # RAG ENGINE (INLINE)
# # ============================================================
# def get_student_info(db, student_name_query: str):
#     """
#     Returns:
#       - "SYSTEM_MESSAGE: ..." for not found / multiple / missing
#       - JSON string for valid single student
#     """
#     if not student_name_query or not student_name_query.strip():
#         return "SYSTEM_MESSAGE: Student name was not provided."

#     query_regex = {"name": {"$regex": student_name_query, "$options": "i"}}
#     matches = list(db.students.find(query_regex))

#     if len(matches) == 0:
#         return "SYSTEM_MESSAGE: No student found with that name. Please verify the spelling."

#     if len(matches) > 1:
#         candidate_names = [s["name"] for s in matches]
#         return (
#             f"SYSTEM_MESSAGE: Multiple students found matching '{student_name_query}': "
#             f"{', '.join(candidate_names)}. Please specify the full name."
#         )

#     student = matches[0]
#     s_id = student["_id"]
#     grade = student["grade"]

#     academics = db.academic_records.find_one({"student_id": s_id})
#     curriculum = db.curriculum.find_one({"grade": grade})

#     if academics is None or curriculum is None:
#         return "SYSTEM_MESSAGE: Student record exists but academic/curriculum data is missing."

#     info = {
#         "Student Profile": {
#             "Name": student["name"],
#             "Grade": student["grade"],
#             "Section": student["section"],
#             "Roll No": student.get("roll_no"),
#             "Emergency Contact": student["parent_details"]["emergency_contact"],
#             "Bus Details": student["logistics"],
#         },
#         "Academic Performance": {
#             "Attendance %": academics["attendance_summary"]["percentage"],
#             "Attendance Breakdown": academics["attendance_summary"]["monthly_breakdown"],
#             "Latest Report Card": academics["grade_card"],
#             "Pending Homework": academics["pending_assignments"],
#         },
#         "Class Syllabus & Timetable": {
#             "Complete Syllabus": curriculum["syllabus"],
#             "Weekly Timetable": curriculum["timetable"],
#             "Exam Datesheet": curriculum.get("exam_datesheet", {}),
#         },
#     }
#     return json.dumps(info, indent=2)


# def search_general_knowledge(chroma_col, query: str, n_results: int = 8) -> str:
#     if not query or not query.strip():
#         return ""
#     results = chroma_col.query(query_texts=[query], n_results=n_results)
#     docs = results.get("documents", [])
#     if not docs or not docs[0]:
#         return ""
#     return "\n---\n".join(docs[0])


# def seed_chroma_from_mongo_school_info(db, chroma_col):
#     school_docs = list(db.school_info.find({}))
#     if not school_docs:
#         print("No school_info docs found in MongoDB to seed Chroma.")
#         return

#     ids, documents, metadatas = [], [], []

#     for doc in school_docs:
#         category = doc.get("category", "unknown")

#         if category == "policies":
#             title = doc.get("title", "Untitled Policy")
#             content = doc.get("content", "")
#             text = f"[POLICY] {title}\n{content}".strip()
#             ids.append(f"policy::{title}".lower().replace(" ", "_"))
#             documents.append(text)
#             metadatas.append({"category": "policies", "title": title})

#         elif category == "transport":
#             routes = doc.get("routes", [])
#             for r in routes:
#                 route_id = r.get("route_id", "unknown")
#                 driver = r.get("driver_name", "")
#                 contact = r.get("driver_contact", "")
#                 stops = r.get("stops", [])
#                 timings = r.get("timings", {})
#                 text = (
#                     f"[TRANSPORT] {route_id}\n"
#                     f"Driver: {driver} ({contact})\n"
#                     f"Stops: {', '.join(stops)}\n"
#                     f"Timings: {json.dumps(timings, ensure_ascii=False)}"
#                 )
#                 ids.append(f"route::{route_id}".lower())
#                 documents.append(text)
#                 metadatas.append({"category": "transport", "route_id": route_id})

#         elif category == "calendar":
#             events = doc.get("events", [])
#             for e in events:
#                 key = e.get("date") or (e.get("start_date", "") + "_" + e.get("end_date", ""))
#                 event_name = e.get("event", "event")
#                 text = f"[CALENDAR] {json.dumps(e, ensure_ascii=False)}"
#                 ids.append(f"calendar::{key}::{event_name}".lower().replace(" ", "_"))
#                 documents.append(text)
#                 metadatas.append({"category": "calendar"})

#         else:
#             text = f"[SCHOOL_INFO] {json.dumps(doc, default=str, ensure_ascii=False)}"
#             ids.append(f"schoolinfo::{str(doc.get('_id'))}")
#             documents.append(text)
#             metadatas.append({"category": category})

#     chroma_col.upsert(ids=ids, documents=documents, metadatas=metadatas)
#     print(f"Seeded/Upserted {len(ids)} docs into Chroma '{CHROMA_COLLECTION}'.")


# # ============================================================
# # SEED DATABASE (INLINE)
# # ============================================================
# def seed_database():
#     db = get_mongo()

#     db.students.drop()
#     db.academic_records.drop()
#     db.curriculum.drop()
#     db.school_info.drop()

#     print("Cleaning complete. Generating complex real-world data...")

#     subjects_list = ["Mathematics", "Science", "English", "Social Studies", "Hindi", "Computer Science"]

#     chapter_pool = {
#         "Mathematics": ["Real Numbers", "Polynomials", "Linear Equations", "Quadratic Equations", "Arithmetic Progressions", "Triangles", "Coordinate Geometry", "Trigonometry", "Circles", "Constructions", "Areas Related to Circles", "Surface Areas and Volumes", "Statistics", "Probability"],
#         "Science": ["Chemical Reactions", "Acids Bases Salts", "Metals and Non-metals", "Carbon Compounds", "Periodic Classification", "Life Processes", "Control and Coordination", "How Organisms Reproduce", "Heredity", "Light Reflection", "Human Eye", "Electricity", "Magnetic Effects", "Sources of Energy"],
#         "English": ["A Letter to God", "Nelson Mandela", "Two Stories about Flying", "From the Diary of Anne Frank", "The Hundred Dresses I", "The Hundred Dresses II", "Glimpses of India", "Mijbil the Otter", "Madam Rides the Bus", "The Sermon at Benares", "The Proposal", "Dust of Snow"],
#         "Social Studies": ["Rise of Nationalism in Europe", "Nationalism in India", "Making of Global World", "Age of Industrialization", "Resources and Development", "Forest and Wildlife", "Water Resources", "Agriculture", "Minerals and Energy", "Manufacturing Industries", "Lifelines of Economy", "Power Sharing"],
#         "Hindi": ["Pad", "Ram-Lakshman", "Savaiya", "Aatmakathya", "Utsah", "Att Nahi Rahi", "Yah Danturit Muskan", "Chaya Mat Chuna", "Kanyadan", "Sangatkar", "Netaji ka Chasma", "Balgovin Bhagat"],
#         "Computer Science": ["Networking Concepts", "HTML and CSS", "Cyber Ethics", "Scratch Programming", "Python Basics", "Conditional Loops", "Lists and Dictionaries", "Database Management", "SQL Commands", "AI Introduction", "Emerging Trends", "Data Visualization"]
#     }

#     # School info
#     bus_routes = []
#     areas = ["Green Valley", "Highland Park", "Sector 15", "Civil Lines", "Model Town", "Railway Colony", "Airport Road", "Tech Park", "River View", "Old City"]
#     for i in range(1, 11):
#         route_id = f"Route_{i:02d}"
#         area = areas[i - 1]
#         stops = [f"{area} Main Gate", f"{area} Market", f"{area} Phase 1", f"{area} Phase 2", "School Drop Point"]
#         bus_routes.append({
#             "route_id": route_id,
#             "driver_name": random.choice(["Ramesh Singh", "Suresh Yadav", "Dalip Kumar", "Rajesh Gill"]),
#             "driver_contact": f"98765432{i:02d}",
#             "stops": stops,
#             "timings": {"pickup_start": "06:45 AM", "school_reach": "07:50 AM", "drop_start": "02:10 PM"}
#         })

#     policies = [
#         {"category": "policies", "title": "Fee Structure 2024-25",
#          "content": "Admission Fee: $500 (One time). Annual Charges: $300. Tuition Fee (Monthly): Grade 1-5: $150, Grade 6-10: $200. Lab Charges: $50/month (Gr 9-10). Transport Fee: varies by route ($80-$120). Late Fee: $10 per day after the 10th of the month."},
#         {"category": "policies", "title": "Uniform Code",
#          "content": "Summer (Mon/Tue/Thu/Fri): White shirt with school logo, Grey trousers/skirt, Black shoes, Grey socks. Winter: Navy Blue Blazer mandatory. Sports (Wed/Sat): House colored T-shirt, White track pants, White canvas shoes."},
#         {"category": "policies", "title": "Assessment & Promotion",
#          "content": "Student must secure 40% in aggregate and 35% in each subject to pass. Attendance requirement is 75% minimum. Medical certificates must be submitted within 3 days of leave."},
#     ]

#     calendar_events = []
#     holidays = {
#         "2024-08-15": "Independence Day",
#         "2024-10-02": "Gandhi Jayanti",
#         "2024-11-01": "Diwali Break Start",
#         "2024-11-05": "Diwali Break End",
#         "2024-12-25": "Christmas"
#     }
#     for date, event in holidays.items():
#         calendar_events.append({"date": date, "event": event, "type": "Holiday", "school_closed": True})

#     calendar_events.append({"start_date": "2024-09-15", "end_date": "2024-09-25", "event": "Half-Yearly Examinations", "type": "Exam"})
#     calendar_events.append({"start_date": "2025-03-01", "end_date": "2025-03-15", "event": "Final Examinations", "type": "Exam"})
#     calendar_events.append({"date": "2024-11-14", "event": "Children's Day Fete", "type": "Celebration", "school_closed": False})
#     calendar_events.append({"date": "2024-12-10", "event": "Annual Sports Day", "type": "Sports", "school_closed": False})

#     db.school_info.insert_many([{"category": "transport", "routes": bus_routes}] + policies + [{"category": "calendar", "events": calendar_events}])

#     # Curriculum
#     for g in range(6, 11):
#         timetable = {}
#         days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
#         periods = ["08:00-08:45", "08:45-09:30", "09:30-10:15", "10:45-11:30", "11:30-12:15", "12:15-01:00"]
#         for day in days:
#             daily_subjects = random.sample(subjects_list, 6)
#             timetable[day] = [{"time": periods[i], "subject": daily_subjects[i]} for i in range(6)]

#         grade_syllabus = {}
#         for sub in subjects_list:
#             num_chapters = random.randint(10, 14)
#             selected = random.sample(chapter_pool[sub], min(num_chapters, len(chapter_pool[sub])))
#             grade_syllabus[sub] = [f"Ch {idx+1}: {name}" for idx, name in enumerate(selected)]

#         db.curriculum.insert_one({
#             "grade": g,
#             "section": "General",
#             "syllabus": grade_syllabus,
#             "timetable": timetable,
#             "exam_datesheet": {"Half-Yearly": {sub: f"2024-09-{random.randint(15,25)}" for sub in subjects_list}}
#         })

#     # Students + academics
#     names = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan",
#              "Diya", "Saanvi", "Ananya", "Aadhya", "Pari", "Kiara", "Myra", "Riya", "Anvi", "Fatima"]
#     surnames = ["Sharma", "Verma", "Gupta", "Malhotra", "Iyer", "Khan", "Patel", "Singh", "Das", "Nair"]

#     student_docs = []
#     academic_docs = []
#     student_counter = 0

#     for grade in range(6, 11):
#         for _ in range(4):
#             student_counter += 1
#             s_id = f"STU_{student_counter:03d}"
#             fname = names[student_counter - 1]
#             lname = random.choice(surnames)

#             stu_doc = {
#                 "_id": s_id,
#                 "name": f"{fname} {lname}",
#                 "grade": grade,
#                 "section": random.choice(["A", "B", "C"]),
#                 "roll_no": random.randint(1, 40),
#                 "dob": "2010-05-20",
#                 "parent_details": {
#                     "father_name": f"Mr. {lname}",
#                     "mother_name": f"Mrs. {lname}",
#                     "primary_email": f"parent.{fname.lower()}@example.com",
#                     "emergency_contact": f"98765{random.randint(10000, 99999)}"
#                 },
#                 "logistics": {"mode": "School Bus", "route_id": f"Route_{random.randint(1,10):02d}", "stop_name": "Market Stop"}
#             }
#             student_docs.append(stu_doc)

#             months = ["June", "July", "August", "September", "October"]
#             attendance_log = {}
#             total_present = 0
#             total_working = 0
#             for m in months:
#                 working_days = 24
#                 present = random.randint(18, 24)
#                 attendance_log[m] = {"working_days": working_days, "present": present}
#                 total_present += present
#                 total_working += working_days

#             grade_card = []
#             for sub in subjects_list:
#                 grade_card.append({
#                     "subject": sub,
#                     "unit_test_1": random.randint(15, 25),
#                     "half_yearly": random.randint(60, 100),
#                     "project_score": random.randint(15, 20),
#                     "remarks": random.choice([
#                         "Participates well",
#                         "Needs to submit homework on time",
#                         "Excellent concept clarity",
#                         "Distracted in class"
#                     ])
#                 })

#             acad_doc = {
#                 "student_id": s_id,
#                 "academic_year": "2024-25",
#                 "class_teacher": "Mrs. Anderson",
#                 "attendance_summary": {
#                     "total_working_days": total_working,
#                     "total_present": total_present,
#                     "percentage": round((total_present / total_working) * 100, 1),
#                     "monthly_breakdown": attendance_log
#                 },
#                 "grade_card": grade_card,
#                 "pending_assignments": [
#                     {"subject": "Science", "title": "Model of Atom", "due_date": "2024-11-20", "status": "Pending"},
#                     {"subject": "English", "title": "Essay on Pollution", "due_date": "2024-11-18", "status": "Pending"}
#                 ]
#             }
#             academic_docs.append(acad_doc)

#     db.students.insert_many(student_docs)
#     db.academic_records.insert_many(academic_docs)

#     print("DATABASE GENERATION SUCCESSFUL")
#     chroma_col = get_chroma_collection()
#     seed_chroma_from_mongo_school_info(db, chroma_col)
#     print("✅ Chroma seeding complete.")


# # ============================================================
# # TRANSLATION (NLLB OFFLINE) - FIXED (NO lang_code_to_id)
# # ============================================================
# @st.cache_resource
# def load_translation_model():
#     model_name = "facebook/nllb-200-distilled-600M"
#     tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
#     return tok, model

# def _get_lang_id(tok, lang_code: str) -> int:
#     lang_id = tok.convert_tokens_to_ids(lang_code)
#     if lang_id is None or lang_id == tok.unk_token_id:
#         raise ValueError(f"Language code token not found: {lang_code}")
#     return int(lang_id)

# def translate_text(tok, model, text: str, src: str, tgt: str) -> str:
#     if not text or not text.strip():
#         return text
#     if src not in LANG_MAP or tgt not in LANG_MAP or src == tgt:
#         return text

#     src_code = LANG_MAP[src]
#     tgt_code = LANG_MAP[tgt]

#     if hasattr(tok, "src_lang"):
#         tok.src_lang = src_code

#     inputs = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
#     forced_bos_token_id = _get_lang_id(tok, tgt_code)

#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             forced_bos_token_id=forced_bos_token_id,
#             max_length=512,
#             num_beams=4
#         )
#     return tok.decode(output[0], skip_special_tokens=True)

# def detect_language(text: str) -> str:
#     try:
#         lang = detect(text)
#         if lang in SUPPORTED_LANGS:
#             return lang
#         return "en"
#     except:
#         # Devanagari fallback (Hindi/Marathi share script)
#         if any("\u0900" <= ch <= "\u097F" for ch in text):
#             return "hi"
#         return "en"


# # ============================================================
# # LLM + TTS
# # ============================================================
# def generate_llm_response(prompt: str) -> str:
#     resp = ollama.chat(
#         model=OLLAMA_MODEL,
#         messages=[
#             {"role": "system", "content": SYSTEM_INSTRUCTIONS},
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return resp["message"]["content"]

# def text_to_audio(text, lang):
#     audio = BytesIO()
#     # gTTS has poor mr support; fallback to hi
#     safe_lang = "hi" if lang == "mr" else lang
#     gTTS(text=text, lang=safe_lang).write_to_fp(audio)
#     audio.seek(0)
#     return audio


# # ============================================================
# # STUDENT NAME EXTRACTION
# # ============================================================
# KNOWN_FIRST_NAMES = [
#     "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun",
#     "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan",
#     "Diya", "Saanvi", "Ananya", "Aadhya", "Pari",
#     "Kiara", "Myra", "Riya", "Anvi", "Fatima"
# ]

# def extract_student_name(text: str):
#     t = text.lower()
#     for s in KNOWN_FIRST_NAMES:
#         if s.lower() in t:
#             return s
#     return None


# # ============================================================
# # VISUALIZATION HELPERS
# # ============================================================
# def plot_attendance(monthly_breakdown: dict):
#     # monthly_breakdown: {"June": {"working_days": 24, "present": 22}, ...}
#     months = list(monthly_breakdown.keys())
#     present = [monthly_breakdown[m]["present"] for m in months]
#     working = [monthly_breakdown[m]["working_days"] for m in months]

#     df = pd.DataFrame({"Month": months, "Present": present, "Working Days": working})
#     df["Attendance %"] = (df["Present"] / df["Working Days"]) * 100

#     fig, ax = plt.subplots()
#     ax.bar(df["Month"], df["Attendance %"])
#     ax.set_ylabel("Attendance %")
#     ax.set_title("Monthly Attendance %")
#     ax.set_ylim(0, 100)
#     plt.xticks(rotation=30)
#     st.pyplot(fig)

#     return df

# def plot_marks(grade_card: list):
#     # grade_card items: subject, unit_test_1(out of 25), half_yearly(out of 100), project_score(out of 20)
#     subjects = [x["subject"] for x in grade_card]
#     # normalize to 100
#     unit_norm = [round((x["unit_test_1"] / 25) * 100, 1) for x in grade_card]
#     proj_norm = [round((x["project_score"] / 20) * 100, 1) for x in grade_card]
#     half = [x["half_yearly"] for x in grade_card]

#     df = pd.DataFrame({
#         "Subject": subjects,
#         "Unit Test (scaled/100)": unit_norm,
#         "Half Yearly (/100)": half,
#         "Project (scaled/100)": proj_norm,
#     })

#     fig, ax = plt.subplots()
#     ax.bar(df["Subject"], df["Half Yearly (/100)"])
#     ax.set_ylabel("Marks")
#     ax.set_title("Half-Yearly Marks by Subject")
#     ax.set_ylim(0, 100)
#     plt.xticks(rotation=30)
#     st.pyplot(fig)

#     return df


# # ============================================================
# # PDF GENERATION
# # ============================================================
# def _wrap_text(c, text, x, y, max_width, line_height=14):
#     """
#     Simple text wrap for reportlab canvas.
#     """
#     words = text.split()
#     line = ""
#     for w in words:
#         test = (line + " " + w).strip()
#         if c.stringWidth(test, "Helvetica", 10) <= max_width:
#             line = test
#         else:
#             c.drawString(x, y, line)
#             y -= line_height
#             line = w
#     if line:
#         c.drawString(x, y, line)
#         y -= line_height
#     return y

# def build_student_pdf(student_dict: dict, attendance_df: pd.DataFrame, marks_df: pd.DataFrame, lang_label="en"):
#     """
#     Returns PDF bytes.
#     """
#     buffer = BytesIO()
#     c = canvas.Canvas(buffer, pagesize=A4)
#     width, height = A4

#     # Header
#     c.setFont("Helvetica-Bold", 16)
#     c.drawString(40, height - 50, "EduBot Student Report")
#     c.setFont("Helvetica", 10)
#     c.drawString(40, height - 70, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

#     y = height - 100
#     c.setFont("Helvetica-Bold", 12)
#     c.drawString(40, y, "Student Profile")
#     y -= 18

#     profile = student_dict.get("Student Profile", {})
#     c.setFont("Helvetica", 10)
#     for k in ["Name", "Grade", "Section", "Roll No", "Emergency Contact"]:
#         v = profile.get(k, "")
#         c.drawString(50, y, f"{k}: {v}")
#         y -= 14

#     # Bus details
#     bus = profile.get("Bus Details", {})
#     c.drawString(50, y, f"Bus Route: {bus.get('route_id','')}, Stop: {bus.get('stop_name','')}")
#     y -= 20

#     # Attendance
#     c.setFont("Helvetica-Bold", 12)
#     c.drawString(40, y, "Attendance Summary")
#     y -= 16
#     c.setFont("Helvetica", 10)

#     perf = student_dict.get("Academic Performance", {})
#     c.drawString(50, y, f"Overall Attendance %: {perf.get('Attendance %','')}")
#     y -= 14

#     # Table-like lines for attendance
#     c.drawString(50, y, "Monthly Attendance %:")
#     y -= 14
#     for _, row in attendance_df.iterrows():
#         c.drawString(60, y, f"{row['Month']}: {row['Attendance %']:.1f}% (Present {int(row['Present'])}/{int(row['Working Days'])})")
#         y -= 12
#         if y < 80:
#             c.showPage()
#             y = height - 60

#     y -= 10

#     # Marks
#     c.setFont("Helvetica-Bold", 12)
#     c.drawString(40, y, "Academic Marks (Half-Yearly)")
#     y -= 16
#     c.setFont("Helvetica", 10)

#     for _, row in marks_df.iterrows():
#         c.drawString(50, y, f"{row['Subject']}: {row['Half Yearly (/100)']}/100")
#         y -= 12
#         if y < 80:
#             c.showPage()
#             y = height - 60

#     y -= 10

#     # Pending homework
#     c.setFont("Helvetica-Bold", 12)
#     c.drawString(40, y, "Pending Homework")
#     y -= 16
#     c.setFont("Helvetica", 10)

#     pending = perf.get("Pending Homework", [])
#     if not pending:
#         c.drawString(50, y, "None")
#         y -= 14
#     else:
#         for p in pending:
#             line = f"{p.get('subject','')}: {p.get('title','')} (Due: {p.get('due_date','')})"
#             y = _wrap_text(c, line, 50, y, max_width=520, line_height=12)
#             if y < 80:
#                 c.showPage()
#                 y = height - 60

#     c.save()
#     buffer.seek(0)
#     return buffer.getvalue()


# # ============================================================
# # STREAMLIT APP
# # ============================================================
# def render_student_dashboard(student_dict: dict):
#     """
#     Visual dashboard (charts + tables).
#     Returns: (attendance_df, marks_df)
#     """
#     profile = student_dict.get("Student Profile", {})
#     perf = student_dict.get("Academic Performance", {})
#     timetable = student_dict.get("Class Syllabus & Timetable", {}).get("Weekly Timetable", {})

#     # Top cards
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Student", profile.get("Name", ""))
#     col2.metric("Grade / Section", f"{profile.get('Grade','')} - {profile.get('Section','')}")
#     col3.metric("Attendance %", perf.get("Attendance %", ""))

#     st.divider()

#     # Attendance chart
#     st.subheader("📊 Attendance")
#     monthly = perf.get("Attendance Breakdown", {})
#     attendance_df = plot_attendance(monthly)
#     st.dataframe(attendance_df, use_container_width=True)

#     st.divider()

#     # Marks chart + table
#     st.subheader("📚 Marks")
#     grade_card = perf.get("Latest Report Card", [])
#     marks_df = plot_marks(grade_card)
#     st.dataframe(pd.DataFrame(grade_card), use_container_width=True)

#     st.divider()

#     # Pending assignments
#     st.subheader("📝 Pending Homework")
#     pending = perf.get("Pending Homework", [])
#     if pending:
#         st.dataframe(pd.DataFrame(pending), use_container_width=True)
#     else:
#         st.info("No pending homework found.")

#     st.divider()

#     # Timetable
#     st.subheader("🗓️ Weekly Timetable")
#     # flatten timetable dict to dataframe
#     rows = []
#     for day, slots in timetable.items():
#         for slot in slots:
#             rows.append({"Day": day, "Time": slot.get("time"), "Subject": slot.get("subject")})
#     if rows:
#         st.dataframe(pd.DataFrame(rows), use_container_width=True)
#     else:
#         st.info("Timetable data not available.")

#     return attendance_df, marks_df


# def run_app():
#     st.set_page_config(page_title="EduBot - School Assistant", page_icon="🎓", layout="centered")
#     st.title("🎓 EduBot: Parent Assistant")

#     tok, trans_model = load_translation_model()
#     db = get_mongo()
#     chroma_col = get_chroma_collection()

#     if "messages" not in st.session_state:
#         st.session_state.messages = [{"role": "assistant", "content": "Namaste! I am EduBot. How can I help you today?"}]

#     st.markdown("### 🎙️ Voice Input")
#     voice_text = speech_to_text(
#         language="en-IN",
#         start_prompt="Click to Speak",
#         stop_prompt="Stop Recording",
#         just_once=True,
#         key="STT"
#     )

#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])
#             if "audio" in msg:
#                 st.audio(msg["audio"], format="audio/mp3")

#     chat_input = st.chat_input("Ask about grades, attendance, syllabus, transport...")

#     prompt = voice_text if voice_text else chat_input
#     if not prompt:
#         return

#     with st.chat_message("user"):
#         st.markdown(prompt)
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     user_lang = detect_language(prompt)
#     prompt_en = translate_text(tok, trans_model, prompt, user_lang, "en")

#     context_parts = []
#     student_system_message = None
#     student_json = None  # store actual student record json string

#     with st.spinner("Searching school records..."):
#         general_info = search_general_knowledge(chroma_col, prompt_en)
#         if general_info:
#             context_parts.append(f"SCHOOL KNOWLEDGE:\n{general_info}")

#         # IMPORTANT: extract student from ORIGINAL prompt (not translated) so names match
#         student_name = extract_student_name(prompt)
#         if student_name:
#             student_info = get_student_info(db, student_name)
#             if isinstance(student_info, str) and student_info.startswith("SYSTEM_MESSAGE:"):
#                 student_system_message = student_info.replace("SYSTEM_MESSAGE:", "").strip()
#             else:
#                 student_json = student_info
#                 context_parts.append(f"STUDENT RECORD:\n{student_info}")

#     # System message -> direct reply
#     if student_system_message:
#         reply_en = student_system_message
#         reply = translate_text(tok, trans_model, reply_en, "en", user_lang)
#         audio = text_to_audio(reply, user_lang)
#         with st.chat_message("assistant"):
#             st.markdown(reply)
#             st.audio(audio, format="audio/mp3")
#         st.session_state.messages.append({"role": "assistant", "content": reply, "audio": audio})
#         return

#     context = "\n\n".join(context_parts).strip()

#     # No context
#     if not context:
#         msg_en = "Sorry, I don’t have that information in the school records right now."
#         msg = translate_text(tok, trans_model, msg_en, "en", user_lang)
#         audio = text_to_audio(msg, user_lang)
#         with st.chat_message("assistant"):
#             st.markdown(msg)
#             st.audio(audio, format="audio/mp3")
#         st.session_state.messages.append({"role": "assistant", "content": msg, "audio": audio})
#         return

#     # If student record exists -> show dashboard + pdf button
#     if student_json:
#         try:
#             student_dict = json.loads(student_json)
#         except Exception:
#             student_dict = None

#         # Use LLM for natural language answer (still context restricted)
#         final_prompt = f"""
# Answer ONLY using the CONTEXT below.
# If the answer is not present, say politely that the data is unavailable.

# CONTEXT:
# {context}

# QUESTION:
# {prompt_en}
# """.strip()

#         with st.chat_message("assistant"):
#             placeholder = st.empty()
#             response_en = generate_llm_response(final_prompt)
#             final_response = translate_text(tok, trans_model, response_en, "en", user_lang)

#             animated = ""
#             for word in final_response.split():
#                 animated += word + " "
#                 time.sleep(0.01)
#                 placeholder.markdown(animated + "▌")
#             placeholder.markdown(animated)

#             audio = text_to_audio(final_response, user_lang)
#             st.audio(audio, format="audio/mp3")

#         st.session_state.messages.append({"role": "assistant", "content": final_response, "audio": audio})

#         # Dashboard UI (visualizations)
#         st.divider()
#         st.header("📌 Student Dashboard")

#         if student_dict:
#             attendance_df, marks_df = render_student_dashboard(student_dict)

#             # PDF download button
#             pdf_bytes = build_student_pdf(student_dict, attendance_df, marks_df)
#             st.download_button(
#                 label="📄 Download Student Report (PDF)",
#                 data=pdf_bytes,
#                 file_name=f"{student_dict.get('Student Profile', {}).get('Name','student')}_report.pdf".replace(" ", "_"),
#                 mime="application/pdf",
#             )
#         else:
#             st.warning("Could not parse student record into dashboard format.")

#         return

#     # Otherwise: general school knowledge only -> LLM response
#     final_prompt = f"""
# Answer ONLY using the CONTEXT below.
# If the answer is not present, say politely that the data is unavailable.

# CONTEXT:
# {context}

# QUESTION:
# {prompt_en}
# """.strip()

#     with st.chat_message("assistant"):
#         placeholder = st.empty()
#         response_en = generate_llm_response(final_prompt)
#         final_response = translate_text(tok, trans_model, response_en, "en", user_lang)

#         animated = ""
#         for word in final_response.split():
#             animated += word + " "
#             time.sleep(0.01)
#             placeholder.markdown(animated + "▌")
#         placeholder.markdown(animated)

#         audio = text_to_audio(final_response, user_lang)
#         st.audio(audio, format="audio/mp3")

#     st.session_state.messages.append({"role": "assistant", "content": final_response, "audio": audio})


# # ============================================================
# # ENTRYPOINT
# # ============================================================
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--seed", action="store_true", help="Seed MongoDB + ChromaDB and exit.")
#     args, _ = parser.parse_known_args()

#     if args.seed:
#         seed_database()
#         return

#     run_app()

# if __name__ == "__main__":
#     main()







# pdf modify is here



# ============================================================
# EduBot - Complete Updated Code (Beautiful PDF + Charts + Logo)
# ============================================================

# import os
# import sys
# import json
# import time
# import random
# import argparse
# from io import BytesIO
# from datetime import datetime

# import streamlit as st
# from langdetect import detect

# import pymongo
# import chromadb
# from chromadb.utils import embedding_functions

# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# import ollama
# from gtts import gTTS
# from streamlit_mic_recorder import speech_to_text

# import pandas as pd
# import matplotlib.pyplot as plt

# # ReportLab: pretty PDFs (Platypus)
# from reportlab.lib.pagesizes import A4
# from reportlab.lib import colors
# from reportlab.lib.units import inch
# from reportlab.platypus import (
#     SimpleDocTemplate,
#     Paragraph,
#     Spacer,
#     Table,
#     TableStyle,
#     Image,
#     PageBreak,
# )
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# # ============================================================
# # CONFIG
# # ============================================================

# MONGO_URI = "mongodb://localhost:27017/"
# DB_NAME = "school_rag_db"

# CHROMA_PATH = "./chroma_db"
# CHROMA_COLLECTION = "school_knowledge"
# EMBED_MODEL = "all-MiniLM-L6-v2"

# OLLAMA_MODEL = "llama3:8b"

# SYSTEM_INSTRUCTIONS = """
# You are EduBot, a warm and professional school assistant.

# Rules:
# 1. Reply in the SAME language as the user.
# 2. Be encouraging if academic performance is low.
# 3. Answer strictly using ONLY the given context.
# 4. If information is missing, politely say it is unavailable.
# """

# SUPPORTED_LANGS = ["en", "hi", "mr"]
# LANG_MAP = {"en": "eng_Latn", "hi": "hin_Deva", "mr": "mar_Deva"}
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ---------- PDF Branding / Theme ----------
# SCHOOL_NAME = "EduBot International School"
# SCHOOL_LOGO_PATH = "https://myelin.co.in/wp-content/uploads/2025/11/logo-purple-scaled.png"  # ✅ put your logo here (PNG recommended)

# THEME_PRIMARY = colors.HexColor("#1F4E79")  # deep blue
# THEME_ACCENT = colors.HexColor("#2E86AB")   # lighter blue
# THEME_BG = colors.HexColor("#F5F7FB")       # soft background
# THEME_MUTED = colors.HexColor("#6B7280")    # gray text


# # ============================================================
# # DB + CHROMA CLIENTS
# # ============================================================

# def get_mongo():
#     client = pymongo.MongoClient(MONGO_URI)
#     return client[DB_NAME]


# def get_chroma_collection():
#     chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
#     embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
#     col = chroma_client.get_or_create_collection(
#         name=CHROMA_COLLECTION,
#         embedding_function=embed_fn
#     )
#     return col


# # ============================================================
# # RAG ENGINE (INLINE)
# # ============================================================

# def get_student_info(db, student_name_query: str):
#     """
#     Returns:
#       - "SYSTEM_MESSAGE: ..." for not found / multiple / missing
#       - JSON string for valid single student
#     """
#     if not student_name_query or not student_name_query.strip():
#         return "SYSTEM_MESSAGE: Student name was not provided."

#     query_regex = {"name": {"$regex": student_name_query, "$options": "i"}}
#     matches = list(db.students.find(query_regex))

#     if len(matches) == 0:
#         return "SYSTEM_MESSAGE: No student found with that name. Please verify the spelling."

#     if len(matches) > 1:
#         candidate_names = [s["name"] for s in matches]
#         return (
#             f"SYSTEM_MESSAGE: Multiple students found matching '{student_name_query}': "
#             f"{', '.join(candidate_names)}. Please specify the full name."
#         )

#     student = matches[0]
#     s_id = student["_id"]
#     grade = student["grade"]

#     academics = db.academic_records.find_one({"student_id": s_id})
#     curriculum = db.curriculum.find_one({"grade": grade})

#     if academics is None or curriculum is None:
#         return "SYSTEM_MESSAGE: Student record exists but academic/curriculum data is missing."

#     info = {
#         "Student Profile": {
#             "Name": student["name"],
#             "Grade": student["grade"],
#             "Section": student["section"],
#             "Roll No": student.get("roll_no"),
#             "Emergency Contact": student["parent_details"]["emergency_contact"],
#             "Bus Details": student["logistics"],
#         },
#         "Academic Performance": {
#             "Attendance %": academics["attendance_summary"]["percentage"],
#             "Attendance Breakdown": academics["attendance_summary"]["monthly_breakdown"],
#             "Latest Report Card": academics["grade_card"],
#             "Pending Homework": academics["pending_assignments"],
#         },
#         "Class Syllabus & Timetable": {
#             "Complete Syllabus": curriculum["syllabus"],
#             "Weekly Timetable": curriculum["timetable"],
#             "Exam Datesheet": curriculum.get("exam_datesheet", {}),
#         },
#     }
#     return json.dumps(info, indent=2)


# def search_general_knowledge(chroma_col, query: str, n_results: int = 8) -> str:
#     if not query or not query.strip():
#         return ""
#     results = chroma_col.query(query_texts=[query], n_results=n_results)
#     docs = results.get("documents", [])
#     if not docs or not docs[0]:
#         return ""
#     return "\n---\n".join(docs[0])


# def seed_chroma_from_mongo_school_info(db, chroma_col):
#     school_docs = list(db.school_info.find({}))
#     if not school_docs:
#         print("No school_info docs found in MongoDB to seed Chroma.")
#         return

#     ids, documents, metadatas = [], [], []

#     for doc in school_docs:
#         category = doc.get("category", "unknown")

#         if category == "policies":
#             title = doc.get("title", "Untitled Policy")
#             content = doc.get("content", "")
#             text = f"[POLICY] {title}\n{content}".strip()
#             ids.append(f"policy::{title}".lower().replace(" ", "_"))
#             documents.append(text)
#             metadatas.append({"category": "policies", "title": title})

#         elif category == "transport":
#             routes = doc.get("routes", [])
#             for r in routes:
#                 route_id = r.get("route_id", "unknown")
#                 driver = r.get("driver_name", "")
#                 contact = r.get("driver_contact", "")
#                 stops = r.get("stops", [])
#                 timings = r.get("timings", {})
#                 text = (
#                     f"[TRANSPORT] {route_id}\n"
#                     f"Driver: {driver} ({contact})\n"
#                     f"Stops: {', '.join(stops)}\n"
#                     f"Timings: {json.dumps(timings, ensure_ascii=False)}"
#                 )
#                 ids.append(f"route::{route_id}".lower())
#                 documents.append(text)
#                 metadatas.append({"category": "transport", "route_id": route_id})

#         elif category == "calendar":
#             events = doc.get("events", [])
#             for e in events:
#                 key = e.get("date") or (e.get("start_date", "") + "_" + e.get("end_date", ""))
#                 event_name = e.get("event", "event")
#                 text = f"[CALENDAR] {json.dumps(e, ensure_ascii=False)}"
#                 ids.append(f"calendar::{key}::{event_name}".lower().replace(" ", "_"))
#                 documents.append(text)
#                 metadatas.append({"category": "calendar"})

#         else:
#             text = f"[SCHOOL_INFO] {json.dumps(doc, default=str, ensure_ascii=False)}"
#             ids.append(f"schoolinfo::{str(doc.get('_id'))}")
#             documents.append(text)
#             metadatas.append({"category": category})

#     chroma_col.upsert(ids=ids, documents=documents, metadatas=metadatas)
#     print(f"Seeded/Upserted {len(ids)} docs into Chroma '{CHROMA_COLLECTION}'.")


# # ============================================================
# # SEED DATABASE (INLINE)
# # ============================================================

# def seed_database():
#     db = get_mongo()

#     db.students.drop()
#     db.academic_records.drop()
#     db.curriculum.drop()
#     db.school_info.drop()

#     print("Cleaning complete. Generating complex real-world data...")

#     subjects_list = ["Mathematics", "Science", "English", "Social Studies", "Hindi", "Computer Science"]

#     chapter_pool = {
#         "Mathematics": ["Real Numbers", "Polynomials", "Linear Equations", "Quadratic Equations", "Arithmetic Progressions",
#                         "Triangles", "Coordinate Geometry", "Trigonometry", "Circles", "Constructions", "Areas Related to Circles",
#                         "Surface Areas and Volumes", "Statistics", "Probability"],
#         "Science": ["Chemical Reactions", "Acids Bases Salts", "Metals and Non-metals", "Carbon Compounds", "Periodic Classification",
#                     "Life Processes", "Control and Coordination", "How Organisms Reproduce", "Heredity", "Light Reflection",
#                     "Human Eye", "Electricity", "Magnetic Effects", "Sources of Energy"],
#         "English": ["A Letter to God", "Nelson Mandela", "Two Stories about Flying", "From the Diary of Anne Frank",
#                     "The Hundred Dresses I", "The Hundred Dresses II", "Glimpses of India", "Mijbil the Otter",
#                     "Madam Rides the Bus", "The Sermon at Benares", "The Proposal", "Dust of Snow"],
#         "Social Studies": ["Rise of Nationalism in Europe", "Nationalism in India", "Making of Global World", "Age of Industrialization",
#                            "Resources and Development", "Forest and Wildlife", "Water Resources", "Agriculture", "Minerals and Energy",
#                            "Manufacturing Industries", "Lifelines of Economy", "Power Sharing"],
#         "Hindi": ["Pad", "Ram-Lakshman", "Savaiya", "Aatmakathya", "Utsah", "Att Nahi Rahi", "Yah Danturit Muskan",
#                   "Chaya Mat Chuna", "Kanyadan", "Sangatkar", "Netaji ka Chasma", "Balgovin Bhagat"],
#         "Computer Science": ["Networking Concepts", "HTML and CSS", "Cyber Ethics", "Scratch Programming", "Python Basics",
#                              "Conditional Loops", "Lists and Dictionaries", "Database Management", "SQL Commands",
#                              "AI Introduction", "Emerging Trends", "Data Visualization"]
#     }

#     # School info
#     bus_routes = []
#     areas = ["Green Valley", "Highland Park", "Sector 15", "Civil Lines", "Model Town",
#              "Railway Colony", "Airport Road", "Tech Park", "River View", "Old City"]

#     for i in range(1, 11):
#         route_id = f"Route_{i:02d}"
#         area = areas[i - 1]
#         stops = [f"{area} Main Gate", f"{area} Market", f"{area} Phase 1", f"{area} Phase 2", "School Drop Point"]
#         bus_routes.append({
#             "route_id": route_id,
#             "driver_name": random.choice(["Ramesh Singh", "Suresh Yadav", "Dalip Kumar", "Rajesh Gill"]),
#             "driver_contact": f"98765432{i:02d}",
#             "stops": stops,
#             "timings": {"pickup_start": "06:45 AM", "school_reach": "07:50 AM", "drop_start": "02:10 PM"}
#         })

#     policies = [
#         {"category": "policies", "title": "Fee Structure 2024-25",
#          "content": "Admission Fee: $500 (One time). Annual Charges: $300. Tuition Fee (Monthly): Grade 1-5: $150, Grade 6-10: $200. Lab Charges: $50/month (Gr 9-10). Transport Fee: varies by route ($80-$120). Late Fee: $10 per day after the 10th of the month."},
#         {"category": "policies", "title": "Uniform Code",
#          "content": "Summer (Mon/Tue/Thu/Fri): White shirt with school logo, Grey trousers/skirt, Black shoes, Grey socks. Winter: Navy Blue Blazer mandatory. Sports (Wed/Sat): House colored T-shirt, White track pants, White canvas shoes."},
#         {"category": "policies", "title": "Assessment & Promotion",
#          "content": "Student must secure 40% in aggregate and 35% in each subject to pass. Attendance requirement is 75% minimum. Medical certificates must be submitted within 3 days of leave."},
#     ]

#     calendar_events = []
#     holidays = {
#         "2024-08-15": "Independence Day",
#         "2024-10-02": "Gandhi Jayanti",
#         "2024-11-01": "Diwali Break Start",
#         "2024-11-05": "Diwali Break End",
#         "2024-12-25": "Christmas"
#     }
#     for date, event in holidays.items():
#         calendar_events.append({"date": date, "event": event, "type": "Holiday", "school_closed": True})

#     calendar_events.append({"start_date": "2024-09-15", "end_date": "2024-09-25", "event": "Half-Yearly Examinations", "type": "Exam"})
#     calendar_events.append({"start_date": "2025-03-01", "end_date": "2025-03-15", "event": "Final Examinations", "type": "Exam"})
#     calendar_events.append({"date": "2024-11-14", "event": "Children's Day Fete", "type": "Celebration", "school_closed": False})
#     calendar_events.append({"date": "2024-12-10", "event": "Annual Sports Day", "type": "Sports", "school_closed": False})

#     db.school_info.insert_many([{"category": "transport", "routes": bus_routes}] + policies + [{"category": "calendar", "events": calendar_events}])

#     # Curriculum
#     for g in range(6, 11):
#         timetable = {}
#         days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
#         periods = ["08:00-08:45", "08:45-09:30", "09:30-10:15", "10:45-11:30", "11:30-12:15", "12:15-01:00"]
#         for day in days:
#             daily_subjects = random.sample(subjects_list, 6)
#             timetable[day] = [{"time": periods[i], "subject": daily_subjects[i]} for i in range(6)]

#         grade_syllabus = {}
#         for sub in subjects_list:
#             num_chapters = random.randint(10, 14)
#             selected = random.sample(chapter_pool[sub], min(num_chapters, len(chapter_pool[sub])))
#             grade_syllabus[sub] = [f"Ch {idx+1}: {name}" for idx, name in enumerate(selected)]

#         db.curriculum.insert_one({
#             "grade": g,
#             "section": "General",
#             "syllabus": grade_syllabus,
#             "timetable": timetable,
#             "exam_datesheet": {"Half-Yearly": {sub: f"2024-09-{random.randint(15,25)}" for sub in subjects_list}}
#         })

#     # Students + academics
#     names = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan",
#              "Diya", "Saanvi", "Ananya", "Aadhya", "Pari", "Kiara", "Myra", "Riya", "Anvi", "Fatima"]
#     surnames = ["Sharma", "Verma", "Gupta", "Malhotra", "Iyer", "Khan", "Patel", "Singh", "Das", "Nair"]

#     student_docs = []
#     academic_docs = []
#     student_counter = 0

#     for grade in range(6, 11):
#         for _ in range(4):
#             student_counter += 1
#             s_id = f"STU_{student_counter:03d}"
#             fname = names[student_counter - 1]
#             lname = random.choice(surnames)

#             stu_doc = {
#                 "_id": s_id,
#                 "name": f"{fname} {lname}",
#                 "grade": grade,
#                 "section": random.choice(["A", "B", "C"]),
#                 "roll_no": random.randint(1, 40),
#                 "dob": "2010-05-20",
#                 "parent_details": {
#                     "father_name": f"Mr. {lname}",
#                     "mother_name": f"Mrs. {lname}",
#                     "primary_email": f"parent.{fname.lower()}@example.com",
#                     "emergency_contact": f"98765{random.randint(10000, 99999)}"
#                 },
#                 "logistics": {"mode": "School Bus", "route_id": f"Route_{random.randint(1,10):02d}", "stop_name": "Market Stop"}
#             }
#             student_docs.append(stu_doc)

#             months = ["June", "July", "August", "September", "October"]
#             attendance_log = {}
#             total_present = 0
#             total_working = 0
#             for m in months:
#                 working_days = 24
#                 present = random.randint(18, 24)
#                 attendance_log[m] = {"working_days": working_days, "present": present}
#                 total_present += present
#                 total_working += working_days

#             grade_card = []
#             for sub in subjects_list:
#                 grade_card.append({
#                     "subject": sub,
#                     "unit_test_1": random.randint(15, 25),
#                     "half_yearly": random.randint(60, 100),
#                     "project_score": random.randint(15, 20),
#                     "remarks": random.choice([
#                         "Participates well",
#                         "Needs to submit homework on time",
#                         "Excellent concept clarity",
#                         "Distracted in class"
#                     ])
#                 })

#             acad_doc = {
#                 "student_id": s_id,
#                 "academic_year": "2024-25",
#                 "class_teacher": "Mrs. Anderson",
#                 "attendance_summary": {
#                     "total_working_days": total_working,
#                     "total_present": total_present,
#                     "percentage": round((total_present / total_working) * 100, 1),
#                     "monthly_breakdown": attendance_log
#                 },
#                 "grade_card": grade_card,
#                 "pending_assignments": [
#                     {"subject": "Science", "title": "Model of Atom", "due_date": "2024-11-20", "status": "Pending"},
#                     {"subject": "English", "title": "Essay on Pollution", "due_date": "2024-11-18", "status": "Pending"}
#                 ]
#             }
#             academic_docs.append(acad_doc)

#     db.students.insert_many(student_docs)
#     db.academic_records.insert_many(academic_docs)

#     print("DATABASE GENERATION SUCCESSFUL")
#     chroma_col = get_chroma_collection()
#     seed_chroma_from_mongo_school_info(db, chroma_col)
#     print("✅ Chroma seeding complete.")


# # ============================================================
# # TRANSLATION (NLLB OFFLINE) - FIXED (NO lang_code_to_id)
# # ============================================================

# @st.cache_resource
# def load_translation_model():
#     model_name = "facebook/nllb-200-distilled-600M"
#     tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
#     return tok, model


# def _get_lang_id(tok, lang_code: str) -> int:
#     lang_id = tok.convert_tokens_to_ids(lang_code)
#     if lang_id is None or lang_id == tok.unk_token_id:
#         raise ValueError(f"Language code token not found: {lang_code}")
#     return int(lang_id)


# def translate_text(tok, model, text: str, src: str, tgt: str) -> str:
#     if not text or not text.strip():
#         return text
#     if src not in LANG_MAP or tgt not in LANG_MAP or src == tgt:
#         return text

#     src_code = LANG_MAP[src]
#     tgt_code = LANG_MAP[tgt]

#     if hasattr(tok, "src_lang"):
#         tok.src_lang = src_code

#     inputs = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
#     forced_bos_token_id = _get_lang_id(tok, tgt_code)

#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             forced_bos_token_id=forced_bos_token_id,
#             max_length=512,
#             num_beams=4
#         )
#     return tok.decode(output[0], skip_special_tokens=True)


# def detect_language(text: str) -> str:
#     try:
#         lang = detect(text)
#         if lang in SUPPORTED_LANGS:
#             return lang
#         return "en"
#     except Exception:
#         # Devanagari fallback (Hindi/Marathi share script)
#         if any("\u0900" <= ch <= "\u097F" for ch in text):
#             return "hi"
#         return "en"


# # ============================================================
# # LLM + TTS
# # ============================================================

# def generate_llm_response(prompt: str) -> str:
#     resp = ollama.chat(
#         model=OLLAMA_MODEL,
#         messages=[
#             {"role": "system", "content": SYSTEM_INSTRUCTIONS},
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return resp["message"]["content"]


# def text_to_audio(text, lang):
#     audio = BytesIO()
#     # gTTS has poor mr support; fallback to hi
#     safe_lang = "hi" if lang == "mr" else lang
#     gTTS(text=text, lang=safe_lang).write_to_fp(audio)
#     audio.seek(0)
#     return audio


# # ============================================================
# # STUDENT NAME EXTRACTION
# # ============================================================

# KNOWN_FIRST_NAMES = [
#     "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun",
#     "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan",
#     "Diya", "Saanvi", "Ananya", "Aadhya", "Pari",
#     "Kiara", "Myra", "Riya", "Anvi", "Fatima"
# ]


# def extract_student_name(text: str):
#     t = text.lower()
#     for s in KNOWN_FIRST_NAMES:
#         if s.lower() in t:
#             return s
#     return None


# # ============================================================
# # VISUALIZATION HELPERS (Streamlit UI)
# # ============================================================

# def plot_attendance(monthly_breakdown: dict):
#     months = list(monthly_breakdown.keys())
#     present = [monthly_breakdown[m]["present"] for m in months]
#     working = [monthly_breakdown[m]["working_days"] for m in months]

#     df = pd.DataFrame({"Month": months, "Present": present, "Working Days": working})
#     df["Attendance %"] = (df["Present"] / df["Working Days"]) * 100

#     fig, ax = plt.subplots()
#     ax.bar(df["Month"], df["Attendance %"])
#     ax.set_ylabel("Attendance %")
#     ax.set_title("Monthly Attendance %")
#     ax.set_ylim(0, 100)
#     plt.xticks(rotation=30)
#     st.pyplot(fig)

#     return df


# def plot_marks(grade_card: list):
#     subjects = [x["subject"] for x in grade_card]
#     half = [x["half_yearly"] for x in grade_card]

#     df = pd.DataFrame({
#         "Subject": subjects,
#         "Half Yearly (/100)": half,
#     })

#     fig, ax = plt.subplots()
#     ax.bar(df["Subject"], df["Half Yearly (/100)"])
#     ax.set_ylabel("Marks")
#     ax.set_title("Half-Yearly Marks by Subject")
#     ax.set_ylim(0, 100)
#     plt.xticks(rotation=30)
#     st.pyplot(fig)

#     return df


# # ============================================================
# # BEAUTIFUL PDF HELPERS (Charts + Tables + Header/Footer)
# # ============================================================

# def _fig_to_rl_image(fig, width=6.5 * inch):
#     """
#     Convert a matplotlib figure to a ReportLab Image flowable.
#     """
#     buf = BytesIO()
#     fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
#     plt.close(fig)
#     buf.seek(0)

#     img = Image(buf)
#     img.drawWidth = width
#     img.drawHeight = (width * 0.55)  # nice visual ratio
#     return img


# def make_attendance_bar_chart(monthly_breakdown: dict):
#     months = list(monthly_breakdown.keys())
#     present = [monthly_breakdown[m]["present"] for m in months]
#     working = [monthly_breakdown[m]["working_days"] for m in months]
#     pct = [(p / w) * 100 for p, w in zip(present, working)]

#     fig, ax = plt.subplots()
#     ax.bar(months, pct)
#     ax.set_ylim(0, 100)
#     ax.set_ylabel("Attendance %")
#     ax.set_title("Monthly Attendance (%)")
#     ax.tick_params(axis="x", rotation=30)
#     return fig


# def make_attendance_pie_chart(attendance_pct: float):
#     try:
#         present = max(0.0, min(100.0, float(attendance_pct)))
#     except Exception:
#         present = 0.0
#     absent = 100.0 - present

#     fig, ax = plt.subplots()
#     ax.pie([present, absent], autopct="%1.0f%%", startangle=90)
#     ax.set_title("Attendance Split (Present vs Absent)")
#     return fig


# def make_marks_bar_chart(grade_card: list):
#     subjects = [x["subject"] for x in grade_card]
#     half = [x["half_yearly"] for x in grade_card]

#     fig, ax = plt.subplots()
#     ax.bar(subjects, half)
#     ax.set_ylim(0, 100)
#     ax.set_ylabel("Marks (/100)")
#     ax.set_title("Half-Yearly Marks by Subject")
#     ax.tick_params(axis="x", rotation=30)
#     return fig


# def _styled_table(data, col_widths=None, header_bg=THEME_PRIMARY):
#     t = Table(data, colWidths=col_widths, hAlign="LEFT")
#     style = TableStyle([
#         ("BACKGROUND", (0, 0), (-1, 0), header_bg),
#         ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
#         ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
#         ("FONTSIZE", (0, 0), (-1, 0), 10),

#         ("BACKGROUND", (0, 1), (-1, -1), colors.white),
#         ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
#         ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
#         ("FONTSIZE", (0, 1), (-1, -1), 9),

#         ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
#         ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
#         ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
#         ("LEFTPADDING", (0, 0), (-1, -1), 6),
#         ("RIGHTPADDING", (0, 0), (-1, -1), 6),
#         ("TOPPADDING", (0, 0), (-1, -1), 5),
#         ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
#     ])
#     t.setStyle(style)
#     return t


# def _draw_header_footer(canvas, doc):
#     canvas.saveState()

#     w, h = A4

#     # top band
#     canvas.setFillColor(THEME_PRIMARY)
#     canvas.rect(0, h - 70, w, 70, stroke=0, fill=1)

#     # logo
#     if SCHOOL_LOGO_PATH and os.path.exists(SCHOOL_LOGO_PATH):
#         try:
#             canvas.drawImage(SCHOOL_LOGO_PATH, 30, h - 62, width=40, height=40, mask="auto")
#         except Exception:
#             pass

#     canvas.setFillColor(colors.white)
#     canvas.setFont("Helvetica-Bold", 16)
#     canvas.drawString(80, h - 45, f"{SCHOOL_NAME}")

#     canvas.setFont("Helvetica", 9)
#     canvas.drawString(80, h - 60, "Student Performance Report")

#     # footer
#     canvas.setFillColor(colors.HexColor("#111827"))
#     canvas.setFont("Helvetica", 8)
#     canvas.drawString(30, 25, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
#     canvas.drawRightString(w - 30, 25, f"Page {doc.page}")

#     canvas.restoreState()


# # ============================================================
# # PDF GENERATION (BEAUTIFUL)
# # ============================================================

# def build_student_pdf(student_dict: dict, attendance_df: pd.DataFrame, marks_df: pd.DataFrame, lang_label="en"):
#     """
#     Returns PDF bytes (beautiful styled report with charts, tables, logo).
#     """
#     buffer = BytesIO()

#     doc = SimpleDocTemplate(
#         buffer,
#         pagesize=A4,
#         leftMargin=30,
#         rightMargin=30,
#         topMargin=85,
#         bottomMargin=45
#     )

#     styles = getSampleStyleSheet()
#     styles.add(ParagraphStyle(
#         name="H1",
#         parent=styles["Heading1"],
#         fontName="Helvetica-Bold",
#         fontSize=16,
#         textColor=THEME_PRIMARY,
#         spaceAfter=10
#     ))
#     styles.add(ParagraphStyle(
#         name="H2",
#         parent=styles["Heading2"],
#         fontName="Helvetica-Bold",
#         fontSize=12,
#         textColor=THEME_ACCENT,
#         spaceBefore=10,
#         spaceAfter=8
#     ))
#     styles.add(ParagraphStyle(
#         name="Body",
#         parent=styles["BodyText"],
#         fontSize=9.5,
#         leading=13,
#         textColor=colors.HexColor("#111827")
#     ))
#     styles.add(ParagraphStyle(
#         name="Muted",
#         parent=styles["BodyText"],
#         fontSize=9,
#         leading=12,
#         textColor=THEME_MUTED
#     ))

#     profile = student_dict.get("Student Profile", {})
#     perf = student_dict.get("Academic Performance", {})
#     attendance_pct = perf.get("Attendance %", 0)

#     story = []

#     # --- Title block
#     student_name = profile.get("Name", "Student")
#     grade = profile.get("Grade", "")
#     section = profile.get("Section", "")
#     roll_no = profile.get("Roll No", "")

#     story.append(Paragraph(f"<b>Student:</b> {student_name}", styles["H1"]))
#     story.append(Paragraph(f"<b>Grade/Section:</b> {grade}-{section} &nbsp;&nbsp; <b>Roll No:</b> {roll_no}", styles["Muted"]))
#     story.append(Spacer(1, 10))

#     # --- Student info "card"
#     bus = profile.get("Bus Details", {})
#     emergency = profile.get("Emergency Contact", "")

#     info_data = [
#         ["Student Information", ""],
#         ["Emergency Contact", str(emergency)],
#         ["Bus Route", f"{bus.get('route_id','')}"],
#         ["Stop", f"{bus.get('stop_name','')}"],
#     ]
#     info_table = _styled_table(info_data, col_widths=[2.0 * inch, 4.6 * inch], header_bg=THEME_PRIMARY)
#     story.append(info_table)
#     story.append(Spacer(1, 12))

#     # --- Attendance section
#     story.append(Paragraph("Attendance Overview", styles["H2"]))

#     monthly = perf.get("Attendance Breakdown", {}) or {}
#     fig_bar = make_attendance_bar_chart(monthly) if monthly else None
#     fig_pie = make_attendance_pie_chart(attendance_pct)

#     charts_row = []
#     if fig_bar:
#         charts_row.append(_fig_to_rl_image(fig_bar, width=3.2 * inch))
#     else:
#         charts_row.append(Paragraph("Monthly attendance data not available.", styles["Muted"]))

#     charts_row.append(_fig_to_rl_image(fig_pie, width=3.2 * inch))

#     charts_table = Table([charts_row], colWidths=[3.25 * inch, 3.25 * inch])
#     charts_table.setStyle(TableStyle([
#         ("VALIGN", (0, 0), (-1, -1), "TOP"),
#         ("BACKGROUND", (0, 0), (-1, -1), colors.white),
#         ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
#         ("LEFTPADDING", (0, 0), (-1, -1), 6),
#         ("RIGHTPADDING", (0, 0), (-1, -1), 6),
#         ("TOPPADDING", (0, 0), (-1, -1), 6),
#         ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
#     ]))
#     story.append(charts_table)
#     story.append(Spacer(1, 10))

#     # Attendance table
#     if attendance_df is not None and len(attendance_df) > 0:
#         att_table_data = [["Month", "Present", "Working Days", "Attendance %"]]
#         for _, r in attendance_df.iterrows():
#             att_table_data.append([
#                 str(r["Month"]),
#                 str(int(r["Present"])),
#                 str(int(r["Working Days"])),
#                 f"{float(r['Attendance %']):.1f}%"
#             ])
#         story.append(_styled_table(att_table_data, col_widths=[1.4 * inch, 1.2 * inch, 1.5 * inch, 1.6 * inch], header_bg=THEME_ACCENT))
#     else:
#         story.append(Paragraph("Attendance table not available.", styles["Muted"]))

#     story.append(Spacer(1, 14))

#     # --- Marks section
#     story.append(Paragraph("Academic Performance (Half-Yearly)", styles["H2"]))

#     grade_card = perf.get("Latest Report Card", []) or []
#     if grade_card:
#         fig_marks = make_marks_bar_chart(grade_card)
#         story.append(_fig_to_rl_image(fig_marks, width=6.5 * inch))
#         story.append(Spacer(1, 8))

#         marks_table_data = [["Subject", "UT1 (/25)", "Half-Yearly (/100)", "Project (/20)", "Remark"]]
#         for x in grade_card:
#             marks_table_data.append([
#                 x.get("subject", ""),
#                 str(x.get("unit_test_1", "")),
#                 str(x.get("half_yearly", "")),
#                 str(x.get("project_score", "")),
#                 x.get("remarks", ""),
#             ])
#         story.append(_styled_table(marks_table_data, col_widths=[1.45 * inch, 0.9 * inch, 1.2 * inch, 1.0 * inch, 2.0 * inch], header_bg=THEME_PRIMARY))
#     else:
#         story.append(Paragraph("Marks data not available.", styles["Muted"]))

#     story.append(Spacer(1, 14))

#     # --- Pending homework section
#     story.append(Paragraph("Pending Homework", styles["H2"]))
#     pending = perf.get("Pending Homework", []) or []

#     if pending:
#         hw_table_data = [["Subject", "Title", "Due Date", "Status"]]
#         for p in pending:
#             hw_table_data.append([
#                 p.get("subject", ""),
#                 p.get("title", ""),
#                 p.get("due_date", ""),
#                 p.get("status", ""),
#             ])
#         story.append(_styled_table(hw_table_data, col_widths=[1.2 * inch, 3.4 * inch, 1.1 * inch, 0.8 * inch], header_bg=THEME_ACCENT))
#     else:
#         story.append(Paragraph("No pending homework.", styles["Body"]))

#     # Build
#     doc.build(story, onFirstPage=_draw_header_footer, onLaterPages=_draw_header_footer)

#     buffer.seek(0)
#     return buffer.getvalue()


# # ============================================================
# # STREAMLIT DASHBOARD
# # ============================================================

# def render_student_dashboard(student_dict: dict):
#     """
#     Visual dashboard (charts + tables).
#     Returns: (attendance_df, marks_df)
#     """
#     profile = student_dict.get("Student Profile", {})
#     perf = student_dict.get("Academic Performance", {})
#     timetable = student_dict.get("Class Syllabus & Timetable", {}).get("Weekly Timetable", {})

#     col1, col2, col3 = st.columns(3)
#     col1.metric("Student", profile.get("Name", ""))
#     col2.metric("Grade / Section", f"{profile.get('Grade','')} - {profile.get('Section','')}")
#     col3.metric("Attendance %", perf.get("Attendance %", ""))

#     st.divider()

#     st.subheader("📊 Attendance")
#     monthly = perf.get("Attendance Breakdown", {})
#     attendance_df = plot_attendance(monthly) if monthly else pd.DataFrame()
#     if not attendance_df.empty:
#         st.dataframe(attendance_df, use_container_width=True)
#     else:
#         st.info("Attendance data not available.")

#     st.divider()

#     st.subheader("📚 Marks")
#     grade_card = perf.get("Latest Report Card", [])
#     marks_df = plot_marks(grade_card) if grade_card else pd.DataFrame()
#     if grade_card:
#         st.dataframe(pd.DataFrame(grade_card), use_container_width=True)
#     else:
#         st.info("Marks data not available.")

#     st.divider()

#     st.subheader("📝 Pending Homework")
#     pending = perf.get("Pending Homework", [])
#     if pending:
#         st.dataframe(pd.DataFrame(pending), use_container_width=True)
#     else:
#         st.info("No pending homework found.")

#     st.divider()

#     st.subheader("🗓️ Weekly Timetable")
#     rows = []
#     for day, slots in (timetable or {}).items():
#         for slot in slots:
#             rows.append({"Day": day, "Time": slot.get("time"), "Subject": slot.get("subject")})

#     if rows:
#         st.dataframe(pd.DataFrame(rows), use_container_width=True)
#     else:
#         st.info("Timetable data not available.")

#     return attendance_df, marks_df


# # ============================================================
# # APP
# # ============================================================

# def run_app():
#     st.set_page_config(page_title="EduBot - School Assistant", page_icon="🎓", layout="centered")
#     st.title("🎓 EduBot: Parent Assistant")

#     tok, trans_model = load_translation_model()
#     db = get_mongo()
#     chroma_col = get_chroma_collection()

#     if "messages" not in st.session_state:
#         st.session_state.messages = [{"role": "assistant", "content": "Namaste! I am EduBot. How can I help you today?"}]

#     st.markdown("### 🎙️ Voice Input")
#     voice_text = speech_to_text(
#         language="en-IN",
#         start_prompt="Click to Speak",
#         stop_prompt="Stop Recording",
#         just_once=True,
#         key="STT"
#     )

#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])
#             if "audio" in msg:
#                 st.audio(msg["audio"], format="audio/mp3")

#     chat_input = st.chat_input("Ask about grades, attendance, syllabus, transport...")

#     prompt = voice_text if voice_text else chat_input
#     if not prompt:
#         return

#     with st.chat_message("user"):
#         st.markdown(prompt)
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     user_lang = detect_language(prompt)
#     prompt_en = translate_text(tok, trans_model, prompt, user_lang, "en")

#     context_parts = []
#     student_system_message = None
#     student_json = None

#     with st.spinner("Searching school records..."):
#         general_info = search_general_knowledge(chroma_col, prompt_en)
#         if general_info:
#             context_parts.append(f"SCHOOL KNOWLEDGE:\n{general_info}")

#         # extract student from ORIGINAL prompt (so names match)
#         student_name = extract_student_name(prompt)
#         if student_name:
#             student_info = get_student_info(db, student_name)
#             if isinstance(student_info, str) and student_info.startswith("SYSTEM_MESSAGE:"):
#                 student_system_message = student_info.replace("SYSTEM_MESSAGE:", "").strip()
#             else:
#                 student_json = student_info
#                 context_parts.append(f"STUDENT RECORD:\n{student_info}")

#     # System message -> direct reply
#     if student_system_message:
#         reply_en = student_system_message
#         reply = translate_text(tok, trans_model, reply_en, "en", user_lang)
#         audio = text_to_audio(reply, user_lang)
#         with st.chat_message("assistant"):
#             st.markdown(reply)
#             st.audio(audio, format="audio/mp3")
#         st.session_state.messages.append({"role": "assistant", "content": reply, "audio": audio})
#         return

#     context = "\n\n".join(context_parts).strip()

#     if not context:
#         msg_en = "Sorry, I don’t have that information in the school records right now."
#         msg = translate_text(tok, trans_model, msg_en, "en", user_lang)
#         audio = text_to_audio(msg, user_lang)
#         with st.chat_message("assistant"):
#             st.markdown(msg)
#             st.audio(audio, format="audio/mp3")
#         st.session_state.messages.append({"role": "assistant", "content": msg, "audio": audio})
#         return

#     # If student record exists -> show dashboard + pdf button
#     if student_json:
#         try:
#             student_dict = json.loads(student_json)
#         except Exception:
#             student_dict = None

#         final_prompt = f"""
# Answer ONLY using the CONTEXT below. If the answer is not present, say politely that the data is unavailable.

# CONTEXT:
# {context}

# QUESTION:
# {prompt_en}
#         """.strip()

#         with st.chat_message("assistant"):
#             placeholder = st.empty()
#             response_en = generate_llm_response(final_prompt)
#             final_response = translate_text(tok, trans_model, response_en, "en", user_lang)

#             animated = ""
#             for word in final_response.split():
#                 animated += word + " "
#                 time.sleep(0.01)
#                 placeholder.markdown(animated + "▌")
#             placeholder.markdown(animated)

#             audio = text_to_audio(final_response, user_lang)
#             st.audio(audio, format="audio/mp3")

#         st.session_state.messages.append({"role": "assistant", "content": final_response, "audio": audio})

#         st.divider()
#         st.header("📌 Student Dashboard")

#         if student_dict:
#             attendance_df, marks_df = render_student_dashboard(student_dict)

#             # ✅ Beautiful PDF
#             pdf_bytes = build_student_pdf(student_dict, attendance_df, marks_df)

#             st.download_button(
#                 label="📄 Download Student Report (PDF)",
#                 data=pdf_bytes,
#                 file_name=f"{student_dict.get('Student Profile', {}).get('Name','student')}_report.pdf".replace(" ", "_"),
#                 mime="application/pdf",
#             )
#         else:
#             st.warning("Could not parse student record into dashboard format.")
#         return

#     # Otherwise: general school knowledge only -> LLM response
#     final_prompt = f"""
# Answer ONLY using the CONTEXT below. If the answer is not present, say politely that the data is unavailable.

# CONTEXT:
# {context}

# QUESTION:
# {prompt_en}
#     """.strip()

#     with st.chat_message("assistant"):
#         placeholder = st.empty()
#         response_en = generate_llm_response(final_prompt)
#         final_response = translate_text(tok, trans_model, response_en, "en", user_lang)

#         animated = ""
#         for word in final_response.split():
#             animated += word + " "
#             time.sleep(0.01)
#             placeholder.markdown(animated + "▌")
#         placeholder.markdown(animated)

#         audio = text_to_audio(final_response, user_lang)
#         st.audio(audio, format="audio/mp3")

#     st.session_state.messages.append({"role": "assistant", "content": final_response, "audio": audio})


# # ============================================================
# # ENTRYPOINT
# # ============================================================

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--seed", action="store_true", help="Seed MongoDB + ChromaDB and exit.")
#     args, _ = parser.parse_known_args()

#     if args.seed:
#         seed_database()
#         return

#     run_app()


# if __name__ == "__main__":
#     main()


## nice 
# ============================================================
# EduBot - Complete Updated Code
# - MongoDB + Chroma RAG
# - Voice input + multilingual (NLLB)
# - Ollama response (context-only)
# - ChatGPT-like typing delay (jitter)
# - Streamlit top-left responsive logo
# - Beautiful PDF (ReportLab Platypus)
#   * Top-left logo visible in header
#   * Center watermark transparent behind all text/charts
#   * Auto-download + cache logo from URL
# ============================================================

import os
import sys
import json
import time
import random
import argparse
from io import BytesIO
from datetime import datetime
import urllib.request

import streamlit as st
from langdetect import detect

import pymongo
import chromadb
from chromadb.utils import embedding_functions

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import ollama
from gtts import gTTS
from streamlit_mic_recorder import speech_to_text

import pandas as pd
import matplotlib.pyplot as plt

# ReportLab: pretty PDFs (Platypus)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader


# ============================================================
# CONFIG
# ============================================================

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "school_rag_db"

CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = "school_knowledge"
EMBED_MODEL = "all-MiniLM-L6-v2"

OLLAMA_MODEL = "llama3:8b"

SYSTEM_INSTRUCTIONS = """
You are EduBot, a warm and professional school assistant.

Rules:
1. Reply in the SAME language as the user.
2. Be encouraging if academic performance is low.
3. Answer strictly using ONLY the given context.
4. If information is missing, politely say it is unavailable.
"""

SUPPORTED_LANGS = ["en", "hi", "mr"]
LANG_MAP = {"en": "eng_Latn", "hi": "hin_Deva", "mr": "mar_Deva"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Branding ----------
SCHOOL_NAME = "EduBot International School"

# Logo (auto-download + cache)
SCHOOL_LOGO_URL = "https://myelin.co.in/wp-content/uploads/2025/11/logo-purple-scaled.png"
SCHOOL_LOGO_PATH = "./assets/myelin_logo.png"

# PDF Theme
THEME_PRIMARY = colors.HexColor("#1F4E79")  # deep blue header
THEME_ACCENT = colors.HexColor("#2E86AB")   # accent tables
THEME_MUTED = colors.HexColor("#6B7280")    # muted text


def ensure_logo_downloaded():
    """Downloads logo once and caches it locally for UI + PDF."""
    os.makedirs(os.path.dirname(SCHOOL_LOGO_PATH), exist_ok=True)
    if not os.path.exists(SCHOOL_LOGO_PATH):
        urllib.request.urlretrieve(SCHOOL_LOGO_URL, SCHOOL_LOGO_PATH)


# ============================================================
# DB + CHROMA CLIENTS
# ============================================================

def get_mongo():
    client = pymongo.MongoClient(MONGO_URI)
    return client[DB_NAME]


def get_chroma_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    col = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embed_fn
    )
    return col


# ============================================================
# RAG ENGINE (INLINE)
# ============================================================

def get_student_info(db, student_name_query: str):
    """
    Returns:
      - "SYSTEM_MESSAGE: ..." for not found / multiple / missing
      - JSON string for valid single student
    """
    if not student_name_query or not student_name_query.strip():
        return "SYSTEM_MESSAGE: Student name was not provided."

    query_regex = {"name": {"$regex": student_name_query, "$options": "i"}}
    matches = list(db.students.find(query_regex))

    if len(matches) == 0:
        return "SYSTEM_MESSAGE: No student found with that name. Please verify the spelling."

    if len(matches) > 1:
        candidate_names = [s["name"] for s in matches]
        return (
            f"SYSTEM_MESSAGE: Multiple students found matching '{student_name_query}': "
            f"{', '.join(candidate_names)}. Please specify the full name."
        )

    student = matches[0]
    s_id = student["_id"]
    grade = student["grade"]

    academics = db.academic_records.find_one({"student_id": s_id})
    curriculum = db.curriculum.find_one({"grade": grade})

    if academics is None or curriculum is None:
        return "SYSTEM_MESSAGE: Student record exists but academic/curriculum data is missing."

    info = {
        "Student Profile": {
            "Name": student["name"],
            "Grade": student["grade"],
            "Section": student["section"],
            "Roll No": student.get("roll_no"),
            "Emergency Contact": student["parent_details"]["emergency_contact"],
            "Bus Details": student["logistics"],
        },
        "Academic Performance": {
            "Attendance %": academics["attendance_summary"]["percentage"],
            "Attendance Breakdown": academics["attendance_summary"]["monthly_breakdown"],
            "Latest Report Card": academics["grade_card"],
            "Pending Homework": academics["pending_assignments"],
        },
        "Class Syllabus & Timetable": {
            "Complete Syllabus": curriculum["syllabus"],
            "Weekly Timetable": curriculum["timetable"],
            "Exam Datesheet": curriculum.get("exam_datesheet", {}),
        },
    }
    return json.dumps(info, indent=2)


def search_general_knowledge(chroma_col, query: str, n_results: int = 8) -> str:
    if not query or not query.strip():
        return ""
    results = chroma_col.query(query_texts=[query], n_results=n_results)
    docs = results.get("documents", [])
    if not docs or not docs[0]:
        return ""
    return "\n---\n".join(docs[0])


def seed_chroma_from_mongo_school_info(db, chroma_col):
    school_docs = list(db.school_info.find({}))
    if not school_docs:
        print("No school_info docs found in MongoDB to seed Chroma.")
        return

    ids, documents, metadatas = [], [], []

    for doc in school_docs:
        category = doc.get("category", "unknown")

        if category == "policies":
            title = doc.get("title", "Untitled Policy")
            content = doc.get("content", "")
            text = f"[POLICY] {title}\n{content}".strip()
            ids.append(f"policy::{title}".lower().replace(" ", "_"))
            documents.append(text)
            metadatas.append({"category": "policies", "title": title})

        elif category == "transport":
            routes = doc.get("routes", [])
            for r in routes:
                route_id = r.get("route_id", "unknown")
                driver = r.get("driver_name", "")
                contact = r.get("driver_contact", "")
                stops = r.get("stops", [])
                timings = r.get("timings", {})
                text = (
                    f"[TRANSPORT] {route_id}\n"
                    f"Driver: {driver} ({contact})\n"
                    f"Stops: {', '.join(stops)}\n"
                    f"Timings: {json.dumps(timings, ensure_ascii=False)}"
                )
                ids.append(f"route::{route_id}".lower())
                documents.append(text)
                metadatas.append({"category": "transport", "route_id": route_id})

        elif category == "calendar":
            events = doc.get("events", [])
            for e in events:
                key = e.get("date") or (e.get("start_date", "") + "_" + e.get("end_date", ""))
                event_name = e.get("event", "event")
                text = f"[CALENDAR] {json.dumps(e, ensure_ascii=False)}"
                ids.append(f"calendar::{key}::{event_name}".lower().replace(" ", "_"))
                documents.append(text)
                metadatas.append({"category": "calendar"})

        else:
            text = f"[SCHOOL_INFO] {json.dumps(doc, default=str, ensure_ascii=False)}"
            ids.append(f"schoolinfo::{str(doc.get('_id'))}")
            documents.append(text)
            metadatas.append({"category": category})

    chroma_col.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Seeded/Upserted {len(ids)} docs into Chroma '{CHROMA_COLLECTION}'.")


# ============================================================
# SEED DATABASE (INLINE)
# ============================================================

def seed_database():
    db = get_mongo()

    db.students.drop()
    db.academic_records.drop()
    db.curriculum.drop()
    db.school_info.drop()

    print("Cleaning complete. Generating complex real-world data...")

    subjects_list = ["Mathematics", "Science", "English", "Social Studies", "Hindi", "Computer Science"]

    chapter_pool = {
        "Mathematics": ["Real Numbers", "Polynomials", "Linear Equations", "Quadratic Equations", "Arithmetic Progressions",
                        "Triangles", "Coordinate Geometry", "Trigonometry", "Circles", "Constructions", "Areas Related to Circles",
                        "Surface Areas and Volumes", "Statistics", "Probability"],
        "Science": ["Chemical Reactions", "Acids Bases Salts", "Metals and Non-metals", "Carbon Compounds", "Periodic Classification",
                    "Life Processes", "Control and Coordination", "How Organisms Reproduce", "Heredity", "Light Reflection",
                    "Human Eye", "Electricity", "Magnetic Effects", "Sources of Energy"],
        "English": ["A Letter to God", "Nelson Mandela", "Two Stories about Flying", "From the Diary of Anne Frank",
                    "The Hundred Dresses I", "The Hundred Dresses II", "Glimpses of India", "Mijbil the Otter",
                    "Madam Rides the Bus", "The Sermon at Benares", "The Proposal", "Dust of Snow"],
        "Social Studies": ["Rise of Nationalism in Europe", "Nationalism in India", "Making of Global World", "Age of Industrialization",
                           "Resources and Development", "Forest and Wildlife", "Water Resources", "Agriculture", "Minerals and Energy",
                           "Manufacturing Industries", "Lifelines of Economy", "Power Sharing"],
        "Hindi": ["Pad", "Ram-Lakshman", "Savaiya", "Aatmakathya", "Utsah", "Att Nahi Rahi", "Yah Danturit Muskan",
                  "Chaya Mat Chuna", "Kanyadan", "Sangatkar", "Netaji ka Chasma", "Balgovin Bhagat"],
        "Computer Science": ["Networking Concepts", "HTML and CSS", "Cyber Ethics", "Scratch Programming", "Python Basics",
                             "Conditional Loops", "Lists and Dictionaries", "Database Management", "SQL Commands",
                             "AI Introduction", "Emerging Trends", "Data Visualization"]
    }

    # School info
    bus_routes = []
    areas = ["Green Valley", "Highland Park", "Sector 15", "Civil Lines", "Model Town",
             "Railway Colony", "Airport Road", "Tech Park", "River View", "Old City"]

    for i in range(1, 11):
        route_id = f"Route_{i:02d}"
        area = areas[i - 1]
        stops = [f"{area} Main Gate", f"{area} Market", f"{area} Phase 1", f"{area} Phase 2", "School Drop Point"]
        bus_routes.append({
            "route_id": route_id,
            "driver_name": random.choice(["Ramesh Singh", "Suresh Yadav", "Dalip Kumar", "Rajesh Gill"]),
            "driver_contact": f"98765432{i:02d}",
            "stops": stops,
            "timings": {"pickup_start": "06:45 AM", "school_reach": "07:50 AM", "drop_start": "02:10 PM"}
        })

    policies = [
        {"category": "policies", "title": "Fee Structure 2024-25",
         "content": "Admission Fee: $500 (One time). Annual Charges: $300. Tuition Fee (Monthly): Grade 1-5: $150, Grade 6-10: $200. Lab Charges: $50/month (Gr 9-10). Transport Fee: varies by route ($80-$120). Late Fee: $10 per day after the 10th of the month."},
        {"category": "policies", "title": "Uniform Code",
         "content": "Summer (Mon/Tue/Thu/Fri): White shirt with school logo, Grey trousers/skirt, Black shoes, Grey socks. Winter: Navy Blue Blazer mandatory. Sports (Wed/Sat): House colored T-shirt, White track pants, White canvas shoes."},
        {"category": "policies", "title": "Assessment & Promotion",
         "content": "Student must secure 40% in aggregate and 35% in each subject to pass. Attendance requirement is 75% minimum. Medical certificates must be submitted within 3 days of leave."},
    ]

    calendar_events = []
    holidays = {
        "2024-08-15": "Independence Day",
        "2024-10-02": "Gandhi Jayanti",
        "2024-11-01": "Diwali Break Start",
        "2024-11-05": "Diwali Break End",
        "2024-12-25": "Christmas"
    }
    for date, event in holidays.items():
        calendar_events.append({"date": date, "event": event, "type": "Holiday", "school_closed": True})

    calendar_events.append({"start_date": "2024-09-15", "end_date": "2024-09-25", "event": "Half-Yearly Examinations", "type": "Exam"})
    calendar_events.append({"start_date": "2025-03-01", "end_date": "2025-03-15", "event": "Final Examinations", "type": "Exam"})
    calendar_events.append({"date": "2024-11-14", "event": "Children's Day Fete", "type": "Celebration", "school_closed": False})
    calendar_events.append({"date": "2024-12-10", "event": "Annual Sports Day", "type": "Sports", "school_closed": False})

    db.school_info.insert_many([{"category": "transport", "routes": bus_routes}] + policies + [{"category": "calendar", "events": calendar_events}])

    # Curriculum
    for g in range(6, 11):
        timetable = {}
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        periods = ["08:00-08:45", "08:45-09:30", "09:30-10:15", "10:45-11:30", "11:30-12:15", "12:15-01:00"]
        for day in days:
            daily_subjects = random.sample(subjects_list, 6)
            timetable[day] = [{"time": periods[i], "subject": daily_subjects[i]} for i in range(6)]

        grade_syllabus = {}
        for sub in subjects_list:
            num_chapters = random.randint(10, 14)
            selected = random.sample(chapter_pool[sub], min(num_chapters, len(chapter_pool[sub])))
            grade_syllabus[sub] = [f"Ch {idx+1}: {name}" for idx, name in enumerate(selected)]

        db.curriculum.insert_one({
            "grade": g,
            "section": "General",
            "syllabus": grade_syllabus,
            "timetable": timetable,
            "exam_datesheet": {"Half-Yearly": {sub: f"2024-09-{random.randint(15,25)}" for sub in subjects_list}}
        })

    # Students + academics
    names = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan",
             "Diya", "Saanvi", "Ananya", "Aadhya", "Pari", "Kiara", "Myra", "Riya", "Anvi", "Fatima"]
    surnames = ["Sharma", "Verma", "Gupta", "Malhotra", "Iyer", "Khan", "Patel", "Singh", "Das", "Nair"]

    student_docs = []
    academic_docs = []
    student_counter = 0

    for grade in range(6, 11):
        for _ in range(4):
            student_counter += 1
            s_id = f"STU_{student_counter:03d}"
            fname = names[student_counter - 1]
            lname = random.choice(surnames)

            stu_doc = {
                "_id": s_id,
                "name": f"{fname} {lname}",
                "grade": grade,
                "section": random.choice(["A", "B", "C"]),
                "roll_no": random.randint(1, 40),
                "dob": "2010-05-20",
                "parent_details": {
                    "father_name": f"Mr. {lname}",
                    "mother_name": f"Mrs. {lname}",
                    "primary_email": f"parent.{fname.lower()}@example.com",
                    "emergency_contact": f"98765{random.randint(10000, 99999)}"
                },
                "logistics": {"mode": "School Bus", "route_id": f"Route_{random.randint(1,10):02d}", "stop_name": "Market Stop"}
            }
            student_docs.append(stu_doc)

            months = ["June", "July", "August", "September", "October"]
            attendance_log = {}
            total_present = 0
            total_working = 0
            for m in months:
                working_days = 24
                present = random.randint(18, 24)
                attendance_log[m] = {"working_days": working_days, "present": present}
                total_present += present
                total_working += working_days

            grade_card = []
            for sub in subjects_list:
                grade_card.append({
                    "subject": sub,
                    "unit_test_1": random.randint(15, 25),
                    "half_yearly": random.randint(60, 100),
                    "project_score": random.randint(15, 20),
                    "remarks": random.choice([
                        "Participates well",
                        "Needs to submit homework on time",
                        "Excellent concept clarity",
                        "Distracted in class"
                    ])
                })

            acad_doc = {
                "student_id": s_id,
                "academic_year": "2024-25",
                "class_teacher": "Mrs. Anderson",
                "attendance_summary": {
                    "total_working_days": total_working,
                    "total_present": total_present,
                    "percentage": round((total_present / total_working) * 100, 1),
                    "monthly_breakdown": attendance_log
                },
                "grade_card": grade_card,
                "pending_assignments": [
                    {"subject": "Science", "title": "Model of Atom", "due_date": "2024-11-20", "status": "Pending"},
                    {"subject": "English", "title": "Essay on Pollution", "due_date": "2024-11-18", "status": "Pending"}
                ]
            }
            academic_docs.append(acad_doc)

    db.students.insert_many(student_docs)
    db.academic_records.insert_many(academic_docs)

    print("DATABASE GENERATION SUCCESSFUL")
    chroma_col = get_chroma_collection()
    seed_chroma_from_mongo_school_info(db, chroma_col)
    print("✅ Chroma seeding complete.")


# ============================================================
# TRANSLATION (NLLB OFFLINE) - FIXED
# ============================================================

@st.cache_resource
def load_translation_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    return tok, model


def _get_lang_id(tok, lang_code: str) -> int:
    lang_id = tok.convert_tokens_to_ids(lang_code)
    if lang_id is None or lang_id == tok.unk_token_id:
        raise ValueError(f"Language code token not found: {lang_code}")
    return int(lang_id)


def translate_text(tok, model, text: str, src: str, tgt: str) -> str:
    if not text or not text.strip():
        return text
    if src not in LANG_MAP or tgt not in LANG_MAP or src == tgt:
        return text

    src_code = LANG_MAP[src]
    tgt_code = LANG_MAP[tgt]

    if hasattr(tok, "src_lang"):
        tok.src_lang = src_code

    inputs = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    forced_bos_token_id = _get_lang_id(tok, tgt_code)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=512,
            num_beams=4
        )
    return tok.decode(output[0], skip_special_tokens=True)


def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        if lang in SUPPORTED_LANGS:
            return lang
        return "en"
    except Exception:
        # Devanagari fallback
        if any("\u0900" <= ch <= "\u097F" for ch in text):
            return "hi"
        return "en"


# ============================================================
# LLM + TTS
# ============================================================

def generate_llm_response(prompt: str) -> str:
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": prompt}
        ]
    )
    return resp["message"]["content"]


def text_to_audio(text, lang):
    audio = BytesIO()
    safe_lang = "hi" if lang == "mr" else lang
    gTTS(text=text, lang=safe_lang).write_to_fp(audio)
    audio.seek(0)
    return audio


# ============================================================
# STUDENT NAME EXTRACTION
# ============================================================

KNOWN_FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun",
    "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan",
    "Diya", "Saanvi", "Ananya", "Aadhya", "Pari",
    "Kiara", "Myra", "Riya", "Anvi", "Fatima"
]


def extract_student_name(text: str):
    t = text.lower()
    for s in KNOWN_FIRST_NAMES:
        if s.lower() in t:
            return s
    return None


# ============================================================
# VISUALIZATION HELPERS (Streamlit UI)
# ============================================================

def plot_attendance(monthly_breakdown: dict):
    months = list(monthly_breakdown.keys())
    present = [monthly_breakdown[m]["present"] for m in months]
    working = [monthly_breakdown[m]["working_days"] for m in months]

    df = pd.DataFrame({"Month": months, "Present": present, "Working Days": working})
    df["Attendance %"] = (df["Present"] / df["Working Days"]) * 100

    fig, ax = plt.subplots()
    ax.bar(df["Month"], df["Attendance %"])
    ax.set_ylabel("Attendance %")
    ax.set_title("Monthly Attendance %")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=30)
    st.pyplot(fig)

    return df


def plot_marks(grade_card: list):
    subjects = [x["subject"] for x in grade_card]
    half = [x["half_yearly"] for x in grade_card]

    df = pd.DataFrame({
        "Subject": subjects,
        "Half Yearly (/100)": half,
    })

    fig, ax = plt.subplots()
    ax.bar(df["Subject"], df["Half Yearly (/100)"])
    ax.set_ylabel("Marks")
    ax.set_title("Half-Yearly Marks by Subject")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=30)
    st.pyplot(fig)

    return df


# ============================================================
# BEAUTIFUL PDF HELPERS (Charts + Tables + Watermark)
# ============================================================

def _fig_to_rl_image(fig, width=6.5 * inch):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    img = Image(buf)
    img.drawWidth = width
    img.drawHeight = (width * 0.55)
    return img


def make_attendance_bar_chart(monthly_breakdown: dict):
    months = list(monthly_breakdown.keys())
    present = [monthly_breakdown[m]["present"] for m in months]
    working = [monthly_breakdown[m]["working_days"] for m in months]
    pct = [(p / w) * 100 for p, w in zip(present, working)]

    fig, ax = plt.subplots()
    ax.bar(months, pct)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Attendance %")
    ax.set_title("Monthly Attendance (%)")
    ax.tick_params(axis="x", rotation=30)
    return fig


def make_attendance_pie_chart(attendance_pct: float):
    try:
        present = max(0.0, min(100.0, float(attendance_pct)))
    except Exception:
        present = 0.0
    absent = 100.0 - present

    fig, ax = plt.subplots()
    ax.pie([present, absent], autopct="%1.0f%%", startangle=90)
    ax.set_title("Attendance Split (Present vs Absent)")
    return fig


def make_marks_bar_chart(grade_card: list):
    subjects = [x["subject"] for x in grade_card]
    half = [x["half_yearly"] for x in grade_card]

    fig, ax = plt.subplots()
    ax.bar(subjects, half)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Marks (/100)")
    ax.set_title("Half-Yearly Marks by Subject")
    ax.tick_params(axis="x", rotation=30)
    return fig


def _styled_table(data, col_widths=None, header_bg=THEME_PRIMARY):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),

        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),

        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])
    t.setStyle(style)
    return t


def _draw_header_footer(canvas, doc):
    """
    Draw watermark (center, transparent, behind everything) + header logo left.
    """
    canvas.saveState()
    w, h = A4

    # ---- WATERMARK (behind everything) ----
    if SCHOOL_LOGO_PATH and os.path.exists(SCHOOL_LOGO_PATH):
        try:
            if hasattr(canvas, "setFillAlpha"):
                canvas.setFillAlpha(0.08)  # transparency

            img = ImageReader(SCHOOL_LOGO_PATH)
            iw, ih = img.getSize()

            target_w = w * 0.55
            scale = target_w / float(iw)
            target_h = ih * scale

            x = (w - target_w) / 2
            y = (h - target_h) / 2

            canvas.drawImage(
                img,
                x, y,
                width=target_w,
                height=target_h,
                mask="auto",
                preserveAspectRatio=True,
                anchor="c",
            )

            if hasattr(canvas, "setFillAlpha"):
                canvas.setFillAlpha(1)
        except Exception:
            if hasattr(canvas, "setFillAlpha"):
                canvas.setFillAlpha(1)

    # ---- HEADER BAND ----
    canvas.setFillColor(THEME_PRIMARY)
    canvas.rect(0, h - 70, w, 70, stroke=0, fill=1)

    # ---- TOP-LEFT LOGO (clear) ----
    if SCHOOL_LOGO_PATH and os.path.exists(SCHOOL_LOGO_PATH):
        try:
            canvas.drawImage(
                SCHOOL_LOGO_PATH,
                25, h - 62,
                width=48, height=48,
                mask="auto",
                preserveAspectRatio=True
            )
        except Exception:
            pass

    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 16)
    canvas.drawString(80, h - 45, f"{SCHOOL_NAME}")

    canvas.setFont("Helvetica", 9)
    canvas.drawString(80, h - 60, "Student Performance Report")

    # ---- FOOTER ----
    canvas.setFillColor(colors.HexColor("#111827"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(30, 25, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    canvas.drawRightString(w - 30, 25, f"Page {doc.page}")

    canvas.restoreState()


def build_student_pdf(student_dict: dict, attendance_df: pd.DataFrame, marks_df: pd.DataFrame, lang_label="en"):
    """
    Returns PDF bytes (styled report with charts, tables, logo + watermark).
    """
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=30,
        rightMargin=30,
        topMargin=85,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=THEME_PRIMARY,
        spaceAfter=10
    ))
    styles.add(ParagraphStyle(
        name="H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=THEME_ACCENT,
        spaceBefore=10,
        spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name="Body",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#111827")
    ))
    styles.add(ParagraphStyle(
        name="Muted",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
        textColor=THEME_MUTED
    ))

    profile = student_dict.get("Student Profile", {})
    perf = student_dict.get("Academic Performance", {})
    attendance_pct = perf.get("Attendance %", 0)

    story = []

    # --- Title
    student_name = profile.get("Name", "Student")
    grade = profile.get("Grade", "")
    section = profile.get("Section", "")
    roll_no = profile.get("Roll No", "")

    story.append(Paragraph(f"<b>Student:</b> {student_name}", styles["H1"]))
    story.append(Paragraph(f"<b>Grade/Section:</b> {grade}-{section} &nbsp;&nbsp; <b>Roll No:</b> {roll_no}", styles["Muted"]))
    story.append(Spacer(1, 10))

    # --- Student Info Card
    bus = profile.get("Bus Details", {})
    emergency = profile.get("Emergency Contact", "")

    info_data = [
        ["Student Information", ""],
        ["Emergency Contact", str(emergency)],
        ["Bus Route", f"{bus.get('route_id','')}"],
        ["Stop", f"{bus.get('stop_name','')}"],
    ]
    info_table = _styled_table(info_data, col_widths=[2.0 * inch, 4.6 * inch], header_bg=THEME_PRIMARY)
    story.append(info_table)
    story.append(Spacer(1, 12))

    # --- Attendance
    story.append(Paragraph("Attendance Overview", styles["H2"]))

    monthly = perf.get("Attendance Breakdown", {}) or {}
    fig_bar = make_attendance_bar_chart(monthly) if monthly else None
    fig_pie = make_attendance_pie_chart(attendance_pct)

    charts_row = []
    if fig_bar:
        charts_row.append(_fig_to_rl_image(fig_bar, width=3.2 * inch))
    else:
        charts_row.append(Paragraph("Monthly attendance data not available.", styles["Muted"]))
    charts_row.append(_fig_to_rl_image(fig_pie, width=3.2 * inch))

    charts_table = Table([charts_row], colWidths=[3.25 * inch, 3.25 * inch])
    charts_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(charts_table)
    story.append(Spacer(1, 10))

    # Attendance table
    if attendance_df is not None and len(attendance_df) > 0:
        att_table_data = [["Month", "Present", "Working Days", "Attendance %"]]
        for _, r in attendance_df.iterrows():
            att_table_data.append([
                str(r["Month"]),
                str(int(r["Present"])),
                str(int(r["Working Days"])),
                f"{float(r['Attendance %']):.1f}%"
            ])
        story.append(_styled_table(
            att_table_data,
            col_widths=[1.4 * inch, 1.2 * inch, 1.5 * inch, 1.6 * inch],
            header_bg=THEME_ACCENT
        ))
    else:
        story.append(Paragraph("Attendance table not available.", styles["Muted"]))

    story.append(Spacer(1, 14))

    # --- Marks
    story.append(Paragraph("Academic Performance (Half-Yearly)", styles["H2"]))

    grade_card = perf.get("Latest Report Card", []) or []
    if grade_card:
        fig_marks = make_marks_bar_chart(grade_card)
        story.append(_fig_to_rl_image(fig_marks, width=6.5 * inch))
        story.append(Spacer(1, 8))

        marks_table_data = [["Subject", "UT1 (/25)", "Half-Yearly (/100)", "Project (/20)", "Remark"]]
        for x in grade_card:
            marks_table_data.append([
                x.get("subject", ""),
                str(x.get("unit_test_1", "")),
                str(x.get("half_yearly", "")),
                str(x.get("project_score", "")),
                x.get("remarks", ""),
            ])
        story.append(_styled_table(
            marks_table_data,
            col_widths=[1.45 * inch, 0.9 * inch, 1.2 * inch, 1.0 * inch, 2.0 * inch],
            header_bg=THEME_PRIMARY
        ))
    else:
        story.append(Paragraph("Marks data not available.", styles["Muted"]))

    story.append(Spacer(1, 14))

    # --- Pending Homework
    story.append(Paragraph("Pending Homework", styles["H2"]))
    pending = perf.get("Pending Homework", []) or []

    if pending:
        hw_table_data = [["Subject", "Title", "Due Date", "Status"]]
        for p in pending:
            hw_table_data.append([
                p.get("subject", ""),
                p.get("title", ""),
                p.get("due_date", ""),
                p.get("status", ""),
            ])
        story.append(_styled_table(
            hw_table_data,
            col_widths=[1.2 * inch, 3.4 * inch, 1.1 * inch, 0.8 * inch],
            header_bg=THEME_ACCENT
        ))
    else:
        story.append(Paragraph("No pending homework.", styles["Body"]))

    doc.build(story, onFirstPage=_draw_header_footer, onLaterPages=_draw_header_footer)

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# STREAMLIT DASHBOARD
# ============================================================

def render_student_dashboard(student_dict: dict):
    profile = student_dict.get("Student Profile", {})
    perf = student_dict.get("Academic Performance", {})
    timetable = student_dict.get("Class Syllabus & Timetable", {}).get("Weekly Timetable", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("Student", profile.get("Name", ""))
    col2.metric("Grade / Section", f"{profile.get('Grade','')} - {profile.get('Section','')}")
    col3.metric("Attendance %", perf.get("Attendance %", ""))

    st.divider()

    st.subheader("📊 Attendance")
    monthly = perf.get("Attendance Breakdown", {})
    attendance_df = plot_attendance(monthly) if monthly else pd.DataFrame()
    if not attendance_df.empty:
        st.dataframe(attendance_df, use_container_width=True)
    else:
        st.info("Attendance data not available.")

    st.divider()

    st.subheader("📚 Marks")
    grade_card = perf.get("Latest Report Card", [])
    marks_df = plot_marks(grade_card) if grade_card else pd.DataFrame()
    if grade_card:
        st.dataframe(pd.DataFrame(grade_card), use_container_width=True)
    else:
        st.info("Marks data not available.")

    st.divider()

    st.subheader("📝 Pending Homework")
    pending = perf.get("Pending Homework", [])
    if pending:
        st.dataframe(pd.DataFrame(pending), use_container_width=True)
    else:
        st.info("No pending homework found.")

    st.divider()

    st.subheader("🗓️ Weekly Timetable")
    rows = []
    for day, slots in (timetable or {}).items():
        for slot in slots:
            rows.append({"Day": day, "Time": slot.get("time"), "Subject": slot.get("subject")})

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Timetable data not available.")

    return attendance_df, marks_df


# ============================================================
# TYPING EFFECT (ChatGPT-ish)
# ============================================================

def stream_typing_effect(text: str, placeholder, min_delay=0.02, max_delay=0.06):
    """
    Word-by-word typing with jitter, like ChatGPT.
    """
    animated = ""
    for word in text.split():
        animated += word + " "
        time.sleep(random.uniform(min_delay, max_delay))
        placeholder.markdown(animated + "▌")
    placeholder.markdown(animated)


# ============================================================
# APP
# ============================================================

def run_app():
    st.set_page_config(page_title="EduBot - School Assistant", page_icon="🎓", layout="centered")

    # Ensure logo is present
    ensure_logo_downloaded()

    # Header row: logo top-left + title
    c1, c2 = st.columns([1, 6], vertical_alignment="center")
    with c1:
        if os.path.exists(SCHOOL_LOGO_PATH):
            st.image(SCHOOL_LOGO_PATH, width=90)
    with c2:
        st.title("🎓 EduBot: Parent Assistant")

    tok, trans_model = load_translation_model()
    db = get_mongo()
    chroma_col = get_chroma_collection()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Namaste! I am EduBot. How can I help you today?"}]

    st.markdown("### 🎙️ Voice Input")
    voice_text = speech_to_text(
        language="en-IN",
        start_prompt="Click to Speak",
        stop_prompt="Stop Recording",
        just_once=True,
        key="STT"
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "audio" in msg:
                st.audio(msg["audio"], format="audio/mp3")

    chat_input = st.chat_input("Ask about grades, attendance, syllabus, transport...")

    prompt = voice_text if voice_text else chat_input
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    user_lang = detect_language(prompt)
    prompt_en = translate_text(tok, trans_model, prompt, user_lang, "en")

    context_parts = []
    student_system_message = None
    student_json = None

    with st.spinner("Searching school records..."):
        general_info = search_general_knowledge(chroma_col, prompt_en)
        if general_info:
            context_parts.append(f"SCHOOL KNOWLEDGE:\n{general_info}")

        student_name = extract_student_name(prompt)  # use original prompt for name matching
        if student_name:
            student_info = get_student_info(db, student_name)
            if isinstance(student_info, str) and student_info.startswith("SYSTEM_MESSAGE:"):
                student_system_message = student_info.replace("SYSTEM_MESSAGE:", "").strip()
            else:
                student_json = student_info
                context_parts.append(f"STUDENT RECORD:\n{student_info}")

    # System message -> direct reply (no LLM)
    if student_system_message:
        reply_en = student_system_message
        reply = translate_text(tok, trans_model, reply_en, "en", user_lang)
        audio = text_to_audio(reply, user_lang)
        with st.chat_message("assistant"):
            st.markdown(reply)
            st.audio(audio, format="audio/mp3")
        st.session_state.messages.append({"role": "assistant", "content": reply, "audio": audio})
        return

    context = "\n\n".join(context_parts).strip()

    # No context at all
    if not context:
        msg_en = "Sorry, I don’t have that information in the school records right now."
        msg = translate_text(tok, trans_model, msg_en, "en", user_lang)
        audio = text_to_audio(msg, user_lang)
        with st.chat_message("assistant"):
            st.markdown(msg)
            st.audio(audio, format="audio/mp3")
        st.session_state.messages.append({"role": "assistant", "content": msg, "audio": audio})
        return

    # LLM prompt (context-restricted)
    final_prompt = f"""
Answer ONLY using the CONTEXT below. If the answer is not present, say politely that the data is unavailable.

CONTEXT:
{context}

QUESTION:
{prompt_en}
    """.strip()

    with st.chat_message("assistant"):
        placeholder = st.empty()
        response_en = generate_llm_response(final_prompt)
        final_response = translate_text(tok, trans_model, response_en, "en", user_lang)

        # ChatGPT-like typing delay
        stream_typing_effect(final_response, placeholder, min_delay=0.02, max_delay=0.06)

        audio = text_to_audio(final_response, user_lang)
        st.audio(audio, format="audio/mp3")

    st.session_state.messages.append({"role": "assistant", "content": final_response, "audio": audio})

    # If student record exists -> show dashboard + pdf button
    if student_json:
        st.divider()
        st.header("📌 Student Dashboard")

        try:
            student_dict = json.loads(student_json)
        except Exception:
            student_dict = None

        if student_dict:
            attendance_df, marks_df = render_student_dashboard(student_dict)
            pdf_bytes = build_student_pdf(student_dict, attendance_df, marks_df)

            st.download_button(
                label="📄 Download Student Report (PDF)",
                data=pdf_bytes,
                file_name=f"{student_dict.get('Student Profile', {}).get('Name','student')}_report.pdf".replace(" ", "_"),
                mime="application/pdf",
            )
        else:
            st.warning("Could not parse student record into dashboard format.")


# ============================================================
# ENTRYPOINT
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", action="store_true", help="Seed MongoDB + ChromaDB and exit.")
    args, _ = parser.parse_known_args()

    if args.seed:
        seed_database()
        return

    run_app()


if __name__ == "__main__":
    main()
