import pymongo
import random
from datetime import datetime, timedelta

# 1. SETUP CONNECTION
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["school_rag_db"]

# CLEAN SLATE
db.students.drop()
db.academic_records.drop()
db.curriculum.drop()
db.school_info.drop()

print("Cleaning complete. Generating complex real-world data...")

# ==========================================
# DATA HELPER LISTS (For Realism)
# ==========================================
subjects_list = ["Mathematics", "Science", "English", "Social Studies", "Hindi", "Computer Science"]

chapter_pool = {
    "Mathematics": ["Real Numbers", "Polynomials", "Linear Equations", "Quadratic Equations", "Arithmetic Progressions", "Triangles", "Coordinate Geometry", "Trigonometry", "Circles", "Constructions", "Areas Related to Circles", "Surface Areas and Volumes", "Statistics", "Probability"],
    "Science": ["Chemical Reactions", "Acids Bases Salts", "Metals and Non-metals", "Carbon Compounds", "Periodic Classification", "Life Processes", "Control and Coordination", "How Organisms Reproduce", "Heredity", "Light Reflection", "Human Eye", "Electricity", "Magnetic Effects", "Sources of Energy"],
    "English": ["A Letter to God", "Nelson Mandela", "Two Stories about Flying", "From the Diary of Anne Frank", "The Hundred Dresses I", "The Hundred Dresses II", "Glimpses of India", "Mijbil the Otter", "Madam Rides the Bus", "The Sermon at Benares", "The Proposal", "Dust of Snow"],
    "Social Studies": ["Rise of Nationalism in Europe", "Nationalism in India", "Making of Global World", "Age of Industrialization", "Resources and Development", "Forest and Wildlife", "Water Resources", "Agriculture", "Minerals and Energy", "Manufacturing Industries", "Lifelines of Economy", "Power Sharing"],
    "Hindi": ["Pad", "Ram-Lakshman", "Savaiya", "Aatmakathya", "Utsah", "Att Nahi Rahi", "Yah Danturit Muskan", "Chaya Mat Chuna", "Kanyadan", "Sangatkar", "Netaji ka Chasma", "Balgovin Bhagat"],
    "Computer Science": ["Networking Concepts", "HTML and CSS", "Cyber Ethics", "Scratch Programming", "Python Basics", "Conditional Loops", "Lists and Dictionaries", "Database Management", "SQL Commands", "AI Introduction", "Emerging Trends", "Data Visualization"]
}

# ==========================================
# 2. SCHOOL KNOWLEDGE BASE (Global Info)
# ==========================================

# A. COMPLEX BUS ROUTES
bus_routes = []
areas = ["Green Valley", "Highland Park", "Sector 15", "Civil Lines", "Model Town", "Railway Colony", "Airport Road", "Tech Park", "River View", "Old City"]
for i in range(1, 11):
    route_id = f"Route_{i:02d}"
    area = areas[i-1]
    stops = [f"{area} Main Gate", f"{area} Market", f"{area} Phase 1", f"{area} Phase 2", "School Drop Point"]
    bus_routes.append({
        "route_id": route_id,
        "driver_name": random.choice(["Ramesh Singh", "Suresh Yadav", "Dalip Kumar", "Rajesh Gill"]),
        "driver_contact": f"98765432{i:02d}",
        "stops": stops,
        "timings": {"pickup_start": "06:45 AM", "school_reach": "07:50 AM", "drop_start": "02:10 PM"}
    })

# B. DETAILED POLICIES
policies = [
    {
        "category": "policies", "title": "Fee Structure 2024-25",
        "content": "Admission Fee: $500 (One time). Annual Charges: $300. Tuition Fee (Monthly): Grade 1-5: $150, Grade 6-10: $200. Lab Charges: $50/month (Gr 9-10). Transport Fee: varies by route ($80-$120). Late Fee: $10 per day after the 10th of the month."
    },
    {
        "category": "policies", "title": "Uniform Code",
        "content": "Summer (Mon/Tue/Thu/Fri): White shirt with school logo, Grey trousers/skirt, Black shoes, Grey socks. Winter: Navy Blue Blazer mandatory. Sports (Wed/Sat): House colored T-shirt, White track pants, White canvas shoes."
    },
    {
        "category": "policies", "title": "Assessment & Promotion",
        "content": "Student must secure 40% in aggregate and 35% in each subject to pass. Attendance requirement is 75% minimum. Medical certificates must be submitted within 3 days of leave."
    }
]

# C. RICH CALENDAR
calendar_events = []
# Holidays
holidays = {"2024-08-15": "Independence Day", "2024-10-02": "Gandhi Jayanti", "2024-11-01": "Diwali Break Start", "2024-11-05": "Diwali Break End", "2024-12-25": "Christmas"}
for date, event in holidays.items():
    calendar_events.append({"date": date, "event": event, "type": "Holiday", "school_closed": True})

# Exams
calendar_events.append({"start_date": "2024-09-15", "end_date": "2024-09-25", "event": "Half-Yearly Examinations", "type": "Exam"})
calendar_events.append({"start_date": "2025-03-01", "end_date": "2025-03-15", "event": "Final Examinations", "type": "Exam"})

# Events
calendar_events.append({"date": "2024-11-14", "event": "Children's Day Fete", "type": "Celebration", "school_closed": False})
calendar_events.append({"date": "2024-12-10", "event": "Annual Sports Day", "type": "Sports", "school_closed": False})

# Insert Global Data
db.school_info.insert_many([{"category": "transport", "routes": bus_routes}] + policies + [{"category": "calendar", "events": calendar_events}])

# ==========================================
# 3. CURRICULUM & TIMETABLES (Per Grade)
# ==========================================
grades_data = []

for g in range(6, 11): # Grades 6 to 10
    # Generate Timetable
    timetable = {}
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    periods = ["08:00-08:45", "08:45-09:30", "09:30-10:15", "10:45-11:30", "11:30-12:15", "12:15-01:00"]
    
    for day in days:
        daily_schedule = []
        daily_subjects = random.sample(subjects_list, 6) # Shuffle subjects for the day
        for i, period in enumerate(periods):
            daily_schedule.append({"time": period, "subject": daily_subjects[i]})
        timetable[day] = daily_schedule

    # Generate Syllabus (10-14 chapters per subject)
    grade_syllabus = {}
    for sub in subjects_list:
        num_chapters = random.randint(10, 14)
        # Pick random chapters from pool and add 'Chapter X' prefix
        selected_chaps = random.sample(chapter_pool[sub], min(num_chapters, len(chapter_pool[sub])))
        grade_syllabus[sub] = [f"Ch {idx+1}: {name}" for idx, name in enumerate(selected_chaps)]

    db.curriculum.insert_one({
        "grade": g,
        "section": "General", # Applying to all sections for now
        "syllabus": grade_syllabus,
        "timetable": timetable,
        "exam_datesheet": {
             "Half-Yearly": {sub: f"2024-09-{random.randint(15,25)}" for sub in subjects_list}
        }
    })

# ==========================================
# 4. STUDENTS & ACADEMIC RECORDS (Dynamic)
# ==========================================
names = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan",
         "Diya", "Saanvi", "Ananya", "Aadhya", "Pari", "Kiara", "Myra", "Riya", "Anvi", "Fatima"]
surnames = ["Sharma", "Verma", "Gupta", "Malhotra", "Iyer", "Khan", "Patel", "Singh", "Das", "Nair"]

student_docs = []
academic_docs = []

# Distribute 20 students across grades 6-10 (4 students per grade)
student_counter = 0
for grade in range(6, 11):
    for _ in range(4): # 4 students per grade
        student_counter += 1
        s_id = f"STU_{student_counter:03d}"
        fname = names[student_counter-1]
        lname = random.choice(surnames)
        
        # 1. Student Personal Doc
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
            "logistics": {
                "mode": "School Bus",
                "route_id": f"Route_{random.randint(1,10):02d}", # Link to complex routes
                "stop_name": "Market Stop"
            }
        }
        student_docs.append(stu_doc)

        # 2. Academic Record Doc
        # Generate Monthly Attendance
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

        # Generate Grades for all 6 subjects
        grade_card = []
        for sub in subjects_list:
            grade_card.append({
                "subject": sub,
                "unit_test_1": random.randint(15, 25), # Out of 25
                "half_yearly": random.randint(60, 100), # Out of 100
                "project_score": random.randint(15, 20), # Out of 20
                "remarks": random.choice(["Participates well", "Needs to submit homework on time", "Excellent concept clarity", "Distracted in class"])
            })

        acad_doc = {
            "student_id": s_id,
            "academic_year": "2024-25",
            "class_teacher": "Mrs. Anderson",
            "attendance_summary": {
                "total_working_days": total_working,
                "total_present": total_present,
                "percentage": round((total_present/total_working)*100, 1),
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

print("----------------------------------------------------------------")
print(f"DATABASE GENERATION SUCCESSFUL")
print(f"1. Students: {len(student_docs)} records (Grades 6-10)")
print(f"2. Academic Records: {len(academic_docs)} records (With monthly attendance & 6 subjects)")
print(f"3. Curriculum: 5 documents (One per grade, with Timetables & Syllabus)")
print(f"4. School Info: Bus Routes (10), Policies (Fees, Uniform, etc), Calendar (Exams, Events)")
print("----------------------------------------------------------------")