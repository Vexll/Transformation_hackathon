from typing import Any
import pandas as pd
import os
import json
from datetime import datetime

DATA_FILE = "database/index.csv"
LOG_FILE = "database/actions_log.json"

DEFAULT_COURSES = "MATH101,PHYS101,COMP101"

def initialize_data_file():
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
        df = pd.DataFrame([{"courses": DEFAULT_COURSES}])
        df.to_csv(DATA_FILE, index=False)
    else:
        df = pd.read_csv(DATA_FILE)
        if "courses" not in df.columns:
            df = pd.DataFrame([{"courses": DEFAULT_COURSES}])
            df.to_csv(DATA_FILE, index=False)

def read_data() -> pd.DataFrame:
    initialize_data_file()
    return pd.read_csv(DATA_FILE).to_dict(orient="records")

def write_data(data):
    df = pd.DataFrame(data)
    df.to_csv(DATA_FILE, index=False)

def log_action(action_name: str, params: dict[str, Any]):
    """يسجل كل أكشن في ملف JSON"""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action_name,
        "params": params
    }
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)

def drop_course(student_id: str, course_code: str):
    data = read_data()
    student = data[0]
    courses = student["courses"].split(',')
    if course_code in courses:
        courses.remove(course_code)
    student["courses"] = ','.join(courses)
    write_data(data)
    log_action("drop_course", {"student_id": student_id, "course_code": course_code})
    return f"✅ Dropped {course_code} for student {student_id}"

def manipulate_course(student_id: str, old_code: str, new_code: str):
    data = read_data()
    student = data[0]
    courses = student["courses"].split(',')
    if old_code in courses:
        courses.remove(old_code)
    courses.append(new_code)
    student["courses"] = ','.join(courses)
    write_data(data)
    log_action("manipulate_course", {"student_id": student_id, "old_code": old_code, "new_code": new_code})
    return f"✅ Changed {old_code} to {new_code} for student {student_id}"


def add_course(student_id: str, course_code: str):
    data = read_data()
    student = data[0]
    courses = student["courses"].split(',')
    courses.append(course_code)
    student["courses"] = ','.join(courses)
    write_data(data)
    log_action("add_course", {"student_id": student_id, "course_code": course_code})
    return f"✅ Added {course_code} for student {student_id}"

def excuse_course(student_id, course_code):
    message = drop_course(student_id, course_code) + " (Excused)"
    log_action("excuse_course", {"student_id": student_id, "course_code": course_code})
    return message
