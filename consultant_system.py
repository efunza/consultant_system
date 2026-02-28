# app.py
import streamlit as st
import pandas as pd
import re
from datetime import datetime, date, time, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

st.set_page_config(page_title="Personalized Planning & Intelligent Consultant", layout="wide")

# -----------------------------
# Data Models
# -----------------------------
@dataclass
class Task:
    title: str
    category: str
    priority: int            # 1 (low) - 5 (high)
    deadline: date
    duration_min: int
    energy: str              # Low / Medium / High
    notes: str = ""
    status: str = "Planned"  # Planned / Done / Skipped

# -----------------------------
# Helpers
# -----------------------------
ENERGY_ORDER = {"Low": 0, "Medium": 1, "High": 2}

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def today_local() -> date:
    return date.today()

def parse_date_guess(text: str) -> date:
    """
    Tries to parse a date from text. Supports:
    - YYYY-MM-DD
    - 'today', 'tomorrow'
    - 'next week' -> today + 7
    Fallback: today + 1
    """
    t = text.strip().lower()
    if "today" in t:
        return today_local()
    if "tomorrow" in t:
        return today_local() + timedelta(days=1)
    if "next week" in t:
        return today_local() + timedelta(days=7)

    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", t)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return date(y, mo, d)
        except ValueError:
            pass

    return today_local() + timedelta(days=1)

def parse_duration_guess(text: str) -> int:
    """
    Supports:
    - '30 min', '45 minutes'
    - '2h', '1.5h', '2 hours'
    Fallback: 30
    """
    t = text.lower().strip()

    # minutes
    m = re.search(r"(\d+)\s*(min|mins|minute|minutes)\b", t)
    if m:
        return clamp(int(m.group(1)), 5, 600)

    # hours like 2h or 1.5h
    m = re.search(r"(\d+(\.\d+)?)\s*h\b", t)
    if m:
        hours = float(m.group(1))
        return clamp(int(round(hours * 60)), 5, 600)

    # hours like "2 hours"
    m = re.search(r"(\d+(\.\d+)?)\s*(hour|hours)\b", t)
    if m:
        hours = float(m.group(1))
        return clamp(int(round(hours * 60)), 5, 600)

    return 30

def parse_priority_guess(text: str) -> int:
    """
    Supports:
    - 'priority 4' or 'p4'
    - 'urgent' -> 5
    - 'low' -> 2
    Fallback: 3
    """
    t = text.lower()
    m = re.search(r"\bpriority\s*(\d)\b", t)
    if m:
        return clamp(int(m.group(1)), 1, 5)
    m = re.search(r"\bp(\d)\b", t)
    if m:
        return clamp(int(m.group(1)), 1, 5)

    if "urgent" in t or "asap" in t:
        return 5
    if "high" in t:
        return 4
    if "medium" in t:
        return 3
    if "low" in t:
        return 2
    return 3

def parse_energy_guess(text: str) -> str:
    t = text.lower()
    if "high energy" in t or "deep work" in t:
        return "High"
    if "low energy" in t or "easy" in t:
        return "Low"
    return "Medium"

def parse_category_guess(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["school", "homework", "study", "revision"]):
        return "School"
    if any(k in t for k in ["work", "client", "meeting", "report"]):
        return "Work"
    if any(k in t for k in ["health", "exercise", "gym", "run"]):
        return "Health"
    if any(k in t for k in ["family", "chores", "clean", "errand"]):
        return "Home"
    return "Personal"

def task_score(task: Task, now: date, preferred_energy: str) -> float:
    """
    Hybrid scoring: deadline proximity + priority + energy match.
    Higher score -> earlier scheduling.
    """
    days_left = (task.deadline - now).days
    # closer deadlines => bigger urgency component
    urgency = 0
    if days_left <= 0:
        urgency = 10
    else:
        urgency = 1 / (days_left ** 0.5)

    priority_component = task.priority * 1.5
    energy_match = 1.0 if task.energy == preferred_energy else 0.4

    # small penalty for very long tasks to encourage chunking
    duration_penalty = 0.0008 * task.duration_min

    return (urgency * 6) + priority_component + (energy_match * 2) - duration_penalty

def chunk_task(task: Task, max_chunk_min: int) -> List[Task]:
    """
    Break long tasks into smaller chunks for well-being and realism.
    """
    if task.duration_min <= max_chunk_min:
        return [task]
    chunks = []
    remaining = task.duration_min
    i = 1
    while remaining > 0:
        piece = min(max_chunk_min, remaining)
        chunks.append(Task(
            title=f"{task.title} (Part {i})",
            category=task.category,
            priority=task.priority,
            deadline=task.deadline,
            duration_min=piece,
            energy=task.energy,
            notes=task.notes,
            status=task.status
        ))
        remaining -= piece
        i += 1
    return chunks

def build_time_blocks(
    day: date,
    start_t: time,
    end_t: time,
    break_every_min: int,
    break_min: int,
    focus_block_min: int,
) -> List[Tuple[datetime, datetime, str]]:
    """
    Create alternating focus and break blocks for a single day.
    """
    blocks = []
    cur = datetime.combine(day, start_t)
    end = datetime.combine(day, end_t)

    # Ensure sensible:
    focus_block_min = clamp(focus_block_min, 15, 180)
    break_every_min = clamp(break_every_min, 30, 240)
    break_min = clamp(break_min, 5, 30)

    focus_counter = 0

    while cur < end:
        # determine focus duration remaining until break
        remaining_until_break = break_every_min - focus_counter
        focus_duration = min(focus_block_min, remaining_until_break)

        # if not enough time for a real focus block, finish
        if cur + timedelta(minutes=15) > end:
            break

        focus_end = min(end, cur + timedelta(minutes=focus_duration))
        blocks.append((cur, focus_end, "Focus"))
        focus_counter += int((focus_end - cur).total_seconds() // 60)
        cur = focus_end

        if cur >= end:
            break

        # add break if time allows and we hit break threshold
        if focus_counter >= break_every_min and (cur + timedelta(minutes=break_min) <= end):
            b_end = cur + timedelta(minutes=break_min)
            blocks.append((cur, b_end, "Break"))
            cur = b_end
            focus_counter = 0

    return blocks

def generate_plan(
    tasks: List[Task],
    week_start: date,
    days: int,
    work_start: time,
    work_end: time,
    break_every_min: int,
    break_min: int,
    focus_block_min: int,
    preferred_energy: str,
    max_chunk_min: int,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Main planner:
    - chunk long tasks
    - score and order tasks daily
    - fill focus blocks with tasks using duration fitting
    - keep breaks pre-scheduled
    """
    warnings = []
    now = today_local()

    # Expand tasks by chunking
    expanded = []
    for t in tasks:
        expanded.extend(chunk_task(t, max_chunk_min=max_chunk_min))

    # Remove already done/skipped from scheduling
    expanded = [t for t in expanded if t.status == "Planned"]

    schedule_rows = []

    for day_offset in range(days):
        day = week_start + timedelta(days=day_offset)

        blocks = build_time_blocks(
            day=day,
            start_t=work_start,
            end_t=work_end,
            break_every_min=break_every_min,
            break_min=break_min,
            focus_block_min=focus_block_min
        )

        # daily filter: tasks not past deadline by too much; still allow overdue
        # but prioritize them higher via scoring
        daily_tasks = sorted(
            expanded,
            key=lambda x: task_score(x, now=day, preferred_energy=preferred_energy),
            reverse=True
        )

        for (b_start, b_end, b_type) in blocks:
            if b_type == "Break":
                schedule_rows.append({
                    "Date": day.isoformat(),
                    "Start": b_start.strftime("%H:%M"),
                    "End": b_end.strftime("%H:%M"),
                    "Type": "Break",
                    "Task": "Break / Reset",
                    "Category": "",
                    "Priority": "",
                    "Deadline": "",
                    "Energy": "",
                    "Duration(min)": int((b_end - b_start).total_seconds() // 60),
                })
                continue

            # focus block duration
            block_min = int((b_end - b_start).total_seconds() // 60)
            if block_min < 10:
                continue

            # pick best-fitting task (duration <= block)
            chosen_idx = None
            for i, t in enumerate(daily_tasks):
                if t.duration_min <= block_min:
                    chosen_idx = i
                    break

            if chosen_idx is None:
                # no fitting task; leave as buffer
                schedule_rows.append({
                    "Date": day.isoformat(),
                    "Start": b_start.strftime("%H:%M"),
                    "End": b_end.strftime("%H:%M"),
                    "Type": "Buffer",
                    "Task": "Buffer / Admin / Light catch-up",
                    "Category": "",
                    "Priority": "",
                    "Deadline": "",
                    "Energy": "",
                    "Duration(min)": block_min,
                })
                continue

            t = daily_tasks.pop(chosen_idx)
            # remove from expanded too (first match by title+deadline+duration)
            for j, et in enumerate(expanded):
                if (et.title == t.title and et.deadline == t.deadline and et.duration_min == t.duration_min):
                    expanded.pop(j)
                    break

            schedule_rows.append({
                "Date": day.isoformat(),
                "Start": b_start.strftime("%H:%M"),
                "End": (b_start + timedelta(minutes=t.duration_min)).strftime("%H:%M"),
                "Type": "Task",
                "Task": t.title,
                "Category": t.category,
                "Priority": t.priority,
                "Deadline": t.deadline.isoformat(),
                "Energy": t.energy,
                "Duration(min)": t.duration_min,
            })

            # leftover time in focus block becomes buffer
            remaining = block_min - t.duration_min
            if remaining >= 10:
                rb_start = b_start + timedelta(minutes=t.duration_min)
                rb_end = b_end
                schedule_rows.append({
                    "Date": day.isoformat(),
                    "Start": rb_start.strftime("%H:%M"),
                    "End": rb_end.strftime("%H:%M"),
                    "Type": "Buffer",
                    "Task": "Buffer / Stretch / Quick messages",
                    "Category": "",
                    "Priority": "",
                    "Deadline": "",
                    "Energy": "",
                    "Duration(min)": int((rb_end - rb_start).total_seconds() // 60),
                })

        # warn if there are urgent tasks not scheduled for that day but deadline is today/overdue
        urgent_unscheduled = [t for t in daily_tasks if (t.deadline - day).days <= 0 and t.priority >= 4]
        if urgent_unscheduled:
            warnings.append(
                f"{day.isoformat()}: Some urgent/overdue high-priority tasks weren't scheduled. "
                f"Consider extending work hours or reducing break frequency."
            )

    df = pd.DataFrame(schedule_rows)
    return df, warnings

def tasks_to_df(tasks: List[Task]) -> pd.DataFrame:
    if not tasks:
        return pd.DataFrame(columns=["title", "category", "priority", "deadline", "duration_min", "energy", "notes", "status"])
    return pd.DataFrame([asdict(t) for t in tasks])

def df_to_tasks(df: pd.DataFrame) -> List[Task]:
    tasks = []
    for _, r in df.iterrows():
        tasks.append(Task(
            title=str(r["title"]),
            category=str(r["category"]),
            priority=int(r["priority"]),
            deadline=pd.to_datetime(r["deadline"]).date(),
            duration_min=int(r["duration_min"]),
            energy=str(r["energy"]),
            notes=str(r.get("notes", "")),
            status=str(r.get("status", "Planned")),
        ))
    return tasks

# -----------------------------
# Session State
# -----------------------------
if "tasks" not in st.session_state:
    st.session_state.tasks = [
        Task("Math revision", "School", 4, today_local() + timedelta(days=2), 60, "High", "Past paper practice"),
        Task("Write project abstract", "School", 5, today_local() + timedelta(days=1), 45, "Medium", "Keep it concise"),
        Task("Workout", "Health", 3, today_local() + timedelta(days=1), 30, "Medium", "Light session"),
    ]
if "last_plan" not in st.session_state:
    st.session_state.last_plan = None
if "feedback" not in st.session_state:
    st.session_state.feedback = {"satisfaction": 3, "notes": ""}

# Adaptive knobs (simple “learning from feedback”)
if "adaptive" not in st.session_state:
    st.session_state.adaptive = {
        "break_every_min": 60,
        "break_min": 10,
        "focus_block_min": 45,
        "preferred_energy": "High",
        "max_chunk_min": 60,
    }

# -----------------------------
# UI
# -----------------------------
st.title("📅 Personalized Planning & Intelligent Consultant System (Streamlit Prototype)")
st.caption(
    "A privacy-first prototype that builds daily/weekly schedules using a hybrid of rule-based planning and adaptive feedback."
)

left, right = st.columns([1.05, 1.2], gap="large")

with left:
    st.subheader("1) Goals & Preferences")

    goal = st.text_area(
        "Your main goal (this week)",
        value="Finish my school tasks early and keep a healthy balance.",
        height=80
    )

    colA, colB = st.columns(2)
    with colA:
        plan_horizon = st.selectbox("Plan horizon", ["Today", "Next 3 days", "This week (7 days)"], index=2)
        if plan_horizon == "Today":
            days = 1
        elif plan_horizon == "Next 3 days":
            days = 3
        else:
            days = 7

        start_hour = st.slider("Work start hour", 5, 12, 8)
        end_hour = st.slider("Work end hour", 13, 23, 18)

    with colB:
        preferred_energy = st.selectbox(
            "Preferred energy for hardest tasks",
            ["Low", "Medium", "High"],
            index=["Low", "Medium", "High"].index(st.session_state.adaptive["preferred_energy"])
        )
        max_chunk_min = st.slider("Max single-task chunk (min)", 20, 120, st.session_state.adaptive["max_chunk_min"], step=5)
        focus_block_min = st.slider("Focus block length (min)", 15, 120, st.session_state.adaptive["focus_block_min"], step=5)

    st.subheader("2) Break & Well-being Settings")
    colC, colD = st.columns(2)
    with colC:
        break_every_min = st.slider("Add a break every (min of focus)", 30, 180, st.session_state.adaptive["break_every_min"], step=5)
    with colD:
        break_min = st.slider("Break duration (min)", 5, 30, st.session_state.adaptive["break_min"], step=1)

    st.info(
        "Privacy note: This prototype stores your inputs only in the current Streamlit session. "
        "No external API calls are required."
    )

    st.subheader("3) Add Tasks")

    with st.expander("Add task (form)", expanded=True):
        t_title = st.text_input("Task title", value="")
        t_category = st.selectbox("Category", ["School", "Work", "Health", "Home", "Personal"], index=0)
        t_priority = st.slider("Priority (1 low → 5 high)", 1, 5, 3)
        t_deadline = st.date_input("Deadline", value=today_local() + timedelta(days=1))
        t_duration = st.number_input("Duration (minutes)", min_value=5, max_value=600, value=45, step=5)
        t_energy = st.selectbox("Energy required", ["Low", "Medium", "High"], index=1)
        t_notes = st.text_input("Notes (optional)", value="")

        if st.button("➕ Add task"):
            if not t_title.strip():
                st.warning("Please enter a task title.")
            else:
                st.session_state.tasks.append(Task(
                    title=t_title.strip(),
                    category=t_category,
                    priority=int(t_priority),
                    deadline=t_deadline,
                    duration_min=int(t_duration),
                    energy=t_energy,
                    notes=t_notes.strip(),
                ))
                st.success("Task added.")

    with st.expander("Add tasks conversationally (NLP-ish)", expanded=False):
        st.write("Examples you can type:")
        st.code(
            "Finish biology notes tomorrow 45 min priority 4\n"
            "Workout next week 30min\n"
            "Write report 2h urgent\n"
            "Study math 1.5h high energy"
        )
        chat_in = st.text_area("Type one task per line", height=120, placeholder="Enter tasks here...")
        if st.button("🧠 Parse & add from text"):
            lines = [ln.strip() for ln in chat_in.splitlines() if ln.strip()]
            added = 0
            for ln in lines:
                new_task = Task(
                    title=re.sub(r"\b(today|tomorrow|next week|\d{4}-\d{2}-\d{2})\b", "", ln, flags=re.I).strip() or "Untitled task",
                    category=parse_category_guess(ln),
                    priority=parse_priority_guess(ln),
                    deadline=parse_date_guess(ln),
                    duration_min=parse_duration_guess(ln),
                    energy=parse_energy_guess(ln),
                    notes="Added via text input"
                )
                st.session_state.tasks.append(new_task)
                added += 1
            st.success(f"Added {added} task(s).")

    st.subheader("4) Task List")
    df_tasks = tasks_to_df(st.session_state.tasks)

    # Editable task table
    edited = st.data_editor(
        df_tasks,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "priority": st.column_config.NumberColumn(min_value=1, max_value=5, step=1),
            "duration_min": st.column_config.NumberColumn("duration_min", min_value=5, max_value=600, step=5),
            "deadline": st.column_config.DateColumn("deadline"),
            "status": st.column_config.SelectboxColumn("status", options=["Planned", "Done", "Skipped"]),
            "energy": st.column_config.SelectboxColumn("energy", options=["Low", "Medium", "High"]),
            "category": st.column_config.SelectboxColumn("category", options=["School", "Work", "Health", "Home", "Personal"]),
        }
    )

    colE, colF, colG = st.columns(3)
    with colE:
        if st.button("💾 Save edits"):
            try:
                st.session_state.tasks = df_to_tasks(edited)
                st.success("Saved.")
            except Exception as e:
                st.error(f"Could not save edits: {e}")

    with colF:
        if st.button("🧹 Clear all data"):
            st.session_state.tasks = []
            st.session_state.last_plan = None
            st.session_state.feedback = {"satisfaction": 3, "notes": ""}
            st.success("Cleared.")

    with colG:
        csv = edited.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export tasks CSV", data=csv, file_name="tasks.csv", mime="text/csv")

with right:
    st.subheader("5) Generate Personalized Plan")

    week_start = today_local()
    work_start = time(start_hour, 0)
    work_end = time(end_hour, 0)

    # Basic constraints check
    if end_hour <= start_hour:
        st.error("Work end hour must be later than start hour.")
    else:
        if st.button("✨ Generate plan"):
            plan_df, warnings = generate_plan(
                tasks=st.session_state.tasks,
                week_start=week_start,
                days=days,
                work_start=work_start,
                work_end=work_end,
                break_every_min=break_every_min,
                break_min=break_min,
                focus_block_min=focus_block_min,
                preferred_energy=preferred_energy,
                max_chunk_min=max_chunk_min,
            )
            st.session_state.last_plan = plan_df

            # Update adaptive settings (store latest selections)
            st.session_state.adaptive.update({
                "break_every_min": break_every_min,
                "break_min": break_min,
                "focus_block_min": focus_block_min,
                "preferred_energy": preferred_energy,
                "max_chunk_min": max_chunk_min,
            })

            if warnings:
                st.warning("Warnings:\n- " + "\n- ".join(warnings))
            else:
                st.success("Plan generated.")

    st.markdown("**Goal (used as context):**")
    st.write(goal)

    st.divider()

    plan_df = st.session_state.last_plan
    if plan_df is None or plan_df.empty:
        st.info("No plan yet. Click **Generate plan** to create your schedule.")
    else:
        st.subheader("📌 Schedule")
        # Display grouped by date
        for d, g in plan_df.groupby("Date"):
            st.markdown(f"### {d}")
            st.dataframe(g[["Start", "End", "Type", "Task", "Category", "Priority", "Deadline", "Energy", "Duration(min)"]],
                         use_container_width=True)

        st.download_button(
            "⬇️ Export schedule CSV",
            data=plan_df.to_csv(index=False).encode("utf-8"),
            file_name="schedule.csv",
            mime="text/csv"
        )

        st.divider()

        st.subheader("6) Feedback & Adaptation")
        st.write("Rate how the plan feels. The system will adjust break frequency and focus blocks next time.")

        satisfaction = st.slider("Satisfaction (1 low → 5 high)", 1, 5, st.session_state.feedback["satisfaction"])
        fb_notes = st.text_area("Notes (what should change?)", value=st.session_state.feedback["notes"], height=90)

        if st.button("✅ Save feedback & adapt"):
            st.session_state.feedback = {"satisfaction": satisfaction, "notes": fb_notes}

            # Simple “learning”: if low satisfaction, add more breaks and shorten focus blocks.
            # if high satisfaction, slightly reduce breaks and allow longer focus blocks.
            be = st.session_state.adaptive["break_every_min"]
            bm = st.session_state.adaptive["break_min"]
            fb = st.session_state.adaptive["focus_block_min"]

            if satisfaction <= 2:
                be = clamp(be - 10, 30, 180)
                bm = clamp(bm + 2, 5, 30)
                fb = clamp(fb - 5, 15, 120)
            elif satisfaction >= 4:
                be = clamp(be + 10, 30, 180)
                bm = clamp(bm - 1, 5, 30)
                fb = clamp(fb + 5, 15, 120)

            st.session_state.adaptive.update({
                "break_every_min": be,
                "break_min": bm,
                "focus_block_min": fb,
            })

            st.success(
                f"Adapted settings → break every {be} min, break {bm} min, focus block {fb} min."
            )

        st.divider()

        st.subheader("7) Evaluation Dashboard (Prototype Metrics)")

        # Completion metrics from tasks table
        tdf = tasks_to_df(st.session_state.tasks)
        if len(tdf) == 0:
            st.info("Add tasks to see metrics.")
        else:
            total = len(tdf)
            done = int((tdf["status"] == "Done").sum())
            skipped = int((tdf["status"] == "Skipped").sum())
            planned = int((tdf["status"] == "Planned").sum())

            completion_rate = (done / total) if total else 0.0

            # "Recommendation accuracy" proxy:
            # how many scheduled task blocks correspond to high priority tasks (>=4)
            scheduled_tasks = plan_df[plan_df["Type"] == "Task"].copy()
            if not scheduled_tasks.empty:
                high_pri_blocks = (scheduled_tasks["Priority"].fillna(0).astype(int) >= 4).sum()
                rec_accuracy_proxy = high_pri_blocks / max(1, len(scheduled_tasks))
            else:
                rec_accuracy_proxy = 0.0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Tasks total", total)
            col2.metric("Completion rate", f"{completion_rate*100:.1f}%")
            col3.metric("High-priority focus share", f"{rec_accuracy_proxy*100:.1f}%")
            col4.metric("Satisfaction", str(st.session_state.feedback["satisfaction"]))

            st.caption(
                "Note: These are prototype metrics. In a full study, you’d log outcomes over time and compute "
                "recommendation accuracy and user satisfaction more rigorously (with consent)."
            )

st.divider()
st.markdown(
    """
### What to include in your report (quick mapping)
- **Rule-based + ML framing:** This prototype uses scoring + constraints as the rule engine; feedback adaptation simulates learning.
- **Well-being:** Break intervals, task chunking, and buffer blocks prevent overload.
- **NLP:** Conversational task entry parses dates/durations/priorities from text.
- **Privacy:** Local session state; export is optional; no external calls required.
"""
)
