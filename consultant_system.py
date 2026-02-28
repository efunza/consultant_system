# app.py
# Project 3: Personalized Planning & Intelligent Consultant System (Upgraded)
# Single-file Streamlit app with:
# - Better scheduling: time windows, recurring tasks, dependencies, max daily load, anti-burnout rules
# - Smarter chunking: optional “subtasks” splitting + max chunk length
# - Stronger NLP: conversational parsing + confirmation step before saving
# - Explainability: “Why this slot?” reasons for each scheduled task
# - Evaluation: completion/adherence metrics + trends (session-local)
# - Privacy-first: local-only session state + JSON export/import + clear data

import streamlit as st
import pandas as pd
import json
import re
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Optional, Tuple

st.set_page_config(page_title="Personalized Planning & Intelligent Consultant (Upgraded)", layout="wide")

# -----------------------------
# Models
# -----------------------------
@dataclass
class Task:
    id: str
    title: str
    category: str
    priority: int                 # 1-5
    deadline: date
    duration_min: int
    energy: str                   # Low/Medium/High
    notes: str = ""
    status: str = "Planned"       # Planned/Done/Skipped

    # Upgrades
    earliest_start: Optional[time] = None  # time window start (optional)
    latest_end: Optional[time] = None      # time window end (optional)
    depends_on: List[str] = None           # list of task ids
    recurrence: str = "None"               # None/Daily/Weekly
    weekly_days: List[int] = None          # 0=Mon ... 6=Sun
    split_mode: str = "Auto"               # Auto/Manual
    subtasks: str = ""                     # newline-separated subtasks (optional)
    estimate_min: Optional[int] = None     # optional “best guess”
    estimate_max: Optional[int] = None     # optional “worst-case”

@dataclass
class ScheduledItem:
    day: date
    start_dt: datetime
    end_dt: datetime
    item_type: str                 # Task/Break/Buffer/Recovery
    task_id: str = ""
    task_title: str = ""
    category: str = ""
    priority: Optional[int] = None
    deadline: Optional[date] = None
    energy: str = ""
    duration_min: int = 0
    reason: str = ""


# -----------------------------
# Constants / Helpers
# -----------------------------
CATEGORIES = ["School", "Work", "Health", "Home", "Personal"]
ENERGIES = ["Low", "Medium", "High"]
RECURRENCES = ["None", "Daily", "Weekly"]

ENERGY_ORDER = {"Low": 0, "Medium": 1, "High": 2}

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def iso(d: date) -> str:
    return d.isoformat()

def today_local() -> date:
    return date.today()

def time_or_none(hhmm: str) -> Optional[time]:
    hhmm = (hhmm or "").strip()
    if not hhmm:
        return None
    try:
        parts = hhmm.split(":")
        if len(parts) != 2:
            return None
        h, m = int(parts[0]), int(parts[1])
        return time(h, m)
    except Exception:
        return None

def safe_list(x):
    return x if isinstance(x, list) else []

def parse_date_guess(text: str) -> date:
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
    t = text.lower().strip()

    m = re.search(r"(\d+)\s*(min|mins|minute|minutes)\b", t)
    if m:
        return clamp(int(m.group(1)), 5, 600)

    m = re.search(r"(\d+(\.\d+)?)\s*h\b", t)
    if m:
        return clamp(int(round(float(m.group(1)) * 60)), 5, 600)

    m = re.search(r"(\d+(\.\d+)?)\s*(hour|hours)\b", t)
    if m:
        return clamp(int(round(float(m.group(1)) * 60)), 5, 600)

    return 30

def parse_priority_guess(text: str) -> int:
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
    if any(k in t for k in ["school", "homework", "study", "revision", "exam", "assignment"]):
        return "School"
    if any(k in t for k in ["work", "client", "meeting", "report", "proposal"]):
        return "Work"
    if any(k in t for k in ["health", "exercise", "gym", "run", "walk"]):
        return "Health"
    if any(k in t for k in ["family", "chores", "clean", "errand", "laundry"]):
        return "Home"
    return "Personal"

def parse_time_window(text: str) -> Tuple[Optional[time], Optional[time]]:
    """
    Tries patterns like:
    - "after 4pm"
    - "before 6pm"
    - "between 14:00-16:00"
    - "14:00-16:30"
    """
    t = text.lower()

    def parse_ampm(s: str) -> Optional[time]:
        s = s.strip()
        m = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", s)
        if not m:
            return None
        hh = int(m.group(1))
        mm = int(m.group(2) or "0")
        ap = m.group(3)
        if ap == "pm" and hh != 12:
            hh += 12
        if ap == "am" and hh == 12:
            hh = 0
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return time(hh, mm)
        return None

    # between 14:00-16:00
    m = re.search(r"between\s+(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})", t)
    if m:
        return time_or_none(m.group(1)), time_or_none(m.group(2))

    # plain 14:00-16:00
    m = re.search(r"\b(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\b", t)
    if m:
        return time_or_none(m.group(1)), time_or_none(m.group(2))

    # after 4pm
    m = re.search(r"after\s+(\d{1,2}(?::\d{2})?\s*(am|pm))", t)
    if m:
        return parse_ampm(m.group(1)), None

    # before 6pm
    m = re.search(r"before\s+(\d{1,2}(?::\d{2})?\s*(am|pm))", t)
    if m:
        return None, parse_ampm(m.group(1))

    return None, None

def recurrence_instances(task: Task, start_day: date, days: int) -> List[Task]:
    """
    Expand a recurring task into per-day instances (still linked via original id in depends).
    Each instance gets its own id but keeps notes referencing recurrence.
    """
    if task.recurrence == "None":
        return [task]

    instances = []
    for offset in range(days):
        d = start_day + timedelta(days=offset)
        if task.recurrence == "Daily":
            ok = True
        elif task.recurrence == "Weekly":
            wd = d.weekday()
            ok = wd in (task.weekly_days or [])
        else:
            ok = False

        if not ok:
            continue

        inst = Task(
            id=str(uuid.uuid4()),
            title=f"{task.title} ({d.isoformat()})",
            category=task.category,
            priority=task.priority,
            deadline=min(task.deadline, d) if task.deadline else d,
            duration_min=task.duration_min,
            energy=task.energy,
            notes=(task.notes + " | Recurring instance").strip(),
            status=task.status,
            earliest_start=task.earliest_start,
            latest_end=task.latest_end,
            depends_on=safe_list(task.depends_on),
            recurrence="None",
            weekly_days=[],
            split_mode=task.split_mode,
            subtasks=task.subtasks,
            estimate_min=task.estimate_min,
            estimate_max=task.estimate_max,
        )
        instances.append(inst)

    return instances

def split_into_subtasks(task: Task, max_chunk_min: int) -> List[Task]:
    """
    Smarter splitting:
    - If user provided manual subtasks (lines), split duration across them (roughly evenly).
    - Else fall back to max_chunk splitting.
    """
    subtasks = [s.strip() for s in (task.subtasks or "").splitlines() if s.strip()]
    if task.split_mode == "Manual" and subtasks:
        n = len(subtasks)
        base = max(10, int(round(task.duration_min / n)))
        pieces = []
        remaining = task.duration_min
        for i, name in enumerate(subtasks, start=1):
            dur = base if i < n else remaining
            dur = clamp(dur, 10, 600)
            remaining -= dur
            pieces.append(Task(
                id=str(uuid.uuid4()),
                title=f"{task.title}: {name}",
                category=task.category,
                priority=task.priority,
                deadline=task.deadline,
                duration_min=dur,
                energy=task.energy,
                notes=(task.notes + " | Subtask").strip(),
                status=task.status,
                earliest_start=task.earliest_start,
                latest_end=task.latest_end,
                depends_on=safe_list(task.depends_on),
                recurrence="None",
                weekly_days=[],
                split_mode="Auto",
                subtasks="",
                estimate_min=None,
                estimate_max=None,
            ))
        return pieces

    # Auto split by max chunk
    if task.duration_min <= max_chunk_min:
        return [task]

    pieces = []
    remaining = task.duration_min
    i = 1
    while remaining > 0:
        dur = min(max_chunk_min, remaining)
        pieces.append(Task(
            id=str(uuid.uuid4()),
            title=f"{task.title} (Part {i})",
            category=task.category,
            priority=task.priority,
            deadline=task.deadline,
            duration_min=dur,
            energy=task.energy,
            notes=(task.notes + " | Auto-split").strip(),
            status=task.status,
            earliest_start=task.earliest_start,
            latest_end=task.latest_end,
            depends_on=safe_list(task.depends_on),
            recurrence="None",
            weekly_days=[],
            split_mode="Auto",
            subtasks="",
            estimate_min=None,
            estimate_max=None,
        ))
        remaining -= dur
        i += 1
    return pieces

def build_day_blocks(
    day: date,
    start_t: time,
    end_t: time,
    break_every_min: int,
    break_min: int,
    focus_block_min: int,
) -> List[Tuple[datetime, datetime, str]]:
    """
    Alternating focus blocks with breaks. Break inserted after break_every_min focus minutes.
    """
    blocks = []
    cur = datetime.combine(day, start_t)
    end = datetime.combine(day, end_t)

    focus_block_min = clamp(focus_block_min, 15, 180)
    break_every_min = clamp(break_every_min, 30, 240)
    break_min = clamp(break_min, 5, 30)

    focused_since_break = 0

    while cur < end:
        remaining_until_break = break_every_min - focused_since_break
        focus_duration = min(focus_block_min, remaining_until_break)

        if cur + timedelta(minutes=15) > end:
            break

        focus_end = min(end, cur + timedelta(minutes=focus_duration))
        blocks.append((cur, focus_end, "Focus"))
        focused_since_break += int((focus_end - cur).total_seconds() // 60)
        cur = focus_end

        if cur >= end:
            break

        if focused_since_break >= break_every_min and (cur + timedelta(minutes=break_min) <= end):
            b_end = cur + timedelta(minutes=break_min)
            blocks.append((cur, b_end, "Break"))
            cur = b_end
            focused_since_break = 0

    return blocks

def task_score(task: Task, on_day: date, preferred_energy: str) -> float:
    """
    Score combines:
    - urgency (deadline proximity)
    - priority
    - energy match
    - light penalty for long tasks
    """
    days_left = (task.deadline - on_day).days
    if days_left <= 0:
        urgency = 10.0
    else:
        urgency = 1.0 / (days_left ** 0.5)

    priority_component = task.priority * 1.7
    energy_match = 1.0 if task.energy == preferred_energy else 0.5
    duration_penalty = 0.0008 * task.duration_min

    return (urgency * 6.5) + priority_component + (energy_match * 2.2) - duration_penalty

def within_time_window(block_start: datetime, block_end: datetime, t: Task) -> bool:
    if t.earliest_start is not None:
        earliest = datetime.combine(block_start.date(), t.earliest_start)
        if block_end <= earliest:
            return False
        if block_start < earliest and block_end - earliest < timedelta(minutes=10):
            return False
    if t.latest_end is not None:
        latest = datetime.combine(block_start.date(), t.latest_end)
        if block_start >= latest:
            return False
        if block_end > latest and latest - block_start < timedelta(minutes=10):
            return False
    return True

def dependencies_satisfied(task: Task, done_ids: set) -> bool:
    deps = safe_list(task.depends_on)
    return all(d in done_ids for d in deps)

def recommend_break_style(satisfaction: int) -> str:
    if satisfaction <= 2:
        return "More recovery: shorter focus blocks + more frequent breaks."
    if satisfaction >= 4:
        return "You’re doing well: slightly longer focus blocks and fewer breaks may work."
    return "Balanced: keep current pacing."

def tasks_to_df(tasks: List[Task]) -> pd.DataFrame:
    if not tasks:
        return pd.DataFrame(columns=[
            "id","title","category","priority","deadline","duration_min","energy","notes","status",
            "earliest_start","latest_end","depends_on","recurrence","weekly_days",
            "split_mode","subtasks","estimate_min","estimate_max"
        ])
    rows = []
    for t in tasks:
        d = asdict(t)
        d["deadline"] = t.deadline.isoformat()
        d["earliest_start"] = t.earliest_start.strftime("%H:%M") if t.earliest_start else ""
        d["latest_end"] = t.latest_end.strftime("%H:%M") if t.latest_end else ""
        d["depends_on"] = ",".join(safe_list(t.depends_on))
        d["weekly_days"] = ",".join(str(x) for x in safe_list(t.weekly_days))
        rows.append(d)
    return pd.DataFrame(rows)

def df_to_tasks(df: pd.DataFrame) -> List[Task]:
    tasks = []
    for _, r in df.iterrows():
        deps = [x.strip() for x in str(r.get("depends_on","")).split(",") if x.strip()]
        wdays = []
        for x in str(r.get("weekly_days","")).split(","):
            x = x.strip()
            if x.isdigit():
                wdays.append(int(x))
        tasks.append(Task(
            id=str(r.get("id") or uuid.uuid4()),
            title=str(r.get("title","")).strip() or "Untitled task",
            category=str(r.get("category","Personal")),
            priority=int(clamp(int(r.get("priority",3)), 1, 5)),
            deadline=pd.to_datetime(r.get("deadline", today_local())).date(),
            duration_min=int(clamp(int(r.get("duration_min",30)), 5, 600)),
            energy=str(r.get("energy","Medium")),
            notes=str(r.get("notes","")),
            status=str(r.get("status","Planned")),
            earliest_start=time_or_none(str(r.get("earliest_start",""))),
            latest_end=time_or_none(str(r.get("latest_end",""))),
            depends_on=deps,
            recurrence=str(r.get("recurrence","None")),
            weekly_days=wdays,
            split_mode=str(r.get("split_mode","Auto")),
            subtasks=str(r.get("subtasks","")),
            estimate_min=(int(r.get("estimate_min")) if str(r.get("estimate_min","")).strip().isdigit() else None),
            estimate_max=(int(r.get("estimate_max")) if str(r.get("estimate_max","")).strip().isdigit() else None),
        ))
    return tasks

def schedule_plan(
    tasks: List[Task],
    start_day: date,
    days: int,
    work_start: time,
    work_end: time,
    break_every_min: int,
    break_min: int,
    focus_block_min: int,
    preferred_energy: str,
    max_chunk_min: int,
    max_daily_focus_min: int,
    max_high_energy_in_row: int,
    recovery_after_high: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    warnings = []

    # 1) Expand recurring tasks
    expanded: List[Task] = []
    for t in tasks:
        expanded.extend(recurrence_instances(t, start_day, days))

    # 2) Remove non-planned
    expanded = [t for t in expanded if t.status == "Planned"]

    # 3) Split tasks into chunks/subtasks
    split_tasks: List[Task] = []
    for t in expanded:
        split_tasks.extend(split_into_subtasks(t, max_chunk_min=max_chunk_min))

    # Quick lookup for original titles in explanations
    task_by_id: Dict[str, Task] = {t.id: t for t in split_tasks}

    schedule_items: List[ScheduledItem] = []

    # Track completion for dependency satisfaction (within generated plan, treat scheduled as "planned", not done)
    done_ids_global: set = set()

    for day_offset in range(days):
        day = start_day + timedelta(days=day_offset)
        blocks = build_day_blocks(day, work_start, work_end, break_every_min, break_min, focus_block_min)

        daily_focus_used = 0
        high_energy_streak = 0

        # Recompute ordering each day (dynamic)
        candidates = sorted(
            split_tasks,
            key=lambda x: task_score(x, on_day=day, preferred_energy=preferred_energy),
            reverse=True
        )

        for (b_start, b_end, b_type) in blocks:
            block_minutes = int((b_end - b_start).total_seconds() // 60)

            if b_type == "Break":
                schedule_items.append(ScheduledItem(
                    day=day, start_dt=b_start, end_dt=b_end, item_type="Break",
                    task_title="Break / Reset", duration_min=block_minutes,
                    reason="Well-being: scheduled break interval."
                ))
                high_energy_streak = 0
                continue

            # Max daily focus load
            if daily_focus_used >= max_daily_focus_min:
                schedule_items.append(ScheduledItem(
                    day=day, start_dt=b_start, end_dt=b_end, item_type="Buffer",
                    task_title="Buffer / Low load (daily focus limit reached)",
                    duration_min=block_minutes,
                    reason="Well-being: max daily focus limit."
                ))
                continue

            # Choose best fitting task with constraints
            chosen_idx = None
            chosen = None
            chosen_reason = ""

            for i, t in enumerate(candidates):
                # Fit inside block
                if t.duration_min > block_minutes:
                    continue

                # Time window constraint
                if not within_time_window(b_start, b_end, t):
                    continue

                # Dependency constraint (uses global done set; for planning we treat dependencies as needing completion,
                # but in a prototype we allow dependency satisfaction based on "Done" tasks only.
                # You can tighten this later by allowing "scheduled earlier in plan" as satisfied.)
                if not dependencies_satisfied(t, done_ids_global):
                    continue

                # Anti-burnout: limit high-energy streak
                if t.energy == "High" and high_energy_streak >= max_high_energy_in_row:
                    continue

                # If block would exceed daily focus limit, skip
                if daily_focus_used + t.duration_min > max_daily_focus_min:
                    continue

                chosen_idx = i
                chosen = t

                # Explanation
                days_left = (t.deadline - day).days
                urgency_text = "overdue" if days_left < 0 else ("due today" if days_left == 0 else f"due in {days_left} day(s)")
                energy_text = "energy match" if t.energy == preferred_energy else "acceptable energy"
                tw = ""
                if t.earliest_start or t.latest_end:
                    tw = " + time window respected"
                chosen_reason = f"Priority {t.priority} + {urgency_text} + {energy_text}{tw}."
                break

            if chosen is None:
                schedule_items.append(ScheduledItem(
                    day=day, start_dt=b_start, end_dt=b_end, item_type="Buffer",
                    task_title="Buffer / Admin / Light catch-up",
                    duration_min=block_minutes,
                    reason="No task fit this slot (constraints/time windows/dependencies/durations)."
                ))
                high_energy_streak = 0
                continue

            # Schedule chosen task
            task_end = b_start + timedelta(minutes=chosen.duration_min)
            schedule_items.append(ScheduledItem(
                day=day, start_dt=b_start, end_dt=task_end, item_type="Task",
                task_id=chosen.id,
                task_title=chosen.title,
                category=chosen.category,
                priority=chosen.priority,
                deadline=chosen.deadline,
                energy=chosen.energy,
                duration_min=chosen.duration_min,
                reason=chosen_reason
            ))

            daily_focus_used += chosen.duration_min

            # Update streak logic
            if chosen.energy == "High":
                high_energy_streak += 1
            else:
                high_energy_streak = 0

            # Remove from pools
            candidates.pop(chosen_idx)
            # Remove one matching task from split_tasks (by id)
            split_tasks = [x for x in split_tasks if x.id != chosen.id]

            # Optional: recovery block after high-energy task (if time left)
            remaining = block_minutes - chosen.duration_min
            if recovery_after_high and chosen.energy == "High" and remaining >= 10:
                rb_start = task_end
                rb_end = b_end
                schedule_items.append(ScheduledItem(
                    day=day, start_dt=rb_start, end_dt=rb_end, item_type="Recovery",
                    task_title="Recovery / Walk / Water / Stretch",
                    duration_min=int((rb_end - rb_start).total_seconds() // 60),
                    reason="Well-being: recovery after high-energy work."
                ))
                high_energy_streak = 0
            elif remaining >= 10:
                rb_start = task_end
                rb_end = b_end
                schedule_items.append(ScheduledItem(
                    day=day, start_dt=rb_start, end_dt=rb_end, item_type="Buffer",
                    task_title="Buffer / Quick messages / Stretch",
                    duration_min=int((rb_end - rb_start).total_seconds() // 60),
                    reason="Extra time left in focus block."
                ))

        # Warnings: urgent tasks left unscheduled
        unscheduled_urgent = [t for t in candidates if (t.deadline - day).days <= 0 and t.priority >= 4]
        if unscheduled_urgent:
            warnings.append(
                f"{day.isoformat()}: Some urgent/overdue high-priority tasks were not placed. "
                f"Try extending work hours, raising max daily focus, or reducing constraints."
            )

    # Build DataFrame
    rows = []
    for it in schedule_items:
        rows.append({
            "Date": it.day.isoformat(),
            "Start": it.start_dt.strftime("%H:%M"),
            "End": it.end_dt.strftime("%H:%M"),
            "Type": it.item_type,
            "Task": it.task_title,
            "Category": it.category,
            "Priority": it.priority if it.priority is not None else "",
            "Deadline": it.deadline.isoformat() if it.deadline else "",
            "Energy": it.energy,
            "Duration(min)": it.duration_min,
            "Reason": it.reason
        })
    df = pd.DataFrame(rows)
    return df, warnings

def build_export_payload() -> dict:
    return {
        "version": 2,
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "tasks": tasks_to_df(st.session_state.tasks).to_dict(orient="records"),
        "history": st.session_state.history,
        "adaptive": st.session_state.adaptive,
        "feedback": st.session_state.feedback,
    }

def load_import_payload(payload: dict):
    tasks_records = payload.get("tasks", [])
    df = pd.DataFrame(tasks_records)
    st.session_state.tasks = df_to_tasks(df) if not df.empty else []
    st.session_state.history = payload.get("history", [])
    st.session_state.adaptive = payload.get("adaptive", st.session_state.adaptive)
    st.session_state.feedback = payload.get("feedback", st.session_state.feedback)


# -----------------------------
# Session State
# -----------------------------
if "tasks" not in st.session_state:
    st.session_state.tasks = [
        Task(
            id=str(uuid.uuid4()),
            title="Math revision",
            category="School",
            priority=4,
            deadline=today_local() + timedelta(days=2),
            duration_min=90,
            energy="High",
            notes="Past paper practice",
            earliest_start=None,
            latest_end=None,
            depends_on=[],
            recurrence="None",
            weekly_days=[],
            split_mode="Auto",
            subtasks="",
            estimate_min=60,
            estimate_max=120
        ),
        Task(
            id=str(uuid.uuid4()),
            title="Write project abstract",
            category="School",
            priority=5,
            deadline=today_local() + timedelta(days=1),
            duration_min=60,
            energy="Medium",
            notes="Keep it concise",
            depends_on=[],
            recurrence="None",
            weekly_days=[],
            split_mode="Manual",
            subtasks="Outline key points\nWrite first draft\nEdit and tighten",
            estimate_min=45,
            estimate_max=90
        ),
        Task(
            id=str(uuid.uuid4()),
            title="Workout",
            category="Health",
            priority=3,
            deadline=today_local() + timedelta(days=7),
            duration_min=30,
            energy="Medium",
            notes="Light session",
            depends_on=[],
            recurrence="Weekly",
            weekly_days=[0, 2, 4],  # Mon Wed Fri
            split_mode="Auto",
            subtasks="",
            estimate_min=25,
            estimate_max=40
        ),
    ]

if "last_plan" not in st.session_state:
    st.session_state.last_plan = None

if "feedback" not in st.session_state:
    st.session_state.feedback = {"satisfaction": 3, "notes": ""}

if "adaptive" not in st.session_state:
    st.session_state.adaptive = {
        "break_every_min": 60,
        "break_min": 10,
        "focus_block_min": 45,
        "preferred_energy": "High",
        "max_chunk_min": 60,
        "max_daily_focus_min": 240,
        "max_high_energy_in_row": 2,
        "recovery_after_high": True,
    }

if "history" not in st.session_state:
    # Each entry: {"ts": "...", "completion_rate": float, "satisfaction": int}
    st.session_state.history = []

if "nlp_candidates" not in st.session_state:
    st.session_state.nlp_candidates = pd.DataFrame()

# -----------------------------
# UI
# -----------------------------
st.title("📅 Personalized Planning & Intelligent Consultant System — Upgraded Prototype")
st.caption("Hybrid planning + human-centered well-being + explainable recommendations + privacy-first local data.")

left, right = st.columns([1.05, 1.25], gap="large")

# -----------------------------
# Left panel: Preferences + Tasks
# -----------------------------
with left:
    st.subheader("1) Goal & Preferences")

    goal = st.text_area(
        "Your main goal (used as context)",
        value="Finish important tasks early, avoid burnout, and keep a healthy balance.",
        height=80
    )

    colA, colB = st.columns(2)
    with colA:
        plan_horizon = st.selectbox("Plan horizon", ["Today", "Next 3 days", "This week (7 days)"], index=2)
        days = 1 if plan_horizon == "Today" else (3 if plan_horizon == "Next 3 days" else 7)

        start_hour = st.slider("Work start hour", 5, 12, 8)
        end_hour = st.slider("Work end hour", 13, 23, 18)

    with colB:
        preferred_energy = st.selectbox(
            "Hardest tasks preferred energy",
            ENERGIES,
            index=ENERGIES.index(st.session_state.adaptive["preferred_energy"])
        )
        max_chunk_min = st.slider(
            "Max chunk length (min)",
            20, 120,
            st.session_state.adaptive["max_chunk_min"],
            step=5
        )
        focus_block_min = st.slider(
            "Focus block length (min)",
            15, 120,
            st.session_state.adaptive["focus_block_min"],
            step=5
        )

    st.subheader("2) Well-being & Load Constraints")
    colC, colD = st.columns(2)
    with colC:
        break_every_min = st.slider("Break after (min focus)", 30, 180, st.session_state.adaptive["break_every_min"], step=5)
        max_daily_focus_min = st.slider("Max daily focus minutes", 60, 600, st.session_state.adaptive["max_daily_focus_min"], step=15)
    with colD:
        break_min = st.slider("Break duration (min)", 5, 30, st.session_state.adaptive["break_min"], step=1)
        max_high_energy_in_row = st.slider("Max HIGH-energy tasks in a row", 1, 4, st.session_state.adaptive["max_high_energy_in_row"], step=1)

    recovery_after_high = st.toggle("Add recovery time after HIGH-energy task (if time left)", value=st.session_state.adaptive["recovery_after_high"])

    st.info(
        "Privacy: your data stays in this session unless you export it. "
        "Use Export/Import JSON to move data between devices."
    )

    st.subheader("3) Add Task (Advanced)")
    with st.expander("Add task (form)", expanded=True):
        t_title = st.text_input("Task title", value="")
        t_category = st.selectbox("Category", CATEGORIES, index=0)
        t_priority = st.slider("Priority (1 low → 5 high)", 1, 5, 3)
        t_deadline = st.date_input("Deadline", value=today_local() + timedelta(days=1))
        t_duration = st.number_input("Duration (minutes)", min_value=5, max_value=600, value=45, step=5)
        t_energy = st.selectbox("Energy required", ENERGIES, index=1)
        t_notes = st.text_input("Notes (optional)", value="")

        st.markdown("**Time window (optional):**")
        colTW1, colTW2 = st.columns(2)
        with colTW1:
            earliest = st.text_input("Earliest start (HH:MM)", value="", placeholder="e.g., 16:00")
        with colTW2:
            latest = st.text_input("Latest end (HH:MM)", value="", placeholder="e.g., 18:30")

        st.markdown("**Recurring (optional):**")
        rec = st.selectbox("Recurrence", RECURRENCES, index=0)
        weekly_days = []
        if rec == "Weekly":
            weekly_days = st.multiselect(
                "Which days?",
                options=[("Mon",0),("Tue",1),("Wed",2),("Thu",3),("Fri",4),("Sat",5),("Sun",6)],
                default=[("Mon",0),("Wed",2),("Fri",4)],
                format_func=lambda x: x[0],
            )
            weekly_days = [x[1] for x in weekly_days]

        st.markdown("**Dependencies (optional):**")
        existing_tasks = st.session_state.tasks
        dep_options = [(t.id, t.title) for t in existing_tasks]
        deps_selected = st.multiselect(
            "This task depends on (must be DONE first):",
            options=dep_options,
            format_func=lambda x: x[1],
        )
        deps_selected = [x[0] for x in deps_selected]

        st.markdown("**Smarter splitting (optional):**")
        split_mode = st.radio("Split mode", ["Auto", "Manual"], horizontal=True, index=0)
        subtasks_text = ""
        if split_mode == "Manual":
            subtasks_text = st.text_area(
                "Subtasks (one per line). Duration will be split across them.",
                value="",
                height=90
            )

        st.markdown("**Estimation (optional):**")
        colEst1, colEst2 = st.columns(2)
        with colEst1:
            est_min = st.number_input("Best-case minutes", min_value=0, max_value=1000, value=0, step=5)
        with colEst2:
            est_max = st.number_input("Worst-case minutes", min_value=0, max_value=1000, value=0, step=5)

        if st.button("➕ Add task"):
            if not t_title.strip():
                st.warning("Please enter a task title.")
            else:
                task = Task(
                    id=str(uuid.uuid4()),
                    title=t_title.strip(),
                    category=t_category,
                    priority=int(t_priority),
                    deadline=t_deadline,
                    duration_min=int(t_duration),
                    energy=t_energy,
                    notes=t_notes.strip(),
                    status="Planned",
                    earliest_start=time_or_none(earliest),
                    latest_end=time_or_none(latest),
                    depends_on=deps_selected,
                    recurrence=rec,
                    weekly_days=weekly_days,
                    split_mode=split_mode,
                    subtasks=subtasks_text.strip(),
                    estimate_min=(int(est_min) if est_min > 0 else None),
                    estimate_max=(int(est_max) if est_max > 0 else None),
                )
                st.session_state.tasks.append(task)
                st.success("Task added.")

    st.subheader("4) Conversational Input (with confirmation)")
    with st.expander("Parse tasks from text", expanded=False):
        st.write("Examples:")
        st.code(
            "Finish biology notes tomorrow 45 min priority 4\n"
            "Workout next week 30min weekly Mon Wed Fri\n"
            "Write report 2h urgent between 14:00-16:30\n"
            "Study math 1.5h high energy after 4pm"
        )
        chat_in = st.text_area("Type one task per line", height=120, placeholder="Enter tasks here...")
        if st.button("🧠 Parse (preview before saving)"):
            lines = [ln.strip() for ln in chat_in.splitlines() if ln.strip()]
            rows = []
            for ln in lines:
                es, le = parse_time_window(ln)
                rec = "None"
                weekly_days = []
                if "daily" in ln.lower():
                    rec = "Daily"
                if "weekly" in ln.lower():
                    rec = "Weekly"
                    # quick weekday extraction
                    wd_map = {"mon":0,"tue":1,"wed":2,"thu":3,"fri":4,"sat":5,"sun":6}
                    found = set()
                    for k,v in wd_map.items():
                        if re.search(rf"\b{k}\b", ln.lower()):
                            found.add(v)
                    weekly_days = sorted(list(found)) if found else [0,2,4]

                title_clean = re.sub(r"\b(today|tomorrow|next week|daily|weekly|\d{4}-\d{2}-\d{2})\b", "", ln, flags=re.I).strip()
                title_clean = re.sub(r"\bpriority\s*\d\b|\bp\d\b|\burgent\b|\basap\b", "", title_clean, flags=re.I).strip()
                title_clean = re.sub(r"\bbetween\s+\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\b", "", title_clean, flags=re.I).strip()
                title_clean = re.sub(r"\b\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\b", "", title_clean, flags=re.I).strip()
                title_clean = re.sub(r"\bafter\s+\d{1,2}(:\d{2})?\s*(am|pm)\b", "", title_clean, flags=re.I).strip()
                title_clean = re.sub(r"\bbefore\s+\d{1,2}(:\d{2})?\s*(am|pm)\b", "", title_clean, flags=re.I).strip()
                title_clean = title_clean or "Untitled task"

                rows.append({
                    "add": True,
                    "title": title_clean,
                    "category": parse_category_guess(ln),
                    "priority": parse_priority_guess(ln),
                    "deadline": parse_date_guess(ln).isoformat(),
                    "duration_min": parse_duration_guess(ln),
                    "energy": parse_energy_guess(ln),
                    "notes": "Added via text input",
                    "earliest_start": es.strftime("%H:%M") if es else "",
                    "latest_end": le.strftime("%H:%M") if le else "",
                    "recurrence": rec,
                    "weekly_days": ",".join(str(x) for x in weekly_days),
                    "split_mode": "Auto",
                    "subtasks": "",
                    "estimate_min": "",
                    "estimate_max": "",
                    "depends_on": "",
                })
            st.session_state.nlp_candidates = pd.DataFrame(rows)
            st.success("Parsed. Review and confirm below.")

        if isinstance(st.session_state.nlp_candidates, pd.DataFrame) and not st.session_state.nlp_candidates.empty:
            st.write("✅ Confirm parsed tasks (uncheck **add** to skip):")
            edited = st.data_editor(
                st.session_state.nlp_candidates,
                use_container_width=True,
                column_config={
                    "add": st.column_config.CheckboxColumn("add"),
                    "priority": st.column_config.NumberColumn(min_value=1, max_value=5, step=1),
                    "duration_min": st.column_config.NumberColumn(min_value=5, max_value=600, step=5),
                    "energy": st.column_config.SelectboxColumn(options=ENERGIES),
                    "category": st.column_config.SelectboxColumn(options=CATEGORIES),
                    "recurrence": st.column_config.SelectboxColumn(options=RECURRENCES),
                }
            )
            colN1, colN2 = st.columns(2)
            with colN1:
                if st.button("➕ Add selected tasks"):
                    added = 0
                    for _, r in edited.iterrows():
                        if not bool(r.get("add", True)):
                            continue
                        st.session_state.tasks.append(Task(
                            id=str(uuid.uuid4()),
                            title=str(r["title"]).strip() or "Untitled task",
                            category=str(r["category"]),
                            priority=int(clamp(int(r["priority"]), 1, 5)),
                            deadline=pd.to_datetime(r["deadline"]).date(),
                            duration_min=int(clamp(int(r["duration_min"]), 5, 600)),
                            energy=str(r["energy"]),
                            notes=str(r.get("notes","")),
                            status="Planned",
                            earliest_start=time_or_none(str(r.get("earliest_start",""))),
                            latest_end=time_or_none(str(r.get("latest_end",""))),
                            depends_on=[x.strip() for x in str(r.get("depends_on","")).split(",") if x.strip()],
                            recurrence=str(r.get("recurrence","None")),
                            weekly_days=[int(x) for x in str(r.get("weekly_days","")).split(",") if x.strip().isdigit()],
                            split_mode=str(r.get("split_mode","Auto")),
                            subtasks=str(r.get("subtasks","")),
                            estimate_min=(int(r.get("estimate_min")) if str(r.get("estimate_min","")).strip().isdigit() else None),
                            estimate_max=(int(r.get("estimate_max")) if str(r.get("estimate_max","")).strip().isdigit() else None),
                        ))
                        added += 1
                    st.session_state.nlp_candidates = pd.DataFrame()
                    st.success(f"Added {added} task(s).")
            with colN2:
                if st.button("🗑️ Discard parsed preview"):
                    st.session_state.nlp_candidates = pd.DataFrame()
                    st.info("Discarded preview.")

    st.subheader("5) Task List (editable)")
    df_tasks = tasks_to_df(st.session_state.tasks)

    edited_tasks = st.data_editor(
        df_tasks,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "priority": st.column_config.NumberColumn(min_value=1, max_value=5, step=1),
            "duration_min": st.column_config.NumberColumn(min_value=5, max_value=600, step=5),
            "energy": st.column_config.SelectboxColumn(options=ENERGIES),
            "category": st.column_config.SelectboxColumn(options=CATEGORIES),
            "status": st.column_config.SelectboxColumn(options=["Planned", "Done", "Skipped"]),
            "recurrence": st.column_config.SelectboxColumn(options=RECURRENCES),
            "split_mode": st.column_config.SelectboxColumn(options=["Auto", "Manual"]),
        }
    )

    colS1, colS2, colS3 = st.columns(3)
    with colS1:
        if st.button("💾 Save edits"):
            try:
                st.session_state.tasks = df_to_tasks(edited_tasks)
                st.success("Saved.")
            except Exception as e:
                st.error(f"Could not save edits: {e}")

    with colS2:
        if st.button("🧹 Clear all data"):
            st.session_state.tasks = []
            st.session_state.last_plan = None
            st.session_state.feedback = {"satisfaction": 3, "notes": ""}
            st.session_state.history = []
            st.session_state.nlp_candidates = pd.DataFrame()
            st.success("Cleared.")

    with colS3:
        export_payload = build_export_payload()
        st.download_button(
            "⬇️ Export JSON (backup)",
            data=json.dumps(export_payload, indent=2).encode("utf-8"),
            file_name="planning_assistant_backup.json",
            mime="application/json"
        )

    st.subheader("Import JSON (restore)")
    up = st.file_uploader("Upload a backup JSON", type=["json"])
    if up is not None:
        try:
            payload = json.loads(up.read().decode("utf-8"))
            if st.button("↩️ Restore from uploaded JSON"):
                load_import_payload(payload)
                st.success("Restored.")
        except Exception as e:
            st.error(f"Import failed: {e}")


# -----------------------------
# Right panel: Generate plan + Explain + Metrics
# -----------------------------
with right:
    st.subheader("6) Generate Personalized Plan")

    if end_hour <= start_hour:
        st.error("Work end hour must be later than start hour.")
    else:
        start_day = today_local()
        work_start = time(start_hour, 0)
        work_end = time(end_hour, 0)

        if st.button("✨ Generate plan (with explanations)"):
            plan_df, warnings = schedule_plan(
                tasks=st.session_state.tasks,
                start_day=start_day,
                days=days,
                work_start=work_start,
                work_end=work_end,
                break_every_min=break_every_min,
                break_min=break_min,
                focus_block_min=focus_block_min,
                preferred_energy=preferred_energy,
                max_chunk_min=max_chunk_min,
                max_daily_focus_min=max_daily_focus_min,
                max_high_energy_in_row=max_high_energy_in_row,
                recovery_after_high=recovery_after_high,
            )
            st.session_state.last_plan = plan_df

            # update adaptive defaults to last used
            st.session_state.adaptive.update({
                "break_every_min": break_every_min,
                "break_min": break_min,
                "focus_block_min": focus_block_min,
                "preferred_energy": preferred_energy,
                "max_chunk_min": max_chunk_min,
                "max_daily_focus_min": max_daily_focus_min,
                "max_high_energy_in_row": max_high_energy_in_row,
                "recovery_after_high": recovery_after_high,
            })

            if warnings:
                st.warning("Warnings:\n- " + "\n- ".join(warnings))
            else:
                st.success("Plan generated.")

    st.markdown("**Goal context:**")
    st.write(goal)

    st.divider()

    plan_df = st.session_state.last_plan
    if plan_df is None or plan_df.empty:
        st.info("No plan yet. Click **Generate plan**.")
    else:
        st.subheader("📌 Schedule (Calendar-like view)")
        show_reasons = st.toggle("Show explanations (why each task was placed)", value=True)

        for d, g in plan_df.groupby("Date"):
            st.markdown(f"### {d}")
            cols = ["Start","End","Type","Task","Category","Priority","Deadline","Energy","Duration(min)"]
            if show_reasons:
                cols += ["Reason"]
            st.dataframe(g[cols], use_container_width=True)

        st.download_button(
            "⬇️ Export schedule CSV",
            data=plan_df.to_csv(index=False).encode("utf-8"),
            file_name="schedule.csv",
            mime="text/csv"
        )

        st.divider()

        st.subheader("7) Feedback & Adaptation")
        st.write("Rate how the plan feels. The system adjusts pacing and load suggestions next time.")
        satisfaction = st.slider("Satisfaction (1 low → 5 high)", 1, 5, st.session_state.feedback["satisfaction"])
        fb_notes = st.text_area("Notes (what should change?)", value=st.session_state.feedback["notes"], height=90)

        if st.button("✅ Save feedback & adapt"):
            st.session_state.feedback = {"satisfaction": satisfaction, "notes": fb_notes}

            # “Learning” rules (still lightweight, but more meaningful):
            # - low satisfaction: reduce daily focus limit + more breaks + shorter focus blocks
            # - high satisfaction: allow slightly more focus, fewer breaks, longer focus blocks
            a = st.session_state.adaptive
            if satisfaction <= 2:
                a["break_every_min"] = clamp(a["break_every_min"] - 10, 30, 180)
                a["break_min"] = clamp(a["break_min"] + 2, 5, 30)
                a["focus_block_min"] = clamp(a["focus_block_min"] - 5, 15, 120)
                a["max_daily_focus_min"] = clamp(a["max_daily_focus_min"] - 30, 60, 600)
            elif satisfaction >= 4:
                a["break_every_min"] = clamp(a["break_every_min"] + 10, 30, 180)
                a["break_min"] = clamp(a["break_min"] - 1, 5, 30)
                a["focus_block_min"] = clamp(a["focus_block_min"] + 5, 15, 120)
                a["max_daily_focus_min"] = clamp(a["max_daily_focus_min"] + 30, 60, 600)

            st.success("Saved feedback. " + recommend_break_style(satisfaction))

            # add history point
            # completion rate from tasks
            tdf = tasks_to_df(st.session_state.tasks)
            total = len(tdf)
            done = int((tdf["status"] == "Done").sum()) if total else 0
            completion_rate = done / total if total else 0.0

            st.session_state.history.append({
                "ts": datetime.utcnow().isoformat() + "Z",
                "completion_rate": completion_rate,
                "satisfaction": satisfaction,
            })

        st.divider()

        st.subheader("8) Evaluation Dashboard (Prototype Metrics)")

        tdf = tasks_to_df(st.session_state.tasks)
        total = len(tdf)
        done = int((tdf["status"] == "Done").sum()) if total else 0
        skipped = int((tdf["status"] == "Skipped").sum()) if total else 0
        planned = int((tdf["status"] == "Planned").sum()) if total else 0
        completion_rate = (done / total) if total else 0.0

        scheduled_tasks = plan_df[plan_df["Type"] == "Task"].copy()
        if not scheduled_tasks.empty:
            high_pri_blocks = (scheduled_tasks["Priority"].replace("", 0).astype(int) >= 4).sum()
            rec_accuracy_proxy = high_pri_blocks / max(1, len(scheduled_tasks))
        else:
            rec_accuracy_proxy = 0.0

        colM1, colM2, colM3, colM4 = st.columns(4)
        colM1.metric("Tasks total", total)
        colM2.metric("Completion rate", f"{completion_rate*100:.1f}%")
        colM3.metric("High-priority focus share", f"{rec_accuracy_proxy*100:.1f}%")
        colM4.metric("Satisfaction", str(st.session_state.feedback["satisfaction"]))

        st.caption(
            "Prototype metrics are simplified. For a research evaluation, track outcomes over time and compute adherence, "
            "prediction accuracy, and satisfaction trends with user consent."
        )

        # Trend chart (session-local)
        if st.session_state.history:
            hist_df = pd.DataFrame(st.session_state.history)
            hist_df["ts"] = pd.to_datetime(hist_df["ts"])
            hist_df = hist_df.sort_values("ts")
            trend = hist_df.set_index("ts")[["completion_rate","satisfaction"]]
            st.line_chart(trend)

st.divider()
st.markdown(
    """
### What you upgraded (for your report)
- **Scheduling engine:** deadlines + priority + energy matching + constraints (time windows, max daily load, anti-burnout rules).
- **Well-being:** breaks + recovery blocks + daily focus limit + chunking.
- **NLP interaction:** parse tasks from text + **confirmation** before saving.
- **Explainability:** every scheduled task has a “Reason”.
- **Adaptation:** feedback updates pacing + load settings and logs a trend history.
- **Privacy:** local state + export/import JSON + clear button.
"""
)
