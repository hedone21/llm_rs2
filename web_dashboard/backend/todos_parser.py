"""Parser for .agent/todos/*.md TODO files."""
import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TODOS_DIR = _PROJECT_ROOT / ".agent" / "todos"

_ROLE_MAP = {
    "architect.md": "Architect",
    "rust_developer.md": "Rust Developer",
    "frontend_developer.md": "Frontend Developer",
    "tester.md": "Tester",
    "tech_writer.md": "Technical Writer",
    "backlog.md": "Backlog",
}

_TASK_RE = re.compile(r"^##\s*\[(?P<priority>P\d)\]\s*(?P<title>.+)$")
_FIELD_RE = re.compile(r"^-\s*\*\*(?P<key>[^*]+)\*\*:\s*(?P<value>.*)$")
_DESC_RE = re.compile(r"^>\s*\*\*역할\*\*:\s*(?P<desc>.+)$")


def _parse_file(path: Path) -> dict:
    """Parse a single TODO markdown file and return role info with tasks."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    description = ""
    tasks = []
    current = None

    for line in lines:
        # Role description from blockquote header
        m = _DESC_RE.match(line)
        if m:
            description = m.group("desc").strip()
            continue

        # Task header: ## [P#] Title
        m = _TASK_RE.match(line)
        if m:
            if current:
                tasks.append(current)
            current = {
                "priority": m.group("priority"),
                "title": m.group("title").strip(),
                "status": "TODO",
                "sprint": "",
                "dependencies": "",
                "description": "",
                "acceptance_criteria": "",
                "notes": "",
            }
            continue

        # Field: - **Key**: Value
        if current:
            m = _FIELD_RE.match(line)
            if m:
                key = m.group("key").strip().lower().replace(" ", "_")
                value = m.group("value").strip()
                if key in current:
                    current[key] = value

    if current:
        tasks.append(current)

    return {"description": description, "tasks": tasks}


def load_todos():
    """Load all TODO files and return structured data.

    Returns dict with summary, roles, and by_sprint breakdown,
    or None if the todos directory doesn't exist.
    """
    if not _TODOS_DIR.exists():
        return None

    roles = {}
    all_tasks = []

    for filename, role_name in _ROLE_MAP.items():
        fpath = _TODOS_DIR / filename
        if not fpath.exists():
            continue
        parsed = _parse_file(fpath)
        roles[role_name] = {
            "file": filename,
            "description": parsed["description"],
            "tasks": parsed["tasks"],
        }
        all_tasks.extend(parsed["tasks"])

    if not all_tasks:
        return None

    # Summary counts
    total = len(all_tasks)
    done = sum(1 for t in all_tasks if t["status"].upper() == "DONE")
    in_progress = sum(1 for t in all_tasks if t["status"].upper() == "IN_PROGRESS")
    blocked = sum(1 for t in all_tasks if t["status"].upper() == "BLOCKED")
    todo = total - done - in_progress - blocked
    completion_pct = round(done / total * 100, 1) if total else 0.0

    # Sprint breakdown
    by_sprint = {}
    for t in all_tasks:
        s = t.get("sprint", "").strip() or "unassigned"
        by_sprint[s] = by_sprint.get(s, 0) + 1

    return {
        "summary": {
            "total": total,
            "done": done,
            "in_progress": in_progress,
            "todo": todo,
            "blocked": blocked,
            "completion_pct": completion_pct,
        },
        "roles": roles,
        "by_sprint": by_sprint,
    }
