"""Support-service lookup and general wellbeing guidance."""

from __future__ import annotations

from ..safety import helplines
from ..schemas import ToolResult

CAMPUS = [
    {"name": "Campus counselling cell", "contact": "Student Welfare / Dean's office",
     "note": "Most Indian universities provide free, confidential counselling."},
    {"name": "Faculty mentor / class teacher", "contact": "As allotted",
     "note": "Useful when academic load is part of the problem."},
    {"name": "Hostel warden or resident tutor", "contact": "Hostel office",
     "note": "First point of contact after hours."},
]

TIPS = {
    "sleep": [
        "Keep the wake-up time fixed even after a bad night; the body anchors to "
        "waking, not sleeping.",
        "Screens out of bed — if you must, at least stop scrolling in the dark.",
        "If you're awake past twenty minutes, get up, sit somewhere dim, go back "
        "when sleepy.",
        "Caffeine has a half-life of about five hours; an evening chai is a 2 a.m. "
        "problem.",
    ],
    "focus": [
        "One task, twenty-five minutes, phone in another room. Repeat with a real break.",
        "Write the next physical action, not the goal — 'open the file' beats "
        "'finish the project'.",
        "Lower the bar to start. Starting badly still beats not starting.",
    ],
    "routine": [
        "Anchor the day with three fixed points: wake, one meal, one walk outside.",
        "Daylight within an hour of waking does more for mood than most apps.",
        "Movement counts even at ten minutes; the dose-response curve is steepest "
        "at the low end.",
    ],
}


def find_support(region: str = "IN", kind: str = "all") -> ToolResult:
    crisis = helplines(region)
    data = {"crisis": crisis, "campus": CAMPUS}
    if kind == "crisis":
        payload, lines = crisis, crisis[:4]
    elif kind == "campus":
        payload, lines = CAMPUS, CAMPUS
    else:
        payload, lines = data, crisis[:3] + CAMPUS[:1]
    display = "Support you can reach today: " + "; ".join(
        f"{r['name']} ({r['contact']})" for r in lines
    )
    return ToolResult(name="find_support", ok=True, output=payload, display=display)


def wellbeing_tips(issue: str = "sleep") -> ToolResult:
    issue = issue if issue in TIPS else "routine"
    tips = TIPS[issue]
    return ToolResult(name="sleep_hygiene", ok=True,
                      output={"issue": issue, "tips": tips},
                      display=f"A few things that help with {issue}: " + " ".join(tips))
