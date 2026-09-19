"""Safety layer — the highest-privilege component in MITRA.

Design principle: *safety is not a prompt, it is code*. An LLM prompt can be
talked out of its instructions; a deterministic pre- and post-filter cannot.
Every user turn passes through `assess()` before any generative agent runs, and
every candidate reply passes through `screen_reply()` before it reaches the user.

The risk model is intentionally transparent (weighted lexical + contextual
features) rather than a black box, because a viva examiner — and, more
importantly, a user — should be able to ask "why did it escalate?" and get an
answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .schemas import RiskLevel

# --------------------------------------------------------------------------
# Crisis lexicon
# --------------------------------------------------------------------------
# NOTE: this project deliberately stores *ideation* phrases only. It contains no
# method, means, dosage or lethality information anywhere in the codebase, and
# the agent is forbidden from generating any. Detection never requires knowing
# how someone might act, only that they are in danger.

CRISIS_PHRASES: Dict[str, float] = {
    r"\bkill(ing)? (my ?self|me)\b": 1.0,
    r"\bend (my|it all|my own) (life|existence)\b": 1.0,
    r"\bsuicid(e|al)\b": 0.9,
    r"\bdon'?t want to (live|be alive|wake up)\b": 0.9,
    r"\bwant(s|ing|ed)? to die\b": 0.95,
    r"\bbetter off (with ?out me|dead)\b": 0.9,
    r"\bno reason to (live|go on|continue)\b": 0.85,
    r"\bself[- ]?harms?\b": 0.85,
    r"\b(hurt|harm|injur)(ing|s|ed)? +my ?self\b": 0.85,
    r"\bcut(ting)? my ?self\b": 0.9,
    r"\bnot waking up\b": 0.8,
    r"\bdisappear (forever|for ?ever)\b": 0.75,
    r"\bi (can'?t|cannot) (go on|do this any ?more|take it any ?more)\b": 0.7,
    r"\bgoodbye (forever|everyone)\b": 0.8,
    r"\bwriting (a|my) (note|letter) to my family\b": 0.7,
    r"\bnobody would (ever |even )?(miss|notice) me\b": 0.7,
    r"\bif i (was|were) gone\b": 0.7,
    r"\b(hurt|harm|kill) (him|her|them|someone|people)\b": 0.85,
}

DISTRESS_PHRASES: Dict[str, float] = {
    r"\bhopeless\b": 0.32,
    r"\bworthless\b": 0.32,
    r"\bempty inside\b": 0.30,
    r"\bnumb\b": 0.26,
    r"\bnothing (feels|seems) (worth|good|real)\b": 0.28,
    r"\bi hate my ?self\b": 0.34,
    r"\bburn(t|ed) out\b": 0.22,
    r"\bcan'?t sleep\b": 0.20,
    r"\b(haven'?t|not) been (eating|sleeping)\b": 0.24,
    r"\bnot (eating|sleeping)\b": 0.22,
    r"\bpanic attacks?\b": 0.32,
    r"\bbreaking down\b": 0.30,
    r"\bfalling apart\b": 0.28,
    r"\btrapped\b": 0.24,
    r"\bno way out\b": 0.26,
    r"\bgiving up\b": 0.30,
    r"\balone\b": 0.16,
    r"\bno ?one (cares|understands|gets it)\b": 0.30,
    r"\bfeel invisible\b": 0.22,
    r"\bfailed everything\b": 0.24,
    r"\bcry(ing)? (every ?day|all the time)\b": 0.30,
    r"\bexhausted from pretending\b": 0.28,
    r"\bcan'?t tell anyone\b": 0.22,
    r"\bpretending (to be|i'?m) (ok|okay|fine)\b": 0.26,
    r"\bstopped (replying|talking|going out|answering|attending)\b": 0.28,
    r"\btoo much effort\b": 0.20,
    r"\bcan'?t remember the last time\b": 0.26,
    r"\b(feels?|everything is) (grey|gray|pointless|meaningless)\b": 0.26,
    r"\bdrowning\b": 0.28,
    r"\b(never|not) good enough\b": 0.24,
    r"\bexhaust(ed|ing)\b": 0.16,
}

# Stage B: compositional concept rules.
#
# Stage A (the lexicon above) is high precision but low recall: it only fires on
# phrasings it has seen. Real disclosures are paraphrased, hedged and
# colloquial. These rules instead look for a *concept pair* — a self-reference
# plus a cessation, harm or burdensomeness concept — within a bounded window,
# which generalises to phrasings never written into the lexicon.
CONCEPT_RULES: List[Tuple[str, float, str]] = [
    (r"\b(wish|wishing|hope|hoping|want|wanting)\b[^.?!]{0,30}\b(i )?"
     r"(would ?n'?t|do ?n'?t|never)\b[^.?!]{0,14}\bwake up\b", 0.90,
     "wish not to wake"),
    (r"\bno (point|reason)\b[^.?!]{0,32}\b(me (being|existing)|being here|"
     r"living|my life|going on|carrying on|any ?more)\b", 0.85,
     "life-not-worth-continuing"),
    (r"\b(thoughts? of|thinking about|keep (having|thinking))\b[^.?!]{0,28}"
     r"\b(end(ing)? (my life|it all|things)|dying|die)\b", 0.95,
     "ideation reported as thoughts"),
    (r"\b(if i (was ?n'?t|were ?n'?t|was not) (around|here|alive)|with ?out me)\b"
     r"[^.?!]{0,32}\b(better|cope|fine|happier|easier)\b", 0.85,
     "perceived burdensomeness"),
    (r"\b(better|cope better|happier|easier|fine)\b[^.?!]{0,32}"
     r"\bif i (was ?n'?t|were ?n'?t|was not) (around|here|alive)\b", 0.85,
     "perceived burdensomeness"),
    (r"\b(nobody|no ?one) would (ever |even )?(miss|notice|care about)\b"
     r"[^.?!]{0,20}\b(me|my absence)\b", 0.85, "perceived burdensomeness"),
    (r"\bif i (was|were) gone\b", 0.82, "cessation framing"),
    (r"\bstop(ped)? (existing|being here)\b", 0.85, "cessation wish"),
    (r"\b(give|giving|gave) up on (life|living|everything)\b", 0.60,
     "global giving up"),
    (r"\b(can'?t|cannot) (see|imagine) (a|any) (future|way forward)\b", 0.62,
     "foreshortened future"),
]

PROTECTIVE_PHRASES: Dict[str, float] = {
    r"\bi (am|'m) safe\b": -0.35,
    r"\bnot going to (hurt|harm) my ?self\b": -0.45,
    r"\btalking to (a|my) (counsell?or|therapist|doctor|psychiatrist)\b": -0.25,
    r"\bfeeling better\b": -0.20,
    r"\bi'?m (ok|okay|fine|alright) now\b": -0.25,
    r"\bmy (friend|family|parents|partner) (support|help)s? me\b": -0.15,
}

# Turns that quote, study or discuss the topic rather than disclose it.
ACADEMIC_CONTEXT = re.compile(
    r"\b(essay|assignment|research|paper|presentation|survey|statistics|"
    r"awareness campaign|for my (project|class)|movie|novel|character)\b",
    re.I,
)
THIRD_PARTY = re.compile(
    r"\b(my (friend|roommate|brother|sister|cousin|classmate)|someone i know)\b", re.I
)
PAST_TENSE = re.compile(
    r"\b(used to|last year|when i was|years ago|in school|back then|have recovered|"
    r"went through|came out of|back in (school|college|\d)|"
    r"in \d+(st|nd|rd|th) (standard|grade)|that was (before|years))\b", re.I
)
NEGATION_TOKENS = re.compile(
    r"\b(never|not|no|don'?t|doesn'?t|wouldn'?t|won'?t|would not)\b\s*$", re.I
)


def _locally_negated(text: str, patterns: List[str]) -> bool:
    """True only when a negation sits immediately before the matched phrase.

    A sentence-level negation test is wrong here: "I don't want to be alive"
    contains a negation *inside* the risk phrase, and treating it as a denial
    inverts the meaning of the most serious utterance the system can receive.
    """
    for pattern in patterns:
        m = re.search(pattern, text, re.I)
        if not m:
            continue
        window = text[max(0, m.start() - 24): m.start()]
        if not NEGATION_TOKENS.search(window):
            return False          # at least one un-negated risk phrase stands
    return bool(patterns)

# Requests the agent must refuse outright, regardless of risk band.
FORBIDDEN_REQUESTS = [
    (re.compile(r"\b(what|which) (medicine|medication|tablet|drug|dose|dosage)\b", re.I),
     "medication_advice"),
    (re.compile(r"\b(should i (stop|skip|double)|can i stop) (taking )?(my )?(meds|medicine|medication)\b", re.I),
     "medication_advice"),
    (re.compile(r"\b(diagnos(e|is)|do i have) (depression|bipolar|adhd|ocd|schizophrenia|anxiety disorder)\b", re.I),
     "diagnosis_request"),
    (re.compile(r"\b(how (to|do i)|ways to|best way to) (harm|hurt) my ?self\b", re.I),
     "method_request"),
    (re.compile(r"\bhow (many|much) .{0,24}(pills|tablets) .{0,18}(to|would)\b", re.I),
     "method_request"),
    (re.compile(r"\b(how to|tips to) (lose|purge|starve|not eat)\b", re.I),
     "disordered_eating"),
]

# Content the agent must never emit, checked on the way out.
UNSAFE_OUTPUT = [
    (re.compile(r"\byou (have|are suffering from|are clinically)\b.{0,24}"
                r"(depression|bipolar|anxiety disorder|ptsd|ocd|adhd)", re.I),
     "implied diagnosis"),
    (re.compile(r"\b(take|try|start|stop) (\d+\s?mg|sertraline|fluoxetine|alprazolam|"
                r"xanax|prozac|escitalopram|clonazepam)", re.I),
     "medication guidance"),
    (re.compile(r"\byou don'?t need (a )?(therapist|doctor|professional help)\b", re.I),
     "discourages professional help"),
    (re.compile(r"\bi am (a|your) (human|doctor|therapist|psychiatrist)\b", re.I),
     "false identity claim"),
    (re.compile(r"\b(i promise|i guarantee) (you|that) (nothing|everything)\b", re.I),
     "unfounded reassurance"),
]


@dataclass
class RiskAssessment:
    level: RiskLevel
    score: float
    signals: List[str]
    blocked_request: str | None = None


# --------------------------------------------------------------------------
# Helplines
# --------------------------------------------------------------------------

HELPLINES: Dict[str, List[Dict[str, str]]] = {
    "IN": [
        {"name": "Tele-MANAS (Government of India, 24x7)", "contact": "14416 or 1800-891-4416",
         "note": "Free, confidential, available in 20 Indian languages."},
        {"name": "KIRAN Mental Health Helpline (MSJE, 24x7)", "contact": "1800-599-0019",
         "note": "Toll-free counselling and referral in 13 languages."},
        {"name": "AASRA (24x7)", "contact": "+91 98204 66726",
         "note": "Volunteer-run emotional support line."},
        {"name": "Vandrevala Foundation (24x7)", "contact": "+91 99996 66555",
         "note": "Call or WhatsApp for free counselling."},
        {"name": "iCALL, TISS", "contact": "+91 91529 87821",
         "note": "Psychosocial helpline, Mon-Sat, 10 a.m. to 8 p.m."},
        {"name": "Emergency services", "contact": "112",
         "note": "If there is immediate danger to life."},
    ],
    "GLOBAL": [
        {"name": "findahelpline.com", "contact": "https://findahelpline.com",
         "note": "Verified crisis lines for almost every country."},
        {"name": "International Association for Suicide Prevention",
         "contact": "https://www.iasp.info/crisis-centres-helplines/",
         "note": "Directory of crisis centres worldwide."},
    ],
}


def helplines(region: str = "IN") -> List[Dict[str, str]]:
    return HELPLINES.get(region.upper(), []) + HELPLINES["GLOBAL"]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _match(patterns: Dict[str, float], text: str) -> List[Tuple[str, float]]:
    hits = []
    for pattern, weight in patterns.items():
        if re.search(pattern, text, re.I):
            hits.append((pattern, weight))
    return hits


def assess(text: str, history: List[str] | None = None,
           crisis_threshold: float = 0.55,
           elevated_threshold: float = 0.30) -> RiskAssessment:
    """Score a single user turn for risk.

    Returns a transparent assessment: the score, the band and the human-readable
    signals that produced it. Context modifiers can only *reduce* a crisis score
    when the disclosure is clearly academic, third-party or historical — and even
    then the score is floored at the elevated band, never at zero.
    """
    text = (text or "").strip()
    if not text:
        return RiskAssessment(RiskLevel.SUPPORTIVE, 0.0, [])

    signals: List[str] = []
    score = 0.0

    # 1. Explicit refusal categories -------------------------------------
    blocked = None
    for pattern, label in FORBIDDEN_REQUESTS:
        if pattern.search(text):
            blocked = label
            signals.append(f"blocked request category: {label}")
            if label in {"method_request", "disordered_eating"}:
                score = max(score, 0.85)
            break

    # 2. Crisis lexicon ---------------------------------------------------
    crisis_hits = _match(CRISIS_PHRASES, text)
    if crisis_hits:
        score = max(score, max(w for _, w in crisis_hits))
        signals.append(f"{len(crisis_hits)} crisis-lexicon match(es)")

    # 2b. Compositional concept rules (Stage B) ---------------------------
    concept_hits = []
    for pattern, weight, label in CONCEPT_RULES:
        if re.search(pattern, text, re.I):
            concept_hits.append((pattern, weight))
            signals.append(f"concept rule: {label}")
    if concept_hits:
        score = max(score, max(w for _, w in concept_hits))
        crisis_hits = crisis_hits + concept_hits

    # 3. Distress lexicon (additive, capped) ------------------------------
    distress_hits = _match(DISTRESS_PHRASES, text)
    if distress_hits:
        added = min(0.45, sum(w for _, w in distress_hits))
        score += added
        signals.append(f"{len(distress_hits)} distress marker(s) (+{added:.2f})")

    # 4. Protective factors ----------------------------------------------
    for pattern, weight in _match(PROTECTIVE_PHRASES, text):
        score += weight
        signals.append(f"protective factor ({weight:+.2f})")

    # 5. Contextual de-escalation ----------------------------------------
    if crisis_hits:
        if ACADEMIC_CONTEXT.search(text):
            score = min(score, 0.45)
            signals.append("academic/fictional framing detected — de-escalated")
        elif THIRD_PARTY.search(text):
            score = min(score, 0.50)
            signals.append("third-party disclosure — de-escalated but still elevated")
        elif PAST_TENSE.search(text):
            score = min(score, 0.48)
            signals.append("historical framing — de-escalated")
        elif _locally_negated(text, [p for p, _ in crisis_hits]):
            score = min(score, 0.40)
            signals.append("locally negated statement — de-escalated")

    # 6. Conversation-level persistence -----------------------------------
    # Distress that repeats across turns matters more than a single bad line.
    if history:
        recent = " ".join(history[-4:])
        prior = _match(CRISIS_PHRASES, recent)
        if prior and score >= elevated_threshold:
            score = min(1.0, score + 0.15)
            signals.append("risk language recurring across turns (+0.15)")

    # Crisis requires *explicit* risk language or a high-confidence concept
    # rule. Accumulated distress markers alone can reach the elevated band but
    # must not tip into crisis on their own — that is how false escalations are
    # manufactured, and a system that cries wolf gets ignored when it matters.
    explicit = any(w >= 0.80 for _, w in crisis_hits)
    if not explicit and score >= crisis_threshold:
        score = crisis_threshold - 0.01
        signals.append("no explicit risk phrase — capped below the crisis band")

    # Safety floor: once explicit risk language has appeared, protective factors
    # and context may soften the score but must never return it to "no concern".
    if crisis_hits and score < elevated_threshold:
        score = elevated_threshold
        signals.append("risk language present — floored at the elevated band")

    score = max(0.0, min(1.0, round(score, 3)))

    if score >= crisis_threshold:
        level = RiskLevel.CRISIS
    elif score >= elevated_threshold:
        level = RiskLevel.ELEVATED
    else:
        level = RiskLevel.SUPPORTIVE

    return RiskAssessment(level, score, signals, blocked)


# --------------------------------------------------------------------------
# Outbound filter
# --------------------------------------------------------------------------

def screen_reply(reply: str) -> Tuple[bool, List[str]]:
    """Post-generation guardrail. Returns (is_safe, violations)."""
    violations = [label for pattern, label in UNSAFE_OUTPUT if pattern.search(reply or "")]
    return (not violations), violations


CRISIS_TEMPLATE = (
    "I'm really glad you told me, and I want to stop and stay with this for a "
    "moment. What you're carrying sounds unbearably heavy, and you should not "
    "have to hold it on your own.\n\n"
    "I'm an AI project and I'm not able to keep you safe by myself — a person "
    "can. Please reach out right now to one of these, they are free, "
    "confidential and open 24x7:\n\n{lines}\n\n"
    "If you are in immediate danger, please call 112 or ask someone nearby to "
    "sit with you while you call.\n\n"
    "Would you be willing to tell one person you trust — a friend, a family "
    "member, your warden or a counsellor on campus — how you're feeling today? "
    "I'll stay here with you either way."
)


def crisis_reply(region: str = "IN") -> Tuple[str, List[Dict[str, str]]]:
    res = helplines(region)
    lines = "\n".join(f"- {r['name']}: {r['contact']}" for r in res[:4])
    return CRISIS_TEMPLATE.format(lines=lines), res


REFUSAL_TEMPLATES = {
    "medication_advice": (
        "I can't advise on medicines or doses — that has to come from the doctor "
        "who knows your history, because getting it wrong can genuinely hurt you. "
        "What I can do is help you write down what you want to ask them, or sit "
        "with whatever is making the medication feel like a question right now."
    ),
    "diagnosis_request": (
        "I'm not able to diagnose anything, and I'd be doing you a disservice if I "
        "guessed. A registered psychologist or psychiatrist can assess this properly, "
        "and campus counselling is usually free. If it helps, I can listen to what "
        "you've been noticing in yourself, and we can put it into words you could take "
        "to them."
    ),
    "method_request": (
        "I'm not going to help with that, and I'm not going anywhere either. "
        "It sounds like you're in a lot of pain right now. Please talk to someone "
        "who can actually be with you in it — Tele-MANAS is free and open 24x7 on "
        "14416."
    ),
    "disordered_eating": (
        "I can't help with restricting food or weight-loss targets — that's an area "
        "where well-meant advice does real harm. If eating has started to feel like "
        "a battleground, that's worth talking through with a doctor or counsellor, "
        "and I'm happy to stay with how it's been feeling for you."
    ),
}
