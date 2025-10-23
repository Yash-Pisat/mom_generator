import re, json, asyncio, httpx, streamlit as st

st.set_page_config(page_title="Minutes of Meeting Generator", layout="wide")

# ----- Secrets (set in Streamlit Cloud: Settings → Secrets) -----
# Required:
#   GOOGLE_API_KEY = "your_gemini_api_key"
# Optional:
#   GEMINI_MODEL   = "gemini-1.5-flash"  (default below)
#   DEBUG          = "true" to show raw error details in UI
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
GEMINI_MODEL   = st.secrets.get("GEMINI_MODEL", "gemini-1.5-flash")
DEBUG          = (st.secrets.get("DEBUG") or "false").lower() == "true"

# ----- VTT parsing & chunking (minimal, no extra packages) -----
CUE_RE = re.compile(r"(?P<start>\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})")
SPEAKER_COLON = re.compile(r"^\s*([A-Z][\w .'\-]{0,40}):\s*(.*)$")

def parse_vtt(raw: str):
    lines = raw.splitlines()
    out, i = [], 0
    while i < len(lines):
        m = CUE_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        start, end = m.groups()
        j = i + 1
        texts = []
        while j < len(lines) and lines[j].strip() != "" and not CUE_RE.match(lines[j]):
            texts.append(lines[j].strip())
            j += 1
        text = " ".join(texts)
        spkr, body = None, text
        m1 = SPEAKER_COLON.match(text)
        if m1:
            spkr, body = m1.group(1), m1.group(2)
        out.append({"start": start, "end": end, "speaker": spkr, "text": body})
        i = j
    return out

def _sec(ts): h,m,s = ts.split(":"); s,ms = s.split("."); return int(h)*3600+int(m)*60+int(s)+int(ms)/1000
def merge_short(segs, gap=3):
    merged=[]
    for s in segs:
        if merged and s["speaker"]==merged[-1]["speaker"]:
            if 0 <= _sec(s["start"]) - _sec(merged[-1]["end"]) <= gap:
                merged[-1]["end"]=s["end"]; merged[-1]["text"]+=" "+s["text"]; continue
        merged.append(s)
    return merged

def chunk(segs, window=360, overlap=20):  # slightly smaller windows to avoid token limits
    if not segs: return []
    start_all, end_all = _sec(segs[0]["start"]), _sec(segs[-1]["end"])
    out=[]; cur=start_all
    while cur < end_all:
        wend = cur + window
        items = [s for s in segs if _sec(s["start"]) < wend and _sec(s["end"]) > cur]
        out.append({"start":cur,"end":min(wend,end_all),"items":items})
        cur = wend - overlap
    return out

def fmt_chunk(ch):
    lines=[]
    for s in ch["items"]:
        sp = s["speaker"] or "Unknown"
        lines.append(f'[{s["start"]}] {sp}: {s["text"]}')
    return "\n".join(lines)

# ----- Prompt (detailed discussion + crisp actions) -----
def build_prompt(chunk_text: str) -> str:
    return f"""
You are a professional business meeting summarizer.

Your goal is to generate structured minutes of meeting in detailed yet clear form.

Return JSON in this structure:
{{
  "topics": [
    {{
      "title": "string",
      "discussion": [
        "Detailed bullet capturing what was said, the reasoning, decisions, and relevant context."
      ],
      "actions": [
        {{"task": "short imperative action <= 20 words", "owner": "person/team or Unassigned", "due": "date if given or null"}}
      ]
    }}
  ]
}}

Guidelines:
- Discussion bullets should be rich and explanatory (who said what, why, context, decisions).
- Actions must be crisp, imperative, and omit rationale (keep rationale in discussion).
- Only include owners/dates when explicitly stated; otherwise owner="Unassigned", due=null.
- Do not invent facts.

Transcript:
{chunk_text}
"""

# ----- Gemini call with robust error reporting -----
def _gemini_error_text(resp_json: dict) -> str:
    if resp_json is None:
        return "Unknown error (no response body)."
    parts = []
    if "promptFeedback" in resp_json:
        pf = resp_json["promptFeedback"]
        if pf.get("blockReason"):
            parts.append(f"Blocked: {pf.get('blockReason')}")
        if pf.get("safetyRatings"):
            cats = [r.get("category","") for r in pf["safetyRatings"] if r.get("probability")]
            if cats: parts.append("Safety: " + ", ".join(cats))
    if "error" in resp_json:
        e = resp_json["error"]
        parts.append(f"{e.get('status','ERROR')}: {e.get('message','')}")
    if not parts and not resp_json.get("candidates"):
        parts.append("No candidates returned (quota/tokens/size?).")
    return " | ".join(parts) or "Unspecified Gemini error."

async def call_gemini(prompt: str):
    if not GOOGLE_API_KEY:
        return None, "GOOGLE_API_KEY is missing in Streamlit Secrets."
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    payload = {
        "contents":[{"parts":[{"text": prompt + "\n\nReturn valid JSON only."}]}],
        "generationConfig":{"temperature":0.2,"responseMimeType":"application/json"}
    }
    headers = {"x-goog-api-key": GOOGLE_API_KEY}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            try:
                body = r.json()
            except Exception:
                body = None
            return None, f"HTTP {r.status_code}: {_gemini_error_text(body)}"
        try:
            data = r.json()
            cands = data.get("candidates") or []
            if not cands:
                return None, _gemini_error_text(data)
            txt = cands[0]["content"]["parts"][0]["text"]
            return json.loads(txt), None
        except Exception as e:
            if DEBUG:
                st.exception(e)
            return None, f"Parse error: {repr(e)} (model returned non-JSON?)"

# ----- Extraction orchestration (collects errors) -----
async def extract_topics(chunks):
    topics, errors = [], []
    async def run(c):
        prompt = build_prompt(fmt_chunk(c))
        result, err = await call_gemini(prompt)
        if err: errors.append(err)
        return result or {"topics": []}
    results = await asyncio.gather(*[run(c) for c in chunks])
    for r in results:
        topics.extend(r.get("topics", []))
    return topics, errors

# ----- UI -----
st.title("📋 Minutes of Meeting")

meeting_title = st.text_input("Meeting title", "Project Discussion")
vtt = st.file_uploader("Upload .vtt transcript", type=["vtt"])

if st.button("Generate Minutes", type="primary") and vtt:
    raw = vtt.getvalue().decode("utf-8", "ignore")
    segs = merge_short(parse_vtt(raw))
    if not segs:
        st.error("No cues found in file.")
        st.stop()

    duration_min = int(round((_sec(segs[-1]['end']) - _sec(segs[0]['start']))/60))
    chs = chunk(segs, window=360, overlap=20)

    with st.spinner("Summarizing..."):
        topics, errors = asyncio.run(extract_topics(chs))

    # Structured results
    st.subheader("Structured Summary")
    if not topics:
        st.warning("No topics were extracted. See error details below.")
    for i, t in enumerate(topics, 1):
        with st.expander(f"{i}. {t.get('title','(untitled)')}", expanded=(i==1)):
            for d in t.get("discussion", []):
                st.markdown(f"- {d}")
            if t.get("actions"):
                st.markdown("**Actions**")
                for a in t["actions"]:
                    task = a.get("task","")
                    own  = a.get("owner","Unassigned")
                    due  = a.get("due") or "-"
                    st.write(f"- **{task}** — Owner: *{own}* (Due: {due})")

    # Email draft (always produces output, includes reason if empty)
    st.divider()
    st.subheader("✉️ Email Draft")
    lines = [
        f"Subject: Minutes of Meeting – {meeting_title}",
        "",
        "Please find the meeting minutes below:",
        "",
        "**Key Discussion Points, Actions & Decisions -**",
        ""
    ]
    if topics:
        for t in topics:
            lines.append(f"**{t.get('title','(untitled)')}**")
            for d in t.get("discussion", []):
                lines.append(f"*   {d}")
            for a in t.get("actions", []):
                task=a.get("task",""); own=a.get("owner","Unassigned"); due=a.get("due") or "-"
                lines.append(f"*   **[Action] {task} — Owner: {own}, Due: {due}.**")
            lines.append("")
    else:
        reason = errors[0] if errors else "No extractable content detected or input too large."
        lines.append(f"_Note: No topics could be extracted. Reason: {reason}_")
        lines.append("")

    draft = "\n".join(lines + ["Regards,", "Automated MoM Assistant"])
    st.text_area("Email Draft", value=draft, height=380)

    if errors:
        st.error("Gemini reported the following issue(s):")
        for e in errors:
            st.write(f"- {e}")
        if DEBUG:
            st.caption("DEBUG is true — raw error details were printed above (if any).")

    st.download_button(
        "Download minutes.json",
        data=json.dumps({
            "meeting_title": meeting_title,
            "duration_min": duration_min,
            "topics": topics,
            "email_draft": draft,
            "errors": errors
        }, indent=2),
        file_name="minutes.json",
        mime="application/json"
    )
