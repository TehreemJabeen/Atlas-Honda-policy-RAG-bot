import base64
from pathlib import Path
import streamlit as st
from PIL import Image
from chatbot import ask_bot, PolicyResponse

# ---- Base64 Image Loader Helper ----
def get_base64_image(image_path: str) -> str:
    """Encodes a local image to base64 for reliable HTML embedding in Streamlit."""
    path = Path(__file__).parent / image_path
    if path.exists():
        with open(path, "rb") as f:
            data = f.read()
        return f"data:image/png;base64,{base64.b64encode(data).decode()}"
    return ""

# Load logo as Base64 string
LOGO_BASE64 = get_base64_image("logo.png")

# ---- Exact Brand Spec ----
RED = "#E60012"
RED_DARK = "#B8000E"
SIDEBAR_BG = "#111111"
MAIN_BG = "#F8F9FA"
CARD_BG = "#FFFFFF"

st.set_page_config(
    page_title="Atlas Honda | Policy Assistant",
    page_icon="logo.png" if Path("logo.png").exists() else "🏍️",
    layout="centered"
)

# ---- Advanced CSS UI Styling ----
st.markdown(f"""
<style>
/* ---- Font & Global Imports ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

/* ---- Main Container Background ---- */
[data-testid="stAppViewContainer"] {{
    background-color: {MAIN_BG};
}}
[data-testid="stHeader"] {{
    background-color: {MAIN_BG};
}}

/* ---- Dark Sidebar Styling ---- */
[data-testid="stSidebar"] {{
    background-color: {SIDEBAR_BG};
    border-right: 1px solid #222222;
}}
[data-testid="stSidebar"] * {{
    color: #E0E0E0 !important;
}}
[data-testid="stSidebar"] hr {{
    border-color: #2A2A2A;
}}

/* ---- Brand Header Banner Card ---- */
.brand-header {{
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 22px 28px;
    background-color: {CARD_BG};
    border-radius: 16px;
    border-left: 6px solid {RED};
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    margin-bottom: 24px;
}}
.brand-logo {{
    height: 72px;
    width: auto;
    object-fit: contain;
    mix-blend-mode: multiply;
    flex-shrink: 0;
}}
.brand-title {{
    color: #111111;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: 0.5px;
    margin: 0;
    line-height: 1.1;
    text-transform: uppercase;
}}
.brand-subtitle {{
    color: #666666;
    font-size: 13.5px;
    margin: 0;
    margin-top: 4px;
    font-weight: 400;
}}

/* ---- Chat Bubbles & Layout ---- */
.chat-row {{
    display: flex;
    margin-bottom: 18px;
}}
.chat-row.user {{
    justify-content: flex-end;
}}
.chat-row.bot {{
    justify-content: flex-start;
}}

/* User Bubble */
.bubble-user {{
    background: linear-gradient(135deg, {RED} 0%, {RED_DARK} 100%);
    color: #FFFFFF;
    padding: 12px 20px;
    border-radius: 20px 20px 4px 20px;
    max-width: 75%;
    font-size: 14.5px;
    font-weight: 500;
    line-height: 1.4;
    box-shadow: 0 4px 12px rgba(230, 0, 18, 0.18);
}}

/* Bot Bubble Container */
.bubble-bot-wrapper {{
    max-width: 88%;
}}
.bot-label {{
    display: flex;
    align-items: center;
    gap: 8px;
    color: {RED};
    font-weight: 700;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}}
.bot-avatar {{
    height: 22px;
    width: auto;
    object-fit: contain;
    mix-blend-mode: multiply;
    flex-shrink: 0;
}}
.bubble-bot {{
    background-color: {CARD_BG};
    color: #1A1A1A;
    padding: 18px 22px;
    border-radius: 4px 20px 20px 20px;
    border: 1px solid #EAEEF2;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.03);
    font-size: 14.5px;
    line-height: 1.5;
}}

/* Scenario Cards (inside bot responses) */
.scenario-card {{
    border: 1px solid #EAEAEA;
    border-left: 4px solid {RED};
    border-radius: 10px;
    padding: 14px 18px;
    margin-top: 12px;
    margin-bottom: 6px;
    background-color: #FAFAFA;
}}
.citation-pill {{
    display: inline-block;
    background-color: #FFF0F1;
    color: {RED_DARK};
    border: 1px solid #FFCCD0;
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 6px;
    margin-top: 8px;
}}

/* ---- Modernized Chat Input ---- */
[data-testid="stChatInput"] {{
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px !important;
    background-color: #FFFFFF !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04) !important;
    transition: all 0.2s ease;
}}
[data-testid="stChatInput"]:focus-within {{
    border-color: {RED} !important;
    box-shadow: 0 4px 20px rgba(230, 0, 18, 0.12) !important;
}}
[data-testid="stChatInput"] button {{
    background-color: {RED} !important;
    border-radius: 8px !important;
}}

/* ---- Footer ---- */
.brand-footer {{
    text-align: center;
    color: #A0A0A0;
    font-size: 12px;
    font-weight: 500;
    margin-top: 36px;
    padding-top: 16px;
    border-top: 1px solid #EAEAEA;
}}
</style>
""", unsafe_allow_html=True)

# ---- Branded Header Card ----
st.markdown(f"""
<div class="brand-header">
    <img src="{LOGO_BASE64}" alt="Atlas Honda Logo" class="brand-logo">
    <div>
        <p class="brand-title">Atlas Honda Limited</p>
        <p class="brand-subtitle">Policy Assistant — Instant guidance on leaves, benefits & conduct</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 24px; padding-bottom: 12px; border-bottom: 1px solid #222222;">
        <span style="color: #E60012; font-weight: 800; letter-spacing: 1px; font-size: 13px; text-transform: uppercase;">
            Atlas Honda Ltd.
        </span>
        <div style="color: #777777; font-size: 11px; margin-top: 2px;">Internal HR Portal</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("Settings")
    show_debug = st.toggle("Show retrieved sources", value=False)
    st.divider()
    st.caption("This assistant only answers from official ingested policy documents. "
               "If a topic is not explicitly covered, it will inform you rather than guess.")

# ---- Session State ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Empty State Starter Prompts ----
if len(st.session_state.messages) == 0:
    st.markdown("<p style='color:#777777; font-size:13px; font-weight:600; text-align:center; margin-top:20px;'>SUGGESTED QUESTIONS</p>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    if c1.button("📋 Annual Leave", use_container_width=True):
        st.session_state.messages.append({"role": "user", "text": "What is the annual leave quota and policy?"})
        st.rerun()
    if c2.button("🏥 Medical Coverage", use_container_width=True):
        st.session_state.messages.append({"role": "user", "text": "How do medical insurance and health benefits work?"})
        st.rerun()
    if c3.button("🚗 Travel Allowance", use_container_width=True):
        st.session_state.messages.append({"role": "user", "text": "What are the rules for travel expense reimbursement?"})
        st.rerun()

# ---- Render Helpers ----
def render_scenario_html(scenario) -> str:
    html = f'<div class="scenario-card"><b style="color:#111;">{scenario.scenario_title}</b><p style="margin-top:4px;">{scenario.overview}</p>'

    if scenario.eligibility_and_limits:
        html += "<b>Eligibility & Limits</b><ul>"
        html += "".join(f"<li>{item}</li>" for item in scenario.eligibility_and_limits)
        html += "</ul>"

    if scenario.action_steps:
        html += "<b>Action Steps</b><ol>"
        html += "".join(f"<li>{step}</li>" for step in scenario.action_steps)
        html += "</ol>"

    if scenario.notes_or_exceptions:
        html += f'<p>⚠️ <i>{scenario.notes_or_exceptions}</i></p>'

    if scenario.citations:
        html += "".join(f'<span class="citation-pill">{c}</span>' for c in scenario.citations)

    html += "</div>"
    return html

def render_bot_bubble_html(structured: PolicyResponse) -> str:
    is_real_answer = len(structured.scenarios) > 0

    inner = f"<b style='font-size:16px; color:#111;'>{structured.main_heading}</b><p style='margin-top:4px;'>{structured.summary}</p>" \
        if is_real_answer else f"<p style='margin:0;'>{structured.summary}</p>"

    if is_real_answer:
        for scenario in structured.scenarios:
            inner += render_scenario_html(scenario)

    return f"""
    <div class="chat-row bot">
        <div class="bubble-bot-wrapper">
            <div class="bot-label">
                <img src="{LOGO_BASE64}" class="bot-avatar" alt="Logo">
                <span>Atlas Honda Assistant</span>
            </div>
            <div class="bubble-bot">{inner}</div>
        </div>
    </div>
    """

def render_user_bubble_html(text) -> str:
    return f"""
    <div class="chat-row user">
        <div class="bubble-user">{text}</div>
    </div>
    """

def render_debug_panel(retrieved_docs):
    if retrieved_docs:
        with st.expander(f"🔍 Source Verification ({len(retrieved_docs)} chunks retrieved)"):
            for i, doc in enumerate(retrieved_docs, 1):
                source = doc.metadata.get("source", "unknown")
                section = doc.metadata.get("section_path", "")
                st.markdown(f"**[{i}] {source}** — {section}")
                st.text(doc.page_content[:400])
                st.divider()

# ---- Render Past Messages ----
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(render_user_bubble_html(msg["text"]), unsafe_allow_html=True)
    else:
        st.markdown(render_bot_bubble_html(msg["structured"]), unsafe_allow_html=True)
        if show_debug:
            render_debug_panel(msg["retrieved_docs"])

# ---- Chat Input ----
question = st.chat_input("Ask a policy question...")

if question:
    st.session_state.messages.append({"role": "user", "text": question})
    st.rerun()

# Handle pending bot execution after user message append
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_q = st.session_state.messages[-1]["text"]
    
    with st.spinner("Searching policy database..."):
        structured, retrieved_docs = ask_bot(user_q)

    st.session_state.messages.append({
        "role": "assistant",
        "structured": structured,
        "retrieved_docs": retrieved_docs
    })
    st.rerun()

# ---- Footer ----
st.markdown('<div class="brand-footer">Powered by Atlas Honda AI HR Engine</div>', unsafe_allow_html=True)