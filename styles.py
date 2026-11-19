"""Visual design for the Streamlit app: design tokens and component styles.

Kept in one module so ``app.py`` stays about behaviour. ``CSS`` applies everywhere; ``LOGIN_CSS`` is added only
on the sign-in screen (it hides the sidebar and widens the layout).
"""

CSS = """
<style>
:root {
  --ink: #0b1220;
  --body: #334155;
  --muted: #64748b;
  --faint: #94a3b8;
  --line: #e2e8f0;
  --line-strong: #cbd5e1;
  --canvas: #f4f6fb;
  --surface: #ffffff;
  --brand: #2347d9;
  --brand-dark: #1a37ab;
  --brand-soft: #eef2ff;
  --brand-line: #c7d2fe;
  --navy: #0b1f4b;
  --good: #047857; --good-bg: #ecfdf5; --good-line: #a7f3d0;
  --warn: #b45309; --warn-bg: #fffbeb; --warn-line: #fde68a;
  --bad: #b91c1c;  --bad-bg: #fef2f2;  --bad-line: #fecaca;
  --radius: 12px;
  --shadow-sm: 0 1px 2px rgba(15, 23, 42, .05);
  --shadow-md: 0 4px 14px rgba(15, 23, 42, .08);
  --shadow-lg: 0 18px 44px rgba(15, 23, 42, .14);
}
html, body, .stApp, [class*="css"] {
  font-family: "Inter", "Segoe UI", -apple-system, "Helvetica Neue", Arial, sans-serif;
  color: var(--body);
  -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--canvas); }
#MainMenu, footer, [data-testid="stToolbarActions"], [data-testid="stMainMenu"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"] { visibility: visible !important; }
[data-testid="stExpandSidebarButton"] { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; }
.block-container { padding-top: 2.4rem; padding-bottom: 6rem; max-width: 960px; }
h1, h2, h3 { letter-spacing: -0.01em; }

/* ---- Inputs and buttons ---- */
.stTextInput [data-testid="stTextInputRootElement"], .stSelectbox [data-baseweb="select"] > div, .stTextArea textarea {
  border-radius: 10px !important; border: 1px solid var(--line-strong) !important; background: var(--surface) !important; }
.stTextInput [data-testid="stTextInputRootElement"]:focus-within { border-color: var(--brand) !important; box-shadow: 0 0 0 3px rgba(35, 71, 217, .15) !important; }
.stTextInput input { border: 0 !important; box-shadow: none !important; background: transparent !important; }
[data-testid="stSelectbox"] [role="group"] { border: 1px solid var(--line-strong) !important; border-radius: 10px !important; background: var(--surface) !important; }
.stTextInput label p, .stSelectbox label p, .stFileUploader label p { font-weight: 600; font-size: 0.82rem; color: var(--ink); }
.stButton > button, [data-testid="stFormSubmitButton"] > button {
  border-radius: 10px; font-weight: 600; font-size: 0.88rem; border: 1px solid var(--line-strong);
  transition: border-color .15s, box-shadow .15s, background .15s; }
.stButton > button:hover { border-color: var(--brand); color: var(--brand); }
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"], [data-testid="stFormSubmitButton"] > button[data-testid="stBaseButton-primaryFormSubmit"] {
  background: var(--brand); border-color: var(--brand); color: #fff; box-shadow: 0 1px 2px rgba(35, 71, 217, .35); }
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover,
[data-testid="stFormSubmitButton"] > button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
  background: var(--brand-dark); border-color: var(--brand-dark); color: #fff; }

/* ---- Role badges and avatars ---- */
.role-badge { display: inline-block; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
  padding: 3px 9px; border-radius: 999px; border: 1px solid transparent; white-space: nowrap; }
.role-badge.admin { background: #f5f3ff; color: #6d28d9; border-color: #ddd6fe; }
.role-badge.director { background: #eff6ff; color: #1d4ed8; border-color: #bfdbfe; }
.role-badge.senior { background: #f0fdfa; color: #0f766e; border-color: #99f6e4; }
.role-badge.employee { background: #f1f5f9; color: #475569; border-color: #cbd5e1; }
.avatar { width: 38px; height: 38px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 0.82rem; color: #fff; flex: none; }
.avatar.admin { background: #6d28d9; } .avatar.director { background: #1d4ed8; }
.avatar.senior { background: #0f766e; } .avatar.employee { background: #475569; }

/* ---- Page header ---- */
.page-head { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px;
  padding-bottom: 16px; margin-bottom: 18px; border-bottom: 1px solid var(--line); }
.eyebrow { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--brand); }
.page-head h1 { font-size: 1.7rem; line-height: 1.2; font-weight: 750; color: var(--ink); margin: 3px 0 5px 0; padding: 0; }
.page-head p { margin: 0; color: var(--muted); font-size: 0.93rem; }
.scope-pill { font-size: 0.78rem; font-weight: 600; color: var(--brand); background: var(--brand-soft);
  border: 1px solid var(--brand-line); border-radius: 999px; padding: 5px 12px; white-space: nowrap; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
.brand { display: flex; align-items: center; gap: 11px; margin-bottom: 6px; }
.brand-mark { width: 36px; height: 36px; border-radius: 10px; background: linear-gradient(145deg, var(--brand), var(--navy));
  color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1rem;
  box-shadow: 0 2px 6px rgba(35, 71, 217, .35); }
.brand-name { font-weight: 750; color: var(--ink); font-size: 1rem; line-height: 1.1; }
.brand-sub { color: var(--muted); font-size: 0.78rem; }
.profile { display: flex; align-items: center; gap: 11px; padding: 12px; margin: 14px 0 4px 0; border: 1px solid var(--line);
  border-radius: var(--radius); background: linear-gradient(180deg, #fff, var(--canvas)); }
.profile .who { min-width: 0; line-height: 1.25; }
.profile .name { font-weight: 700; color: var(--ink); font-size: 0.9rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.profile .title { color: var(--muted); font-size: 0.75rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 4px; }
.side-title { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--muted); margin: 22px 0 8px 0; }
.status-list { border: 1px solid var(--line); border-radius: 10px; background: var(--surface); }
.status-row { display: flex; align-items: center; justify-content: space-between; gap: 10px;
  padding: 10px 14px; font-size: 0.84rem; color: var(--ink); }
.status-row + .status-row { border-top: 1px solid var(--line); }
.status-row .val { color: var(--muted); font-size: 0.78rem; text-align: right; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; }
.dot.on { background: #10b981; } .dot.idle { background: #f59e0b; } .dot.off { background: #ef4444; }
.dot.busy { background: #3b82f6; animation: pulse 1.2s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
.side-note { color: var(--muted); font-size: 0.76rem; margin-top: 6px; line-height: 1.5; }
.upload-item { font-size: 0.84rem; color: var(--ink); font-weight: 600; line-height: 1.25; }
.upload-item span { display: block; color: var(--muted); font-weight: 400; font-size: 0.75rem; }

/* ---- Admin views ---- */
.kpi-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 4px 0 18px 0; }
.kpi { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 14px 16px; box-shadow: var(--shadow-sm); }
.kpi .label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.kpi .value { font-size: 1.65rem; font-weight: 750; color: var(--ink); line-height: 1.2; margin-top: 4px; }
.kpi .hint { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }
.panel-title { font-size: 1rem; font-weight: 700; color: var(--ink); margin: 6px 0 2px 0; }
.panel-sub { font-size: 0.86rem; color: var(--muted); margin: 0 0 12px 0; }
.st-key-admin_nav { margin-bottom: 14px; }
.st-key-admin_panel, .st-key-kb_upload, .st-key-user_admin {
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 18px 20px; box-shadow: var(--shadow-sm); }
@media (max-width: 760px) { .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }

table.coverage { width: 100%; border-collapse: separate; border-spacing: 0; border: 1px solid var(--line); border-radius: 10px;
  background: var(--surface); font-size: 0.84rem; overflow: hidden; }
table.coverage th { text-align: left; font-weight: 600; color: var(--muted); background: var(--canvas); padding: 9px 12px; border-bottom: 1px solid var(--line); }
table.coverage td { padding: 9px 12px; color: var(--ink); border-bottom: 1px solid var(--line); line-height: 1.35; }
table.coverage tr:last-child td { border-bottom: 0; }
table.coverage .num { text-align: right; white-space: nowrap; width: 1%; }

/* ---- Banners ---- */
.banner { border-radius: 10px; padding: 10px 14px; font-size: 0.86rem; margin-bottom: 14px; border: 1px solid; }
.banner.info { background: var(--brand-soft); border-color: var(--brand-line); color: #1e3a8a; }
.banner.error { background: var(--bad-bg); border-color: var(--bad-line); color: var(--bad); }

/* ---- Empty state ---- */
.welcome { background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 22px 26px; margin-bottom: 16px;
  box-shadow: var(--shadow-sm); border-left: 4px solid var(--brand); }
.welcome .hello { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.welcome h3 { margin: 0; color: var(--ink); font-size: 1.12rem; font-weight: 750; }
.welcome p { margin: 0; color: var(--muted); font-size: 0.92rem; line-height: 1.6; }
.st-key-starters .stButton > button {
  width: 100%; height: 100%; min-height: 98px; text-align: left; justify-content: flex-start;
  border-radius: var(--radius); border: 1px solid var(--line); background: var(--surface); color: var(--ink);
  padding: 13px 15px; font-weight: 500; box-shadow: var(--shadow-sm); transition: border-color .15s, box-shadow .15s, transform .15s;
}
.st-key-starters [data-testid="stHorizontalBlock"] { align-items: stretch; }
.st-key-starters [data-testid="stColumn"] > div, .st-key-starters .stButton { height: 100%; }
.st-key-starters .stButton > button > div { width: 100%; justify-content: flex-start; text-align: left; }
.st-key-starters .stButton > button [data-testid="stMarkdownContainer"] { width: 100%; text-align: left; display: flex; flex-direction: column; align-items: flex-start; gap: 0; }
.st-key-starters .stButton > button p { white-space: normal; text-align: left; margin: 0; line-height: 1.4; font-size: 0.88rem; }
.st-key-starters .stButton > button p:first-child { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em;
  text-transform: uppercase; color: var(--brand); margin-bottom: 5px; }
.st-key-starters .stButton > button:hover { border-color: var(--brand); box-shadow: var(--shadow-md); color: var(--brand); transform: translateY(-1px); }
.starter-label { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin: 6px 0 -2px 2px; }

.topic-label { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin: 14px 0 6px 0; }
.topic-chips { display: flex; flex-wrap: wrap; gap: 6px; }
.topic-chip { font-size: 0.8rem; color: var(--brand); background: var(--brand-soft); border: 1px solid var(--brand-line); border-radius: 999px; padding: 3px 11px; }
[class*="st-key-followups_"] .stButton > button { justify-content: flex-start; text-align: left; border-radius: 12px; border: 1px solid var(--brand-line);
  background: var(--surface); color: var(--ink); font-weight: 500; padding: 9px 14px; }
[class*="st-key-followups_"] .stButton > button p { text-align: left; font-size: 0.88rem; }
[class*="st-key-followups_"] .stButton > button:hover { border-color: var(--brand); color: var(--brand); background: var(--brand-soft); }

/* ---- Chat ---- */
[data-testid="stChatMessage"] { background: var(--surface); border: 1px solid var(--line); border-radius: 16px;
  padding: 15px 19px; margin-bottom: 12px; box-shadow: var(--shadow-sm); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: var(--brand-soft); border-color: var(--brand-line); }
[data-testid="stChatInput"] { border-radius: 16px; box-shadow: var(--shadow-md); }
.meta-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 8px 0; }
.tag { display: inline-block; padding: 2px 9px; border-radius: 6px; font-size: 0.72rem; font-weight: 600; border: 1px solid transparent; }
.tag.neutral { background: #f1f5f9; color: var(--body); border-color: var(--line); }
.tag.brand { background: var(--brand-soft); color: var(--brand); border-color: var(--brand-line); }
.tag.high { background: var(--good-bg); color: var(--good); border-color: var(--good-line); }
.tag.medium { background: var(--warn-bg); color: var(--warn); border-color: var(--warn-line); }
.tag.warn { background: var(--warn-bg); color: var(--warn); border-color: var(--warn-line); }
.tag.good { background: var(--good-bg); color: var(--good); border-color: var(--good-line); }
.tag.bad { background: var(--bad-bg); color: var(--bad); border-color: var(--bad-line); }
.tag.low { background: var(--bad-bg); color: var(--bad); border-color: var(--bad-line); }

/* ---- Source cards ---- */
.source-card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 11px 14px; margin: 8px 0; }
.source-card[open] { border-color: var(--brand-line); }
.source-card summary { cursor: pointer; list-style: none; font-weight: 600; font-size: 0.9rem; color: var(--ink); }
.source-card summary::-webkit-details-marker { display: none; }
.source-card .sub { color: var(--muted); font-weight: 400; font-size: 0.78rem; margin-top: 3px; }
.source-card .scores { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 8px 0; }
.source-card .chunk { background: var(--canvas); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px;
  font-size: 0.83rem; line-height: 1.55; color: var(--body); }
.bar { height: 4px; background: var(--line); border-radius: 4px; overflow: hidden; margin-top: 9px; }
.bar > div { height: 100%; background: var(--brand); }

.st-key-top_actions { position: fixed; top: 10px; right: 76px; z-index: 1000001; width: auto !important; flex-direction: row !important;
  align-items: center; gap: 8px !important; }
.st-key-top_actions button { border-radius: 999px; padding: 4px 14px; font-weight: 600; font-size: 0.84rem; background: var(--surface);
  border: 1px solid var(--line); color: var(--ink); }
.st-key-top_actions .stButton > button { min-height: 2.25rem; }
[data-testid="stPopoverBody"] .profile { border: 0; box-shadow: none; padding: 4px 0 10px 0; margin: 0; }
section[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
section[data-testid="stSidebar"] [role="radiogroup"] label { padding: 8px 10px; border-radius: 9px; width: 100%; }
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: var(--brand-soft, #e0e7ff); font-weight: 650; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child { display: none; }
section[data-testid="stSidebar"] .stButton > button, section[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] > button {
  border-radius: 9px; font-weight: 600; font-size: 0.84rem; }
</style>
"""

DARK_CSS = """
<style>
:root {
  color-scheme: dark;
  --ink: #f1f5f9; --body: #cbd5e1; --muted: #94a3b8; --faint: #64748b;
  --line: #243049; --line-strong: #33415c;
  --canvas: #0a1020; --surface: #121a2e;
  --brand: #7b97ff; --brand-dark: #6384f5; --brand-soft: #1a2750; --brand-line: #34498f;
  --good: #34d399; --good-bg: #0b2a22; --good-line: #14604a;
  --warn: #fbbf24; --warn-bg: #2d2208; --warn-line: #6b4f12;
  --bad: #f87171;  --bad-bg: #2f1214;  --bad-line: #7a2a2e;
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, .4); --shadow-md: 0 4px 14px rgba(0, 0, 0, .45); --shadow-lg: 0 18px 44px rgba(0, 0, 0, .55);
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stBottom"], [data-testid="stBottom"] > div,
header[data-testid="stHeader"] { background: var(--canvas) !important; }
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div { background: var(--surface) !important; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp span, [data-testid="stMarkdownContainer"] { color: var(--body); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp strong { color: var(--ink); }
[data-testid="stCaptionContainer"], .stApp small { color: var(--muted); }
.stApp a { color: var(--brand); }

.stButton > button, [data-testid="stFormSubmitButton"] > button { background: var(--surface); color: var(--ink); border-color: var(--line-strong); }
.stButton > button p, .stButton > button span { color: inherit; }
.stButton > button:hover { border-color: var(--brand); color: var(--brand); }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"],
[data-testid="stFormSubmitButton"] > button[data-testid="stBaseButton-primaryFormSubmit"] { background: #2f55e6; border-color: #2f55e6; color: #fff; }
.stButton > button[kind="primary"] p, .stButton > button[data-testid="stBaseButton-primary"] p { color: #fff; }
.st-key-auth_card .stButton > button[kind="tertiary"], .st-key-auth_card .stButton > button[data-testid="stBaseButton-tertiary"] { background: none !important; color: var(--brand); border: 0 !important; }

input, textarea, [data-baseweb="select"] div, [data-baseweb="input"] input { color: var(--ink) !important; -webkit-text-fill-color: var(--ink); }
input::placeholder, textarea::placeholder { color: var(--faint) !important; -webkit-text-fill-color: var(--faint); }
[data-baseweb="popover"] > div, [data-baseweb="menu"], [data-baseweb="popover"] ul, [data-testid="stPopoverBody"] { background: var(--surface) !important; color: var(--body); border-color: var(--line); }
[data-baseweb="popover"] li:hover, [data-baseweb="menu"] li:hover { background: var(--brand-soft) !important; }
[data-testid="stChatInput"], [data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea { background: var(--surface) !important; }
[data-testid="stFileUploaderDropzone"] { background: var(--canvas) !important; border-color: var(--line-strong); }
[data-testid="stFileUploaderDropzone"] button { background: var(--surface) !important; color: var(--ink) !important; border-color: var(--line-strong) !important; }
[data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] span { color: var(--muted) !important; }
[data-testid="stSelectbox"] [role="group"] { background: var(--surface) !important; border-color: var(--line-strong) !important; }
[data-testid="stSelectbox"] input, [data-testid="stSelectbox"] [role="group"] * { color: var(--ink) !important; -webkit-text-fill-color: var(--ink); }
[data-testid="stSelectbox"] svg { fill: var(--muted); }
[role="listbox"], [role="listbox"] [role="option"], [data-testid="stSelectboxVirtualDropdown"] { background: var(--surface) !important; color: var(--ink) !important; }
[role="listbox"] [role="option"]:hover, [role="listbox"] [role="option"][data-focused], [role="listbox"] [role="option"][aria-selected="true"] { background: var(--brand-soft) !important; }
[data-testid="stTooltipIcon"] svg, [data-testid="stTooltipHoverTarget"] svg { color: var(--muted) !important; fill: var(--muted) !important; }
[data-testid="stSliderThumbValue"], [data-testid="stSlider"] p { color: var(--ink) !important; }
[data-testid="stExpander"], [data-testid="stExpander"] details { background: var(--surface); border-color: var(--line); }
[data-testid="stAlert"] { background: var(--brand-soft); color: var(--ink); }
[data-testid="stCode"] pre, [data-testid="stCode"] code { background: var(--canvas) !important; color: var(--body); }
[data-testid="stDataFrame"] { filter: invert(.92) hue-rotate(180deg); border-radius: 10px; }
[data-testid="stSlider"] [data-testid="stTickBar"] { color: var(--muted); }
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: var(--brand-soft); }

.role-badge.admin { background: #2a1d52; color: #c4b5fd; border-color: #4c3a8f; }
.role-badge.director { background: #14264f; color: #93c5fd; border-color: #2c4a8f; }
.role-badge.senior { background: #0c2f2c; color: #5eead4; border-color: #1c6a63; }
.role-badge.employee { background: #1e293b; color: #cbd5e1; border-color: #3b4a63; }
.profile { background: linear-gradient(180deg, var(--surface), var(--canvas)); }
.tag.neutral { background: #1e293b; color: var(--body); }
.banner.info { color: #bfd0ff; }
.demo-card { background: var(--canvas); }
</style>
"""

LOGIN_CSS = """
<style>
section[data-testid="stSidebar"], [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"],
header[data-testid="stHeader"] { display: none !important; }
html, body, .stApp, [data-testid="stMain"] { overflow: hidden !important; }
.stApp { background: var(--surface); }
.block-container { max-width: 100% !important; padding: 0 !important; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { gap: 0 !important; }
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] style) { display: none; }
[data-testid="stHorizontalBlock"] { gap: 0 !important; align-items: stretch; }

.st-key-auth_brand { background: #0f172a; height: 100vh !important; min-height: 100vh !important; display: flex !important; flex-direction: column; align-items: center; justify-content: center !important; }
.auth-logo { text-align: center; }
.auth-logo .mark { width: 56px; height: 56px; margin: 0 auto 16px auto; border-radius: 14px; background: #2347d9; color: #fff;
  display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.5rem; }
.auth-logo .name { color: #fff; font-weight: 700; font-size: 1.4rem; letter-spacing: -0.01em; }
.auth-logo .sub { color: #94a3b8; font-size: 0.9rem; margin-top: 2px; }

.st-key-auth_card { height: 100vh !important; min-height: 100vh !important; display: flex !important; flex-direction: column; justify-content: center !important; padding: 0 12% !important; }
.auth-title { font-size: 1.6rem; font-weight: 750; color: var(--ink); margin: 0 0 14px 0; letter-spacing: -0.01em; }
.st-key-auth_card [data-testid="stForm"] { padding: 0; }
.st-key-auth_card .stButton > button[kind="tertiary"] { padding: 0 !important; border: 0 !important; background: none !important; box-shadow: none !important; font-size: 0.85rem; color: var(--brand); min-height: 0; justify-content: flex-start; }

.demo-head { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin: 14px 0 6px 0; }
.st-key-demo_grid .stButton > button { font-size: 0.8rem; padding: 4px 8px; min-height: 2.1rem; }
@media (max-width: 900px) { .st-key-auth_brand { display: none; } html, body, .stApp, [data-testid="stMain"] { overflow: auto !important; } }
</style>
"""
