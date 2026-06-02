CSS = """
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; }
.stApp { background: linear-gradient(135deg,#0a1a0d 0%,#0f2016 50%,#0a1a0d 100%); min-height:100vh; }
.hero-header { text-align:center; padding:2rem 1rem 1.5rem;
  background:linear-gradient(180deg,rgba(34,197,94,0.08) 0%,transparent 100%);
  border-bottom:1px solid rgba(34,197,94,0.15); margin-bottom:2rem; }
.hero-header h1 { font-family:Georgia,serif; font-size:2.6rem; font-weight:700;
  color:#e8f5e9; letter-spacing:-0.5px; margin:0; }
.hero-header p { color:#6ab87a; font-size:1rem; margin-top:0.4rem; font-weight:300; }
.vbadge { display:inline-block; background:rgba(34,197,94,0.15);
  border:1px solid rgba(34,197,94,0.3); color:#4ade80; font-size:0.72rem; font-weight:600;
  padding:0.15rem 0.6rem; border-radius:999px; letter-spacing:1px; text-transform:uppercase; margin-top:0.6rem; }
.metric-card { background:rgba(255,255,255,0.04); border:1px solid rgba(34,197,94,0.2);
  border-radius:12px; padding:1.2rem 1.5rem; margin-bottom:0.8rem; }
.metric-label { color:#6ab87a; font-size:0.72rem; text-transform:uppercase;
  letter-spacing:1.5px; font-weight:600; margin-bottom:0.3rem; }
.metric-value { color:#e8f5e9; font-size:1.55rem; font-weight:600; }
.metric-sub { color:#4a8a58; font-size:0.78rem; margin-top:0.2rem; }
.result-healthy { background:linear-gradient(135deg,rgba(34,197,94,0.18),rgba(21,128,61,0.12));
  border:2px solid #22c55e; border-radius:16px; padding:2rem; text-align:center; }
.result-unhealthy { background:linear-gradient(135deg,rgba(239,68,68,0.18),rgba(153,27,27,0.12));
  border:2px solid #ef4444; border-radius:16px; padding:2rem; text-align:center; }
.result-title { font-family:Georgia,serif; font-size:2.2rem; font-weight:700; margin:0.5rem 0; }
.insight-box { background:rgba(255,255,255,0.03); border-left:3px solid #22c55e;
  border-radius:0 10px 10px 0; padding:1.1rem 1.4rem; margin:0.8rem 0;
  color:#c8e6c9; font-size:0.93rem; line-height:1.7; }
.section-title { font-family:Georgia,serif; color:#a5d6a7; font-size:1.25rem; font-weight:700;
  margin:1.4rem 0 0.7rem; padding-bottom:0.4rem; border-bottom:1px solid rgba(34,197,94,0.2); }
.param-row { display:flex; justify-content:space-between; align-items:center;
  padding:0.5rem 0; border-bottom:1px solid rgba(255,255,255,0.05); color:#c8e6c9; font-size:0.88rem; }
.param-pass { color:#4ade80; font-weight:600; }
.param-fail { color:#f87171; font-weight:600; }
.badge { display:inline-block; padding:0.2rem 0.7rem; border-radius:999px;
  font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.8px; }
div[data-testid="stSidebar"] { background:#07130a !important; border-right:1px solid rgba(34,197,94,0.15) !important; }
.stButton > button { background:linear-gradient(135deg,#16a34a,#15803d); color:white;
  border:none; border-radius:8px; font-weight:600; padding:0.6rem 2rem; width:100%;
  font-size:0.95rem; transition:all 0.2s; }
.stButton > button:hover { background:linear-gradient(135deg,#15803d,#166534);
  transform:translateY(-1px); box-shadow:0 4px 20px rgba(34,197,94,0.3); }
.stProgress > div > div { background:#22c55e !important; }
.warn-box { background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.3);
  border-radius:10px; padding:1rem 1.2rem; color:#fde68a; font-size:0.88rem; margin:0.8rem 0; }
.pipeline-step { background:rgba(255,255,255,0.03); border:1px solid rgba(34,197,94,0.15);
  border-radius:12px; padding:1.4rem; margin:0.8rem 0; }
.pipeline-num { background:linear-gradient(135deg,#22c55e,#15803d); color:white;
  width:28px; height:28px; border-radius:50%; display:inline-flex; align-items:center;
  justify-content:center; font-weight:700; font-size:0.82rem; margin-right:0.7rem; }
.chip { display:inline-block; background:rgba(255,255,255,0.06);
  border:1px solid rgba(34,197,94,0.2); border-radius:7px; padding:0.35rem 0.7rem;
  font-size:0.78rem; color:#c8e6c9; margin:0.2rem; }
.chip b { color:#e8f5e9; }
</style>
"""