#!/usr/bin/env python3
"""Generate delivery-lifecycle SVGs in the house style of Tim's architecture SVGs:
dark navy cards, accent strokes per stage, animated dashed wires, rounded corners."""

import os

OUT = os.path.dirname(os.path.abspath(__file__))
BASE = "https://github.com/timwukp/MLOps-agent-skills/blob/main/"

STYLE = """
  <defs>
    <marker id="ahB" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#5b8cff"/>
    </marker>
    <marker id="ahG" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#3ecf8e"/>
    </marker>
    <marker id="ahO" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#ffb454"/>
    </marker>
    <marker id="ahR" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#ff6b81"/>
    </marker>
    <style>
      .bg    { fill:#0b1020; }
      .card  { fill:#141b33; stroke-width:2; rx:12; }
      .cDesign{ stroke:#b48cff; } .cData{ stroke:#5b8cff; } .cTrain{ stroke:#ffb454; }
      .cDeploy{ stroke:#3ecf8e; } .cOps{ stroke:#ff6b81; } .cCross{ stroke:#26305a; }
      .title { fill:#e7ecff; font-size:14px; font-weight:650; }
      .sub   { fill:#8592c0; font-size:10px; }
      .band  { fill:#101731; stroke:#26305a; stroke-width:1.5; }
      .bandT { fill:#8592c0; font-size:11px; font-weight:600; letter-spacing:1px; }
      .lgd   { fill:#8592c0; font-size:11px; }
      .wire  { stroke:#5b8cff; stroke-width:2.5; fill:none; stroke-dasharray:7 6; animation:dash 1.1s linear infinite; }
      .wireG { stroke:#3ecf8e; stroke-width:2.5; fill:none; stroke-dasharray:7 6; animation:dash 1.1s linear infinite; }
      .wireO { stroke:#ffb454; stroke-width:2.5; fill:none; stroke-dasharray:7 6; animation:dash 1.1s linear infinite; }
      .wireR { stroke:#ff6b81; stroke-width:2.5; fill:none; stroke-dasharray:7 6; animation:dash 1.4s linear infinite; }
      .wireDim { stroke:#26305a; stroke-width:2; fill:none; stroke-dasharray:4 5; }
      @keyframes dash { to { stroke-dashoffset:-13; } }
      a { cursor:pointer; }
    </style>
  </defs>
"""

def card(x, y, w, h, cls, icon, name, sub, href):
    cx = x + w / 2
    return f'''  <a href="{BASE}{href}" target="_top">
    <g>
      <rect class="card {cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="12"/>
      <text class="title" x="{cx}" y="{y+26}" text-anchor="middle">{icon} {name}</text>
      <text class="sub" x="{cx}" y="{y+44}" text-anchor="middle">{sub}</text>
    </g>
  </a>
'''

def wire(x1, y1, x2, y2, cls="wire", marker="ahB", curve=None):
    if curve:
        return f'  <path class="{cls}" d="{curve}" marker-end="url(#{marker})"/>\n'
    return f'  <path class="{cls}" d="M{x1},{y1} L{x2},{y2}" marker-end="url(#{marker})"/>\n'

# ============ MLOps chain (1240 x 620) ============
W, H = 1240, 620
CW, CH = 176, 58   # card size
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,-apple-system,\'Segoe UI\',Roboto,sans-serif">']
svg.append(STYLE)
svg.append(f'  <rect class="bg" x="0" y="0" width="{W}" height="{H}" rx="16"/>')
svg.append(f'  <text class="title" x="30" y="42" font-size="18">MLOps Delivery Chain</text>')
svg.append(f'  <text class="sub" x="30" y="62">click any card to open its SKILL.md</text>')

# Row 1 (y=90): design + data stages
y1 = 90
xs = [30, 236, 442, 648, 854]
svg.append(card(xs[0], y1, CW, CH, "cDesign", "🎯", "ml-solution-design", "intake · architecture", "skills/mlops/ml-solution-design/SKILL.md"))
svg.append(card(xs[1], y1, CW, CH, "cData", "📥", "data-ingestion", "batch · streaming", "skills/mlops/data-ingestion/SKILL.md"))
svg.append(card(xs[2], y1, CW, CH, "cData", "✅", "data-validation", "contracts · quality", "skills/mlops/data-validation/SKILL.md"))
svg.append(card(xs[3], y1, CW, CH, "cData", "🧪", "feature-engineering", "transform · select", "skills/mlops/feature-engineering/SKILL.md"))
svg.append(card(xs[4], y1, CW, CH, "cData", "🗄️", "feature-store", "Feast · SageMaker FS", "skills/mlops/feature-store/SKILL.md"))
for i in range(4):
    svg.append(wire(xs[i]+CW, y1+CH/2, xs[i+1]-8, y1+CH/2))

# Row 2 (y=230): training stage
y2 = 230
svg.append(card(30, y2, CW, CH, "cTrain", "🏋️", "model-training", "HPO · distributed", "skills/mlops/model-training/SKILL.md"))
svg.append(card(236, y2, CW, CH, "cTrain", "📊", "ml-experiment-tracking", "MLflow · W&amp;B", "skills/mlops/ml-experiment-tracking/SKILL.md"))
svg.append(card(442, y2, CW, CH, "cTrain", "📦", "model-registry", "aliases · promotion", "skills/mlops/model-registry/SKILL.md"))
svg.append(card(648, y2, CW, CH, "cDeploy", "🚀", "ml-cicd", "GitHub Actions · Pipelines", "skills/mlops/ml-cicd/SKILL.md"))
svg.append(card(854, y2, CW, CH, "cDeploy", "🌐", "model-serving", "FastAPI · endpoints", "skills/mlops/model-serving/SKILL.md"))
# feature-eng down to training (elbow)
svg.append(wire(0,0,0,0, curve=f"M{648+CW/2},{y1+CH} C {648+CW/2},{y1+CH+45} {30+CW/2},{y2-45} {30+CW/2},{y2-3}"))
# tracking parallel (dim, bidirectional feel)
svg.append(f'  <path class="wireDim" d="M{30+CW},{y2+CH/2} L{236-8},{y2+CH/2}"/>\n')
svg.append(wire(236+CW, y2+CH/2, 442-8, y2+CH/2, "wireO", "ahO"))
svg.append(wire(442+CW, y2+CH/2, 648-8, y2+CH/2, "wireO", "ahO"))
svg.append(wire(648+CW, y2+CH/2, 854-8, y2+CH/2, "wireG", "ahG"))

# Row 3 (y=370): ops
y3 = 370
svg.append(card(648, y3, CW, CH, "cOps", "📈", "model-monitoring", "Evidently · Model Monitor", "skills/mlops/model-monitoring/SKILL.md"))
svg.append(card(854, y3, CW, CH, "cOps", "🌊", "model-drift-detection", "PSI · KS · triggers", "skills/mlops/model-drift-detection/SKILL.md"))
svg.append(card(236, y3, CW, CH, "cCross", "🔍", "model-observability", "SHAP · fairness", "skills/mlops/model-observability/SKILL.md"))
svg.append(wire(0,0,0,0, "wireG", "ahG", curve=f"M{854+CW/2},{y2+CH} L{854+CW/2},{y3-8}").replace('class="wire"','class="wireG"'))
# serving->monitoring
svg.append(wire(0,0,0,0, "wireG", "ahG", curve=f"M{854+CW/2},{y2+CH} C {854+CW/2},{y2+CH+40} {648+CW/2},{y3-40} {648+CW/2},{y3-8}"))
# monitoring->drift (single direction; the reverse leg is the retraining loop below)
svg.append(wire(648+CW, y3+CH/2, 854-4, y3+CH/2, "wireR", "ahR"))
# observability dim link to monitoring
svg.append(f'  <path class="wireDim" d="M{236+CW},{y3+CH/2} L{648-8},{y3+CH/2}"/>\n')
# retraining loop: drift -> training (big left elbow)
svg.append(wire(0,0,0,0, "wireR", "ahR", curve=(
    f"M{854+CW/2},{y3+CH} "
    f"C {854+CW/2},{y3+CH+16} {854+CW/2-10},{449} {854+CW/2-26},{449} "
    f"L {40},{449} "
    f"C {14},{449} {14},{449-26} {14},{423} "
    f"L {14},{y2+CH/2+26} "
    f"C {14},{y2+CH/2} {14+10},{y2+CH/2} {30-4},{y2+CH/2}")))
svg.append(f'  <text class="sub" x="500" y="{444}" text-anchor="middle" fill="#ff6b81">retraining trigger</text>')

# Cross-cutting band (y=470)
yb = 470
svg.append(f'  <rect class="band" x="30" y="{yb}" width="{W-60}" height="100" rx="14"/>')
svg.append(f'  <text class="bandT" x="50" y="{yb+26}">CROSS-CUTTING — EVERY STAGE</text>')
cc = [
    ("🧰 ml-testing", "quality gates", "skills/mlops/ml-testing/SKILL.md", 50),
    ("🛡️ ml-security", "privacy · compliance", "skills/mlops/ml-security/SKILL.md", 290),
    ("💰 ml-cost-optimization", "sizing · FinOps", "skills/mlops/ml-cost-optimization/SKILL.md", 530),
    ("🧬 ml-pipeline-orchestration", "Airflow · SageMaker Pipelines", "skills/mlops/ml-pipeline-orchestration/SKILL.md", 800),
]
for icon_name, sub, href, x in cc:
    svg.append(f'''  <a href="{BASE}{href}" target="_top"><g>
      <rect class="card cCross" x="{x}" y="{yb+40}" width="225" height="44" rx="10"/>
      <text class="title" x="{x+112}" y="{yb+58}" text-anchor="middle" font-size="12">{icon_name}</text>
      <text class="sub" x="{x+112}" y="{yb+74}" text-anchor="middle">{sub}</text>
    </g></a>\n''')
svg.append('</svg>')
open(os.path.join(OUT, "mlops-lifecycle.svg"), "w").write("\n".join(svg))

# ============ LLMOps chain (1240 x 540) ============
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1240 540" font-family="system-ui,-apple-system,\'Segoe UI\',Roboto,sans-serif">']
svg.append(STYLE)
svg.append(f'  <rect class="bg" x="0" y="0" width="1240" height="540" rx="16"/>')
svg.append(f'  <text class="title" x="30" y="42" font-size="18">LLMOps Delivery Chain</text>')
svg.append(f'  <text class="sub" x="30" y="62">click any card to open its SKILL.md</text>')

y1 = 100
svg.append(card(30, y1+40, CW, CH, "cDesign", "🎯", "ml-solution-design", "intake · architecture", "skills/mlops/ml-solution-design/SKILL.md"))
svg.append(card(250, y1+40, CW, CH, "cData", "🧹", "llm-data-preparation", "synthetic · dedup", "skills/llmops/llm-data-preparation/SKILL.md"))
svg.append(card(470, y1, CW, CH, "cTrain", "🎛️", "llm-fine-tuning", "LoRA · DPO", "skills/llmops/llm-fine-tuning/SKILL.md"))
svg.append(card(470, y1+84, CW, CH, "cTrain", "🔎", "llm-rag", "chunking · hybrid search", "skills/llmops/llm-rag/SKILL.md"))
svg.append(card(690, y1+40, CW, CH, "cTrain", "🧾", "llm-evaluation", "RAGAS · LLM-as-judge", "skills/llmops/llm-evaluation/SKILL.md"))
svg.append(card(910, y1+40, CW, CH, "cDeploy", "🌐", "llm-deployment", "vLLM · Bedrock", "skills/llmops/llm-deployment/SKILL.md"))
svg.append(wire(30+CW, y1+40+CH/2, 250-8, y1+40+CH/2))
# prep -> fine-tuning & rag (branch)
svg.append(wire(0,0,0,0, curve=f"M{250+CW},{y1+40+CH/2} C {250+CW+30},{y1+40+CH/2} {470-40},{y1+CH/2} {470-8},{y1+CH/2}"))
svg.append(wire(0,0,0,0, curve=f"M{250+CW},{y1+40+CH/2} C {250+CW+30},{y1+40+CH/2} {470-40},{y1+84+CH/2} {470-8},{y1+84+CH/2}"))
# merge -> evaluation
svg.append(wire(0,0,0,0, "wireO", "ahO", curve=f"M{470+CW},{y1+CH/2} C {470+CW+30},{y1+CH/2} {690-40},{y1+40+CH/2} {690-8},{y1+40+CH/2}"))
svg.append(wire(0,0,0,0, "wireO", "ahO", curve=f"M{470+CW},{y1+84+CH/2} C {470+CW+30},{y1+84+CH/2} {690-40},{y1+40+CH/2} {690-8},{y1+40+CH/2}"))
svg.append(wire(690+CW, y1+40+CH/2, 910-8, y1+40+CH/2, "wireG", "ahG"))

# Row 2: app layer + ops
y2 = 300
svg.append(card(910, y2, CW, CH, "cDeploy", "🤖", "llm-agent-orchestration", "LangGraph · tools", "skills/llmops/llm-agent-orchestration/SKILL.md"))
svg.append(card(690, y2, CW, CH, "cOps", "🛡️", "llm-guardrails", "PII · Bedrock Guardrails", "skills/llmops/llm-guardrails/SKILL.md"))
svg.append(card(470, y2, CW, CH, "cOps", "📈", "llm-observability", "tokens · latency · traces", "skills/llmops/llm-observability/SKILL.md"))
svg.append(card(250, y2, CW, CH, "cOps", "💰", "llm-cost-optimization", "routing · caching", "skills/llmops/llm-cost-optimization/SKILL.md"))
svg.append(card(30, y2, CW, CH, "cCross", "✍️", "llm-prompt-engineering", "cross-cutting", "skills/llmops/llm-prompt-engineering/SKILL.md"))
# deployment down to agent-orchestration
svg.append(wire(0,0,0,0, "wireG", "ahG", curve=f"M{910+CW/2},{y1+40+CH} L{910+CW/2},{y2-8}"))
# agent-orch -> guardrails -> observability -> cost (right to left flow)
svg.append(wire(910-0, y2+CH/2, 690+CW+8, y2+CH/2, "wireR", "ahR", curve=f"M{910-8},{y2+CH/2} L{690+CW+4},{y2+CH/2}"))
svg.append(wire(690, y2+CH/2, 470+CW+8, y2+CH/2, "wireR", "ahR", curve=f"M{690-8},{y2+CH/2} L{470+CW+4},{y2+CH/2}"))
svg.append(wire(470, y2+CH/2, 250+CW+8, y2+CH/2, "wireR", "ahR", curve=f"M{470-8},{y2+CH/2} L{250+CW+4},{y2+CH/2}"))
# optimization loop back to deployment: own corridor under row 2, up the right margin,
# into llm-deployment's RIGHT edge -- never crosses another wire or a card.
svg.append(wire(0,0,0,0, "wireR", "ahR", curve=(
    f"M{250+CW/2},{y2+CH} "
    f"C {250+CW/2},{y2+CH+14} {250+CW/2+14},{396} {250+CW/2+40},{396} "
    f"L {1134},{396} "
    f"C {1160},{396} {1160},{370} {1160},{344} "
    f"L {1160},{y1+40+CH/2+26} "
    f"C {1160},{y1+40+CH/2} {1160-14},{y1+40+CH/2} {1090},{y1+40+CH/2}")))
svg.append(f'  <text class="sub" x="640" y="{388}" text-anchor="middle" fill="#ff6b81">optimization loop → cheaper routing, caching, compression</text>')

# bridges band
yb = 440
svg.append(f'  <rect class="band" x="30" y="{yb}" width="1180" height="70" rx="14"/>')
svg.append(f'  <text class="bandT" x="50" y="{yb+26}">BRIDGES TO CLASSIC MLOPS</text>')
svg.append(f'  <text class="sub" x="50" y="{yb+46}">fine-tuned models flow through:</text>')
br = [("📦 model-registry", "skills/mlops/model-registry/SKILL.md", 560),
      ("🌐 model-serving", "skills/mlops/model-serving/SKILL.md", 780),
      ("📈 model-monitoring", "skills/mlops/model-monitoring/SKILL.md", 1000)]
for name, href, x in br:
    svg.append(f'''  <a href="{BASE}{href}" target="_top"><g>
      <rect class="card cCross" x="{x}" y="{yb+16}" width="190" height="38" rx="10"/>
      <text class="title" x="{x+95}" y="{yb+40}" text-anchor="middle" font-size="12">{name}</text>
    </g></a>\n''')
svg.append('</svg>')
open(os.path.join(OUT, "llmops-lifecycle.svg"), "w").write("\n".join(svg))
print("SVGs written")
