"""Layout regression test for the lifecycle diagrams.

Flattens every flow wire in docs/*-lifecycle.svg into polyline segments and
asserts (a) no two wires intersect and (b) no wire passes through a card's
interior. Crossing dashed lines make the diagrams hard to read, so this guards
the layout whenever docs/gen_lifecycle_svg.py is edited.

Usage: python3 tests/check_svg_geometry.py [svg ...]   # defaults to docs/*-lifecycle.svg
"""
import itertools
import os
import re
import sys

def parse_path(d):
    """Return list of points approximating the path (M/L/C only)."""
    toks = re.findall(r'[MLC]|-?\d+\.?\d*', d)
    pts, i, cur, cmd = [], 0, None, None
    while i < len(toks):
        t = toks[i]
        if t in 'MLC':
            cmd = t; i += 1; continue
        if cmd == 'M':
            cur = (float(toks[i]), float(toks[i+1])); pts.append(cur); i += 2; cmd='L'
        elif cmd == 'L':
            cur = (float(toks[i]), float(toks[i+1])); pts.append(cur); i += 2
        elif cmd == 'C':
            p1=(float(toks[i]),float(toks[i+1])); p2=(float(toks[i+2]),float(toks[i+3])); p3=(float(toks[i+4]),float(toks[i+5]))
            p0=cur
            for k in range(1, 13):
                s=k/12; m=1-s
                x=m**3*p0[0]+3*m*m*s*p1[0]+3*m*s*s*p2[0]+s**3*p3[0]
                y=m**3*p0[1]+3*m*m*s*p1[1]+3*m*s*s*p2[1]+s**3*p3[1]
                pts.append((x,y))
            cur=p3; i += 6
        else:
            i += 1
    return pts

def segs(pts):
    return [(pts[i], pts[i+1]) for i in range(len(pts)-1)]

def inter(a, b):
    (x1,y1),(x2,y2) = a; (x3,y3),(x4,y4) = b
    d = (x2-x1)*(y4-y3) - (y2-y1)*(x4-x3)
    if abs(d) < 1e-12: return None
    t = ((x3-x1)*(y4-y3) - (y3-y1)*(x4-x3)) / d
    u = ((x3-x1)*(y2-y1) - (y3-y1)*(x2-x1)) / d
    EPS = 1e-6
    if EPS < t < 1-EPS and EPS < u < 1-EPS:
        return (x1+t*(x2-x1), y1+t*(y2-y1))
    return None

def check(path):
    raw = open(path).read()
    wires = [(m.group(1), m.group(2)) for m in
             re.finditer(r'<path class="(wire\w*)" d="([^"]+)"', raw)]
    rects = [(float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
             for m in re.finditer(r'<rect class="card [^"]*" x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"', raw)]
    poly = [(c, segs(parse_path(d))) for c, d in wires]

    print(f"\n== {path.split('/')[-1]}: {len(poly)} wires, {len(rects)} cards ==")
    bad = 0
    for (ca, sa), (cb, sb) in itertools.combinations(poly, 2):
        hits = [p for x in sa for y in sb if (p := inter(x, y))]
        if hits:
            bad += 1
            print(f"  CROSS {ca} x {cb} at " + ", ".join(f"({h[0]:.0f},{h[1]:.0f})" for h in hits[:3]))
    # wire cutting through a card interior (allow 6px margin so edge-anchored arrows are fine)
    for cls, ss in poly:
        for (x, y, w, h) in rects:
            for (p, q) in ss:
                for pt in (p, q):
                    if x+6 < pt[0] < x+w-6 and y+6 < pt[1] < y+h-6:
                        bad += 1
                        print(f"  THROUGH-CARD {cls} at ({pt[0]:.0f},{pt[1]:.0f}) inside rect({x:.0f},{y:.0f},{w:.0f},{h:.0f})")
                        break
                else: continue
                break
    print("  CLEAN — no wire crossings, no wire through a card" if bad == 0 else f"  {bad} issue(s)")
    return bad

targets = sys.argv[1:]
if not targets:
    docs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
    targets = [os.path.join(docs, f) for f in ("mlops-lifecycle.svg", "llmops-lifecycle.svg")]

total = sum(check(p) for p in targets)
print("\nAll layout checks passed!" if total == 0 else f"\n{total} layout issue(s) found.")
sys.exit(1 if total else 0)
