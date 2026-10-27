"""
Modern DAG Dependency Graph Visualizer for DA-RAG.
Generates interactive, lightweight, responsive SVG/HTML diagrams displaying
the target entity and its repository dependency graph relations.
"""

from typing import List, Dict, Any


def render_dag_graph_svg(
    target_name: str,
    packed_items: List[Dict[str, Any]],
    tokens: int = 0,
    n_units: int = 0,
    width: int = 860,
    height: int = 280
) -> str:
    """
    Renders an ultra-modern SVG DAG graph representation of target entity and
    its retrieved dependency nodes, with edge connections and mode badges.
    Ensures zero leading whitespace on any line to prevent Markdown code-block parsing.
    """
    short_target = target_name.split('.')[-1]
    if not short_target:
        short_target = target_name

    # If no items packed, show single isolated node with baseline indicator
    if not packed_items:
        raw_html = f"""<div class="comparison-card comparison-card-dag">
<div class="comparison-header comparison-header-dag">
<div class="comparison-header-title">🌿 DA-RAG · Dependency Graph Context</div>
<span class="token-chip token-chip-dag">{tokens} / 2,048 tok · {n_units} units</span>
</div>
<div style="padding: 30px 20px; text-align: center; color: #94A3B8;">
<div style="font-size: 28px; margin-bottom: 8px;">🚫</div>
<div style="font-weight: 700; font-size: 15px; color: #F1F5F9; letter-spacing: -0.01em;">No Dependency Graph Context Retrieved</div>
<div style="font-size: 13px; margin-top: 6px; color: #94A3B8; max-width: 480px; margin-left: auto; margin-right: auto; line-height: 1.5;">Zero repository context units retrieved. Model operates in isolation.</div>
</div>
</div>"""
        return "\n".join(line.strip() for line in raw_html.splitlines() if line.strip())

    # Layout nodes: Target on the left, dependencies spaced along right
    n_items = min(len(packed_items), 5)  # Show up to 5 key nodes
    svg_h = max(260, 60 + n_items * 55)
    
    target_x, target_y = 130, svg_h // 2
    
    nodes_svg = []
    edges_svg = []
    
    # Target Node with subtle glow
    target_node = f"""<g class="target-node">
<rect x="{target_x - 100}" y="{target_y - 30}" width="200" height="60" rx="12" fill="#0F172A" stroke="#38BDF8" stroke-width="2" filter="url(#glow-cyan)"/>
<circle cx="{target_x - 75}" cy="{target_y - 2}" r="6" fill="#38BDF8"/>
<text x="{target_x - 60}" y="{target_y - 8}" fill="#38BDF8" font-size="10" font-weight="700" letter-spacing="1">QUERY TARGET</text>
<text x="{target_x - 60}" y="{target_y + 14}" fill="#F8FAFC" font-size="13" font-weight="600" font-family="'JetBrains Mono', monospace">{short_target[:18]}</text>
</g>"""
    nodes_svg.append(target_node)

    dep_x = width - 180
    step_y = (svg_h - 60) / max(n_items - 1, 1) if n_items > 1 else 0
    start_y = 50 if n_items > 1 else svg_h // 2

    for i, item in enumerate(packed_items[:n_items]):
        node_y = int(start_y + i * step_y) if n_items > 1 else svg_h // 2
        
        mode = item.get("mode", "full")
        cost = item.get("cost", 0)
        sym = item.get("symbol") or item.get("unit_id", "").split("::")[-1] or f"unit_{i+1}"
        short_sym = sym.split(".")[-1][:22]
        
        rel = item.get("dependency_rel") or "depends_on"
        
        # Color coding
        if mode == "full":
            border_color = "#10B981"
            badge_text = "FULL DEF"
            badge_bg = "rgba(16, 185, 129, 0.2)"
            badge_border = "#10B981"
            badge_color = "#34D399"
        else:
            border_color = "#818CF8"
            badge_text = "STUB"
            badge_bg = "rgba(99, 102, 241, 0.2)"
            badge_border = "#818CF8"
            badge_color = "#A5B4FC"
            
        # Bezier curve edge with gradient
        mid_x = (target_x + dep_x) // 2
        edge_path = f"M {target_x + 100} {target_y} C {mid_x} {target_y}, {mid_x} {node_y}, {dep_x - 120} {node_y}"
        
        edge_svg = f"""<g class="dag-edge">
<path d="{edge_path}" fill="none" stroke="{border_color}" stroke-width="2" stroke-opacity="0.45" stroke-dasharray="5,4"/>
<circle cx="{dep_x - 120}" cy="{node_y}" r="4" fill="{border_color}"/>
<rect x="{mid_x - 40}" y="{(target_y + node_y)//2 - 11}" width="80" height="18" rx="4" fill="#0B0F19" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>
<text x="{mid_x}" y="{(target_y + node_y)//2 + 2}" fill="#94A3B8" font-size="9.5" font-weight="600" text-anchor="middle" font-family="'JetBrains Mono', monospace">{rel[:12]}</text>
</g>"""
        edges_svg.append(edge_svg)
        
        # Dependency Node
        node_svg = f"""<g class="dep-node">
<rect x="{dep_x - 120}" y="{node_y - 24}" width="240" height="48" rx="10" fill="#0F172A" stroke="{border_color}" stroke-width="1.8" filter="drop-shadow(0 4px 8px rgba(0,0,0,0.4))"/>
<rect x="{dep_x - 110}" y="{node_y - 14}" width="54" height="17" rx="4" fill="{badge_bg}" stroke="{badge_border}" stroke-width="1"/>
<text x="{dep_x - 83}" y="{node_y - 2}" fill="{badge_color}" font-size="8.5" font-weight="700" text-anchor="middle">{badge_text}</text>
<text x="{dep_x - 48}" y="{node_y - 1}" fill="#F1F5F9" font-size="12" font-weight="600" font-family="'JetBrains Mono', monospace">{short_sym}</text>
<text x="{dep_x + 105}" y="{node_y + 15}" fill="#64748B" font-size="9.5" font-weight="500" text-anchor="end" font-family="'JetBrains Mono', monospace">{cost} tok</text>
</g>"""
        nodes_svg.append(node_svg)

    svg_content = f"""<div class="comparison-card comparison-card-dag">
<div class="comparison-header comparison-header-dag">
<div class="comparison-header-title">🌿 DA-RAG · Dependency Graph Context</div>
<span class="token-chip token-chip-dag">{tokens} / 2,048 tok · {n_units} units</span>
</div>
<div style="padding: 12px 16px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 10px;">
<div style="display: flex; align-items: center; gap: 8px;">
<span style="font-size: 15px;">🕸️</span>
<span style="font-size: 12px; font-weight: 700; color: #38BDF8; letter-spacing: 0.05em; text-transform: uppercase;">Retrieved AST Dependency Call Graph</span>
</div>
<div style="display: flex; gap: 14px; font-size: 11px;">
<span style="display: inline-flex; align-items: center; gap: 5px; color: #34D399; font-weight: 600;"><span style="width: 7px; height: 7px; background: #10B981; border-radius: 50%;"></span> Full Definition</span>
<span style="display: inline-flex; align-items: center; gap: 5px; color: #A5B4FC; font-weight: 600;"><span style="width: 7px; height: 7px; background: #818CF8; border-radius: 50%;"></span> Interface Stub</span>
<span style="display: inline-flex; align-items: center; gap: 5px; color: #38BDF8; font-weight: 600;"><span style="width: 7px; height: 7px; background: #38BDF8; border-radius: 50%;"></span> Target Entity</span>
</div>
</div>
<svg viewBox="0 0 {width} {svg_h}" width="100%" height="{svg_h}" style="overflow: visible;" xmlns="http://www.w3.org/2000/svg">
<defs>
<filter id="glow-cyan" x="-20%" y="-20%" width="140%" height="140%">
<feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#38BDF8" flood-opacity="0.25"/>
</filter>
<pattern id="dot-grid" width="24" height="24" patternUnits="userSpaceOnUse">
<circle cx="2" cy="2" r="1" fill="rgba(255,255,255,0.06)"/>
</pattern>
</defs>
<rect width="{width}" height="{svg_h}" fill="url(#dot-grid)" rx="8"/>
{"".join(edges_svg)}
{"".join(nodes_svg)}
</svg>
</div>
</div>"""
    # CRITICAL: Strip all leading whitespace from every line so Markdown parser NEVER interprets it as an indented code block
    clean_lines = [line.strip() for line in svg_content.splitlines() if line.strip()]
    return "\n".join(clean_lines)
