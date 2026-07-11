#!/usr/bin/env python3
"""
Extract thermodynamic cycle reaction energies from CSV files and check cycle closure.
"""

# --- repo path bootstrap (auto) ---
from pathlib import Path as _Path
import sys as _sys
_REPO_CAND = _Path(__file__).resolve().parent
while _REPO_CAND != _REPO_CAND.parent and not (_REPO_CAND / "software.yaml").exists():
    _REPO_CAND = _REPO_CAND.parent
if not (_REPO_CAND / "software.yaml").exists():
    raise RuntimeError("Could not locate repo root (software.yaml)")
REPO_ROOT = _REPO_CAND
TOOLS_DIR = REPO_ROOT / "tools"
_sys.path.insert(0, str(TOOLS_DIR))
try:
    from paths import load_software as _load_software
    _SW = _load_software()
except Exception:
    _SW = {}
# --- end bootstrap ---

import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DE_CSV = os.path.join(SCRIPT_DIR, 'dE.csv')
LIG_MIM_CSV = str(REPO_ROOT / "runs" / "lig_exchange/MIM_PBE_STRUC/dE.csv")
LIG_MIMH_CSV = str(REPO_ROOT / "runs" / "lig_exchange/MIMH_PBE_STRUC/dE.csv")

# Reference (CCSD) for errors
REF_PHASE = 'SP_init'
REF_RUN = 'run24'
REF_METHOD = 'DLPNO-CCSD(T) RIJCOSX def2-TZVPPD def2-TZVPPD/C AutoAux'
REF_BSSE = 'yes'

# Selected method/phase groups to compare against CCSD
TARGET_GROUPS = [
    ('SP_init', 'run24', 'DLPNO-CCSD(T) RIJCOSX def2-TZVPPD def2-TZVPPD/C AutoAux', 'yes'),
    ('SP_init', 'run23', 'B3LYP def2-TZVPPD D3BJ AutoAux', 'yes'),
    ('SP_opt', 'run8', 'mDFTB3D3/3ob_prime', 'no'),
    ('SP_opt', 'run17', 'DFTB3 DampingH', 'no'),
    ('SP_opt', 'run27', 'g-xTB', 'no'),

]


def filter_row(df, phase, run, method, bsse=None):
    """Filter by phase, run, method and optionally BSSE."""
    mask = (df['Phase'] == phase) & (df['Run'] == run) & (df['Method'] == method)
    if bsse is not None and 'BSSE' in df.columns:
        mask &= (df['BSSE'] == bsse)
    return df[mask]


def get_value_by_filters(df, col, phase, run, method, bsse=None):
    """Get single value by explicit filters, or None if not found."""
    sub = filter_row(df, phase, run, method, bsse)
    if sub.empty or col not in sub.columns:
        return None
    return sub[col].iloc[0]


def build_method_reaction_row(reaction_label, phase, run, method, bsse,
                              dE_method, dE_ref):
    """Build one reaction row dict using CCSD reference (round to 3 decimals)."""

    def r3(x):
        return round(x, 3) if x is not None else None

    error = None
    if dE_method is not None and dE_ref is not None:
        error = dE_method - dE_ref

    return {
        'Phase': phase,
        'Run': run,
        'Method': method,
        'BSSE': bsse,
        'Reaction': reaction_label,
        'dE_method': r3(dE_method),
        'dE_CCSD_ref': r3(dE_ref),
        'Error_vs_CCSD': r3(error),
    }


def main():
    # Load CSVs
    if not os.path.exists(DE_CSV):
        print(f"ERROR: {DE_CSV} not found")
        return
    if not os.path.exists(LIG_MIM_CSV):
        print(f"ERROR: {LIG_MIM_CSV} not found")
        return
    if not os.path.exists(LIG_MIMH_CSV):
        print(f"ERROR: {LIG_MIMH_CSV} not found")
        return

    df_de = pd.read_csv(DE_CSV)
    df_mim = pd.read_csv(LIG_MIM_CSV)
    df_mimh = pd.read_csv(LIG_MIMH_CSV)

    # Reaction definitions (first four are the basic thermodynamic cycle)
    reaction_configs = [
        ('[Zn·MeOH4]²⁺ + MImH → [Zn·MImH·MeOH3]²⁺ + MeOH', 'mimh', 'dE_0_1'),
        ('[Zn·MImH·MeOH3]²⁺ → [Zn·MIm·MeOH3]⁺ + H⁺', 'de', 'dE_1Zn_1MImH_3MeOH_1Zn_1MIm_3MeOH_H'),
        ('[Zn·MeOH4]²⁺ + MIm⁻ → [Zn·MIm·MeOH3]⁺ + MeOH', 'mim', 'dE_0_1'),
        ('MImH → MIm⁻ + H⁺', 'de', 'dE_MImH_MIm_H'),
        # Additional three reactions used in the extended cycle diagram
        ('MImH + H₂O → MIm⁻ + H₃O⁺', 'de', 'dE_MImH_H2O_MIm_H3O'),
        ('[Zn·MImH·MeOH3]²⁺ + H₂O → [Zn·MIm·MeOH3]⁺ + H₃O⁺', 'de', 'dE_1Zn_1MImH_3MeOH_H2O_1Zn_1MIm_3MeOH_H3O'),
        ('H₂O + H⁺ → H₃O⁺', 'de', 'dE_H2O_H_H3O'),
    ]

    # Map name to actual DataFrame
    df_map = {'de': df_de, 'mim': df_mim, 'mimh': df_mimh}

    rows = []

    for phase, run, method, bsse in TARGET_GROUPS:
        for label, df_key, col in reaction_configs:
            df = df_map[df_key]
            dE_method = get_value_by_filters(df, col, phase, run, method, bsse)
            dE_ref = get_value_by_filters(df, col, REF_PHASE, REF_RUN, REF_METHOD, REF_BSSE)
            if dE_method is None:
                # Skip if this combination does not exist in the CSV
                continue
            rows.append(
                build_method_reaction_row(
                    label, phase, run, method, bsse, dE_method, dE_ref
                )
            )

    out_df = pd.DataFrame(rows)
    out_file = os.path.join(SCRIPT_DIR, 'cycle_reactions_vs_ccsd.csv')
    # utf-8-sig adds BOM so Excel (Windows) opens the file without garbled text
    out_df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"Results saved to {out_file}\n")
    print("dE in kcal/mol, errors relative to CCSD (DLPNO-CCSD(T) RIJCOSX def2-TZVPPD def2-TZVPPD/C AutoAux, SP_init run24, BSSE=yes):\n")
    print(out_df.to_string(index=False))

    # Also generate a thermodynamic cycle diagram for a selected method
    plot_cycle(out_df)

    return out_df


def plot_cycle(out_df: pd.DataFrame) -> None:
    """
    Plot a thermodynamic cycle diagram similar to the provided figure.

    For每个反应，把所有 TARGET_GROUPS (除 CCSD 参考本身) 的结果都列在方程下面，
    每行包含方法名、dE(method)、dE(CCSD) 和 error。
    """

    if len(TARGET_GROUPS) < 1:
        print("TARGET_GROUPS is empty; cannot build diagram.")
        return

    # CCSD 参考（应该和 REF_* 一致）
    ref_phase, ref_run, ref_method, ref_bsse = TARGET_GROUPS[0]
    # 其它要展示的方法（相对于 CCSD 的误差已经存在于 out_df）
    method_groups = TARGET_GROUPS[1:]

    def short_method_name(method: str) -> str:
        """生成简短方法名，用于图中标注。"""
        if not isinstance(method, str):
            return str(method)
        # 特殊处理：把 "DFTB3 DampingH" 显示成 "DFTB3DH"
        if method.startswith("DFTB3 DampingH"):
            return "DFTB3DH"
        token = method.split()[0]
        # 处理带斜杠的 DFTB 名字，比如 mDFTB3D3/3ob_prime
        return token.split('/')[0]

    def build_text_block(reaction_label: str):
        """
        构造这个反应对应的多行文字：
        第一行是反应方程，
        下面每一行是 "<method_short>: dE_method (dE_CCSD) error:err".
        """
        # 参考行
        r_mask = (
            (out_df['Phase'] == ref_phase)
            & (out_df['Run'] == ref_run)
            & (out_df['Method'] == ref_method)
            & (out_df['BSSE'] == ref_bsse)
            & (out_df['Reaction'] == reaction_label)
        )
        if not r_mask.any():
            return None
        r_row = out_df[r_mask].iloc[0]
        dE_ref = r_row['dE_CCSD_ref']

        # 第一行是反应方程；第二行给出 CCSD(T) 数值，格式与其它方法完全一致，
        # 为了保证 dE 和 error 列严格对齐，仍保留 'error = 0.000'，并在末尾标注 (ref)
        ref_line = f"{'CCSD(T)':10s}  dE = {dE_ref:8.3f}   error = {0.0:8.3f} (ref)"
        lines = [
            reaction_label,
            ref_line,
        ]

        for phase, run, method, bsse in method_groups:
            m_mask = (
                (out_df['Phase'] == phase)
                & (out_df['Run'] == run)
                & (out_df['Method'] == method)
                & (out_df['BSSE'] == bsse)
                & (out_df['Reaction'] == reaction_label)
            )
            if not m_mask.any():
                continue
            m_row = out_df[m_mask].iloc[0]
            dE_method = m_row['dE_method']
            err = m_row['Error_vs_CCSD']
            label = short_method_name(method)
            # 使用定宽格式，使不同方法的数字在等宽字体下纵向对齐
            line = f"{label:10s}  dE = {dE_method:8.3f}   error = {err:8.3f}"
            lines.append(line)

        if len(lines) == 1:
            # 只有方程，没有方法行，就不画
            return None
        return "\n".join(lines)

    # Coordinates for arrows – spread out vertically to reduce text overlap
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convenience to draw an annotated arrow
    def arrow_with_label(label, x1, y1, x2, y2, text_y_offset=0.6):
        text_block = build_text_block(label)
        if text_block is None:
            return
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", linewidth=1.2, color="black"),
        )
        x_mid = (x1 + x2) / 2.0
        y_mid = (y1 + y2) / 2.0
        ax.text(
            x_mid,
            y_mid - text_y_offset,
            text_block,
            ha="center",
            va="top",
            fontsize=8,
            fontfamily="monospace",
        )

    # Layout constants
    x_left = 0.0
    x_right = 6.0
    # move left text block further left so it no longer overlaps the left arrow
    x_text_left = -10.0   # left-aligned text block start (left of left arrow)
    x_text_right = 7.8   # text to the right of right arrow

    y_top = 2.5
    y_vert_top = 2.0
    y_vert_bottom = -2.7
    y_bottom = -3.6  # move bottom arrow further down to avoid text overlap

    # Main square cycle (top / bottom / left / right – H+ transfer)
    # Top horizontal arrow with text ABOVE the arrow
    ax.annotate(
        "",
        xy=(x_right, y_top),
        xytext=(x_left, y_top),
        arrowprops=dict(arrowstyle="->", linewidth=1.2, color="black"),
    )
    text_block_top = build_text_block(
        '[Zn·MeOH4]²⁺ + MImH → [Zn·MImH·MeOH3]²⁺ + MeOH'
    )
    if text_block_top is not None:
        ax.text(
            (x_left + x_right) / 2.0,
            y_top + 0.4,
            text_block_top,
            ha="center",
            va="bottom",
            fontsize=8,
            fontfamily="monospace",
        )
    # Left vertical arrow only (no label); labels placed to the LEFT of arrow
    ax.annotate(
        "",
        xy=(x_left, y_vert_bottom),
        xytext=(x_left, y_vert_top),
        arrowprops=dict(arrowstyle="->", linewidth=1.2, color="black"),
    )
    text_block_left = build_text_block('MImH → MIm⁻ + H⁺')
    if text_block_left is not None:
        ax.text(
            x_text_left,
            0.0,
            text_block_left,
            ha="left",
            va="center",
            fontsize=8,
            fontfamily="monospace",
        )
    text_block_left_water = build_text_block('MImH + H₂O → MIm⁻ + H₃O⁺')
    if text_block_left_water is not None:
        ax.text(
            x_text_left,
            -0.9,
            text_block_left_water,
            ha="left",
            va="top",
            fontsize=8,
            fontfamily="monospace",
        )

    arrow_with_label(
        '[Zn·MeOH4]²⁺ + MIm⁻ → [Zn·MIm·MeOH3]⁺ + MeOH',
        x1=x_left,
        y1=y_bottom,
        x2=x_right,
        y2=y_bottom,
    )
    # Right vertical arrow only; labels placed to the RIGHT of arrow
    ax.annotate(
        "",
        xy=(x_right, y_vert_bottom),
        xytext=(x_right, y_vert_top),
        arrowprops=dict(arrowstyle="->", linewidth=1.2, color="black"),
    )
    text_block_right = build_text_block(
        '[Zn·MImH·MeOH3]²⁺ → [Zn·MIm·MeOH3]⁺ + H⁺'
    )
    if text_block_right is not None:
        ax.text(
            x_text_right,
            0.0,
            text_block_right,
            ha="left",
            va="center",
            fontsize=8,
            fontfamily="monospace",
        )
    text_block_right_water = build_text_block(
        '[Zn·MImH·MeOH3]²⁺ + H₂O → [Zn·MIm·MeOH3]⁺ + H₃O⁺'
    )
    if text_block_right_water is not None:
        ax.text(
            x_text_right,
            -0.9,
            text_block_right_water,
            ha="left",
            va="top",
            fontsize=8,
            fontfamily="monospace",
        )

    # Bottom water reaction: text in bottom-left corner, no arrow
    text_block_bottom_water = build_text_block('H₂O + H⁺ → H₃O⁺')
    if text_block_bottom_water is not None:
        ax.text(
            x_text_left,
            -5.0,
            text_block_bottom_water,
            ha="left",
            va="top",
            fontsize=8,
            fontfamily="monospace",
        )

    ax.set_axis_off()
    ax.set_xlim(-9.0, 10.0)
    ax.set_ylim(-5.7, 3.0)
    plt.tight_layout()

    out_png = os.path.join(SCRIPT_DIR, 'cycle_diagram.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"\nCycle diagram saved to {out_png}")


if __name__ == '__main__':
    main()

