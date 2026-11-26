"""Common print style and symbol labels of the publication figures.

Figures are rendered at their final print width so that LaTeX does not scale them. The widths follow the
page geometry of the Elsevier cas-dc class: 210 mm paper width, 18.1 mm side margins, 18 pt column separation.
"""

from pathlib import Path

import matplotlib.pyplot as plt

MM = 1 / 25.4
TEXT_WIDTH = (210.0 - 2 * 18.1) * MM                    # 173.8 mm
COLUMN_WIDTH = (210.0 - 2 * 18.1 - 18 * 25.4 / 72.27) / 2 * MM  # 83.7 mm

PRINT_STYLE = {
    'font.family': 'serif',
    'font.serif': ['STIX Two Text', 'STIXGeneral', 'Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.0,
    'lines.markersize': 3,
    'legend.handlelength': 1.5,
    'pdf.fonttype': 42,
}

# Symbols of the manuscript, keyed by test bench signal (without unit), and the unit shown in brackets.
SYMBOLS = {
    'P.sys': (r'$P_\mathrm{sys}$', 'kW'),
    'eta_sys': (r'$\eta_\mathrm{sys}$', '%'),
    'T_S': (r'$T_\mathrm{S}$', '°C'),
    'p.Si.C': (r'$p_\mathrm{Si,C}$', 'bar'),
    'U.S.AveCell': (r'$U_\mathrm{c}$', 'V'),
    'P.S.el': (r'$P_\mathrm{FC}$', 'kW'),
    'P.CMP.CV': (r'$P_\mathrm{CMP}$', 'kW'),
    'P.ARB.CV': (r'$P_\mathrm{ARB}$', 'kW'),
    'P.FU': (r'$P_\mathrm{CP}$', 'W'),
    'RH.Si.C': (r'$\varphi_\mathrm{Si,C}$', '%'),
    'T.Si.C': (r'$T_\mathrm{Si,C}$', '°C'),
    'T.Si.CL': (r'$T_\mathrm{Si,Cl}$', '°C'),
    'T.So.CL': (r'$T_\mathrm{So,Cl}$', '°C'),
    'p.So.C': (r'$p_\mathrm{So,C}$', 'bar'),
    'pD.S.C': (r'$\Delta p_\mathrm{C}$', 'mbar'),
    'F.Si.Cmp.C': (r'$\dot{m}_\mathrm{C}$', 'kg/h'),
    'I.S.set': (r'$I_\mathrm{set}$', 'A'),
    'p.Si.A.set': (r'$p_\mathrm{A,set}$', 'bar'),
    'p.Si.C.set': (r'$p_\mathrm{C,set}$', 'bar'),
    'C_Stoic': (r'$\lambda_\mathrm{C,set}$', '-'),
    'v.ARB.req': (r'$\omega_\mathrm{ARB,set}$', '1/min'),
    'SS.Delta.T': (r'$\Delta T_\mathrm{Cl,set}$', 'K'),
    'SS.T.Stack': (r'$T_\mathrm{Cl,set}$', '°C'),
    'DPT.Si.C.set': (r'$T_\mathrm{DP,C,set}$', '°C'),
}


def _key(column: str) -> str:
    return column.split(' [')[0]


def symbol(column: str) -> str:
    """Return the manuscript symbol of a signal, or the signal name if it has none."""
    return SYMBOLS.get(_key(column), (_key(column), ''))[0]


def label(column: str, prefix: str = '') -> str:
    """Return 'prefix symbol in unit' of a signal, without unit for dimensionless quantities."""
    sym, unit = SYMBOLS.get(_key(column), (_key(column), column.split(' [')[-1].rstrip(']')))
    return f'{prefix}{sym}' if unit in ('-', '') else f'{prefix}{sym} in {unit}'


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save the figure at its set size as PNG (600 dpi) and as vector PDF next to it."""
    path = Path(path)
    fig.savefig(path.with_suffix('.png'), dpi=600)
    fig.savefig(path.with_suffix('.pdf'), metadata={'CreationDate': None})  # no time stamp, reproducible file
