"""Colours, fonts and the notation of the GUI.

Font sizes are in logical pixels (CustomTkinter scales them with the window). The smallest text is 21 px in a window of
1,200 px: a screenshot of the full window placed at 6.29 in keeps every text at 7.9 pt or more (the paper figure minimum
is 7.85 pt printed). Subscripts are drawn at 75 % of their symbol, as mathtext does in the paper figures.
State colours: the palette of the paper figures (figs_manual/labels.json), one colour per performance state.
"""
BG = '#F4F6F9'
PANEL = '#FFFFFF'
BORDER = '#D5DAE1'
HEAD = '#1F3A5F'           # table header fill
TEXT = '#1B1F24'
MUTED = '#5A6472'
PASS = '#2E7D32'
FAIL = '#C62828'
WARN = '#B26A00'
FAMILY = 'Arial'

SIZE_TITLE = 27
SIZE_HEAD = 23
SIZE_TEXT = 21
SUB_RATIO = 0.75
WINDOW = (1200, 1000)

STATES = ['IO', 'LS', 'CP', 'C']
STATE_NAMES = {'IO': 'Immediate Occupancy', 'LS': 'Life Safety', 'CP': 'Collapse Prevention', 'C': 'Collapse'}
STATE_COLORS = {'IO': '#1F5FA8', 'LS': '#2A9D5C', 'CP': '#D07A00', 'C': '#B31B1B'}
HAZARD_COLORS = {'DD-2': '#1B1B1B', 'DD-1': '#7B4FA0'}      # curves and target markers (not state colours)

# symbol: (base, subscript, unit, base italic, subscript italic); displacements in mm in the GUI
SYMBOLS = {
    'n_story': ('N', 'story', '', True, False), 'n_bay': ('N', 'bay', '', True, False),
    'L_bay': ('L', 'bay', 'm', True, False), 'h1': ('h', '1', 'm', True, False),
    'h_typ': ('h', 'typ', 'm', True, False), 'Htot': ('H', 'tot', 'm', True, False),
    'B_tot': ('B', 'tot', 'm', True, False), 'G_line': ('G', '', 'kN/m', True, False),
    'Q_line': ('Q', '', 'kN/m', True, False), 'M_building': ('M', '', 't', True, False),
    'W': ('W', '', 'kN', True, False),
    'd_col': ('d', 'column', 'mm', True, False), 'bf_col': ('b', 'column', 'mm', True, False),
    'tf_col': ('t', 'f,column', 'mm', True, False), 'tw_col': ('t', 'w,column', 'mm', True, False),
    'd_beam': ('d', 'beam', 'mm', True, False), 'bf_beam': ('b', 'beam', 'mm', True, False),
    'tf_beam': ('t', 'f,beam', 'mm', True, False), 'tw_beam': ('t', 'w,beam', 'mm', True, False),
    'Vy': ('V', 'y', 'kN', True, True), 'dy': ('δ', 'y', 'mm', True, True), 'Vmax': ('V', 'max', 'kN', True, False),
    'dmax': ('δ', 'max', 'mm', True, False), 'V_at_dt': ('V', 'δt', 'kN', True, True), 'dt': ('δ', 't', 'mm', True, True),
    'du': ('δ', 'u', 'mm', True, True), 'T1': ('T', '1', 's', True, False),
    'theta_t': ('θ', 't', '', True, True), 'DCR_delta': ('DCR', 'δ', '', False, True), 'mu_d': ('μ', 'd', '', True, True),
    'V_at_dt_over_Vmax': ('r', 'V', '', True, True), 'Vy_over_W_Sa': ('r', 'W', '', True, True),
}
DISPLAY_SCALE = {'dy': 1000.0, 'dmax': 1000.0, 'dt': 1000.0, 'du': 1000.0}     # m -> mm
DECIMALS = {'Vy': 0, 'Vmax': 0, 'V_at_dt': 0, 'dy': 1, 'dmax': 1, 'dt': 1, 'du': 1, 'T1': 3,
            'theta_t': 4, 'DCR_delta': 3, 'mu_d': 2, 'V_at_dt_over_Vmax': 3, 'Vy_over_W_Sa': 3}
NOT_DEFINED = 'not defined'


def font(size=SIZE_TEXT, weight='normal'):
    import customtkinter as ctk
    return ctk.CTkFont(family=FAMILY, size=size, weight=weight)


def fmt(t: str, v: float) -> str:
    return f'{v * DISPLAY_SCALE.get(t, 1.0):,.{DECIMALS[t]}f}'
