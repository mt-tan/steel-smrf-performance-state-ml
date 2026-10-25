"""Performance State Predictor for Steel Moment Frames: desktop entry point.

    python gui/app.py

Left: frame inputs (geometry, loads, steel grade, soil class, column and beam profiles), the derived model inputs
(read-only) and the validity checks (range check against the database of the paper, closed-form screens of the dataset
generator, the distance to the nearest database frame (core/domain.py), the checks that need an analysis listed as not
checked). Right: for DD-2 and DD-1 (the hazard level is not
an input) the predicted TBEC 2018 performance state with its four class probabilities, the eight predicted pushover
quantities and the five demand ratios, a plot of the predicted characteristic points joined for each hazard level by a
piecewise-linear curve in the order of displacement (core/curve.py; not an analysis result), warnings and the validity
note. CSV export of the results. The predictions are those of the two-stage model of the paper (assets/models/,
gate G1 in model_metadata.json); no structural analysis is run.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
from core.curve import LEGEND as CURVE_LEGEND, TARGET_BEYOND_ULTIMATE, curve_points, target_on_line  # noqa: E402
from core.domain import nearest_distance  # noqa: E402
from core.frame import HAZARDS, FrameInput, spec  # noqa: E402
from core.predictor import RATIOS, STATES, TARGETS, TwoStageModel, predict_frame  # noqa: E402
from core.screens import NOT_CHECKED, NOTE, range_check, screens  # noqa: E402
from ui import theme as T  # noqa: E402

APP_TITLE = 'Performance State Predictor for Steel Moment Frames'
APP_VERSION = '1.1'
PAD = 10
ROWS = TARGETS + RATIOS
UNITS_SI = {'Vy': 'kN', 'dy': 'm', 'Vmax': 'kN', 'dmax': 'm', 'V_at_dt': 'kN', 'dt': 'm', 'du': 'm', 'T1': 's'}
NOT_DEFINED_MSG = 'not defined (target beyond the ultimate displacement)'


class Symbol(ctk.CTkFrame):
    """Symbol of the paper notation: base (italic unless upright), subscript at 75 %, unit."""

    def __init__(self, master, key, size=T.SIZE_TEXT, unit=True, weight='normal', color=T.TEXT, fg='transparent'):
        super().__init__(master, fg_color=fg)
        base, sub, u, base_it, sub_it = T.SYMBOLS[key]
        sub_size = round(size * T.SUB_RATIO)
        ctk.CTkLabel(self, text=base, text_color=color, fg_color='transparent', height=size + 7,
                     font=ctk.CTkFont(family=T.FAMILY, size=size, weight=weight,
                                      slant='italic' if base_it else 'roman')).grid(row=0, column=0, sticky='s')
        col = 1
        if sub:
            ctk.CTkLabel(self, text=sub, text_color=color, fg_color='transparent', height=sub_size + 4,
                         font=ctk.CTkFont(family=T.FAMILY, size=sub_size, weight=weight,
                                          slant='italic' if sub_it else 'roman')).grid(
                row=0, column=col, sticky='s', pady=(0, 1))
            col += 1
        if unit and u:
            ctk.CTkLabel(self, text=f' [{u}]', text_color=color, fg_color='transparent', height=size + 7,
                         font=ctk.CTkFont(family=T.FAMILY, size=size, weight=weight)).grid(row=0, column=col, sticky='s')


class Rich(ctk.CTkFrame):
    """One line of text in pieces: (text, style) with style n (normal), i (italic), s (subscript), si (italic
    subscript), b (bold)."""

    def __init__(self, master, parts, size=T.SIZE_TEXT, color=T.TEXT, fg='transparent'):
        super().__init__(master, fg_color=fg)
        sub = round(size * T.SUB_RATIO)
        for k, (txt, st) in enumerate(parts):
            sz = sub if st in ('s', 'si') else size
            fnt = ctk.CTkFont(family=T.FAMILY, size=sz, weight='bold' if st == 'b' else 'normal',
                              slant='italic' if st in ('i', 'si') else 'roman')
            ctk.CTkLabel(self, text=txt, font=fnt, text_color=color, fg_color='transparent',
                         height=(sz + 4) if st in ('s', 'si') else (sz + 7)).grid(
                row=0, column=k, sticky='s', pady=(0, 1) if st in ('s', 'si') else 0)


class App(ctk.CTk):
    def __init__(self, hidden: bool = False):
        ctk.set_appearance_mode('light')
        ctk.set_default_color_theme('blue')
        super().__init__()
        if hidden:
            self.withdraw()
        self.title(f'{APP_TITLE} (TBEC 2018)')
        self.geometry(f'{T.WINDOW[0]}x{T.WINDOW[1]}')
        self.resizable(False, False)
        self.configure(fg_color=T.BG)
        self.model = TwoStageModel()
        self.spec = spec()
        self.result = None
        self.vars = {}
        self._build()
        self.load_default()

    # ------------------------------------------------------------------ layout
    def _build(self):
        f_text, f_head = T.font(T.SIZE_TEXT), T.font(T.SIZE_HEAD, 'bold')
        head = ctk.CTkFrame(self, fg_color=T.BG)
        head.pack(fill='x', padx=PAD, pady=(6, 2))
        ctk.CTkLabel(head, text=APP_TITLE, font=T.font(T.SIZE_TITLE, 'bold'), text_color=T.HEAD, height=32).grid(
            row=0, column=0, sticky='w')
        btns = ctk.CTkFrame(head, fg_color=T.BG)
        btns.grid(row=0, column=1, sticky='e')
        ctk.CTkButton(btns, text='Predict', font=f_text, width=100, height=32, command=self.update_all).pack(
            side='left')
        ctk.CTkButton(btns, text='Default example', font=f_text, width=170, height=32, fg_color=T.MUTED,
                      command=self.load_default).pack(side='left', padx=8)
        ctk.CTkButton(btns, text='Export CSV', font=f_text, width=130, height=32, fg_color=T.MUTED,
                      command=self.export_dialog).pack(side='left')
        ctk.CTkLabel(head, text='TBEC 2018 performance state at DD-2 and DD-1 from the two-stage model of the paper',
                     font=f_text, text_color=T.MUTED, height=26).grid(row=1, column=0, columnspan=2, sticky='w')
        head.grid_columnconfigure(0, weight=1)
        body = ctk.CTkFrame(self, fg_color=T.BG)
        body.pack(fill='both', expand=True, padx=PAD, pady=(0, PAD))
        left = ctk.CTkFrame(body, fg_color=T.PANEL, border_color=T.BORDER, border_width=1, width=405)
        left.pack(side='left', fill='y', padx=(0, PAD))
        left.pack_propagate(False)
        right = ctk.CTkFrame(body, fg_color=T.PANEL, border_color=T.BORDER, border_width=1)
        right.pack(side='left', fill='both', expand=True)

        # ---- inputs
        grid = ctk.CTkFrame(left, fg_color='transparent')
        grid.pack(fill='x', padx=PAD, pady=(PAD, 0))
        ctk.CTkLabel(grid, text='Frame', font=f_head, text_color=T.HEAD, height=28).grid(row=0, column=0, columnspan=2,
                                                                                        sticky='w')
        rows = [('n_story', 'menu', [str(v) for v in self.spec['story_counts']]),
                ('n_bay', 'menu', [str(v) for v in self.spec['bay_counts']]),
                ('L_bay', 'entry', None), ('h1', 'entry', None), ('h_typ', 'entry', None),
                ('G_line', 'entry', None), ('Q_line', 'entry', None),
                ('grade', 'menu', self.spec['grades']), ('soil', 'menu', self.spec['soils']),
                ('col_profile', 'menu', self.spec['column_profiles']),
                ('beam_profile', 'menu', self.spec['beam_profiles'])]
        names = {'grade': 'Steel grade', 'soil': 'Soil class', 'col_profile': 'Column', 'beam_profile': 'Beam'}
        for i, (key, kind, values) in enumerate(rows, start=1):
            if key in names:
                ctk.CTkLabel(grid, text=names[key], font=f_text, text_color=T.TEXT, height=28).grid(
                    row=i, column=0, sticky='w')
            else:
                Symbol(grid, key).grid(row=i, column=0, sticky='w')
            var = ctk.StringVar()
            self.vars[key] = var
            if kind == 'menu':
                w = ctk.CTkOptionMenu(grid, values=values, variable=var, font=f_text, dropdown_font=f_text, width=170,
                                      height=28, command=lambda _v: self.update_all())
            else:
                w = ctk.CTkEntry(grid, textvariable=var, font=f_text, width=170, height=28)
                w.bind('<Return>', lambda _e: self.update_all())
                w.bind('<FocusOut>', lambda _e: self.update_all())
            w.grid(row=i, column=1, sticky='e', padx=(8, 0), pady=1)
        grid.grid_columnconfigure(0, weight=1)

        # ---- derived inputs (read-only)
        der = ctk.CTkFrame(left, fg_color='transparent')
        der.pack(fill='x', padx=PAD, pady=(4, 0))
        ctk.CTkLabel(der, text='Derived inputs', font=f_head, text_color=T.HEAD, height=28).grid(
            row=0, column=0, columnspan=4, sticky='w')
        self.der_labels = {}
        for key, r, c in (('Htot', 1, 0), ('B_tot', 1, 2), ('M_building', 2, 0), ('W', 2, 2)):
            Symbol(der, key).grid(row=r, column=c, sticky='w', padx=(0 if c == 0 else 14, 0))
            lab = ctk.CTkLabel(der, text='', font=f_text, text_color=T.TEXT, height=27)
            lab.grid(row=r, column=c + 1, sticky='e', padx=(6, 0))
            self.der_labels[key] = lab
        Rich(der, [('d', 'i'), (' / ', 'n'), ('b', 'i'), (' / ', 'n'), ('t', 'i'), ('f', 'si'), (' / ', 'n'),
                   ('t', 'i'), ('w', 'si'), (' [mm]', 'n')], color=T.MUTED).grid(row=3, column=0, columnspan=4,
                                                                                 sticky='w')
        for i, (lab_txt, key) in enumerate([('Column', 'col'), ('Beam', 'beam')]):
            ctk.CTkLabel(der, text=lab_txt, font=f_text, text_color=T.TEXT, height=27).grid(row=4 + i, column=0,
                                                                                           sticky='w')
            lab = ctk.CTkLabel(der, text='', font=f_text, text_color=T.TEXT, height=27)
            lab.grid(row=4 + i, column=1, columnspan=3, sticky='e', padx=(6, 0))
            self.der_labels[key] = lab
        der.grid_columnconfigure(1, weight=1)
        der.grid_columnconfigure(3, weight=1)

        # ---- validity
        val = ctk.CTkFrame(left, fg_color='transparent')
        val.pack(fill='x', padx=PAD, pady=(4, 0))
        ctk.CTkLabel(val, text='Validity checks', font=f_head, text_color=T.HEAD, height=28).grid(
            row=0, column=0, columnspan=2, sticky='w')
        self.range_label = ctk.CTkLabel(val, text='', font=f_text, text_color=T.TEXT, height=27, justify='left',
                                        wraplength=380, anchor='w')
        self.range_label.grid(row=1, column=0, columnspan=2, sticky='w')
        self.screen_labels = []
        short = [[('Ductility, column', 'n')], [('Ductility, beam', 'n')],
                 [('Axial load ', 'n'), ('P', 'i'), ('u', 'si'), ('/', 'n'), ('P', 'i'), ('ye', 's')],
                 [('Gravity ', 'n'), ('P', 'i'), ('g', 'si'), ('/', 'n'), ('P', 'i'), ('c', 's'), (', ground', 'n')],
                 [('Beam gravity flexure', 'n')], [('Clear lengths', 'n')], [('Total height', 'n')]]
        for i, parts in enumerate(short):
            Rich(val, parts).grid(row=2 + i, column=0, sticky='w')
            lab = ctk.CTkLabel(val, text='', font=T.font(T.SIZE_TEXT, 'bold'), height=25)
            lab.grid(row=2 + i, column=1, sticky='e')
            self.screen_labels.append(lab)
        Rich(val, [('Nearest database frame', 'n')]).grid(row=2 + len(short), column=0, sticky='w')
        self.domain_label = ctk.CTkLabel(val, text='', font=T.font(T.SIZE_TEXT, 'bold'), height=25)
        self.domain_label.grid(row=2 + len(short), column=1, sticky='e')
        n0 = 3 + len(short)
        ctk.CTkLabel(val, text='Not checked (need an analysis):', font=f_text, text_color=T.MUTED, height=27).grid(
            row=n0, column=0, columnspan=2, sticky='w')
        not_checked = [[('first-mode mass ratio', 'n')],
                       [('column ', 'n'), ('P', 'i'), ('/', 'n'), ('P', 'i'), ('c', 's'), (' at acceptance', 'n')]]
        for i, parts in enumerate(not_checked):
            Rich(val, parts, color=T.MUTED).grid(row=n0 + 1 + i, column=0, sticky='w', padx=(14, 0))
            ctk.CTkLabel(val, text='not checked', font=f_text, text_color=T.MUTED, height=26).grid(
                row=n0 + 1 + i, column=1, sticky='e')
        val.grid_columnconfigure(0, weight=1)

        # ---- results: table and class probabilities
        top = ctk.CTkFrame(right, fg_color='transparent')
        top.pack(fill='x', padx=PAD, pady=(PAD, 0))
        tab = ctk.CTkFrame(top, fg_color='transparent')
        tab.pack(side='left', anchor='n')
        widths = [132, 112, 112]
        for j, (h, wd) in enumerate(zip(['Quantity', 'DD-2', 'DD-1'], widths)):
            ctk.CTkLabel(tab, text=h, font=T.font(T.SIZE_TEXT, 'bold'), text_color='white', fg_color=T.HEAD,
                         width=wd, height=30, anchor='w' if j == 0 else 'e', padx=6).grid(row=0, column=j, sticky='ew')
        ctk.CTkLabel(tab, text='State', font=T.font(T.SIZE_TEXT, 'bold'), text_color=T.TEXT, fg_color='#E9EDF2',
                     width=widths[0], height=30, anchor='w', padx=6).grid(row=1, column=0, sticky='ew')
        self.cells, self.state_cells = {}, {}
        for j, h in ((1, 'DD-2'), (2, 'DD-1')):
            lab = ctk.CTkLabel(tab, text='', font=T.font(T.SIZE_TEXT, 'bold'), text_color=T.TEXT, fg_color='#E9EDF2',
                               width=widths[j], height=30, anchor='e', padx=6)
            lab.grid(row=1, column=j, sticky='ew')
            self.state_cells[h] = lab
        for i, t in enumerate(ROWS, start=2):
            bg = '#F2F4F7' if i % 2 == 1 else T.PANEL
            cell = ctk.CTkFrame(tab, fg_color=bg, corner_radius=0, height=26, width=widths[0])
            cell.grid(row=i, column=0, sticky='nsew')
            Symbol(cell, t, fg=bg).pack(side='left', padx=(4, 0))
            for j, h in ((1, 'DD-2'), (2, 'DD-1')):
                lab = ctk.CTkLabel(tab, text='', font=f_text, text_color=T.TEXT, fg_color=bg, width=widths[j],
                                   height=26, anchor='e', padx=6)
                lab.grid(row=i, column=j, sticky='ew')
                self.cells[(t, h)] = lab
        from customtkinter.windows.widgets.scaling import ScalingTracker
        s = ScalingTracker.get_widget_scaling(self)
        self.fig_p = Figure(figsize=(380 / 72, 422 / 72), dpi=72 * s, facecolor='white')
        self.ax_p = [self.fig_p.add_subplot(2, 1, k + 1) for k in range(2)]
        self.canvas_p = FigureCanvasTkAgg(self.fig_p, master=top)
        self.canvas_p.get_tk_widget().pack(side='left', anchor='n', padx=(PAD, 0))

        # ---- plot of the characteristic points
        self.fig = Figure(figsize=(746 / 72, 262 / 72), dpi=72 * s, facecolor='white')
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(padx=PAD, pady=(6, 0))
        self.warn_label = ctk.CTkLabel(right, text='', font=f_text, text_color=T.WARN, justify='left',
                                       wraplength=740, anchor='w')
        self.warn_label.pack(fill='x', padx=PAD, pady=(4, 0))
        self.note_label = ctk.CTkLabel(right, text=NOTE, font=f_text, text_color=T.MUTED, justify='left',
                                       wraplength=740, anchor='w')
        self.note_label.pack(fill='x', padx=PAD, pady=(4, 0))
        self.status = ctk.CTkLabel(right, text='', font=f_text, text_color=T.MUTED, anchor='w', height=24)
        self.status.pack(fill='x', padx=PAD, pady=(0, 4))

    # ------------------------------------------------------------------ logic
    def load_default(self):
        d = self.spec['default']
        for k in ('n_story', 'n_bay'):
            self.vars[k].set(str(d[k]))
        for k in ('L_bay', 'h1', 'h_typ'):
            self.vars[k].set(f'{d[k]:.3f}')
        for k in ('G_line', 'Q_line'):
            self.vars[k].set(f'{d[k]:.2f}')
        for k in ('grade', 'soil', 'col_profile', 'beam_profile'):
            self.vars[k].set(d[k])
        self.update_all()

    def frame_input(self) -> FrameInput:
        v = {k: self.vars[k].get().strip() for k in self.vars}
        return FrameInput(n_story=int(v['n_story']), n_bay=int(v['n_bay']), L_bay=float(v['L_bay']),
                          h1=float(v['h1']), h_typ=float(v['h_typ']), G_line=float(v['G_line']),
                          Q_line=float(v['Q_line']), grade=v['grade'], soil=v['soil'], col_profile=v['col_profile'],
                          beam_profile=v['beam_profile'])

    def warnings(self, res) -> list:
        """Every warning of the frame: predictor, curve, V(delta_t) not defined, ranges, screens."""
        warn = list(res.warnings)
        for h in HAZARDS:
            for w in curve_points(res.values[h])[2]:
                warn.append(f'{h}: {w}.')
                if w == TARGET_BEYOND_ULTIMATE:
                    warn.append(f'{h}: V(δt) {NOT_DEFINED_MSG}.')
        outside = [r for r in self.range_rows if not r['inside']]
        if outside:
            warn.append('Inputs outside the ranges of Table 2 of the paper: '
                        + ', '.join(T.SYMBOLS[r['key']][0] + T.SYMBOLS[r['key']][1] for r in outside) + '.')
        if not all(s['passed'] for s in self.screen_rows):
            warn.append('A screen of the database fails: the frame is outside the validated domain.')
        if not self.domain['inside']:
            warn.append(f"The combination of inputs is farther from the database frames than 99 percent of the test "
                        f"frames of the paper (distance {self.domain['distance']:.2f}, limit {self.domain['limit']:.2f}).")
        return warn

    def update_all(self):
        try:
            f = self.frame_input()
        except ValueError:
            self.status.configure(text='Enter numbers for widths, heights and loads.', text_color=T.FAIL)
            return
        res = predict_frame(self.model, f)
        self.result = res
        self.screen_rows = screens(f)
        self.range_rows = range_check(f, self.spec['realized_ranges'], res.derived)
        self.domain = nearest_distance(f, res.derived)
        der = res.derived
        self.der_labels['Htot'].configure(text=f"{der['Htot']:.2f}")
        self.der_labels['B_tot'].configure(text=f"{der['B_tot']:.2f}")
        self.der_labels['M_building'].configure(text=f"{der['M_building']:.1f}")
        self.der_labels['W'].configure(text=f"{res.extra['DD-2']['W']:,.0f}")
        self.der_labels['col'].configure(text=f"{f.col_profile}: {der['d_col']:.0f} / {der['bf_col']:.0f} / "
                                              f"{der['tf_col']:g} / {der['tw_col']:g}")
        self.der_labels['beam'].configure(text=f"{f.beam_profile}: {der['d_beam']:.0f} / {der['bf_beam']:.0f} / "
                                               f"{der['tf_beam']:g} / {der['tw_beam']:g}")
        outside = [r for r in self.range_rows if not r['inside']]
        if outside:
            self.range_label.configure(text='Outside the Table 2 ranges: ' + ', '.join(
                T.SYMBOLS[r['key']][0] + T.SYMBOLS[r['key']][1] for r in outside), text_color=T.FAIL)
        else:
            self.range_label.configure(text='All inputs inside the Table 2 ranges', text_color=T.PASS)
        for lab, s in zip(self.screen_labels, self.screen_rows):
            txt = s['value'] if s['key'].startswith('duct') and s['passed'] else ('pass' if s['passed'] else 'fail')
            lab.configure(text=txt, text_color=T.PASS if s['passed'] else T.FAIL)
        dm = self.domain
        self.domain_label.configure(text=f"{dm['distance']:.2f} {'≤' if dm['inside'] else '>'} {dm['limit']:.2f}",
                                    text_color=T.PASS if dm['inside'] else T.FAIL)
        for h in HAZARDS:
            st = res.state[h]
            self.state_cells[h].configure(text=st, text_color=T.STATE_COLORS[st])
            v = res.values[h]
            for t in TARGETS:
                txt = T.fmt(t, v[t])
                if t == 'V_at_dt' and not target_on_line(v) and v['dt'] > v['du']:
                    txt = T.NOT_DEFINED
                self.cells[(t, h)].configure(text=txt)
            for k in RATIOS:
                self.cells[(k, h)].configure(text=T.fmt(k, res.ratios[h][k]))
        warn = self.warnings(res)
        self.warn_label.configure(text=' '.join(warn) if warn else 'No warning for this frame.',
                                  text_color=T.WARN if warn else T.PASS)
        self.status.configure(text='')
        self._plot_prob(res)
        self._plot(res)

    def _plot_prob(self, res):
        sz = T.SIZE_TEXT
        for ax, h in zip(self.ax_p, HAZARDS):
            ax.clear()
            p = [res.prob[h][s] for s in STATES]
            y = list(range(len(STATES)))[::-1]
            ax.barh(y, p, color=[T.STATE_COLORS[s] for s in STATES], height=0.68)
            for yi, pi in zip(y, p):
                ax.text(min(pi, 1.0) + 0.02, yi, f'{pi:.2f}', va='center', ha='left', fontsize=sz)
            ax.set_yticks(y)
            ax.set_yticklabels(STATES, fontsize=sz)
            ax.set_xlim(0, 1.28)
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_xticklabels(['0', '0.5', '1'], fontsize=sz)
            ax.tick_params(axis='both', labelsize=sz, length=3)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            st = res.state[h]
            ax.set_title(f'{h}: {T.STATE_NAMES[st]}', fontsize=sz, color=T.STATE_COLORS[st], fontweight='bold',
                         loc='left', pad=4)
        self.ax_p[1].set_xlabel('Probability', fontsize=sz)
        self.fig_p.subplots_adjust(left=0.13, right=0.97, top=0.92, bottom=0.12, hspace=0.62)
        self.canvas_p.draw()

    def _plot(self, res):
        ax = self.ax
        ax.clear()
        sz = T.SIZE_TEXT
        mk = {'yield': 'o', 'peak': 's', 'ultimate': 'D'}
        self.curve_lines, self.target_markers = {}, {}
        for h, ls in (('DD-2', '--'), ('DD-1', ':')):
            pts, _, _ = curve_points(res.values[h])
            c = T.HAZARD_COLORS[h]
            self.curve_lines[h], = ax.plot([p[0] * 1000 for p in pts], [p[1] for p in pts], ls, color=c, lw=1.8,
                                           zorder=1)
        for h, filled, ms, z in (('DD-1', False, 15, 2), ('DD-2', True, 9, 3)):
            v, c = res.values[h], T.HAZARD_COLORS[h]
            for name, d, V in (('yield', v['dy'], v['Vy']), ('peak', v['dmax'], v['Vmax']),
                               ('ultimate', v['du'], 0.8 * v['Vmax'])):
                ax.plot(d * 1000, V, mk[name], ms=ms, mfc=c if filled else 'none', mec=c, mew=1.6, zorder=z)
        for h, m in (('DD-2', '^'), ('DD-1', 'v')):
            v, c = res.values[h], T.HAZARD_COLORS[h]
            self.target_markers[h], = ax.plot(v['dt'] * 1000, v['V_at_dt'], m, ms=12, mfc=c, mec=c, zorder=4)
        ax.set_xlabel('Roof displacement [mm]', fontsize=sz)
        ax.set_ylabel('Base shear [kN]', fontsize=sz)
        ax.tick_params(labelsize=sz)
        dmax_all = max(max(res.values[h]['du'], res.values[h]['dt']) for h in HAZARDS) * 1000
        vtop = max(max(res.values[h][t] for t in ('Vmax', 'Vy', 'V_at_dt')) for h in HAZARDS)
        ax.set_xlim(0, 1.06 * dmax_all)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(True, ls='--', lw=0.6, alpha=0.5)
        handles = [Line2D([], [], ls='', marker='o', ms=9, color='0.25', label='yield'),
                   Line2D([], [], ls='', marker='s', ms=9, color='0.25', label='peak'),
                   Line2D([], [], ls='', marker='D', ms=8, color='0.25', label='ultimate'),
                   Line2D([], [], ls='', marker='^', ms=11, color=T.HAZARD_COLORS['DD-2'], label='target DD-2'),
                   Line2D([], [], ls='', marker='v', ms=11, color=T.HAZARD_COLORS['DD-1'], label='target DD-1'),
                   Line2D([], [], ls='', marker='o', ms=13, mfc='none', mec=T.HAZARD_COLORS['DD-1'], mew=1.6,
                          label='DD-1 points')]
        both = tuple(Line2D([], [], ls=ls, lw=1.8, color=T.HAZARD_COLORS[h]) for h, ls in (('DD-2', '--'),
                                                                                           ('DD-1', ':')))
        for old in list(self.fig.legends):
            old.remove()
        self.marker_legend = self.fig.legend(handles=handles, fontsize=sz, loc='upper right', ncol=1, frameon=True,
                                             framealpha=0.95, handletextpad=0.3, borderpad=0.3, labelspacing=0.25,
                                             borderaxespad=0.3)
        self.curve_legend = ax.legend(
            [both], [CURVE_LEGEND.replace(' through', '\nthrough').replace(' (not', '\n(not')],
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)}, fontsize=sz, loc='upper left', frameon=True,
            framealpha=0.95, handlelength=2.6, handletextpad=0.4, borderpad=0.3)
        self._fit_layout(ax, vtop)
        self.canvas.draw()

    def _fit_layout(self, ax, vtop):
        """Axes left of the marker legend (its width is reserved); the y range puts the highest point, padded by half
        the largest marker and 2 px, below the curve legend. The layout is computed last, with the final tick labels,
        and the range is iterated until it no longer changes."""
        r = self.fig.canvas.get_renderer()
        pad = 0.5 * 15 * self.fig.dpi / 72.0 + 2.0
        w = self.marker_legend.get_window_extent(r).width
        ax.set_ylim(0, 1.6 * vtop)
        for _ in range(8):
            self.fig.tight_layout(pad=0.4, rect=(0.0, 0.0, 1.0 - (w + 8.0) / self.fig.bbox.width, 1.0))
            box = self.curve_legend.get_window_extent(r)
            free = max(box.y0 - ax.bbox.y0 - pad, 0.25 * ax.bbox.height)
            top = vtop * ax.bbox.height / free
            if abs(top - ax.get_ylim()[1]) <= 1e-3 * top:
                break
            ax.set_ylim(0, top)

    # ------------------------------------------------------------------ export
    def export_rows(self) -> list:
        res = self.result
        rows = [['section', 'quantity', 'hazard', 'value', 'unit', 'note']]
        for k, v in res.frame.as_dict().items():
            rows.append(['input', k, '', v, '', ''])
        for k, v in res.derived.items():
            rows.append(['derived input', k, '', v, '', ''])
        for h in HAZARDS:
            for k, u in (('W', 'kN'), ('SDS', 'g'), ('SD1', 'g'), ('Sa_T1', 'g')):
                rows.append(['derived input', k, h, res.extra[h][k], u, 'Sa_T1 at the predicted T1' if k == 'Sa_T1'
                             else ''])
        for h in HAZARDS:
            v = res.values[h]
            for t in TARGETS:
                note = ''
                if t == 'V_at_dt' and v['dt'] > v['du']:
                    note = NOT_DEFINED_MSG + '; the value is the model output used by the ratio r_V'
                rows.append(['first stage', t, h, v[t], UNITS_SI[t], note])
            for k in RATIOS:
                rows.append(['demand ratio', k, h, res.ratios[h][k], '', ''])
            for s in STATES:
                rows.append(['class probability', s, h, res.prob[h][s], '', T.STATE_NAMES[s]])
            rows.append(['predicted state', res.state[h], h, T.STATE_NAMES[res.state[h]], '', ''])
        for s in self.screen_rows:
            rows.append(['screen', s['name'], '', 'pass' if s['passed'] else 'fail', s['value'],
                         f"{s['criterion']}; limit {s['limit']}"])
        dm = self.domain
        rows.append(['domain check', 'distance to the nearest database frame', '', dm['distance'],
                     'pass' if dm['inside'] else 'fail',
                     f"standardized frame inputs; limit {dm['limit']} = 99th percentile of the test frames of the paper"])
        for name, crit in NOT_CHECKED:
            rows.append(['screen', name, '', 'not checked', '', crit])
        for r in self.range_rows:
            rows.append(['range check', r['key'], '', r['value'], 'inside' if r['inside'] else 'outside',
                         f"Table 2 range {r['lo']} to {r['hi']}"])
        for w in self.warnings(res):
            rows.append(['warning', w, '', '', '', ''])
        rows.append(['note', NOTE, '', '', '', ''])
        rows.append(['models', 'two-stage model of the paper: first stage the mean of five networks per quantity, '
                     'classifier the soft vote of five networks', '', f'gate {self.model.gate}', '',
                     f'{APP_TITLE} {APP_VERSION}'])
        return rows

    def export_csv(self, path):
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            csv.writer(fh).writerows(self.export_rows())

    def export_dialog(self):
        p = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV', '*.csv')],
                                         initialfile='steel_frame_performance_state.csv')
        if p:
            self.export_csv(p)
            self.status.configure(text='Results written to CSV.', text_color=T.PASS)


if __name__ == '__main__':
    App().mainloop()
