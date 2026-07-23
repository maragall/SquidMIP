"""shadcn/ui-inspired dark design system for SquidHCS, expressed as Qt Style Sheets.

This module translates the shadcn/Tailwind design language (neutral slate palette,
4/8/12/16 spacing scale, 6-8 px radii, 1 px subtle borders, restrained accent,
muted secondary text) into QSS for the PyQt5 app. It is a REFINEMENT of the palette
already in ``squidmip/_qtstyle.py`` (BG #070a0f, border #232b3a, accent #58a6ff),
not a new look: every token below either matches or sits adjacent to a colour the
app already paints, so applying it does not clash with unthemed widgets.

Standalone by design: imports nothing from this package, so it can never
participate in an import cycle and can be adopted one constant at a time.

Integration is find/replace: the compat aliases at the bottom (BTN_QSS, CARD_QSS,
COMBO_QSS, TABS_QSS, BG) carry the same names ``_qtstyle.py`` exports today, so
``_viewer.py`` can later point its ``_BTN_QSS = _qtstyle.BTN_QSS`` lines at
``_theme`` instead with no call-site edits.
"""

from __future__ import annotations

# ======================================================================================
# Tokens
# ======================================================================================

# ---- colour: neutral slate ramp, dark. Matches the app's existing chrome. -----------
BG              = "#070a0f"   # app window background (same as _qtstyle.BG)
SURFACE         = "#0b0e14"   # panes, bars, input fields
SURFACE_RAISED  = "#131824"   # buttons, cards, popups: one step up from SURFACE
SURFACE_HOVER   = "#1a2233"   # hover wash on raised surfaces
SURFACE_PRESSED = "#0d1420"   # pressed: sinks back toward SURFACE
BORDER          = "#232b3a"   # the 1 px hairline used everywhere
BORDER_STRONG   = "#3a4557"   # emphasized border (hover on inputs, dividers that must read)
TEXT            = "#e6edf3"   # primary copy
TEXT_MUTED      = "#8b98ad"   # secondary copy, captions, readouts
TEXT_FAINT      = "#57606a"   # section labels, disabled text
ACCENT          = "#58a6ff"   # the ONE accent: selection, focus, links
ACCENT_HOVER    = "#79b8ff"   # accent hovered
ACCENT_DIM      = "#1c2b44"   # translucent-looking accent wash on solid dark
DANGER          = "#ef4444"   # destructive / failed
SUCCESS         = "#3fb950"   # confirmed / done

# ---- spacing scale (px): Tailwind-style 4-step grid ---------------------------------
SPACE_1, SPACE_2, SPACE_3, SPACE_4 = 4, 8, 12, 16

# ---- radii (px): 8 for interactive chrome, 6 for compact fields, 10 for cards -------
RADIUS_SM = 6
RADIUS    = 8
RADIUS_LG = 10

# ---- type ---------------------------------------------------------------------------
FONT_STACK = "'Inter','SF Pro Text','Segoe UI','Helvetica Neue',sans-serif"
MONO_STACK = "'SF Mono','Menlo','Consolas',monospace"
FONT_SIZE       = 13   # body / controls
FONT_SIZE_SM    = 12   # captions, readouts
FONT_SIZE_LABEL = 10   # uppercase section labels (pairs with letter-spacing)


# ======================================================================================
# Per-widget QSS builders
# ======================================================================================

def base_qss() -> str:
    """App-level base: background, default text, tooltip. Everything else layers on this."""
    return (
        f"QWidget{{background:{BG};color:{TEXT};"
        f"font-family:{FONT_STACK};font-size:{FONT_SIZE}px;}}"
        f"QLabel{{background:transparent;color:{TEXT};}}"
        f"QToolTip{{background:{SURFACE_RAISED};color:{TEXT};"
        f"border:1px solid {BORDER};border-radius:{RADIUS_SM}px;"
        f"padding:{SPACE_1}px {SPACE_2}px;font-size:{FONT_SIZE_SM}px;}}"
    )


def button_qss() -> str:
    """Buttons. Default is the app's standard raised button; two variants are opt-in:

    - primary (accent-filled):  ``btn.setProperty("variant", "primary")``
    - ghost (borderless, quiet): ``btn.setProperty("variant", "ghost")``

    Set the property BEFORE the widget is shown, or call style().unpolish/polish after.
    """
    return (
        # default: raised surface, subtle border, accent border on hover (existing behaviour)
        f"QPushButton{{background:{SURFACE_RAISED};color:{TEXT};"
        f"border:1px solid {BORDER};border-radius:{RADIUS}px;"
        f"padding:7px {SPACE_3}px;font-weight:700;}}"
        f"QPushButton:hover{{border-color:{ACCENT};background:{SURFACE_HOVER};}}"
        f"QPushButton:pressed{{background:{SURFACE_PRESSED};}}"
        f"QPushButton:focus{{border-color:{ACCENT};outline:none;}}"
        f"QPushButton:disabled{{color:{TEXT_FAINT};border-color:#1a2130;}}"
        # primary: filled accent, dark text for contrast on the light-blue fill
        f"QPushButton[variant=\"primary\"]{{background:{ACCENT};color:#0b0e14;"
        f"border:1px solid {ACCENT};}}"
        f"QPushButton[variant=\"primary\"]:hover{{background:{ACCENT_HOVER};"
        f"border-color:{ACCENT_HOVER};}}"
        f"QPushButton[variant=\"primary\"]:pressed{{background:#4993e6;}}"
        f"QPushButton[variant=\"primary\"]:disabled{{background:{SURFACE_RAISED};"
        f"color:{TEXT_FAINT};border-color:{BORDER};}}"
        # ghost: no chrome until hovered
        f"QPushButton[variant=\"ghost\"]{{background:transparent;color:{TEXT_MUTED};"
        f"border:1px solid transparent;font-weight:600;}}"
        f"QPushButton[variant=\"ghost\"]:hover{{background:{SURFACE_HOVER};color:{TEXT};}}"
        f"QPushButton[variant=\"ghost\"]:pressed{{background:{SURFACE_PRESSED};}}"
    )


def card_qss() -> str:
    """Operator 'card' buttons (Process pane): left-aligned, larger radius, quiet until hover."""
    return (
        f"QPushButton[role=\"card\"], QPushButton#card{{background:{SURFACE_PRESSED};"
        f"color:{TEXT};border:1px solid {BORDER};border-radius:{RADIUS_LG}px;"
        f"text-align:left;padding:9px 13px;font-size:{FONT_SIZE}px;font-weight:400;}}"
        f"QPushButton[role=\"card\"]:hover, QPushButton#card:hover"
        f"{{border-color:{ACCENT};background:#111a2b;}}"
        f"QPushButton[role=\"card\"]:disabled, QPushButton#card:disabled"
        f"{{color:{TEXT_FAINT};border-color:#1a2130;}}"
    )


def combo_qss() -> str:
    """QComboBox and its popup list."""
    return (
        f"QComboBox{{background:{SURFACE_PRESSED};color:{TEXT};"
        f"border:1px solid {BORDER};border-radius:{RADIUS_SM}px;"
        f"padding:5px {SPACE_2}px;}}"
        f"QComboBox:hover{{border-color:{BORDER_STRONG};}}"
        f"QComboBox:focus{{border-color:{ACCENT};}}"
        f"QComboBox:disabled{{color:{TEXT_FAINT};}}"
        f"QComboBox::drop-down{{border:none;width:22px;}}"
        f"QComboBox QAbstractItemView{{background:{SURFACE_PRESSED};color:{TEXT};"
        f"border:1px solid {BORDER};selection-background-color:{ACCENT_DIM};"
        f"selection-color:{TEXT};outline:none;}}"
    )


def input_qss() -> str:
    """QLineEdit and QSpinBox/QDoubleSpinBox: same field language as the combo."""
    return (
        f"QLineEdit,QSpinBox,QDoubleSpinBox{{background:{SURFACE_PRESSED};color:{TEXT};"
        f"border:1px solid {BORDER};border-radius:{RADIUS_SM}px;"
        f"padding:5px {SPACE_2}px;selection-background-color:{ACCENT_DIM};}}"
        f"QLineEdit:hover,QSpinBox:hover,QDoubleSpinBox:hover{{border-color:{BORDER_STRONG};}}"
        f"QLineEdit:focus,QSpinBox:focus,QDoubleSpinBox:focus{{border-color:{ACCENT};}}"
        f"QLineEdit:disabled,QSpinBox:disabled,QDoubleSpinBox:disabled{{color:{TEXT_FAINT};}}"
        f"QSpinBox::up-button,QSpinBox::down-button,"
        f"QDoubleSpinBox::up-button,QDoubleSpinBox::down-button"
        f"{{background:{SURFACE_RAISED};border:none;width:16px;}}"
        f"QSpinBox::up-button:hover,QSpinBox::down-button:hover,"
        f"QDoubleSpinBox::up-button:hover,QDoubleSpinBox::down-button:hover"
        f"{{background:{SURFACE_HOVER};}}"
    )


def section_label_qss() -> str:
    """Uppercase micro-headers, applied via objectName: ``lbl.setObjectName("section")``.

    Mirrors the inline style the app repeats today:
    color #57606a, 10 px, 800 weight, 1.5 px letter-spacing.
    """
    return (
        f"QLabel#section{{color:{TEXT_FAINT};font-size:{FONT_SIZE_LABEL}px;"
        f"font-weight:800;letter-spacing:1.5px;padding-top:{SPACE_2}px;}}"
        f"QLabel#muted{{color:{TEXT_MUTED};font-size:{FONT_SIZE_SM}px;}}"
    )


def scroll_qss() -> str:
    """Frameless scroll areas plus thin modern scrollbars (6 px, no arrow buttons)."""
    return (
        f"QScrollArea{{border:none;background:transparent;}}"
        f"QScrollBar:vertical{{background:transparent;width:10px;margin:2px;}}"
        f"QScrollBar::handle:vertical{{background:{BORDER};border-radius:3px;"
        f"min-height:24px;margin:0 2px;}}"
        f"QScrollBar::handle:vertical:hover{{background:{BORDER_STRONG};}}"
        f"QScrollBar:horizontal{{background:transparent;height:10px;margin:2px;}}"
        f"QScrollBar::handle:horizontal{{background:{BORDER};border-radius:3px;"
        f"min-width:24px;margin:2px 0;}}"
        f"QScrollBar::handle:horizontal:hover{{background:{BORDER_STRONG};}}"
        f"QScrollBar::add-line,QScrollBar::sub-line{{width:0;height:0;}}"
        f"QScrollBar::add-page,QScrollBar::sub-page{{background:transparent;}}"
    )


def slider_qss() -> str:
    """Thin groove, accent handle: matches the ndviewer chrome already themed dark."""
    return (
        f"QSlider::groove:horizontal{{background:{BORDER};height:4px;border-radius:2px;}}"
        f"QSlider::sub-page:horizontal{{background:{ACCENT};border-radius:2px;}}"
        f"QSlider::handle:horizontal{{background:{ACCENT};width:12px;margin:-5px 0;"
        f"border-radius:6px;}}"
        f"QSlider::handle:horizontal:hover{{background:{ACCENT_HOVER};}}"
        f"QSlider::groove:vertical{{background:{BORDER};width:4px;border-radius:2px;}}"
        f"QSlider::handle:vertical{{background:{ACCENT};height:12px;margin:0 -5px;"
        f"border-radius:6px;}}"
    )


def tabs_qss() -> str:
    """Tab strip for a pane's OWN tabs. Underline-selected, shadcn-style, kept dark.

    Refines the existing TABS_DARK: drops the white pane outline for the standard
    1 px BORDER hairline, marks selection with a 2 px accent underline.
    """
    return (
        f"QTabWidget{{background:{BG};}}"
        f"QTabWidget::pane{{border:1px solid {BORDER};background:{BG};top:-1px;}}"
        f"QTabBar{{background:{BG};}}"
        f"QTabBar::tab{{background:{SURFACE};color:{TEXT_MUTED};"
        f"padding:6px 13px;border:1px solid {BORDER};border-bottom:none;"
        f"border-top-left-radius:{RADIUS_SM}px;border-top-right-radius:{RADIUS_SM}px;"
        f"margin-right:2px;font-weight:700;font-size:{FONT_SIZE_SM}px;}}"
        f"QTabBar::tab:hover{{color:{TEXT};}}"
        f"QTabBar::tab:selected{{background:#131b2b;color:{TEXT};"
        f"border-bottom:2px solid {ACCENT};}}"
    )


def checkbox_qss() -> str:
    """Checkbox: bordered box, accent fill when checked."""
    return (
        f"QCheckBox{{color:{TEXT};spacing:7px;background:transparent;}}"
        f"QCheckBox:disabled{{color:{TEXT_FAINT};}}"
        f"QCheckBox::indicator{{width:14px;height:14px;border:1px solid {BORDER_STRONG};"
        f"border-radius:3px;background:{SURFACE_PRESSED};}}"
        f"QCheckBox::indicator:hover{{border-color:{ACCENT};}}"
        f"QCheckBox::indicator:checked{{background:{ACCENT};border-color:{ACCENT};}}"
        f"QCheckBox::indicator:disabled{{border-color:{BORDER};background:{SURFACE};}}"
    )


def groupbox_qss() -> str:
    """QGroupBox as a bordered card with a floating faint title, shadcn card style."""
    return (
        f"QGroupBox{{background:{SURFACE};border:1px solid {BORDER};"
        f"border-radius:{RADIUS_LG}px;margin-top:{SPACE_3}px;"
        f"padding:{SPACE_3}px {SPACE_3}px {SPACE_2}px;}}"
        f"QGroupBox::title{{subcontrol-origin:margin;left:{SPACE_3}px;"
        f"padding:0 {SPACE_1}px;color:{TEXT_FAINT};font-size:{FONT_SIZE_LABEL}px;"
        f"font-weight:800;letter-spacing:1.5px;}}"
    )


def menu_qss() -> str:
    """Context menus: raised surface, accent-wash selection."""
    return (
        f"QMenu{{background:{SURFACE_PRESSED};color:{TEXT};border:1px solid {BORDER};"
        f"border-radius:{RADIUS_SM}px;padding:{SPACE_1}px;}}"
        f"QMenu::item{{padding:7px 18px;border-radius:{RADIUS_SM - 2}px;}}"
        f"QMenu::item:selected{{background:{ACCENT_DIM};}}"
        f"QMenu::item:disabled{{color:{TEXT_FAINT};}}"
        f"QMenu::separator{{height:1px;background:{BORDER};margin:{SPACE_1}px {SPACE_2}px;}}"
    )


# ======================================================================================
# Assembly
# ======================================================================================

def build_stylesheet() -> str:
    """The full theme as one QSS string, ready for an app or a widget subtree."""
    return "".join((
        base_qss(),
        button_qss(),
        card_qss(),
        combo_qss(),
        input_qss(),
        section_label_qss(),
        scroll_qss(),
        slider_qss(),
        tabs_qss(),
        checkbox_qss(),
        groupbox_qss(),
        menu_qss(),
    ))


def apply_theme(widget_or_app) -> None:
    """Apply the theme to a QApplication (app-wide) or any QWidget (scoped subtree).

    Note: the app deliberately scopes its dark chrome (see _qtstyle.dark_palette,
    which warns that app-wide theming bled into the embedded ndviewer swatches).
    Prefer applying to top-level windows or panes rather than the QApplication
    until that interaction is re-verified.
    """
    widget_or_app.setStyleSheet(build_stylesheet())


# ======================================================================================
# Compat aliases: same names _qtstyle.py exports today, so _viewer.py's
#   _BTN_QSS = _qtstyle.BTN_QSS   lines can be repointed at _theme with no other edits.
# These are plain per-widget stylesheets (safe for setStyleSheet on a single widget),
# built from the same tokens as the app-level sheet above.
# ======================================================================================

BTN_QSS = (
    f"QPushButton{{background:{SURFACE_RAISED};color:{TEXT};"
    f"border:1px solid {BORDER};border-radius:{RADIUS}px;"
    f"padding:7px {SPACE_3}px;font-weight:700;}}"
    f"QPushButton:hover{{border-color:{ACCENT};background:{SURFACE_HOVER};}}"
    f"QPushButton:pressed{{background:{SURFACE_PRESSED};}}"
    f"QPushButton:disabled{{color:{TEXT_FAINT};}}"
)

CARD_QSS = (
    f"QPushButton{{background:{SURFACE_PRESSED};color:{TEXT};"
    f"border:1px solid {BORDER};border-radius:{RADIUS_LG}px;"
    f"text-align:left;padding:9px 13px;font-size:{FONT_SIZE}px;}}"
    f"QPushButton:hover{{border-color:{ACCENT};background:#111a2b;}}"
    f"QPushButton:disabled{{color:{TEXT_FAINT};border-color:#1a2130;}}"
)

COMBO_QSS = (
    f"QComboBox{{background:{SURFACE_PRESSED};color:{TEXT};"
    f"border:1px solid {BORDER};border-radius:{RADIUS_SM}px;padding:5px {SPACE_2}px;}}"
    f"QComboBox:hover{{border-color:{BORDER_STRONG};}}"
    f"QComboBox:focus{{border-color:{ACCENT};}}"
    f"QComboBox QAbstractItemView{{background:{SURFACE_PRESSED};color:{TEXT};"
    f"border:1px solid {BORDER};selection-background-color:{ACCENT_DIM};"
    f"selection-color:{TEXT};outline:none;}}"
)

TABS_QSS = tabs_qss()

CHECK_QSS = checkbox_qss()
