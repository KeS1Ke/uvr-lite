"""桌面界面的视觉令牌与 Qt 样式。

Qt 样式表不支持 oklch() 与 var()。颜色只在下方令牌里写一次，
样式字符串引用这些名字。数值是对应 OKLCH 的 sRGB 十六进制。
"""

from pathlib import Path

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication, QWidget

_RES = Path(__file__).resolve().parent / "resources"

# --color-paper      oklch(16% 0.012 60)
# --color-paper-2    oklch(21% 0.014 60)
# --color-paper-3    oklch(13% 0.010 60)
# --color-ink        oklch(93% 0.012 80)
# --color-ink-2      oklch(74% 0.022 70)
# --color-rule       oklch(34% 0.016 60)
# --color-accent     oklch(78% 0.120 75)
# --color-accent-2   oklch(84% 0.110 80)   hover
# --color-accent-3   oklch(68% 0.120 70)   pressed
# --color-accent-ink oklch(22% 0.030 60)
# --color-focus      oklch(78% 0.120 75)
TOKENS = {
    "paper": "#1C1916",
    "paper_2": "#2A251F",
    "paper_3": "#161310",
    "paper_hover": "#342E28",
    "ink": "#F4EFE6",
    "ink_2": "#B7AA9A",
    "ink_disabled": "#7A7268",
    "rule": "#4A433B",
    "accent": "#E6B15A",
    "accent_hover": "#F0C27A",
    "accent_pressed": "#C9953E",
    "accent_ink": "#2C2418",
    "accent_wash": "#3A2E1C",
    "focus": "#E6B15A",
}


def _qss() -> str:
    t = TOKENS
    chevron = (_RES / "chevron.svg").as_posix()
    check = (_RES / "check.svg").as_posix()
    return f"""
    QMainWindow {{
        background: {t["paper"]};
    }}
    QWidget#root {{
        background: {t["paper"]};
        color: {t["ink"]};
    }}
    QLabel {{
        color: {t["ink"]};
        background: transparent;
    }}
    QLabel#status {{
        color: {t["ink_2"]};
    }}
    QLabel#dropHint {{
        color: {t["ink_2"]};
        background: {t["paper_3"]};
        border-radius: 8px;
        padding: 24px;
    }}

    QGroupBox {{
        background: {t["paper_2"]};
        color: {t["ink_2"]};
        border: none;
        border-radius: 10px;
        padding: 8px 0 0 0;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: padding;
        subcontrol-position: top left;
        left: 12px;
        top: 0;
        padding: 0;
        color: {t["ink_2"]};
        background: transparent;
    }}
    QGroupBox#files[dragging="true"] {{
        background: {t["accent_wash"]};
    }}

    QFrame#banner {{
        background: {t["accent_wash"]};
        border: none;
        border-radius: 8px;
    }}
    QFrame#banner QLabel {{
        color: {t["ink"]};
    }}

    QLineEdit, QComboBox, QSpinBox {{
        background: {t["paper_3"]};
        color: {t["ink"]};
        border: 2px solid {t["paper_3"]};
        border-radius: 6px;
        padding: 1px 8px;
        selection-background-color: {t["accent_wash"]};
        selection-color: {t["ink"]};
    }}
    QLineEdit:hover, QComboBox:hover, QSpinBox:hover {{
        background: {t["paper"]};
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{
        border-color: {t["focus"]};
    }}
    QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{
        color: {t["ink_disabled"]};
        background: {t["paper_2"]};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 28px;
    }}
    QComboBox::down-arrow {{
        image: url({chevron});
        width: 12px;
        height: 8px;
    }}
    QComboBox QAbstractItemView {{
        background: {t["paper_2"]};
        color: {t["ink"]};
        border: 2px solid {t["rule"]};
        selection-background-color: {t["accent_wash"]};
        selection-color: {t["ink"]};
        outline: 0;
    }}

    QListWidget {{
        background: {t["paper_3"]};
        color: {t["ink"]};
        border: none;
        border-radius: 8px;
        padding: 4px;
        outline: 0;
    }}
    QListWidget::item {{
        padding: 5px 8px;
        border-radius: 4px;
    }}
    QListWidget::item:hover {{
        background: {t["paper_hover"]};
    }}
    QListWidget::item:selected {{
        background: {t["accent_wash"]};
        color: {t["ink"]};
    }}

    QCheckBox {{
        color: {t["ink"]};
        spacing: 8px;
        background: transparent;
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border-radius: 4px;
        background: {t["paper_3"]};
        border: 2px solid {t["rule"]};
    }}
    QCheckBox::indicator:hover {{
        border-color: {t["ink_2"]};
    }}
    QCheckBox::indicator:checked {{
        background: {t["accent"]};
        border-color: {t["accent"]};
        image: url({check});
    }}
    QCheckBox::indicator:focus {{
        border-color: {t["focus"]};
    }}
    QCheckBox:disabled {{
        color: {t["ink_disabled"]};
    }}

    QPushButton {{
        background: {t["paper_3"]};
        color: {t["ink"]};
        border: 2px solid transparent;
        border-radius: 8px;
        padding: 4px 12px;
    }}
    QPushButton:hover {{
        background: {t["paper_hover"]};
    }}
    QPushButton:pressed {{
        background: {t["paper"]};
    }}
    QPushButton:focus {{
        border-color: {t["focus"]};
    }}
    QPushButton:disabled {{
        color: {t["ink_disabled"]};
        background: {t["paper_3"]};
    }}
    QPushButton#primary {{
        background: {t["accent"]};
        color: {t["accent_ink"]};
        font-weight: 600;
        padding: 5px 18px;
    }}
    QPushButton#primary:hover {{
        background: {t["accent_hover"]};
    }}
    QPushButton#primary:pressed {{
        background: {t["accent_pressed"]};
    }}
    QPushButton#primary:disabled {{
        background: {t["rule"]};
        color: {t["ink_disabled"]};
    }}
    QPushButton#quiet {{
        background: transparent;
        color: {t["ink_2"]};
    }}
    QPushButton#quiet:hover {{
        background: {t["paper_2"]};
        color: {t["ink"]};
    }}
    QPushButton#quiet:pressed {{
        background: {t["paper_3"]};
    }}
    QPushButton#quiet:disabled {{
        color: {t["ink_disabled"]};
        background: transparent;
    }}

    QProgressBar {{
        background: {t["paper_3"]};
        color: {t["ink"]};
        border: none;
        border-radius: 5px;
        min-height: 14px;
        max-height: 14px;
        text-align: center;
    }}
    QProgressBar::chunk {{
        background: {t["accent"]};
        border-radius: 5px;
    }}
    QProgressBar:disabled {{
        color: {t["ink_disabled"]};
    }}
    QProgressBar:disabled::chunk {{
        background: {t["rule"]};
    }}
    """


def _palette() -> QPalette:
    t = TOKENS
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(t["paper"]))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(t["ink"]))
    pal.setColor(QPalette.ColorRole.Base, QColor(t["paper_3"]))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(t["paper_2"]))
    pal.setColor(QPalette.ColorRole.Text, QColor(t["ink"]))
    pal.setColor(QPalette.ColorRole.Button, QColor(t["paper_2"]))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(t["ink"]))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(t["accent_wash"]))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(t["ink"]))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(t["ink_2"]))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(t["paper_2"]))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(t["ink"]))
    return pal


def apply(app: QApplication, window: QWidget) -> None:
    """Fusion 暗色底 + 主窗口样式。文件对话框仍走系统原生窗口。"""
    app.setStyle("Fusion")
    app.setPalette(_palette())
    font = QFont()
    font.setFamilies(["Microsoft YaHei UI", "Segoe UI"])
    font.setPointSize(9)
    app.setFont(font)
    window.setStyleSheet(_qss())
