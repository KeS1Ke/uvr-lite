"""uvr-lite 桌面界面（PySide6）。

面向非专业用户：拖放音频、音质三档，设备/位深/BigShifts 等收进「更多选项」。
权重是否就绪看注册表 filename（model_file），不要再拼 .ckpt。
"""

import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread, QUrl
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..download import cuda_torch_installed, model_file, repo_root
from ..models import MODEL_REGISTRY
from .files import dedup_paths, is_audio, precheck_audio, scan_audio_files
from .presets import matching_preset, resolve_quality
from .progress import estimate_eta, summary_text
from .worker import CudaTorchWorker, ModelDownloadWorker, SeparationWorker

_ICON = Path(__file__).resolve().parent / "resources" / "uvr-lite.ico"

PHASE_CN = {"decode": "解码", "infer": "推理", "chunk": "推理", "tta": "增强", "write": "写出"}

MODEL_LABELS = {
    "bs_roformer_ep317": "BS-RoFormer ep317（主力，推荐）",
    "mel_band_karaoke": "Mel-Band RoFormer Karaoke（备选）",
}
# 全量安装包含 CPU/CUDA 两套 torch，应用内选择；mps 仅 macOS 无意义故不列出
DEVICE_CHOICES = ["auto", "cpu", "cuda"]
FORMAT_CHOICES = ["auto", "flac", "wav"]

QUALITY_CHOICES = (
    ("fast", "快速"),
    ("standard", "标准"),
    ("high", "高音质"),
    ("custom", "自定义"),
)
QUALITY_HINTS = {
    "fast": "适合批量，大约快一倍。",
    "standard": "默认音质，速度和质量比较均衡。",
    "high": "更慢一点，分离更干净。",
    "custom": "使用「更多选项」里的重叠窗口、BigShifts 和增强。",
}

# 列表项状态前缀（显示在文件名前）
_PREFIX_PENDING = "⏳ "
_PREFIX_OK = "✓ "
_PREFIX_BAD = "✗ "

_ROLE_PATH = Qt.UserRole
_ROLE_OUTPUT = Qt.UserRole + 1

_PRESET_KEYS = {"fast", "standard", "high", "custom"}

APP_QSS = """
#central { background: #16181d; color: #e8e6e3; }
QGroupBox {
  background: #22262e;
  color: #e8e6e3;
  border: 1px solid #343a46;
  border-radius: 8px;
  margin-top: 14px;
  padding: 12px 10px 10px 10px;
  font-weight: 600;
}
QGroupBox::title {
  subcontrol-origin: margin;
  subcontrol-position: top left;
  left: 12px;
  padding: 0 6px;
  color: #e8e6e3;
  background: #16181d;
}
QGroupBox QLabel { font-weight: 400; }
QLabel { color: #e8e6e3; background: transparent; }
QLabel#hint { color: #b7b3ad; }
QLabel#statusLabel { color: #d4d0cb; }
QPushButton {
  background: #2c313c;
  color: #e8e6e3;
  border: 1px solid #3d4452;
  border-radius: 6px;
  padding: 4px 12px;
  min-height: 32px;
}
QPushButton:hover { background: #363d4c; }
QPushButton:disabled { color: #8a8680; background: #1c1f26; }
QPushButton#startButton {
  background: #e0a45c;
  color: #1a140c;
  border: none;
  font-weight: 700;
  min-height: 36px;
  padding: 6px 18px;
}
QPushButton#startButton:hover { background: #e8b56e; }
QPushButton#startButton:disabled { background: #6d5a40; color: #2c241c; }
QPushButton#advancedToggle {
  background: transparent;
  color: #e0a45c;
  border: 1px solid #6a5438;
  font-weight: 600;
}
QPushButton#advancedToggle:hover { background: #2a261f; }
QPushButton#advancedToggle:checked {
  background: #2c313c;
  color: #e8e6e3;
  border-color: #3d4452;
}
QLineEdit, QComboBox, QSpinBox, QListWidget {
  background: #1b1e25;
  color: #e8e6e3;
  border: 1px solid #3d4452;
  border-radius: 4px;
  padding: 4px 6px;
  min-height: 28px;
  selection-background-color: #4a3b28;
  selection-color: #f6efe4;
}
QComboBox QAbstractItemView {
  background: #22262e;
  color: #e8e6e3;
  selection-background-color: #e0a45c;
  selection-color: #1a140c;
}
QListWidget::item { padding: 4px 6px; }
QListWidget::item:selected { background: #4a3b28; color: #f6efe4; }
QListWidget::item:hover { background: #2a303a; }
QCheckBox { color: #e8e6e3; spacing: 8px; background: transparent; }
QProgressBar {
  background: #1b1e25;
  border: 1px solid #3d4452;
  border-radius: 4px;
  text-align: center;
  color: #e8e6e3;
  min-height: 18px;
}
QProgressBar::chunk { background: #e0a45c; border-radius: 3px; }
"""


def pick_vocal_file(written) -> str | None:
    """写出结果里优先打开文件名含 vocals 的那条，否则第一条。"""
    if written is None:
        return None
    if isinstance(written, (str, Path)):
        items = [written]
    else:
        try:
            items = list(written)
        except TypeError:
            return None
    paths = [Path(p) for p in items if p]
    if not paths:
        return None
    chosen = next((p for p in paths if "vocals" in p.name.lower()), paths[0])
    return str(chosen)


def model_unready_text(model: str) -> str:
    """横幅文案。体积用注册表 size_mb，不要写死 640 MB。"""
    label = MODEL_LABELS.get(model, str(model))
    size = MODEL_REGISTRY.get(model, {}).get("size_mb")
    size_part = f"（约 {size} MB）" if size else ""
    return f"模型「{label}」还没准备好{size_part}，下载后即可分离。"


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class ToggleSelectList(QListWidget):
    """点击式多选：点一次选中、再点取消，各文件互不影响（无需 Ctrl）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # SingleSelection 下 setSelected(True) 会清除其他项；Extended 互不影响
        self.setSelectionMode(QListWidget.ExtendedSelection)

    def mousePressEvent(self, event) -> None:
        item = self.itemAt(event.position().toPoint())
        if item is not None:
            item.setSelected(not item.isSelected())
            return
        # 点击空白：不调用 super()——Qt 默认会清空全部选择（误点风险），
        # 多选场景保留选择更安全
        return


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        app = QApplication.instance()
        if app is not None:
            app.setStyle("Fusion")
        self.settings = QSettings("uvr-lite", "uvr-lite")
        self._paths: list[Path] = []
        # 列表行状态：path -> (前缀, 人声输出路径)。列表重建时按此回放，
        # 否则新增/移除文件会丢掉已完成标记与「双击打开人声」。
        self._item_state: dict[str, tuple[str, str]] = {}
        self._applying_preset = False
        self.setWindowTitle("uvr-lite 人声/伴奏分离")
        self.setWindowIcon(QIcon(str(_ICON)))
        self.setAcceptDrops(True)
        self.setMinimumSize(880, 640)
        self.resize(980, 720)
        self.setStyleSheet("QMainWindow { background-color: #16181d; }")
        self._build_ui()
        self._restore_settings()

    # ---------- 界面 ----------

    def _build_ui(self) -> None:
        central = QWidget(self)
        central.setObjectName("central")
        central.setStyleSheet(APP_QSS)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        # --- 模型状态提示条（缺失时显示，可一键下载）---
        self.banner = QFrame(central)
        self.banner.setObjectName("banner")
        self.banner.setStyleSheet(
            "QFrame#banner { background: #c9843a; border: 1px solid #8a5a1e; "
            "border-radius: 6px; }"
        )
        banner_row = QHBoxLayout(self.banner)
        banner_row.setContentsMargins(10, 8, 10, 8)
        self.label_banner = QLabel(self.banner)
        self.label_banner.setWordWrap(True)
        self.label_banner.setStyleSheet(
            "color: #1a140c; background: transparent; font-weight: 600;"
        )
        self.btn_download = QPushButton("下载模型", self.banner)
        self.btn_download.setStyleSheet(
            "QPushButton { background: #2a2118; color: #f6efe4; border: none; "
            "border-radius: 6px; min-height: 32px; padding: 4px 12px; }"
            "QPushButton:hover { background: #3a2e22; }"
            "QPushButton:disabled { background: #4a3d30; color: #c4b8a8; }"
        )
        self.dl_progress = QProgressBar(self.banner)
        self.dl_progress.setFixedWidth(180)
        self.dl_progress.setVisible(False)
        self.dl_progress.setStyleSheet(
            "QProgressBar { background: #f3e0c4; color: #1a140c; "
            "border: 1px solid #8a5a1e; border-radius: 4px; text-align: center; }"
            "QProgressBar::chunk { background: #2a2118; }"
        )
        banner_row.addWidget(self.label_banner, 1)
        banner_row.addWidget(self.dl_progress)
        banner_row.addWidget(self.btn_download)
        root.addWidget(self.banner)
        self.btn_download.clicked.connect(self._start_download)

        # --- 文件列表 ---
        file_box = QGroupBox("待处理音频", central)
        fl = QVBoxLayout(file_box)
        file_hint = QLabel("可以把文件或文件夹拖进来。分离完成后，双击某一行打开人声。", file_box)
        file_hint.setObjectName("hint")
        file_hint.setWordWrap(True)
        fl.addWidget(file_hint)
        self.list_files = ToggleSelectList(file_box)
        self.list_files.setAlternatingRowColors(True)
        self.list_files.setMinimumHeight(160)
        self.list_files.itemDoubleClicked.connect(self._on_item_double_clicked)
        fl.addWidget(self.list_files)
        btn_row = QHBoxLayout()
        self.btn_add_files = QPushButton("选择文件…", file_box)
        self.btn_add_folder = QPushButton("选择文件夹…", file_box)
        self.btn_remove = QPushButton("移除所选", file_box)
        self.btn_clear = QPushButton("清空", file_box)
        for b in (self.btn_add_files, self.btn_add_folder, self.btn_remove, self.btn_clear):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        fl.addLayout(btn_row)
        self.btn_add_files.clicked.connect(self._add_files_dialog)
        self.btn_add_folder.clicked.connect(self._add_folder_dialog)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear.clicked.connect(self._clear_list)
        root.addWidget(file_box, 1)

        # --- 模型与音质（高级项默认折叠）---
        param_box = QGroupBox("分离", central)
        param_lay = QVBoxLayout(param_box)
        top = QFormLayout()
        self.combo_model = QComboBox(param_box)
        for name in MODEL_REGISTRY:
            self.combo_model.addItem(MODEL_LABELS.get(name, name), name)
        self.combo_quality = QComboBox(param_box)
        for key, label in QUALITY_CHOICES:
            self.combo_quality.addItem(label, key)
        self.combo_quality.setCurrentIndex(self.combo_quality.findData("standard"))
        self.label_quality_hint = QLabel(param_box)
        self.label_quality_hint.setObjectName("hint")
        self.label_quality_hint.setWordWrap(True)
        top.addRow("模型", self.combo_model)
        top.addRow("音质", self.combo_quality)
        top.addRow(self.label_quality_hint)
        param_lay.addLayout(top)

        adv_head = QHBoxLayout()
        self.btn_advanced = QPushButton("更多选项", param_box)
        self.btn_advanced.setObjectName("advancedToggle")
        self.btn_advanced.setCheckable(True)
        self.btn_advanced.setChecked(False)
        adv_head.addWidget(self.btn_advanced)
        adv_head.addStretch(1)
        param_lay.addLayout(adv_head)

        self.advanced_box = QWidget(param_box)
        adv_form = QFormLayout(self.advanced_box)
        adv_form.setContentsMargins(0, 6, 0, 0)
        self.combo_device = QComboBox(self.advanced_box)
        self.combo_device.addItems(DEVICE_CHOICES)
        self.combo_device.currentIndexChanged.connect(self._on_device_changed)
        self.combo_format = QComboBox(self.advanced_box)
        self.combo_format.addItems(FORMAT_CHOICES)
        self.combo_pcm = QComboBox(self.advanced_box)
        self.combo_pcm.addItems(["24", "16"])
        self.spin_bigshifts = QSpinBox(self.advanced_box)
        self.spin_bigshifts.setRange(1, 8)
        self.spin_batch = QSpinBox(self.advanced_box)
        self.spin_batch.setRange(0, 64)
        self.spin_batch.setSpecialValueText("默认（模型配置）")
        self.spin_overlap = QSpinBox(self.advanced_box)
        self.spin_overlap.setRange(0, 8)
        self.spin_overlap.setSpecialValueText("默认（模型配置）")
        self.check_tta = QCheckBox("测试时增强（3 倍耗时，质量更好）", self.advanced_box)
        self.label_cuda_inline = QLabel(
            "显卡加速已安装。设备选「自动」时会优先用显卡。", self.advanced_box
        )
        self.label_cuda_inline.setWordWrap(True)
        self.label_cuda_inline.setVisible(False)
        adv_form.addRow("设备", self.combo_device)
        adv_form.addRow("输出格式", self.combo_format)
        adv_form.addRow("FLAC 位深", self.combo_pcm)
        adv_form.addRow("BigShifts 次数", self.spin_bigshifts)
        adv_form.addRow("批大小（低显存设 1）", self.spin_batch)
        adv_form.addRow("重叠窗口数（1 最快）", self.spin_overlap)
        adv_form.addRow("", self.check_tta)
        adv_form.addRow(self.label_cuda_inline)
        self.advanced_box.setVisible(False)
        param_lay.addWidget(self.advanced_box)
        self.btn_advanced.toggled.connect(self._on_advanced_toggled)
        root.addWidget(param_box)

        # --- 推理引擎：未安装时保留下载；已安装则收成高级里的一行 ---
        engine_box = QGroupBox("显卡加速", central)
        self.engine_box = engine_box
        engine_col = QVBoxLayout(engine_box)
        self.label_engine = QLabel(engine_box)
        self.label_engine.setWordWrap(True)
        self.label_engine.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        engine_row = QHBoxLayout()
        self.cuda_progress = QProgressBar(engine_box)
        self.cuda_progress.setFixedWidth(180)
        self.cuda_progress.setVisible(False)
        self.btn_cuda = QPushButton(engine_box)
        engine_row.addWidget(self.cuda_progress)
        engine_row.addWidget(self.btn_cuda)
        engine_row.addStretch(1)
        engine_col.addWidget(self.label_engine)
        engine_col.addLayout(engine_row)
        root.addWidget(engine_box)
        self.btn_cuda.clicked.connect(self._start_cuda_download)

        # --- 输出目录 ---
        out_box = QGroupBox("输出文件夹", central)
        out_row = QHBoxLayout(out_box)
        self.edit_out = QLineEdit(out_box)
        self.edit_out.setPlaceholderText("未选择（默认：当前目录/output）")
        # 拖放交给主窗口处理；QLineEdit 默认接受拖放会把文件路径吞进输入框
        self.edit_out.setAcceptDrops(False)
        self.btn_out = QPushButton("选择…", out_box)
        out_row.addWidget(self.edit_out)
        out_row.addWidget(self.btn_out)
        root.addWidget(out_box)
        self.btn_out.clicked.connect(self._choose_out_dir)

        # --- 操作区 ---
        action_row = QHBoxLayout()
        self.btn_start = QPushButton("开始分离", central)
        self.btn_start.setObjectName("startButton")
        self.btn_start.setMinimumWidth(140)
        self.btn_cancel = QPushButton("取消", central)
        self.btn_cancel.setEnabled(False)
        action_row.addWidget(self.btn_start)
        action_row.addWidget(self.btn_cancel)
        action_row.addStretch(1)
        root.addLayout(action_row)
        self.btn_start.clicked.connect(self._start_clicked)
        self.btn_cancel.clicked.connect(self._cancel_clicked)

        self.progress = QProgressBarWrap(central)
        self.progress.setEnabled(False)
        root.addWidget(self.progress)

        self.label_status = QLabel("就绪。添加音频文件后即可开始。", central)
        self.label_status.setObjectName("statusLabel")
        self.label_status.setWordWrap(True)
        root.addWidget(self.label_status)

        self.setCentralWidget(central)

        self.spin_overlap.valueChanged.connect(self._on_manual_quality_tweak)
        self.spin_bigshifts.valueChanged.connect(self._on_manual_quality_tweak)
        self.check_tta.toggled.connect(self._on_manual_quality_tweak)
        self.combo_quality.currentIndexChanged.connect(self._on_quality_changed)
        self.combo_model.currentIndexChanged.connect(self._refresh_model_banner)
        self._on_quality_changed()
        self._refresh_model_banner()
        self._refresh_engine_status()

    def _on_advanced_toggled(self, on: bool) -> None:
        self.advanced_box.setVisible(on)
        self.btn_advanced.setText("收起选项" if on else "更多选项")

    def _on_quality_changed(self) -> None:
        preset = self.combo_quality.currentData() or "standard"
        self.label_quality_hint.setText(QUALITY_HINTS.get(preset, QUALITY_HINTS["standard"]))
        if preset == "custom":
            self.btn_advanced.setChecked(True)
            return
        resolved = resolve_quality(str(preset), 0, 1, False)
        blockers = (self.spin_overlap, self.spin_bigshifts, self.check_tta)
        for w in blockers:
            w.blockSignals(True)
        try:
            overlap = resolved["num_overlap"]
            self.spin_overlap.setValue(0 if overlap is None else int(overlap))
            self.spin_bigshifts.setValue(int(resolved["bigshifts"]))
            self.check_tta.setChecked(bool(resolved["tta"]))
        finally:
            for w in blockers:
                w.blockSignals(False)

    def _on_manual_quality_tweak(self, *_args) -> None:
        """高级里改了重叠 / BigShifts / TTA，就切到自定义，避免档位把改动盖掉。"""
        if self._applying_preset:
            return
        if self.combo_quality.currentData() != "custom":
            self._select_data(self.combo_quality, "custom")

    # ---------- 文件列表操作 ----------

    def _add_paths(self, paths: list[Path]) -> None:
        new = dedup_paths(self._paths + paths)
        added = len(new) - len(self._paths)
        self._paths = new
        self._rebuild_list()
        self.label_status.setText(f"已添加 {added} 个文件，共 {len(self._paths)} 个。")

    def _rebuild_list(self) -> None:
        """按 self._paths 重建列表项（行与 _paths 一一对应，状态前缀/输出按 _item_state 回放）。"""
        self.list_files.clear()
        for p in self._paths:
            item = QListWidgetItem(p.name, self.list_files)
            item.setData(_ROLE_PATH, str(p))
            item.setToolTip(str(p))
            prefix, output = self._item_state.get(str(p), ("", ""))
            if prefix:
                item.setText(f"{prefix}{p.name}")
            if output:
                item.setData(_ROLE_OUTPUT, output)

    def _set_item_state(self, path: Path, prefix: str, note: str = "") -> None:
        """更新列表中某文件的显示：prefix 为 ⏳/✓/✗，note 追加说明。"""
        target = str(path)
        _prefix, output = self._item_state.get(target, ("", ""))
        self._item_state[target] = (prefix, output)
        for i in range(self.list_files.count()):
            item = self.list_files.item(i)
            if item.data(_ROLE_PATH) == target:
                base = Path(target).name
                item.setText(f"{prefix}{base}{note}")
                return

    def _remember_output(self, path: Path, written) -> None:
        chosen = pick_vocal_file(written)
        if not chosen:
            return
        target = str(path)
        prefix, _output = self._item_state.get(target, ("", ""))
        self._item_state[target] = (prefix, chosen)
        for i in range(self.list_files.count()):
            item = self.list_files.item(i)
            if item.data(_ROLE_PATH) == target:
                item.setData(_ROLE_OUTPUT, chosen)
                return

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        target = item.data(_ROLE_OUTPUT)
        if not target:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _add_files_dialog(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择音频文件", str(self.settings.value("last_dir", "")),
            "音频文件 (*.mp3 *.flac *.wav *.ogg *.m4a);;所有文件 (*)",
        )
        if files:
            self._add_paths([Path(f) for f in files])

    def _add_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "选择输入文件夹（自动扫描其中音频）", str(self.settings.value("last_dir", "")))
        if folder:
            found = scan_audio_files(Path(folder))
            if not found:
                self.label_status.setText("该文件夹里没有找到音频文件（mp3/flac/wav/ogg/m4a）。")
                return
            self._add_paths(found)
            self.settings.setValue("last_dir", str(Path(folder).resolve()))

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.list_files.selectedIndexes()}, reverse=True)
        for r in rows:
            del self._paths[r]
        self._prune_item_state()
        self._rebuild_list()

    def _clear_list(self) -> None:
        self._paths.clear()
        self._item_state.clear()
        self._rebuild_list()

    def _prune_item_state(self) -> None:
        keep = {str(p) for p in self._paths}
        self._item_state = {k: v for k, v in self._item_state.items() if k in keep}

    def _choose_out_dir(self) -> None:
        start = self.edit_out.text() or str(self.settings.value("last_dir", ""))
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹", start)
        if folder:
            self.edit_out.setText(folder)

    # ---------- 拖拽 ----------

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        if self.btn_cancel.isEnabled():  # 分离进行中不接受拖放
            return
        dropped = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        files: list[Path] = []
        saw_dir = False
        for p in dropped:
            if p.is_dir():
                saw_dir = True
                try:
                    files.extend(scan_audio_files(p))
                except OSError:
                    continue
            # 与「选择文件→所有文件」保持一致：后缀不在白名单但内容可识别的音频也收
            elif p.is_file() and (is_audio(p) or precheck_audio(p)):
                files.append(p)
        if not files:
            if saw_dir:
                self.label_status.setText("文件夹里没有找到音频文件（mp3/flac/wav/ogg/m4a）。")
            elif dropped:
                self.label_status.setText("没有可添加的音频文件。")
            return
        self._add_paths(files)
        event.acceptProposedAction()

    # ---------- 参数记忆 ----------

    def _preset_from_settings(self) -> str:
        """记住的音质档。没有该键的旧用户：对不上三档或开了 TTA 则自定义。"""
        s = self.settings
        saved = s.value("quality_preset", "")
        saved = "" if saved is None else str(saved)
        if saved in _PRESET_KEYS:
            return saved
        legacy = any(s.contains(k) for k in ("num_overlap", "bigshifts", "tta"))
        if not legacy:
            return "standard"
        return matching_preset(
            _as_int(s.value("num_overlap", 0), 0),
            _as_int(s.value("bigshifts", 1), 1),
            _as_bool(s.value("tta", False)),
        )

    def _restore_settings(self) -> None:
        s = self.settings
        self._applying_preset = True
        try:
            self._select_data(self.combo_model, s.value("model", "bs_roformer_ep317"))
            self._select_data(self.combo_device, s.value("device", "auto"))
            self._select_data(self.combo_format, s.value("format", "auto"))
            self._select_data(self.combo_pcm, str(s.value("pcm", "24")))
            self.spin_bigshifts.setValue(_as_int(s.value("bigshifts", 1), 1))
            self.spin_batch.setValue(_as_int(s.value("batch_size", 0), 0))
            self.spin_overlap.setValue(_as_int(s.value("num_overlap", 0), 0))
            self.check_tta.setChecked(_as_bool(s.value("tta", False)))
            out = s.value("out_dir", "")
            if out:
                self.edit_out.setText(str(out))
            self._select_data(self.combo_quality, self._preset_from_settings())
        finally:
            self._applying_preset = False
        # 索引没变时 currentIndexChanged 不会响，这里补一次，把标准档写成 overlap=2
        self._on_quality_changed()

    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("model", self.combo_model.currentData())
        s.setValue("quality_preset", self.combo_quality.currentData() or "standard")
        s.setValue("device", self.combo_device.currentText())
        s.setValue("format", self.combo_format.currentText())
        s.setValue("pcm", self.combo_pcm.currentText())
        s.setValue("bigshifts", self.spin_bigshifts.value())
        s.setValue("batch_size", self.spin_batch.value())
        s.setValue("num_overlap", self.spin_overlap.value())
        s.setValue("tta", self.check_tta.isChecked())
        s.setValue("out_dir", self.edit_out.text())

    @staticmethod
    def _select_data(combo: QComboBox, value) -> None:
        # addItem(label, data) 的项用 data 匹配；addItems 的项无 data，回退 findText
        idx = combo.findData(value)
        if idx < 0:
            idx = combo.findText(str(value))
        if idx >= 0:
            combo.setCurrentIndex(idx)

    # ---------- 推理引擎（torch 二进制）切换 ----------

    def _on_device_changed(self) -> None:
        """设备选择 → 写 torch.ini（启动时据此加载 CPU/CUDA 版 torch）。

        仅安装场景（{app}/torch_cpu|torch_cuda 存在）生效；开发场景忽略。
        切换在下次启动时生效（torch 已在进程内加载）。
        """
        base = Path(__file__).resolve().parents[2]
        if (base / "torch_cpu").exists() or (base / "torch_cuda").exists():
            try:
                (base / "torch.ini").write_text(
                    f"use={self.combo_device.currentText()}\n", encoding="utf-8")
            except OSError:
                return
            self.label_status.setText("推理引擎切换后，重启 uvr-lite 才会生效。")

    # ---------- 模型下载 ----------

    def _refresh_model_banner(self) -> None:
        model = self.combo_model.currentData()
        ready = model_file(model).exists()
        self.banner.setVisible(not ready)
        if not ready:
            self.label_banner.setText(model_unready_text(model))

    def _start_download(self) -> None:
        if self.btn_cancel.isEnabled():  # 分离进行中，先不启动新的重任务
            return
        self.btn_start.setEnabled(False)  # 下载期间不允许开始分离
        model = self.combo_model.currentData()
        self._dl_worker = ModelDownloadWorker(model)
        self._dl_thread = QThread(self)
        self._dl_worker.moveToThread(self._dl_thread)
        self._dl_thread.started.connect(self._dl_worker.run)
        self._dl_worker.progress.connect(self._on_dl_progress)
        self._dl_worker.finished.connect(self._on_dl_finished)
        self._dl_thread.finished.connect(self._dl_thread.deleteLater)
        self._dl_thread.finished.connect(self._dl_worker.deleteLater)

        self.btn_download.setText("取消下载")
        self.btn_download.clicked.disconnect()
        self.btn_download.clicked.connect(self._cancel_download)
        self.dl_progress.setVisible(True)
        self.dl_progress.setValue(0)
        self.label_banner.setText("正在下载模型…（可取消，断点续传）")
        self._dl_thread.start()

    def _cancel_download(self) -> None:
        if self._dl_worker is not None:
            self._dl_worker.cancel()
            self.btn_download.setEnabled(False)
            self.label_banner.setText("正在取消…")

    def _on_dl_progress(self, done, total) -> None:
        pct = int(done / total * 100) if total else 0
        self.dl_progress.setValue(pct)
        self.label_banner.setText(
            f"正在下载模型… {done / 1e6:.0f}/{total / 1e6:.0f} MB（{pct}%，可取消）")

    def _on_dl_finished(self, ok, error) -> None:
        self._dl_thread.quit()
        self._dl_thread.wait(3000)
        self.dl_progress.setVisible(False)
        self.btn_download.setEnabled(True)
        self.btn_download.setText("下载模型")
        self.btn_start.setEnabled(not self.btn_cancel.isEnabled())  # 分离未在跑才恢复
        self.btn_download.clicked.disconnect()
        self.btn_download.clicked.connect(self._start_download)
        if ok:
            self._refresh_model_banner()  # 成功 → 提示条消失
            self.label_status.setText("模型下载完成，可以开始分离了。")
        else:
            self.banner.setVisible(True)
            self.label_banner.setText(f"模型下载未完成：{error}（点击重试）")

    # ---------- CUDA 引擎下载 ----------

    def _refresh_engine_status(self) -> None:
        """已安装：收成高级里的一行。未安装：保留下载按钮。"""
        if cuda_torch_installed():
            self.engine_box.setVisible(False)
            self.label_cuda_inline.setVisible(True)
            self.label_engine.setText("显卡加速已安装。设备选「自动」时会优先用显卡。")
            self.btn_cuda.setText("已安装")
            self.btn_cuda.setEnabled(False)
        else:
            self.engine_box.setVisible(True)
            self.label_cuda_inline.setVisible(False)
            self.label_engine.setText(
                "这台电脑还不能用 NVIDIA 显卡加速。下载约 3.3 GB，装完需要重启。"
            )
            self.btn_cuda.setText("下载")
            self.btn_cuda.setEnabled(True)

    def _start_cuda_download(self) -> None:
        if self.btn_cancel.isEnabled():  # 分离进行中，先不启动下载
            return
        self.btn_start.setEnabled(False)  # 下载/解压期间不允许开始分离
        self._cuda_worker = CudaTorchWorker(repo_root())
        self._cuda_thread = QThread(self)
        self._cuda_worker.moveToThread(self._cuda_thread)
        self._cuda_thread.started.connect(self._cuda_worker.run)
        self._cuda_worker.progress.connect(self._on_cuda_progress)
        self._cuda_worker.finished.connect(self._on_cuda_finished)
        self._cuda_thread.finished.connect(self._cuda_thread.deleteLater)
        self._cuda_thread.finished.connect(self._cuda_worker.deleteLater)

        self.btn_cuda.setText("取消下载")
        self.btn_cuda.clicked.disconnect()
        self.btn_cuda.clicked.connect(self._cancel_cuda_download)
        self.cuda_progress.setVisible(True)
        self.cuda_progress.setValue(0)
        self.label_engine.setText("正在下载 CUDA 引擎…（可取消，断点续传）")
        self._cuda_thread.start()

    def _cancel_cuda_download(self) -> None:
        if self._cuda_worker is not None:
            self._cuda_worker.cancel()
            self.btn_cuda.setEnabled(False)
            self.label_engine.setText("正在取消…")

    def _on_cuda_progress(self, done, total) -> None:
        pct = int(done / total * 100) if total else 0
        self.cuda_progress.setValue(pct)
        self.label_engine.setText(
            f"CUDA 引擎安装中… {pct}%（下载 + 解压，可取消）")

    def _on_cuda_finished(self, ok, error) -> None:
        self._cuda_thread.quit()
        self._cuda_thread.wait(3000)
        self.cuda_progress.setVisible(False)
        self.btn_cuda.setEnabled(True)
        self.btn_cuda.setText("下载")
        self.btn_start.setEnabled(not self.btn_cancel.isEnabled())  # 分离未在跑才恢复
        self.btn_cuda.clicked.disconnect()
        self.btn_cuda.clicked.connect(self._start_cuda_download)
        self._refresh_engine_status()
        if ok:
            QMessageBox.information(
                self, "CUDA 引擎已安装",
                "CUDA 引擎安装完成。重启 uvr-lite 后，设备选择「自动」将优先使用 GPU 加速。")
        else:
            self.label_engine.setText(f"CUDA 引擎下载未完成：{error}（点击重试）")

    # ---------- 任务控制 ----------

    def _start_clicked(self) -> None:
        if not self._paths:
            QMessageBox.information(self, "提示", "请先添加音频文件（选择文件/文件夹或拖拽）。")
            return
        model = self.combo_model.currentData()
        if not model_file(model).exists():
            size = MODEL_REGISTRY.get(model, {}).get("size_mb")
            size_part = f"（约 {size} MB）" if size else ""
            QMessageBox.information(
                self, "模型还没准备好",
                f"请先点击顶部「下载模型」按钮下载权重{size_part}，再开始分离。")
            self._refresh_model_banner()
            return
        out_dir = self.edit_out.text().strip() or str(Path.cwd() / "output")
        out_dir = str(Path(out_dir).resolve())
        self._save_settings()

        # 预检：先快速校验格式——无法识别为音频的文件标 ✗ 且不进队列；
        # 内容真是音频（即使后缀被改）正常通过
        ok_paths, bad_paths = [], []
        for p in self._paths:
            if precheck_audio(p):
                ok_paths.append(p)
            else:
                bad_paths.append(p)
                self._set_item_state(p, _PREFIX_BAD, "（格式不支持）")
        if bad_paths:
            self.label_status.setText(
                f"{len(bad_paths)} 个文件无法识别为音频，已跳过："
                + "、".join(p.name for p in bad_paths[:5])
                + ("…" if len(bad_paths) > 5 else ""))
        if not ok_paths:
            QMessageBox.warning(self, "没有可处理的文件",
                                "所选文件都无法识别为音频格式，请检查文件是否损坏。")
            return

        quality = resolve_quality(
            self.combo_quality.currentData() or "standard",
            self.spin_overlap.value(),
            self.spin_bigshifts.value(),
            self.check_tta.isChecked(),
        )
        params = {
            "model_name": model,
            "device": self.combo_device.currentText(),
            "fmt": self.combo_format.currentText(),
            "pcm": f"PCM_{self.combo_pcm.currentText()}",
            "bigshifts": quality["bigshifts"],
            "batch_size": self.spin_batch.value() or None,
            "num_overlap": quality["num_overlap"],
            "tta": quality["tta"],
        }
        self._worker = SeparationWorker(list(ok_paths), out_dir, params)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.file_failed.connect(self._on_file_failed)
        self._worker.all_finished.connect(self._on_all_finished)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._worker.deleteLater)

        self._out_dir = out_dir
        self._t_start = time.time()
        self._t_file = time.time()
        self._eta_file_idx = -1  # 首个进度回调时重置 _t_file，排除模型加载耗时
        self._file_times: list[float] = []
        self._failed_names: list[str] = []
        self._ok_paths = ok_paths  # 队列索引 → 文件（列表状态标记用）
        for p in ok_paths:
            self._set_item_state(p, _PREFIX_PENDING)
        self._set_busy(True)
        self.progress.bar.setValue(0)
        self.label_status.setText("准备中…")
        self._thread.start()

    def _cancel_clicked(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.btn_cancel.setEnabled(False)
            self.label_status.setText("正在取消…")

    def _on_progress(self, phase, done, total, file_idx, file_total, file_pct) -> None:
        if file_idx != self._eta_file_idx:
            # 本文件首个进度回调：重置计时，避免把 torch 导入/模型加载（可达分钟级）
            # 算进当前文件速度，外推出严重偏大的 ETA
            self._eta_file_idx = file_idx
            self._t_file = time.time()
        global_pct = int((file_idx + file_pct / 100.0) / max(1, file_total) * 100)
        self.progress.bar.setValue(global_pct)
        eta = estimate_eta(
            self._file_times,
            file_idx,
            file_total,
            file_pct / 100.0,
            elapsed_current=time.time() - self._t_file,
        )
        eta_txt = self._fmt_eta(eta) if eta is not None else "计算中…"
        self.label_status.setText(
            f"处理中 {file_idx + 1}/{file_total} · {PHASE_CN.get(phase, phase)}"
            f" {file_pct}% · 预计剩余 {eta_txt}"
        )

    def _on_file_done(self, file_idx, written) -> None:
        self._file_times.append(time.time() - self._t_file)
        self._t_file = time.time()
        if 0 <= file_idx < len(self._ok_paths):
            self._set_item_state(self._ok_paths[file_idx], _PREFIX_OK)
            self._remember_output(self._ok_paths[file_idx], written)

    def _on_file_failed(self, file_idx, error) -> None:
        name = self._ok_paths[file_idx].name
        self._failed_names.append(name)
        if 0 <= file_idx < len(self._ok_paths):
            self._set_item_state(self._ok_paths[file_idx], _PREFIX_BAD)
        self.label_status.setText(f"{name} 处理失败，已跳过（{error[:80]}）")

    def _on_all_finished(self, ok, failed, cancelled) -> None:
        self._set_busy(False)
        # run() 返回后线程事件循环仍在跑，不 quit 会每次都泄漏一个 QThread
        self._thread.quit()
        msg = summary_text(ok, self._failed_names)
        if cancelled:
            msg = f"已取消。{msg}"
        box = QMessageBox(self)
        box.setWindowTitle("分离完成" if not cancelled else "已取消")
        box.setText(msg)
        box.setIcon(QMessageBox.Information if not failed else QMessageBox.Warning)
        btn_open = box.addButton("打开输出文件夹", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Ok)
        box.exec()
        if box.clickedButton() is btn_open:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._out_dir))
        self.label_status.setText("就绪。")

    @staticmethod
    def _fmt_eta(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m} 分 {s} 秒" if m else f"{s} 秒"

    def _set_busy(self, busy: bool) -> None:
        for w in (self.btn_add_files, self.btn_add_folder, self.btn_remove, self.btn_clear,
                  self.combo_model, self.combo_quality, self.btn_advanced,
                  self.combo_device, self.combo_format, self.combo_pcm,
                  self.spin_bigshifts, self.spin_batch, self.spin_overlap, self.check_tta,
                  self.edit_out, self.btn_out, self.btn_download):
            w.setEnabled(not busy)
        self.btn_start.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        self.progress.setEnabled(busy)
        self.list_files.setEnabled(not busy)
        self.setAcceptDrops(not busy)  # 运行中不接受拖放，避免打乱列表状态

    # ---------- 生命周期 ----------

    @staticmethod
    def _thread_running(thread) -> bool:
        """线程是否仍在运行；C++ 对象已被 deleteLater 销毁时按 False 处理。"""
        try:
            return thread is not None and thread.isRunning()
        except RuntimeError:
            return False

    def closeEvent(self, event) -> None:
        self._save_settings()
        if self._thread_running(getattr(self, "_thread", None)) and \
                getattr(self, "_worker", None) is not None:
            self._worker.cancel()
            self._thread.quit()
            self._thread.wait(5000)
        if self._thread_running(getattr(self, "_dl_thread", None)):
            if getattr(self, "_dl_worker", None) is not None:
                self._dl_worker.cancel()
            self._dl_thread.quit()
            self._dl_thread.wait(3000)
        if self._thread_running(getattr(self, "_cuda_thread", None)):
            if getattr(self, "_cuda_worker", None) is not None:
                self._cuda_worker.cancel()
            self._cuda_thread.quit()
            self._cuda_thread.wait(3000)
        super().closeEvent(event)


class QProgressBarWrap(QWidget):
    """进度区：QProgressBar 包装（引擎进度回调 → 全局进度条）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QProgressBar

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.bar = QProgressBar(self)
        lay.addWidget(self.bar)
        self.setEnabled(False)


def run() -> int:
    # 安装场景：--model-dir 指向安装目录下的 models（环境变量让 uvr_lite.download 复用）
    if "--model-dir" in sys.argv:
        idx = sys.argv.index("--model-dir")
        if idx + 1 < len(sys.argv):
            os.environ["UVR_MODEL_DIR"] = sys.argv[idx + 1]
    # Windows 任务栏图标跟随窗口图标（否则显示 Python 默认图标）
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("uvr-lite")
    app = QApplication(sys.argv)
    app.setApplicationName("uvr-lite")
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(str(_ICON)))
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run())
