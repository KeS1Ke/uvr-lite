"""uvr-lite 桌面界面主窗口（PySide6）。

本模块负责行窗口：文件列表与拖拽、参数表单（QSettings 记忆）、模型与
推理引擎的下载状态呈现与触发，以及分离任务的接线（worker 线程 + 进度、
取消、完成汇总）。

torch/引擎不在本模块导入（惰性加载见 ui/worker.py）：界面启动与列表编辑
不付数以秒计的 torch 导入成本。
"""

import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import QEvent, QSettings, Qt, QThread, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..download import (
    cuda_engine_block_reason,
    cuda_engine_supported,
    cuda_torch_installed,
    model_file,
    repo_root,
)
from ..log import ensure_log_dir, get_logger, log_exception, log_hint, log_path
from ..models import MODEL_REGISTRY
from ..quality import (
    QUALITY_CHOICES,
    QUALITY_PRESETS,
    QUALITY_STANDARD,
    quality_from_overlap,
    quality_overlap,
)
from ..stems import StemPair, find_stem_pairs, pair_stems, split_stem
from .files import dedup_paths, is_audio, scan_audio_files
from .progress import PHASE_CN, estimate_eta, summary_text
from .theme import apply as apply_theme
from .worker import (
    CombineParams,
    CombineWorker,
    CudaTorchWorker,
    ModelDownloadWorker,
    SeparationParams,
    SeparationWorker,
)

_ICON = Path(__file__).resolve().parent / "resources" / "uvr-lite.ico"


def _log():
    """惰性取 logger：模块导入时不建 logs/ 目录（导入 UI 不应有文件副作用）。"""
    return get_logger("uvr_lite.ui")


def _log_file_text() -> str:
    """日志文件绝对路径（给 tooltip/无障碍用）；定位失败时返回空串。

    同 log.log_hint：异常目录布局下 repo_root 会抛 RuntimeError，拿路径
    做展示的地方都得自己兜住，不能被"看看日志在哪"这种辅助信息拖崩。
    """
    try:
        return str(log_path())
    except Exception:
        return ""


MODEL_LABELS = {
    "bs_roformer_ep317": "BS-RoFormer ep317（主力，推荐）",
    "mel_band_karaoke": "Mel-Band RoFormer Karaoke（备选）",
}
# 全量安装包含 CPU/CUDA 两套 torch，应用内选择；mps 仅 macOS 无意义故不列出
DEVICE_CHOICES = ["auto", "cpu", "cuda"]
FORMAT_CHOICES = ["auto", "flac", "wav"]

# 功能模式：分离（引擎+模型）/ 合成（人声＋伴奏回混，纯 DSP）
MODE_SEPARATE = 0
MODE_COMBINE = 1
MODE_CHOICES = ["人声/伴奏分离", "音轨合成（人声＋伴奏）"]

# 列表项状态前缀（显示在文件名前）
_PREFIX_PENDING = "⏳ "
_PREFIX_OK = "✓ "
_PREFIX_BAD = "✗ "

# 单行标签留给错误文案的字符数：这一行的宽度要给同一行的按钮腾地方，超出即截断，
# 完整原因在日志里（下载失败常见超长文案：urlopen error [WinError ...]）。
# 取 16 使最坏情况（纯中文、每行都占满字宽）下 banner/engine 两行仍放得下。
_ONE_LINE_ERR = 16


def _short_error(error: str) -> str:
    """压成单行放得下的错误摘要；空值兜一句「未知原因」，太长就截断。"""
    text = (error or "未知原因").strip()
    return text if len(text) <= _ONE_LINE_ERR else text[:_ONE_LINE_ERR] + "…"


def _settings_flag(value: object, default: bool = False) -> bool:
    """QSettings 的布尔在 ini/注册表里经常读回 'true'/'false' 字符串。

    直接 bool('false') 会得到 True，勾选会被记反。
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def _settings_int(value: object, default: int, lo: int, hi: int) -> int:
    """坏注册表值用默认，再夹到控件范围。不能让 MainWindow() 抛出去。"""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, number))


def _settings_float(value: object, default: float, lo: float, hi: float) -> float:
    """同 _settings_int；NaN/Inf 视为坏值。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number == float("inf") or number == float("-inf"):
        return default
    return max(lo, min(hi, number))


_QUALITY_NOTE = (
    "本应用的人声模型已经是 ep317；karaoke 是主唱/和声模型，不是更纯的人声档。"
    "要更干净选「高」，CPU 要速度选「快」。"
)


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
        self.settings = QSettings("uvr-lite", "uvr-lite")
        self._paths: list[Path] = []
        self._tray: QSystemTrayIcon | None = None
        self._tray_menu: QMenu | None = None
        self._busy = False
        self._worker = None
        self._thread = None
        # 合成（人声＋伴奏回混）状态：文件池 → 每次重建时自动配对
        self._mix_files: list[Path] = []
        self._mix_pairs: list[StemPair] = []
        self._mix_unmatched: list[Path] = []
        self._mix_row_paths: list[list[Path]] = []
        self._mix_queue: list[StemPair] = []
        self._mix_worker = None
        self._mix_thread = None
        self._mix_failed_names: list[str] = []
        self._mix_file_times: list[float] = []
        self._mix_out_dir = ""
        self._mix_ignored: list[Path] = []
        # 关窗时 wait 超时：先 ignore，等线程真正结束后再 close 一次
        self._close_after_stop = False
        self._close_guard = False
        self._quit_after_close = False
        self._close_listened: set[int] = set()
        self._precheck_aborted = False
        self._sep_paths: list[Path] = []
        # 版本号放标题：界面里没有别的出口（cli --version 用户看不到），标题截图即报障信息
        self.setWindowTitle(f"uvr-lite 人声分离/合成 {__version__}")
        self.setWindowIcon(QIcon(str(_ICON)))
        self.setAcceptDrops(True)
        self.resize(760, 680)
        self._build_ui()
        self._restore_settings()

    # ---------- 界面 ----------

    def _build_ui(self) -> None:
        central = QWidget(self)
        central.setObjectName("root")
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 14, 16, 10)
        root.setSpacing(8)

        # --- 模型状态提示条（缺失时显示，可一键下载）---
        self.banner = QFrame(central)
        self.banner.setObjectName("banner")
        banner_row = QHBoxLayout(self.banner)
        banner_row.setContentsMargins(8, 6, 8, 6)
        self.label_banner = QLabel(self.banner)
        self.btn_download = QPushButton("下载模型", self.banner)
        self.dl_progress = QProgressBar(self.banner)
        self.dl_progress.setFixedWidth(180)
        self.dl_progress.setVisible(False)
        banner_row.addWidget(self.label_banner)
        banner_row.addWidget(self.dl_progress)
        banner_row.addWidget(self.btn_download)
        banner_row.addStretch(1)
        root.addWidget(self.banner)
        self.btn_download.clicked.connect(self._start_download)

        # --- 功能页（分离 / 合成）：QStackedWidget 切换，不占额外高度 ---
        self.stack = QStackedWidget(central)
        self.sep_page = QWidget(self.stack)
        sep_lay = QVBoxLayout(self.sep_page)
        sep_lay.setContentsMargins(0, 0, 0, 0)
        sep_lay.setSpacing(8)
        self.mix_page = QWidget(self.stack)

        # --- 文件列表 ---
        self.file_box = QGroupBox("待处理音频（可拖拽文件到此处）", central)
        self.file_box.setObjectName("files")
        fl = QVBoxLayout(self.file_box)
        fl.setContentsMargins(12, 8, 12, 10)
        fl.setSpacing(8)
        self.file_stack = QStackedWidget(self.file_box)
        self.file_stack.setMinimumHeight(112)
        self.drop_hint = QLabel("把音频拖到这里\n或用下面的按钮添加", self.file_stack)
        self.drop_hint.setObjectName("dropHint")
        self.drop_hint.setAlignment(Qt.AlignCenter)
        self.list_files = ToggleSelectList(self.file_stack)
        self.file_stack.addWidget(self.drop_hint)
        self.file_stack.addWidget(self.list_files)
        fl.addWidget(self.file_stack, 1)
        btn_row = QHBoxLayout()
        self.btn_add_files = QPushButton("选择文件…", self.file_box)
        self.btn_add_folder = QPushButton("选择文件夹…", self.file_box)
        self.btn_remove = QPushButton("移除所选", self.file_box)
        self.btn_clear = QPushButton("清空", self.file_box)
        for b in (self.btn_add_files, self.btn_add_folder, self.btn_remove, self.btn_clear):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        fl.addLayout(btn_row)
        self.btn_add_files.clicked.connect(self._add_files_dialog)
        self.btn_add_folder.clicked.connect(self._add_folder_dialog)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear.clicked.connect(self._clear_list)
        sep_lay.addWidget(self.file_box, 1)

        # --- 模型与参数 ---
        param_box = QGroupBox("参数", central)
        form = QFormLayout(param_box)
        form.setContentsMargins(12, 8, 12, 10)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(6)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.combo_model = QComboBox(param_box)
        for name in MODEL_REGISTRY:
            self.combo_model.addItem(MODEL_LABELS.get(name, name), name)
        self.combo_device = QComboBox(param_box)
        self.combo_device.addItems(DEVICE_CHOICES)
        self.combo_device.currentIndexChanged.connect(self._on_device_changed)
        self.combo_format = QComboBox(param_box)
        self.combo_format.addItems(FORMAT_CHOICES)
        self.combo_pcm = QComboBox(param_box)
        self.combo_pcm.addItems(["24", "16"])
        self.spin_bigshifts = QSpinBox(param_box)
        self.spin_bigshifts.setRange(1, 8)
        self.spin_batch = QSpinBox(param_box)
        self.spin_batch.setRange(0, 64)
        self.spin_batch.setSpecialValueText("默认（模型配置）")
        self.combo_quality = QComboBox(param_box)
        for name in QUALITY_CHOICES:
            self.combo_quality.addItem(str(QUALITY_PRESETS[name]["menu"]), name)
        self._select_data(self.combo_quality, QUALITY_STANDARD)
        self._sync_quality_tooltip()
        self.combo_quality.currentIndexChanged.connect(self._sync_quality_tooltip)
        self.check_tta = QCheckBox("测试时增强（3 倍耗时，质量更好）", param_box)
        form.addRow("模型", self.combo_model)
        form.addRow("设备", self.combo_device)
        form.addRow("输出格式", self.combo_format)
        form.addRow("FLAC 位深", self.combo_pcm)
        form.addRow("BigShifts 次数", self.spin_bigshifts)
        form.addRow("批大小（低显存设 1）", self.spin_batch)
        form.addRow("质量", self.combo_quality)
        form.addRow("", self.check_tta)
        sep_lay.addWidget(param_box)

        # --- 推理引擎（CPU/CUDA torch）：单包只含 CPU，CUDA 可选下载 ---
        engine_box = QGroupBox("推理引擎", central)
        engine_row = QHBoxLayout(engine_box)
        engine_row.setContentsMargins(12, 8, 12, 10)
        self.label_engine = QLabel(engine_box)
        self.cuda_progress = QProgressBar(engine_box)
        self.cuda_progress.setFixedWidth(180)
        self.cuda_progress.setVisible(False)
        self.btn_cuda = QPushButton(engine_box)
        engine_row.addWidget(self.label_engine)
        engine_row.addWidget(self.cuda_progress)
        engine_row.addWidget(self.btn_cuda)
        engine_row.addStretch(1)
        sep_lay.addWidget(engine_box)
        self.btn_cuda.clicked.connect(self._start_cuda_download)
        self._refresh_engine_status()

        # --- 输出目录 ---
        out_box = QGroupBox("输出文件夹", central)
        out_row = QHBoxLayout(out_box)
        out_row.setContentsMargins(12, 8, 12, 10)
        self.edit_out = QLineEdit(out_box)
        self.edit_out.setPlaceholderText("未选择（默认：当前目录/output）")
        self.btn_out = QPushButton("选择…", out_box)
        out_row.addWidget(self.edit_out)
        out_row.addWidget(self.btn_out)
        sep_lay.addWidget(out_box)
        self.btn_out.clicked.connect(self._choose_out_dir)

        # --- 合成页（人声＋伴奏回混，纯 DSP 不加载模型）---
        mix_lay = QVBoxLayout(self.mix_page)
        mix_lay.setContentsMargins(0, 0, 0, 0)
        mix_lay.setSpacing(8)
        self._build_mix_page(mix_lay)
        self.stack.addWidget(self.sep_page)
        self.stack.addWidget(self.mix_page)
        root.addWidget(self.stack, 1)

        # --- 操作区 ---
        action_row = QHBoxLayout()
        self.label_mode = QLabel("功能", central)
        self.combo_mode = QComboBox(central)
        self.combo_mode.addItems(MODE_CHOICES)
        self.combo_mode.setToolTip("分离：把歌曲拆成人声/伴奏；合成：把两者合回整曲")
        self.btn_start = QPushButton("开始分离", central)
        self.btn_start.setObjectName("primary")
        self.btn_cancel = QPushButton("取消", central)
        self.btn_cancel.setObjectName("quiet")
        self.btn_cancel.setEnabled(False)
        action_row.addWidget(self.label_mode)
        action_row.addWidget(self.combo_mode)
        action_row.addWidget(self.btn_start)
        action_row.addWidget(self.btn_cancel)
        action_row.addStretch(1)
        self.check_minimize_tray = QCheckBox("最小化到系统托盘", central)
        action_row.addWidget(self.check_minimize_tray)
        root.addLayout(action_row)
        self.btn_start.clicked.connect(self._start_clicked)
        self.btn_cancel.clicked.connect(self._cancel_clicked)

        self.progress = QProgressBarWrap(central)
        self.progress.setEnabled(False)
        root.addWidget(self.progress)

        self.label_status = QLabel("就绪。添加音频文件后即可开始。", central)
        self.label_status.setObjectName("status")
        # 「查看日志」：模型下载/CUDA 安装失败只有一行标签写得下，完整原因在日志里，
        # 这里给个固定入口（放在底部状态行，任何失败场景都够得着）
        self.btn_open_log = QPushButton("查看日志", central)
        self.btn_open_log.setObjectName("quiet")
        # 绝对路径不进单行标签（会把「推理引擎」那行撑坏），只放 tooltip：
        # 悬停即可知道日志文件在哪，看不见的屏幕阅读器也有无障碍描述
        tip = "打开日志所在的文件夹，把里面的 uvr-lite.log 发给支持的人就能定位问题。"
        if log_file := _log_file_text():
            tip += f"\n日志文件：{log_file}"
        self.btn_open_log.setToolTip(tip)
        self.btn_open_log.setAccessibleDescription(tip)
        self.btn_open_log.clicked.connect(self._open_log_dir)

        status_row = QHBoxLayout()
        status_row.addWidget(self.label_status, 1)  # 状态文案优先占宽度，按钮靠右不挤
        status_row.addWidget(self.btn_open_log)
        root.addLayout(status_row)

        self.setCentralWidget(central)

        # 控件全部就绪后再挂联动（信号回调里会引用 btn_start/label_status 等）
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.combo_model.currentIndexChanged.connect(self._refresh_model_banner)
        self._refresh_model_banner()
        self._setup_tray()

    def _build_mix_page(self, lay: QVBoxLayout) -> None:
        """合成页：音轨列表（自动配对）+ 增益/格式参数 + 输出目录。"""
        self.mix_file_box = QGroupBox(
            "待合成音轨（人声＋伴奏，可拖拽或选择文件夹自动配对）", self.mix_page)
        self.mix_file_box.setObjectName("files")
        fl = QVBoxLayout(self.mix_file_box)
        fl.setContentsMargins(12, 8, 12, 10)
        fl.setSpacing(8)
        self.mix_stack = QStackedWidget(self.mix_file_box)
        self.mix_stack.setMinimumHeight(112)
        self.mix_drop_hint = QLabel(
            "把「人声」和「伴奏」文件或整个文件夹拖到这里\n"
            "文件名以 -vocals / -instrumental 结尾时自动配对", self.mix_stack)
        self.mix_drop_hint.setObjectName("dropHint")
        self.mix_drop_hint.setAlignment(Qt.AlignCenter)
        self.mix_list = ToggleSelectList(self.mix_stack)
        self.mix_stack.addWidget(self.mix_drop_hint)
        self.mix_stack.addWidget(self.mix_list)
        fl.addWidget(self.mix_stack, 1)
        btn_row = QHBoxLayout()
        self.mix_btn_add_files = QPushButton("选择音轨…", self.mix_file_box)
        self.mix_btn_add_folder = QPushButton("选择文件夹…", self.mix_file_box)
        self.mix_btn_remove = QPushButton("移除所选", self.mix_file_box)
        self.mix_btn_clear = QPushButton("清空", self.mix_file_box)
        for b in (self.mix_btn_add_files, self.mix_btn_add_folder,
                  self.mix_btn_remove, self.mix_btn_clear):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        fl.addLayout(btn_row)
        self.mix_btn_add_files.clicked.connect(self._mix_add_files_dialog)
        self.mix_btn_add_folder.clicked.connect(self._mix_add_folder_dialog)
        self.mix_btn_remove.clicked.connect(self._mix_remove_selected)
        self.mix_btn_clear.clicked.connect(self._mix_clear_list)
        lay.addWidget(self.mix_file_box, 1)

        param_box = QGroupBox("合成参数", self.mix_page)
        form = QFormLayout(param_box)
        form.setContentsMargins(12, 8, 12, 10)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(6)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.spin_vocal_gain = QDoubleSpinBox(param_box)
        self.spin_vocal_gain.setRange(0.0, 4.0)
        self.spin_vocal_gain.setSingleStep(0.05)
        self.spin_vocal_gain.setDecimals(2)
        self.spin_vocal_gain.setValue(1.0)
        self.spin_inst_gain = QDoubleSpinBox(param_box)
        self.spin_inst_gain.setRange(0.0, 4.0)
        self.spin_inst_gain.setSingleStep(0.05)
        self.spin_inst_gain.setDecimals(2)
        self.spin_inst_gain.setValue(1.0)
        self.mix_combo_format = QComboBox(param_box)
        self.mix_combo_format.addItems(FORMAT_CHOICES)
        self.mix_combo_pcm = QComboBox(param_box)
        self.mix_combo_pcm.addItems(["24", "16"])
        self.check_normalize = QCheckBox("峰值归一化（防止相加爆音）", param_box)
        form.addRow("人声音量", self.spin_vocal_gain)
        form.addRow("伴奏音量", self.spin_inst_gain)
        form.addRow("输出格式", self.mix_combo_format)
        form.addRow("FLAC 位深", self.mix_combo_pcm)
        form.addRow("", self.check_normalize)
        lay.addWidget(param_box)

        out_box = QGroupBox("输出文件夹", self.mix_page)
        out_row = QHBoxLayout(out_box)
        out_row.setContentsMargins(12, 8, 12, 10)
        self.mix_edit_out = QLineEdit(out_box)
        self.mix_edit_out.setPlaceholderText("未选择（默认：当前目录/output）")
        self.mix_btn_out = QPushButton("选择…", out_box)
        out_row.addWidget(self.mix_edit_out)
        out_row.addWidget(self.mix_btn_out)
        lay.addWidget(out_box)
        self.mix_btn_out.clicked.connect(self._mix_choose_out_dir)

    # ---------- 文件列表操作 ----------

    def _add_paths(self, paths: list[Path]) -> int:
        """去重追加路径并重建列表；返回本次新增条数（供调用方拼状态栏文案）。"""
        new = dedup_paths(self._paths + paths)
        added = len(new) - len(self._paths)
        self._paths = new
        self._rebuild_list()
        self.label_status.setText(f"已添加 {added} 个文件，共 {len(self._paths)} 个。")
        return added

    def _rebuild_list(self) -> None:
        """按 self._paths 重建列表项（行与 _paths 一一对应，状态前缀保留基础名）。"""
        self.list_files.clear()
        for p in self._paths:
            item = QListWidgetItem(p.name, self.list_files)
            item.setData(Qt.UserRole, str(p))
        self.file_stack.setCurrentWidget(
            self.list_files if self._paths else self.drop_hint)

    def _set_item_state(self, path: Path, prefix: str, note: str = "") -> None:
        """更新列表中某文件的显示：prefix 为 ⏳/✓/✗，note 追加说明。"""
        target = str(path)
        for i in range(self.list_files.count()):
            item = self.list_files.item(i)
            if item.data(Qt.UserRole) == target:
                base = Path(target).name
                item.setText(f"{prefix}{base}{note}")
                return

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
            try:
                found = scan_audio_files(Path(folder))
            except OSError:
                # 不可读目录：异常不能冒到 Qt 事件循环里被吞掉，用户会以为按钮没反应
                self.label_status.setText(
                    f"无法读取文件夹：{Path(folder).name}（权限不足？）。")
                return
            if not found:
                self.label_status.setText("该文件夹里没有找到音频文件（mp3/flac/wav/ogg/m4a）。")
                return
            added = self._add_paths(found)
            if not added:
                self.label_status.setText(
                    f"共 {len(self._paths)} 个文件（无新增，已在列表中）。")
            self.settings.setValue("last_dir", str(Path(folder).resolve()))

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.list_files.selectedIndexes()}, reverse=True)
        for r in rows:
            del self._paths[r]
        self._rebuild_list()

    def _clear_list(self) -> None:
        self._paths.clear()
        self._rebuild_list()

    def _choose_out_dir(self) -> None:
        start = self.edit_out.text() or str(self.settings.value("last_dir", ""))
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹", start)
        if folder:
            self.edit_out.setText(folder)

    # ---------- 合成列表操作（文件池 → 每次重建自动配对） ----------

    def _mix_add_paths(self, paths: list[Path]) -> int:
        """去重追加音轨并重建配对列表；返回新增条数（供调用方拼状态栏文案）。"""
        new = dedup_paths(self._mix_files + paths)
        added = len(new) - len(self._mix_files)
        self._mix_files = new
        self._mix_rebuild_list()
        return added

    def _mix_rebuild_list(self) -> None:
        """按当前文件池重新配对并重建列表：成对、缺另一半、非分离音轨。

        ignored 也要出行，否则拖入普通 song.mp3 会「已添加」但列表空白，
        文件还留在池里删不掉。这类行不是可合成的一对。
        """
        result = pair_stems(self._mix_files)
        self._mix_pairs = result.pairs
        self._mix_unmatched = result.unmatched
        self._mix_ignored = list(result.ignored)
        self.mix_list.clear()
        self._mix_row_paths = []
        for pair in self._mix_pairs:
            base = self._mix_pair_label(pair)
            item = QListWidgetItem(f"{_PREFIX_OK}{base}", self.mix_list)
            item.setData(Qt.UserRole, str(pair.vocals))
            item.setData(Qt.UserRole + 1, base)  # 无前缀原文，用于状态切换
            self._mix_row_paths.append([pair.vocals, pair.instrumental])
        for p in self._mix_unmatched:
            split = split_stem(p)
            missing = "伴奏" if split is not None and split[1] == "vocals" else "人声"
            base = f"{p.name}（缺少{missing}）"
            item = QListWidgetItem(f"{_PREFIX_BAD}{base}", self.mix_list)
            item.setData(Qt.UserRole, str(p))
            item.setData(Qt.UserRole + 1, base)
            self._mix_row_paths.append([p])
        for p in self._mix_ignored:
            base = f"{p.name}（非分离音轨）"
            item = QListWidgetItem(f"{_PREFIX_BAD}{base}", self.mix_list)
            item.setData(Qt.UserRole, str(p))
            item.setData(Qt.UserRole + 1, base)
            self._mix_row_paths.append([p])
        self.mix_stack.setCurrentWidget(
            self.mix_list if self._mix_files else self.mix_drop_hint)

    @staticmethod
    def _mix_pair_label(pair: StemPair) -> str:
        split = split_stem(pair.vocals)
        base = split[0] if split is not None else pair.vocals.stem
        return f"{base}（人声 {pair.vocals.name} ＋ 伴奏 {pair.instrumental.name}）"

    def _set_mix_item_state(self, row: int, prefix: str, note: str = "") -> None:
        """切换合成列表某一行的状态前缀（保留无前缀原文，避免前缀叠加）。"""
        if not (0 <= row < self.mix_list.count()):
            return
        item = self.mix_list.item(row)
        base = item.data(Qt.UserRole + 1) or item.text()
        item.setText(f"{prefix}{base}{note}")

    def _mix_report_added(self, added: int) -> None:
        if added:
            self.label_status.setText(
                f"已添加 {added} 个音轨文件，已配对 {len(self._mix_pairs)} 组。")
        else:
            self.label_status.setText(
                f"无新增音轨（已在列表中），当前已配对 {len(self._mix_pairs)} 组。")

    def _mix_add_files_dialog(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择音轨文件（人声与伴奏）",
            str(self.settings.value("mix_last_dir", "")),
            "音频文件 (*.mp3 *.flac *.wav *.ogg *.m4a);;所有文件 (*)",
        )
        if files:
            added = self._mix_add_paths([Path(f) for f in files])
            self._mix_report_added(added)
            self.settings.setValue("mix_last_dir", str(Path(files[0]).resolve().parent))

    def _mix_add_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "选择文件夹（自动配对 *-vocals / *-instrumental）",
            str(self.settings.value("mix_last_dir", "")))
        if not folder:
            return
        try:
            result = find_stem_pairs(Path(folder))
        except OSError:
            # 不可读目录不能把异常冒到 Qt 事件循环（与分离页同款处理）
            self.label_status.setText(
                f"无法读取文件夹：{Path(folder).name}（权限不足？）。")
            return
        if not result.pairs and not result.unmatched and not result.ignored:
            self.label_status.setText(
                "该文件夹里没有找到分离音轨（*-vocals / *-instrumental）。")
            return
        paths = (
            [p.vocals for p in result.pairs]
            + [p.instrumental for p in result.pairs]
            + list(result.unmatched)
            + list(result.ignored)
        )
        added = self._mix_add_paths(paths)
        self.settings.setValue("mix_last_dir", str(Path(folder).resolve()))
        self._mix_report_added(added)

    def _mix_remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.mix_list.selectedIndexes()}, reverse=True)
        if not rows:
            return
        drop: set[Path] = set()
        for r in rows:
            drop.update(self._mix_row_paths[r])
        self._mix_files = [p for p in self._mix_files if p not in drop]
        self._mix_rebuild_list()

    def _mix_clear_list(self) -> None:
        self._mix_files.clear()
        self._mix_rebuild_list()

    def _mix_choose_out_dir(self) -> None:
        start = self.mix_edit_out.text() or str(self.settings.value("mix_last_dir", ""))
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹", start)
        if folder:
            self.mix_edit_out.setText(folder)

    # ---------- 拖拽 ----------

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            self._set_drop_target(True)
            event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:
        self._set_drop_target(False)
        event.accept()

    def dropEvent(self, event) -> None:
        """按当前功能页分流拖入项：分离页进分离队列，合成页进配对池。

        每一项分流：目录走 scan_audio_files（与「选择文件夹」同一条路），
        文件走 is_audio 过滤。修前只收 is_file() 的项，拖入目录既不扫描也无
        反馈——用户看不出是没处理还是程序坏了。现在目录与文件混拖也能全部入列
        （跨文件夹去重交给 _add_paths）。非本地 URL（http/ftp 等）的
        toLocalFile() 是空串，Path("") 会被当成当前目录扫进去，必须先按
        isLocalFile() 过滤掉。
        """
        self._set_drop_target(False)
        if self.stack.currentWidget() is self.mix_page:
            self._drop_to_mix(event)
            return
        files, empty_dirs, failed_dirs, ignored, local_items = self._scan_dropped(event)

        parts: list[str] = []
        if files:
            # 列表为空时不调 _add_paths：否则会先写「已添加 0 个文件」噪音。
            # 非空时用返回值拼文案，全重复（added=0）要给「无新增」说明。
            added = self._add_paths(files)
            if added:
                parts.append(f"已添加 {added} 个文件，共 {len(self._paths)} 个。")
            else:
                parts.append(f"共 {len(self._paths)} 个文件（无新增，已在列表中）。")
        if empty_dirs:
            names = "、".join(d.name for d in empty_dirs[:3])
            more = f" 等（共 {len(empty_dirs)} 个）" if len(empty_dirs) > 3 else ""
            parts.append(f"文件夹里没有找到音频（mp3/flac/wav/ogg/m4a）：{names}{more}。")
        if failed_dirs:
            names = "、".join(d.name for d in failed_dirs[:3])
            more = f" 等（共 {len(failed_dirs)} 个）" if len(failed_dirs) > 3 else ""
            parts.append(f"无法读取文件夹：{names}{more}（权限不足？）。")
        if ignored:
            parts.append(f"已忽略 {ignored} 个非音频文件。")
        if not parts and local_items:
            parts.append("没有可添加的音频文件（支持 mp3/flac/wav/ogg/m4a）。")
        if parts:
            self.label_status.setText("".join(parts))

    def _scan_dropped(self, event):
        """拖放项分流（分离/合成共用）：返回 (文件, 空目录, 不可读目录, 忽略数, 本地项数)。"""
        files: list[Path] = []
        empty_dirs: list[Path] = []
        failed_dirs: list[Path] = []
        ignored = 0
        local_items = 0
        for u in event.mimeData().urls():
            if not u.isLocalFile() or not u.toLocalFile():
                continue  # 非本地 URL 直接跳过：不计入任何扫描，也不报「没有音频」
            local_items += 1
            p = Path(u.toLocalFile())
            if p.is_dir():
                try:
                    found = scan_audio_files(p)
                except OSError:
                    # 不可读目录（PermissionError 等）：原先异常冲出 dropEvent 被
                    # Qt 打印后吞掉，用户毫无反馈——审计缺陷的原始症状
                    failed_dirs.append(p)
                    continue
                if found:
                    files.extend(found)
                else:
                    empty_dirs.append(p)
            elif p.is_file():
                if is_audio(p):
                    files.append(p)
                else:
                    ignored += 1
        return files, empty_dirs, failed_dirs, ignored, local_items

    def _drop_to_mix(self, event) -> None:
        """合成页拖放：收集音频后交配对池，并给状态栏反馈。"""
        files, empty_dirs, failed_dirs, ignored, local_items = self._scan_dropped(event)
        parts: list[str] = []
        if files:
            added = self._mix_add_paths(files)
            if added:
                parts.append(f"已添加 {added} 个音轨文件。")
            else:
                parts.append("无新增音轨（已在列表中）。")
            parts.append(f"已配对 {len(self._mix_pairs)} 组。")
        if empty_dirs:
            names = "、".join(d.name for d in empty_dirs[:3])
            more = f" 等（共 {len(empty_dirs)} 个）" if len(empty_dirs) > 3 else ""
            parts.append(f"文件夹里没有找到音频（mp3/flac/wav/ogg/m4a）：{names}{more}。")
        if failed_dirs:
            names = "、".join(d.name for d in failed_dirs[:3])
            more = f" 等（共 {len(failed_dirs)} 个）" if len(failed_dirs) > 3 else ""
            parts.append(f"无法读取文件夹：{names}{more}（权限不足？）。")
        if ignored:
            parts.append(f"已忽略 {ignored} 个非音频文件。")
        if not parts and local_items:
            parts.append("没有可添加的音频文件（支持 mp3/flac/wav/ogg/m4a）。")
        if parts:
            self.label_status.setText("".join(parts))

    def _set_drop_target(self, active: bool) -> None:
        box = (self.mix_file_box
               if self.stack.currentWidget() is self.mix_page else self.file_box)
        box.setProperty("dragging", "true" if active else "false")
        box.style().unpolish(box)
        box.style().polish(box)

    def _on_mode_changed(self, index: int) -> None:
        """功能页切换：内容、开始按钮文案与状态提示一起换。"""
        combine = index == MODE_COMBINE
        self.stack.setCurrentWidget(self.mix_page if combine else self.sep_page)
        self.btn_start.setText("开始合成" if combine else "开始分离")
        if not self._busy:
            self.label_status.setText(
                "就绪。添加人声与伴奏（或选择文件夹自动配对）后点击开始合成。"
                if combine else "就绪。添加音频文件后即可开始。")

    # ---------- 参数记忆 ----------

    def _restore_quality(self, s: QSettings) -> None:
        """有 quality 用它；否则用旧的 num_overlap；都没有则标准档。"""
        name = ""
        if s.contains("quality"):
            raw = s.value("quality", "")
            name = str(raw).strip().lower() if raw is not None else ""
            if name not in QUALITY_PRESETS:
                name = ""
        if not name:
            if s.contains("num_overlap"):
                overlap = _settings_int(s.value("num_overlap"), 0, 0, 8)
                name = quality_from_overlap(overlap)
            else:
                name = QUALITY_STANDARD
        self._select_data(self.combo_quality, name)
        self._sync_quality_tooltip()

    def _sync_quality_tooltip(self, *_args) -> None:
        name = self.combo_quality.currentData()
        preset = QUALITY_PRESETS.get(name) if name in QUALITY_PRESETS else None
        hint = str(preset["hint"]) if preset else ""
        tip = f"{hint}\n{_QUALITY_NOTE}" if hint else _QUALITY_NOTE
        self.combo_quality.setToolTip(tip)

    def _restore_settings(self) -> None:
        s = self.settings
        self._select_data(self.combo_model, s.value("model", "bs_roformer_ep317"))
        self._select_data(self.combo_device, s.value("device", "auto"))
        self._select_data(self.combo_format, s.value("format", "auto"))
        self._select_data(self.combo_pcm, str(s.value("pcm", "24")))
        self.spin_bigshifts.setValue(_settings_int(
            s.value("bigshifts", 1), 1,
            self.spin_bigshifts.minimum(), self.spin_bigshifts.maximum()))
        self.spin_batch.setValue(_settings_int(
            s.value("batch_size", 0), 0,
            self.spin_batch.minimum(), self.spin_batch.maximum()))
        self._restore_quality(s)
        self.check_tta.setChecked(_settings_flag(s.value("tta", False)))
        self.check_minimize_tray.setChecked(
            _settings_flag(s.value("minimize_to_tray", False)))
        out = s.value("out_dir", "")
        if out:
            self.edit_out.setText(str(out))
        # 合成页参数（含上次所在功能页）
        mode = _settings_int(
            s.value("mode", MODE_SEPARATE), MODE_SEPARATE, 0, len(MODE_CHOICES) - 1)
        self.combo_mode.setCurrentIndex(mode)
        self._select_data(self.mix_combo_format, s.value("mix_format", "auto"))
        self._select_data(self.mix_combo_pcm, str(s.value("mix_pcm", "24")))
        self.spin_vocal_gain.setValue(_settings_float(
            s.value("mix_vocal_gain", 1.0), 1.0,
            self.spin_vocal_gain.minimum(), self.spin_vocal_gain.maximum()))
        self.spin_inst_gain.setValue(_settings_float(
            s.value("mix_inst_gain", 1.0), 1.0,
            self.spin_inst_gain.minimum(), self.spin_inst_gain.maximum()))
        self.check_normalize.setChecked(_settings_flag(s.value("mix_normalize", False)))
        mix_out = s.value("mix_out_dir", "")
        if mix_out:
            self.mix_edit_out.setText(str(mix_out))

    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("model", self.combo_model.currentData())
        s.setValue("device", self.combo_device.currentText())
        s.setValue("format", self.combo_format.currentText())
        s.setValue("pcm", self.combo_pcm.currentText())
        s.setValue("bigshifts", self.spin_bigshifts.value())
        s.setValue("batch_size", self.spin_batch.value())
        s.setValue("quality", self.combo_quality.currentData())
        s.setValue("tta", self.check_tta.isChecked())
        s.setValue("minimize_to_tray", self.check_minimize_tray.isChecked())
        s.setValue("out_dir", self.edit_out.text())
        s.setValue("mode", self.combo_mode.currentIndex())
        s.setValue("mix_format", self.mix_combo_format.currentText())
        s.setValue("mix_pcm", self.mix_combo_pcm.currentText())
        s.setValue("mix_vocal_gain", self.spin_vocal_gain.value())
        s.setValue("mix_inst_gain", self.spin_inst_gain.value())
        s.setValue("mix_normalize", self.check_normalize.isChecked())
        s.setValue("mix_out_dir", self.mix_edit_out.text())

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

        仅安装场景（{base}/torch_cpu|torch_cuda 存在）生效；开发场景（用环境
        里已装的 torch）忽略。切换在下次启动时生效（torch 已在进程内加载）。

        根目录必须走 repo_root()：安装布局是 {inst}/app/uvr_lite/ui/main.py，
        按 __file__ 上推只会到 {inst}/app，而 torch_cpu/torch_cuda 在 {inst}
        下（installer/install.iss [Files]），判断会恒为假 → 切换静默失效。
        """
        base = repo_root()
        if (base / "torch_cpu").exists() or (base / "torch_cuda").exists():
            try:
                (base / "torch.ini").write_text(
                    f"use={self.combo_device.currentText()}\n", encoding="utf-8")
            except OSError:
                # 原先静默吞掉（只读安装目录时会发生），用户只会看到"切换没生效"
                log_exception("写 torch.ini 失败（推理引擎切换未生效）")
                return
            self.label_status.setText("推理引擎已切换，重启 uvr-lite 后生效。")

    # ---------- 模型下载 ----------

    def _refresh_model_banner(self) -> None:
        model = self.combo_model.currentData()
        ready = model_file(model).exists()
        self.banner.setVisible(not ready)
        if not ready:
            label = MODEL_LABELS.get(model, model)
            self.label_banner.setText(f"模型「{label}」未下载，下载后即可开始分离。")

    def _start_download(self) -> None:
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
        self.btn_download.setEnabled(not self._busy)
        self.btn_download.setText("下载模型")
        self.btn_download.clicked.disconnect()
        self.btn_download.clicked.connect(self._start_download)
        if ok:
            self._refresh_model_banner()  # 成功 → 提示条消失
            self.label_status.setText("模型下载完成，可以开始分离了。")
        else:
            self.banner.setVisible(True)
            self.label_banner.setText(
                f"模型下载未完成：{_short_error(error)}（失败原因见日志，点击重试）")
            # 提示条只有一行、放不下详情（也不该把绝对路径塞进去撑坏这一行）：
            # 完整原因留给日志，用户从底部「查看日志」进入文件夹
            self.label_banner.setToolTip("点窗口底部「查看日志」可以看到完整失败原因。")
            _log().warning("模型下载失败: %s", error)

    # ---------- CUDA 引擎下载 ----------

    def _refresh_engine_status(self) -> None:
        """按安装状态刷新「推理引擎」区：已装 / 可下载 / 当前环境不支持。

        平台不匹配时（内置 wheel 只有 cp312/win_amd64）不装也不报错的静默态，
        而是禁用按钮 + 直说原因——用户点下去才在 worker 线程里看到错误，等于
        白等一趟（见 download.cuda_engine_requirements）。
        """
        if cuda_torch_installed():
            self.label_engine.setText("CUDA 引擎已安装 ✓（设备选「自动」将优先 GPU 加速）")
            self.btn_cuda.setText("已安装")
            self.btn_cuda.setEnabled(False)
        elif not cuda_engine_supported():
            reason = cuda_engine_block_reason()
            # 状态栏一行只放首句；完整文案（含「怎么办」①②）进 tooltip
            self.label_engine.setText(
                f"当前环境不支持 CUDA 引擎 — {reason.split('。')[0]}。")
            self.btn_cuda.setText("不可用")
            self.btn_cuda.setEnabled(False)
            self.btn_cuda.setToolTip(reason)
        else:
            self.label_engine.setText(
                "CUDA 引擎未安装 — 下载约 3.3 GB 后可用 GPU 加速（NVIDIA 显卡）")
            self.btn_cuda.setText("下载 CUDA 引擎")
            self.btn_cuda.setEnabled(True)
            self.btn_cuda.setToolTip("")

    def _start_cuda_download(self) -> None:
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
        self.btn_cuda.setText("下载 CUDA 引擎")
        self.btn_cuda.clicked.disconnect()
        self.btn_cuda.clicked.connect(self._start_cuda_download)
        self._refresh_engine_status()
        if self._busy:
            self.btn_cuda.setEnabled(False)
        if ok:
            QMessageBox.information(
                self, "CUDA 引擎已安装",
                "CUDA 引擎安装完成。重启 uvr-lite 后，设备选择「自动」将优先使用 GPU 加速。")
        else:
            self.label_engine.setText(
                f"CUDA 引擎下载未完成：{_short_error(error)}（失败原因见日志，点击重试）")
            # 这一行塞不下详情也塞不下路径：完整原因在日志里
            self.label_engine.setToolTip("点窗口底部「查看日志」可以看到完整失败原因。")
            # 3.3GB 下载失败的原因（源不可用/磁盘满/解压被拦）只有日志里有
            _log().warning("CUDA 引擎下载失败: %s", error)

    # ---------- 任务控制 ----------

    def _start_clicked(self) -> None:
        if self.combo_mode.currentIndex() == MODE_COMBINE:
            self._mix_start_clicked()
            return
        if not self._paths:
            QMessageBox.information(self, "提示", "请先添加音频文件（选择文件/文件夹或拖拽）。")
            return
        model = self.combo_model.currentData()
        if not model_file(model).exists():
            QMessageBox.information(
                self, "模型未下载",
                "请先点击顶部「下载模型」按钮下载该模型的权重，再开始分离。")
            self._refresh_model_banner()
            return
        out_dir = self.edit_out.text().strip() or str(Path.cwd() / "output")
        out_dir = str(Path(out_dir).resolve())
        self._save_settings()

        device = self.combo_device.currentText()
        # 与 _refresh_engine_status 同一判断：没装 torch_cuda 就不能选 cuda。
        # auto/cpu 不拦截。预检在 worker 里做，这里还没起线程。
        if device == "cuda" and not cuda_torch_installed():
            self.label_status.setText(
                "未安装 CUDA 引擎，无法使用 CUDA 设备。"
                "请先下载 CUDA 引擎，或改用「自动」/「CPU」。"
            )
            return

        quality_name = self.combo_quality.currentData() or QUALITY_STANDARD
        params = SeparationParams(
            model_name=model,
            device=device,
            fmt=self.combo_format.currentText(),
            pcm=f"PCM_{self.combo_pcm.currentText()}",
            bigshifts=self.spin_bigshifts.value(),
            # 0 值（"默认（模型配置）"）→ None：让引擎回落模型自带配置
            batch_size=self.spin_batch.value() or None,
            # 质量档总是明确的 1/2/4，不再把 0 当成「跟模型配置」
            num_overlap=quality_overlap(str(quality_name)),
            tta=self.check_tta.isChecked(),
        )
        # 预检在 worker.run() 里、加载模型之前。这里先把整表交出去。
        self._sep_paths = list(self._paths)
        self._ok_paths = list(self._sep_paths)
        self._precheck_aborted = False
        self._worker = SeparationWorker(list(self._sep_paths), out_dir, params)
        # 上一轮若没走到完成槽，这里仍占着事件循环；先退出再换指针，避免变成孤儿。
        self._release_thread(self._thread)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.file_rejected.connect(self._on_file_rejected)
        self._worker.precheck_ready.connect(self._on_precheck_ready)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.file_failed.connect(self._on_file_failed)
        self._worker.all_finished.connect(self._on_all_finished)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._worker.deleteLater)

        self._out_dir = out_dir
        self._t_start = time.time()
        self._t_file = time.time()
        self._file_times: list[float] = []
        self._failed_names: list[str] = []
        self._set_busy(True)
        self.progress.bar.setValue(0)
        self.label_status.setText("准备中…")
        self._thread.start()

    def _mix_start_clicked(self) -> None:
        """合成任务：校验有配对，构建 CombineWorker（不加载模型）。"""
        self._mix_rebuild_list()
        if not self._mix_pairs:
            if self._mix_files:
                QMessageBox.information(
                    self, "没有可配对的音轨",
                    "列表中没有成对的人声＋伴奏。\n"
                    "文件名以 -vocals 与 -instrumental（或 -inst）结尾时会自动配对。")
            else:
                QMessageBox.information(
                    self, "提示", "请先添加人声与伴奏文件，或选择文件夹自动配对。")
            return
        out_dir = self.mix_edit_out.text().strip() or str(Path.cwd() / "output")
        out_dir = str(Path(out_dir).resolve())
        self._save_settings()

        params = CombineParams(
            vocal_gain=self.spin_vocal_gain.value(),
            inst_gain=self.spin_inst_gain.value(),
            fmt=self.mix_combo_format.currentText(),
            pcm=f"PCM_{self.mix_combo_pcm.currentText()}",
            normalize=self.check_normalize.isChecked(),
        )
        self._mix_queue = list(self._mix_pairs)
        self._mix_worker = CombineWorker(
            [(p.vocals, p.instrumental) for p in self._mix_queue], out_dir, params)
        self._release_thread(self._mix_thread)
        self._mix_thread = QThread(self)
        self._mix_worker.moveToThread(self._mix_thread)
        self._mix_thread.started.connect(self._mix_worker.run)
        self._mix_worker.progress.connect(self._on_mix_progress)
        self._mix_worker.file_done.connect(self._on_mix_file_done)
        self._mix_worker.file_failed.connect(self._on_mix_file_failed)
        self._mix_worker.all_finished.connect(self._on_mix_finished)
        self._mix_thread.finished.connect(self._mix_thread.deleteLater)
        self._mix_thread.finished.connect(self._mix_worker.deleteLater)

        self._mix_out_dir = out_dir
        self._mix_t_file = time.time()
        self._mix_file_times = []
        self._mix_failed_names = []
        for idx in range(len(self._mix_queue)):
            self._set_mix_item_state(idx, _PREFIX_PENDING)
        self._set_busy(True)
        self.progress.bar.setValue(0)
        self.label_status.setText("准备中…")
        self._mix_thread.start()

    def _cancel_clicked(self) -> None:
        if self.combo_mode.currentIndex() == MODE_COMBINE:
            if self._mix_worker is not None:
                self._mix_worker.cancel()
        elif self._worker is not None:
            self._worker.cancel()
        self.btn_cancel.setEnabled(False)
        self.label_status.setText("正在取消…")

    def _on_progress(self, phase, done, total, file_idx, file_total, file_pct) -> None:
        global_pct = int((file_idx + file_pct / 100.0) / max(1, file_total) * 100)
        self.progress.bar.setValue(global_pct)
        eta = estimate_eta(self._file_times, file_idx, file_total, file_pct / 100.0)
        eta_txt = self._fmt_eta(eta) if eta is not None else "计算中…"
        self.label_status.setText(
            f"处理中 {file_idx + 1}/{file_total} · {PHASE_CN.get(phase, phase)}"
            f" {file_pct}% · 预计剩余 {eta_txt}"
        )

    def _on_file_rejected(self, file_idx: int) -> None:
        """预检未过：标「格式不支持」，这一项不进分离队列。"""
        if not (0 <= file_idx < len(self._sep_paths)):
            return
        self._set_item_state(self._sep_paths[file_idx], _PREFIX_BAD, "（格式不支持）")

    def _on_precheck_ready(self, ok_indices) -> None:
        """预检结束。合格项才标成待处理；一个都没有就不该再加载模型。"""
        try:
            indices = [int(i) for i in ok_indices]
        except TypeError:
            indices = []
        ok_set = set(indices)
        bad = [p for i, p in enumerate(self._sep_paths) if i not in ok_set]
        for p in bad:
            self._set_item_state(p, _PREFIX_BAD, "（格式不支持）")
        if bad:
            names = "、".join(p.name for p in bad[:5]) + ("…" if len(bad) > 5 else "")
            self.label_status.setText(
                f"{len(bad)} 个文件无法识别为音频，已跳过：" + names)
            _log().warning("%d 个文件预检失败（非音频或已损坏）: %s", len(bad), names)
        self._ok_paths = [
            self._sep_paths[i] for i in indices if 0 <= i < len(self._sep_paths)]
        for p in self._ok_paths:
            self._set_item_state(p, _PREFIX_PENDING)
        if indices:
            return
        # 必须先于弹窗置位：warning 的嵌套事件循环会把紧随其后的 all_finished 抽走
        self._precheck_aborted = True
        QMessageBox.warning(
            self, "没有可处理的文件",
            "所选文件都无法识别为音频格式，请检查文件是否损坏。" + log_hint())

    def _on_file_done(self, file_idx, written) -> None:
        self._file_times.append(time.time() - self._t_file)
        self._t_file = time.time()
        if 0 <= file_idx < len(self._ok_paths):
            self._set_item_state(self._ok_paths[file_idx], _PREFIX_OK)

    def _on_file_failed(self, file_idx, error) -> None:
        # 先判范围再索引（与 _on_file_done 一致）：槽函数里抛 IndexError 会直接冒到
        # Qt 事件循环，用户看到的是崩溃而不是"某个文件失败"
        path = self._ok_paths[file_idx] if 0 <= file_idx < len(self._ok_paths) else None
        name = path.name if path is not None else f"未知文件（队列索引 {file_idx}）"
        self._failed_names.append(name)
        if path is not None:
            self._set_item_state(path, _PREFIX_BAD)
        # 状态栏只显示 80 字摘要，完整错误（含引擎原始异常）进日志才有得查
        _log().warning("分离失败 %s: %s", name, error)
        self.label_status.setText(f"{name} 处理失败，已跳过（{error[:80]}）")

    @staticmethod
    def _is_thread_running(thread) -> bool:
        """线程是否仍在运行；C++ 对象已被回收时算「不在运行」。

        finished 连着 deleteLater，任务跑完一轮后 self._thread / _mix_thread
        只剩 Python 包装器，裸调 isRunning() 会抛 RuntimeError。closeEvent
        里没兜住就会跳过后面所有线程的 cancel+wait，而带活销毁运行中的
        QThread 会让进程 abort（0xC0000409），写了一半的输出也可能被截断。
        """
        if thread is None:
            return False
        try:
            return thread.isRunning()
        except RuntimeError:
            return False

    def _release_thread(self, thread) -> None:
        """让仍停在 exec() 的任务线程退出。未启动或已销毁则不动。

        started 接到 worker.run：run() 返回后线程还在事件循环里，
        isRunning() 仍为真，finished/deleteLater 不会发生。
        """
        if self._is_thread_running(thread):
            thread.quit()
            thread.wait(5000)

    def _on_all_finished(self, ok, failed, cancelled) -> None:
        self._release_thread(self._thread)
        self._set_busy(False)
        if self._precheck_aborted and not cancelled:
            # 全部预检失败：弹窗和状态栏已在 _on_precheck_ready 里处理，不再套完成框
            self._precheck_aborted = False
            return
        msg = summary_text(ok, self._failed_names)
        if cancelled:
            msg = f"已取消。{msg}"
        if failed:
            # 报错弹窗带上日志路径：非专业用户报障时可以直接把文件发过来
            msg += log_hint()
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

    # ---------- 合成任务的进度 / 完成槽 ----------

    def _on_mix_progress(self, phase, done, total, file_idx, file_total, file_pct) -> None:
        global_pct = int((file_idx + file_pct / 100.0) / max(1, file_total) * 100)
        self.progress.bar.setValue(global_pct)
        eta = estimate_eta(self._mix_file_times, file_idx, file_total, file_pct / 100.0)
        eta_txt = self._fmt_eta(eta) if eta is not None else "计算中…"
        self.label_status.setText(
            f"合成中 {file_idx + 1}/{file_total} · {PHASE_CN.get(phase, phase)}"
            f" {file_pct}% · 预计剩余 {eta_txt}"
        )

    def _on_mix_file_done(self, file_idx, written) -> None:
        self._mix_file_times.append(time.time() - self._mix_t_file)
        self._mix_t_file = time.time()
        self._set_mix_item_state(file_idx, _PREFIX_OK)

    def _on_mix_file_failed(self, file_idx, error) -> None:
        name = (self._mix_queue[file_idx].vocals.name
                if 0 <= file_idx < len(self._mix_queue)
                else f"未知音轨（队列索引 {file_idx}）")
        self._mix_failed_names.append(name)
        self._set_mix_item_state(file_idx, _PREFIX_BAD)
        _log().warning("合成失败 %s: %s", name, error)
        self.label_status.setText(f"{name} 合成失败，已跳过（{error[:80]}）")

    def _on_mix_finished(self, ok, failed, cancelled) -> None:
        self._release_thread(self._mix_thread)
        self._set_busy(False)
        msg = summary_text(ok, self._mix_failed_names)
        if cancelled:
            msg = f"已取消。{msg}"
        if failed:
            msg += log_hint()
        box = QMessageBox(self)
        box.setWindowTitle("合成完成" if not cancelled else "已取消")
        box.setText(msg)
        box.setIcon(QMessageBox.Information if not failed else QMessageBox.Warning)
        btn_open = box.addButton("打开输出文件夹", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Ok)
        box.exec()
        if box.clickedButton() is btn_open:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._mix_out_dir))
        self.label_status.setText("就绪。")

    @staticmethod
    def _fmt_eta(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m} 分 {s} 秒" if m else f"{s} 秒"

    def _cuda_button_allowed(self) -> bool:
        """与 _refresh_engine_status 一致：已安装或不支持时按钮不可点。"""
        return (not cuda_torch_installed()) and cuda_engine_supported()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for w in (self.btn_add_files, self.btn_add_folder, self.btn_remove, self.btn_clear,
                  self.combo_model, self.combo_device, self.combo_format, self.combo_pcm,
                  self.spin_bigshifts, self.spin_batch, self.combo_quality, self.check_tta,
                  self.edit_out, self.btn_out,
                  # 合成页与功能切换同样锁住，防止任务中切换导致按钮语义错位
                  self.combo_mode,
                  self.mix_btn_add_files, self.mix_btn_add_folder, self.mix_btn_remove,
                  self.mix_btn_clear, self.mix_combo_format, self.mix_combo_pcm,
                  self.spin_vocal_gain, self.spin_inst_gain, self.check_normalize,
                  self.mix_edit_out, self.mix_btn_out):
            w.setEnabled(not busy)
        self.btn_download.setEnabled(not busy)
        # 结束忙碌时不能无脑 setEnabled(True)：未安装 / 不支持时 CUDA 按钮仍应是灰的。
        self.btn_cuda.setEnabled(False if busy else self._cuda_button_allowed())
        self.btn_start.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        self.progress.setEnabled(busy)
        self.list_files.setEnabled(not busy)
        self.mix_list.setEnabled(not busy)

    # ---------- 查看日志 ----------

    def _open_log_dir(self) -> None:
        """打开日志所在文件夹（失败排错的统一入口）。

        打开文件夹而不是 .log 文件：日志在用户机器上未必有关联程序，打开
        文件夹一定可用，用户也能顺手把文件发出去。目录不可用（只读安装目录
        等）时不抛异常——"想看日志"不能反过来再崩一次。
        """
        try:
            target = ensure_log_dir()  # 日志文件还没生成时，先把它所在的目录建出来
            opened = target is not None and QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(target)))
        except Exception as e:  # repo_root 抛错/打开失败都降级成一句提示
            opened = False
            log_exception("打开日志文件夹失败", e)
        if opened:
            return
        QMessageBox.information(
            self, "打不开日志文件夹",
            "暂时打不开日志所在的文件夹（可能是安装目录不能写入）。"
            "如果问题一直出现，把这段提示截图发给支持的人就行。")

    # ---------- 最小化到托盘 ----------

    def _setup_tray(self) -> None:
        """托盘图标不常驻：只有最小化且勾选时才出现。系统没有托盘则禁用勾选。"""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self.check_minimize_tray.setEnabled(False)
            return
        tray = QSystemTrayIcon(QIcon(str(_ICON)), self)
        tray.setToolTip("uvr-lite 正在后台运行")
        # setContextMenu 不接管所有权，菜单必须挂住，否则右键是空的
        menu = QMenu(self)
        act_show = menu.addAction("显示主窗口")
        act_quit = menu.addAction("退出")
        act_show.triggered.connect(self._restore_from_tray)
        # 窗口已经隐藏，关掉它不是最后一个可见窗口，quitOnLastWindowClosed 不会结束事件循环。
        act_quit.triggered.connect(self._quit_from_tray)
        tray.setContextMenu(menu)
        tray.activated.connect(self._on_tray_activated)
        self._tray = tray
        self._tray_menu = menu

    def _on_tray_activated(self, reason) -> None:
        # Windows 左键是 Trigger，双击才是 DoubleClick。两种都回到主窗口。
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self._restore_from_tray()

    def _should_minimize_to_tray(self) -> bool:
        if self._tray is None or not QSystemTrayIcon.isSystemTrayAvailable():
            return False
        return self.check_minimize_tray.isChecked()

    def _hide_to_tray(self) -> None:
        # hide 而不是 close：分离线程和事件循环继续跑；真正退出仍走 closeEvent。
        # 未勾选或系统没有托盘时什么都不做，最小化仍留在任务栏。
        if not self._should_minimize_to_tray():
            return
        self._tray.show()
        self.hide()

    def _restore_from_tray(self) -> None:
        self.showNormal()
        self.setWindowState(self.windowState() & ~Qt.WindowMinimized)
        self.raise_()
        self.activateWindow()
        if self._tray is not None:
            self._tray.hide()

    def _quit_from_tray(self) -> None:
        self._quit_after_close = True
        self.close()
        # 关窗被 wait 超时拦住时先不 quit，等线程结束回调里的 close() 成功后再退。
        if self._quit_after_close and not self._close_after_stop:
            self._quit_after_close = False
            QApplication.quit()

    # ---------- 生命周期 ----------

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() != QEvent.Type.WindowStateChange:
            return
        if not (self.isMinimized() and self._should_minimize_to_tray()):
            return
        # changeEvent 里直接 hide() 会把最小化状态机弄乱，等这一轮事件结束再藏
        QTimer.singleShot(0, self._hide_to_tray)

    def _stop_thread_for_close(self, worker, thread, timeout: int) -> bool:
        """cancel/quit/wait 的返回值都要看。线程已停（含已回收）返回 True。

        只有 wait 返回 False 才表示还在跑。cancel/quit 的 None 不是失败。
        已回收线程走 _is_thread_running，不再 quit/wait。
        """
        if not self._is_thread_running(thread):
            return True
        cancelled = worker.cancel() if worker is not None else None
        quitted = thread.quit()
        waited = thread.wait(timeout)
        # wait 成功说明线程已经停了，cancel/quit 的 False 不能把窗口卡住。
        still_running = waited is False or self._is_thread_running(thread)
        stop_failed = waited is False or cancelled is False or quitted is False
        if stop_failed and still_running:
            self._listen_for_thread_exit(thread)
            return False
        return True

    def _listen_for_thread_exit(self, thread) -> None:
        key = id(thread)
        if key in self._close_listened:
            return
        finished = getattr(thread, "finished", None)
        connect = getattr(finished, "connect", None)
        if connect is None:
            self._close_listened.add(key)
            return
        self._close_listened.add(key)
        single = getattr(Qt, "SingleShotConnection", None)
        try:
            if single is not None:
                connect(self._on_task_thread_finished, single)
            else:
                connect(self._on_task_thread_finished)
        except TypeError:
            connect(self._on_task_thread_finished)

    def _any_task_thread_running(self) -> bool:
        for attr in ("_thread", "_mix_thread", "_dl_thread", "_cuda_thread"):
            if self._is_thread_running(getattr(self, attr, None)):
                return True
        return False

    def _on_task_thread_finished(self) -> None:
        """线程真正结束：若关窗还挂着，再 close 一次。同步重入交给定时器。"""
        if not self._close_after_stop or self._any_task_thread_running():
            return
        if self._close_guard:
            QTimer.singleShot(0, self._on_task_thread_finished)
            return
        self.close()
        if self._quit_after_close and not self._close_after_stop:
            self._quit_after_close = False
            QApplication.quit()

    def closeEvent(self, event) -> None:
        # wait 超时的回调里会再 close()。同步重入时不要再走一遍 cancel/wait。
        if self._close_guard:
            event.ignore()
            return
        self._close_guard = True
        try:
            if self._tray is not None:
                self._tray.hide()
            self._save_settings()
            # 统一走 _is_thread_running：任务跑完后 QThread 已被 deleteLater 回收，
            # 裸调 isRunning() 抛的 RuntimeError 会逃出 closeEvent，后面几个线程就
            # 不再 cancel+wait（运行中的 QThread 带活销毁 → 进程 abort）。
            jobs = (
                (getattr(self, "_worker", None), getattr(self, "_thread", None), 5000),
                (getattr(self, "_mix_worker", None), getattr(self, "_mix_thread", None), 5000),
                (getattr(self, "_dl_worker", None), getattr(self, "_dl_thread", None), 3000),
                (getattr(self, "_cuda_worker", None), getattr(self, "_cuda_thread", None), 3000),
            )
            pending = False
            for worker, thread, timeout in jobs:
                if not self._stop_thread_for_close(worker, thread, timeout):
                    pending = True
            if pending:
                self._close_after_stop = True
                self.label_status.setText("正在停止，请稍候…")
                if not self.isVisible():
                    self.show()
                event.ignore()
                return
            self._close_after_stop = False
            super().closeEvent(event)
        finally:
            self._close_guard = False


class QProgressBarWrap(QWidget):
    """整体进度条容器：对外暴露 .bar，由分离任务的进度信号驱动。"""

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
    app.setWindowIcon(QIcon(str(_ICON)))
    win = MainWindow()
    apply_theme(app, win)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run())
