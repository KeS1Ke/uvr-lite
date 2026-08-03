# coding: utf-8
"""uvr-lite 桌面界面主窗口（PySide6）。

票 2：UI 骨架——文件列表（选择文件/选择文件夹/拖拽）、模型与参数表单、
输出目录（QSettings 记忆）、♪ 窗口图标。推理接线在票 3，开始按钮暂禁用。
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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..download import cuda_torch_installed, models_dir, repo_root
from ..models import MODEL_REGISTRY
from .files import AUDIO_EXTS, dedup_paths, is_audio, precheck_audio, scan_audio_files
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

# 列表项状态前缀（显示在文件名前）
_PREFIX_PENDING = "⏳ "
_PREFIX_OK = "✓ "
_PREFIX_BAD = "✗ "


class ToggleSelectList(QListWidget):
    """点击式多选：点一次选中、再点取消，各文件互不影响（无需 Ctrl）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # SingleSelection 下 setSelected(True) 会清除其他项；Extended 互不影响
        self.setSelectionMode(QListWidget.ExtendedSelection)

    def mousePressEvent(self, event) -> None:  # noqa: N802
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
        self.setWindowTitle("uvr-lite 人声/伴奏分离")
        self.setWindowIcon(QIcon(str(_ICON)))
        self.setAcceptDrops(True)
        self.resize(640, 620)
        self._build_ui()
        self._restore_settings()

    # ---------- 界面 ----------

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)

        # --- 模型状态提示条（缺失时显示，可一键下载）---
        self.banner = QFrame(central)
        self.banner.setObjectName("banner")
        self.banner.setStyleSheet(
            "QFrame#banner { background: #FFF8DC; border: 1px solid #E6C300; border-radius: 4px; }"
        )
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

        # --- 文件列表 ---
        file_box = QGroupBox("待处理音频（可拖拽文件到此处）", central)
        fl = QVBoxLayout(file_box)
        self.list_files = ToggleSelectList(file_box)
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
        root.addWidget(file_box)

        # --- 模型与参数 ---
        param_box = QGroupBox("参数", central)
        form = QFormLayout(param_box)
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
        self.spin_overlap = QSpinBox(param_box)
        self.spin_overlap.setRange(0, 8)
        self.spin_overlap.setSpecialValueText("默认（模型配置）")
        self.check_tta = QCheckBox("测试时增强（3 倍耗时，质量更好）", param_box)
        form.addRow("模型", self.combo_model)
        form.addRow("设备", self.combo_device)
        form.addRow("输出格式", self.combo_format)
        form.addRow("FLAC 位深", self.combo_pcm)
        form.addRow("BigShifts 次数", self.spin_bigshifts)
        form.addRow("批大小（低显存设 1）", self.spin_batch)
        form.addRow("重叠窗口数（1 最快）", self.spin_overlap)
        form.addRow("", self.check_tta)
        root.addWidget(param_box)

        # --- 推理引擎（CPU/CUDA torch）：单包只含 CPU，CUDA 可选下载 ---
        engine_box = QGroupBox("推理引擎", central)
        engine_row = QHBoxLayout(engine_box)
        self.label_engine = QLabel(engine_box)
        self.cuda_progress = QProgressBar(engine_box)
        self.cuda_progress.setFixedWidth(180)
        self.cuda_progress.setVisible(False)
        self.btn_cuda = QPushButton(engine_box)
        engine_row.addWidget(self.label_engine)
        engine_row.addWidget(self.cuda_progress)
        engine_row.addWidget(self.btn_cuda)
        engine_row.addStretch(1)
        root.addWidget(engine_box)
        self.btn_cuda.clicked.connect(self._start_cuda_download)
        self._refresh_engine_status()

        # --- 输出目录 ---
        out_box = QGroupBox("输出文件夹", central)
        out_row = QHBoxLayout(out_box)
        self.edit_out = QLineEdit(out_box)
        self.edit_out.setPlaceholderText("未选择（默认：当前目录/output）")
        self.btn_out = QPushButton("选择…", out_box)
        out_row.addWidget(self.edit_out)
        out_row.addWidget(self.btn_out)
        root.addWidget(out_box)
        self.btn_out.clicked.connect(self._choose_out_dir)

        # --- 操作区 ---
        action_row = QHBoxLayout()
        self.btn_start = QPushButton("开始分离", central)
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
        root.addWidget(self.label_status)

        self.setCentralWidget(central)

        # 控件全部就绪后再挂模型状态联动
        self.combo_model.currentIndexChanged.connect(self._refresh_model_banner)
        self._refresh_model_banner()

    # ---------- 文件列表操作 ----------

    def _add_paths(self, paths: list[Path]) -> None:
        new = dedup_paths(self._paths + paths)
        added = len(new) - len(self._paths)
        self._paths = new
        self._rebuild_list()
        self.label_status.setText(f"已添加 {added} 个文件，共 {len(self._paths)} 个。")

    def _rebuild_list(self) -> None:
        """按 self._paths 重建列表项（行与 _paths 一一对应，状态前缀保留基础名）。"""
        self.list_files.clear()
        for p in self._paths:
            item = QListWidgetItem(p.name, self.list_files)
            item.setData(Qt.UserRole, str(p))

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
        self._rebuild_list()

    def _clear_list(self) -> None:
        self._paths.clear()
        self._rebuild_list()

    def _choose_out_dir(self) -> None:
        start = self.edit_out.text() or str(self.settings.value("last_dir", ""))
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹", start)
        if folder:
            self.edit_out.setText(folder)

    # ---------- 拖拽 ----------

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        dropped = [Path(u.toLocalFile()) for u in event.mimeData().urls()]
        files = [p for p in dropped if p.is_file() and is_audio(p)]
        self._add_paths(files)

    # ---------- 参数记忆 ----------

    def _restore_settings(self) -> None:
        s = self.settings
        self._select_data(self.combo_model, s.value("model", "bs_roformer_ep317"))
        self._select_data(self.combo_device, s.value("device", "auto"))
        self._select_data(self.combo_format, s.value("format", "auto"))
        self._select_data(self.combo_pcm, str(s.value("pcm", "24")))
        self.spin_bigshifts.setValue(int(s.value("bigshifts", 1)))
        self.spin_batch.setValue(int(s.value("batch_size", 0)))
        self.spin_overlap.setValue(int(s.value("num_overlap", 0)))
        self.check_tta.setChecked(bool(s.value("tta", False)))
        out = s.value("out_dir", "")
        if out:
            self.edit_out.setText(str(out))

    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("model", self.combo_model.currentData())
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
            self.label_status.setText("推理引擎已切换，重启 uvr-lite 后生效。")

    # ---------- 模型下载 ----------

    def _refresh_model_banner(self) -> None:
        model = self.combo_model.currentData()
        ready = (models_dir() / f"{model}.ckpt").exists()
        self.banner.setVisible(not ready)
        if not ready:
            label = MODEL_LABELS.get(model, model)
            self.label_banner.setText(f"模型「{label}」未下载（约 640 MB），下载后即可开始分离。")

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
        self.btn_download.setEnabled(True)
        self.btn_download.setText("下载模型")
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
        """按安装状态刷新「推理引擎」区：已装 / 可下载。"""
        if cuda_torch_installed():
            self.label_engine.setText("CUDA 引擎已安装 ✓（设备选「自动」将优先 GPU 加速）")
            self.btn_cuda.setText("已安装")
            self.btn_cuda.setEnabled(False)
        else:
            self.label_engine.setText("CUDA 引擎未安装 — 下载约 3.3 GB 后可用 GPU 加速（NVIDIA 显卡）")
            self.btn_cuda.setText("下载 CUDA 引擎")
            self.btn_cuda.setEnabled(True)

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
        if not (models_dir() / f"{model}.ckpt").exists():
            QMessageBox.information(
                self, "模型未下载",
                "请先点击顶部「下载模型」按钮下载权重（约 640 MB），再开始分离。")
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

        params = {
            "model_name": self.combo_model.currentData(),
            "device": self.combo_device.currentText(),
            "fmt": self.combo_format.currentText(),
            "pcm": f"PCM_{self.combo_pcm.currentText()}",
            "bigshifts": self.spin_bigshifts.value(),
            "batch_size": self.spin_batch.value() or None,
            "num_overlap": self.spin_overlap.value() or None,
            "tta": self.check_tta.isChecked(),
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
        self._file_times: list[float] = []
        self._failed_names: list[str] = []
        self._ok_count = 0
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
        global_pct = int((file_idx + file_pct / 100.0) / max(1, file_total) * 100)
        self.progress.bar.setValue(global_pct)
        eta = estimate_eta(self._file_times, file_idx, file_total, file_pct / 100.0)
        eta_txt = self._fmt_eta(eta) if eta is not None else "计算中…"
        self.label_status.setText(
            f"处理中 {file_idx + 1}/{file_total} · {PHASE_CN.get(phase, phase)} {file_pct}% · 预计剩余 {eta_txt}"
        )

    def _on_file_done(self, file_idx, written) -> None:
        self._file_times.append(time.time() - self._t_file)
        self._t_file = time.time()
        self._ok_count += 1
        if 0 <= file_idx < len(self._ok_paths):
            self._set_item_state(self._ok_paths[file_idx], _PREFIX_OK)

    def _on_file_failed(self, file_idx, error) -> None:
        name = self._ok_paths[file_idx].name
        self._failed_names.append(name)
        if 0 <= file_idx < len(self._ok_paths):
            self._set_item_state(self._ok_paths[file_idx], _PREFIX_BAD)
        self.label_status.setText(f"{name} 处理失败，已跳过（{error[:80]}）")

    def _on_all_finished(self, ok, failed, cancelled) -> None:
        self._set_busy(False)
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
                  self.combo_model, self.combo_device, self.combo_format, self.combo_pcm,
                  self.spin_bigshifts, self.spin_batch, self.spin_overlap, self.check_tta,
                  self.edit_out, self.btn_out):
            w.setEnabled(not busy)
        self.btn_start.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        self.progress.setEnabled(busy)
        self.list_files.setEnabled(not busy)

    # ---------- 生命周期 ----------

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_settings()
        if getattr(self, "_worker", None) is not None and self._thread.isRunning():
            self._worker.cancel()
            self._thread.quit()
            self._thread.wait(5000)
        if getattr(self, "_dl_thread", None) is not None and self._dl_thread.isRunning():
            if getattr(self, "_dl_worker", None) is not None:
                self._dl_worker.cancel()
            self._dl_thread.quit()
            self._dl_thread.wait(3000)
        if getattr(self, "_cuda_thread", None) is not None and self._cuda_thread.isRunning():
            if getattr(self, "_cuda_worker", None) is not None:
                self._cuda_worker.cancel()
            self._cuda_thread.quit()
            self._cuda_thread.wait(3000)
        super().closeEvent(event)


class QProgressBarWrap(QWidget):
    """进度区（票 3 接线；骨架期仅占位）。"""

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
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run())
