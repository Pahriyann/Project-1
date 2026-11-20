"""Teachable Machine-style desktop app: objects, hand gestures, and facial expressions."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtMultimedia import QSoundEffect

from face_tracker import FaceTracker
from hand_tracker import HandTracker
from ml_engine import LiveClassifier


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"
OBJECT_MODEL_PATH = MODELS_DIR / "object_classifier.joblib"
GESTURE_MODEL_PATH = MODELS_DIR / "gesture_classifier.joblib"
FACE_MODEL_PATH = MODELS_DIR / "face_classifier.joblib"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
FACE_MODEL_PATH_TASK = MODELS_DIR / "face_landmarker.task"

FACE_LABEL_DEFAULTS = ("Senang", "Sedih", "Terkejut", "Netral")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Teachable Machine Python — Object, Gesture & Face Studio")
        self.resize(1320, 830)
        self.setMinimumSize(1080, 720)

        self.camera: cv2.VideoCapture | None = None
        self.hand_tracker: HandTracker | None = None
        self.face_tracker: FaceTracker | None = None
        self.capture_mode: str | None = None
        self.capture_label = ""
        self.capture_end_time = 0.0
        self.capture_tick = 0
        self.captured_this_session = 0
        self.last_object_prediction = 0.0
        self.last_gesture_prediction = 0.0
        self.last_face_prediction = 0.0
        self.latest_gesture_label: str | None = None
        self.last_reaction_label: str | None = None
        self.candidate_reaction_label: str | None = None
        self.candidate_reaction_frames = 0
        self.last_reaction_time = 0.0
        self.sticker_expiry = 0.0

        self.object_classifier = LiveClassifier("object", minimum_per_class=10)
        self.gesture_classifier = LiveClassifier("gesture", minimum_per_class=10)
        self.face_classifier = LiveClassifier(
            "face", minimum_per_class=10, required_classes=4
        )
        self.object_classifier.load(OBJECT_MODEL_PATH)
        self.gesture_classifier.load(GESTURE_MODEL_PATH)
        self.face_classifier.load(FACE_MODEL_PATH)

        self.sound_effects: dict[str, QSoundEffect] = {}
        self._build_ui()
        self._load_sound_effects()
        self._open_camera()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(33)

    # ----------------------------- UI construction -------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(18, 16, 18, 14)
        main_layout.setSpacing(12)

        header = QtWidgets.QHBoxLayout()
        title_block = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("TEACHABLE MACHINE PYTHON")
        title.setObjectName("mainTitle")
        subtitle = QtWidgets.QLabel(
            "Latih AI benda, gestur tangan, dan ekspresi wajah langsung dari webcam"
        )
        subtitle.setObjectName("subtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block)
        header.addStretch(1)

        self.camera_status = QtWidgets.QLabel("Menghubungkan kamera...")
        self.camera_status.setObjectName("statusPill")
        header.addWidget(self.camera_status)
        self.reconnect_button = QtWidgets.QPushButton("Hubungkan kamera")
        self.reconnect_button.clicked.connect(self._open_camera)
        header.addWidget(self.reconnect_button)
        main_layout.addLayout(header)

        content = QtWidgets.QHBoxLayout()
        content.setSpacing(14)
        main_layout.addLayout(content, 1)

        camera_card = QtWidgets.QFrame()
        camera_card.setObjectName("panel")
        camera_layout = QtWidgets.QVBoxLayout(camera_card)
        camera_layout.setContentsMargins(12, 12, 12, 12)
        camera_layout.setSpacing(8)
        camera_heading = QtWidgets.QLabel("LIVE CAMERA")
        camera_heading.setObjectName("sectionTitle")
        camera_layout.addWidget(camera_heading)

        self.camera_view = QtWidgets.QLabel("Tampilan webcam akan muncul di sini")
        self.camera_view.setObjectName("cameraView")
        self.camera_view.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.camera_view.setMinimumSize(600, 430)
        self.camera_view.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        camera_layout.addWidget(self.camera_view, 1)

        # The sticker is a child of the camera label, so it appears over the preview.
        self.sticker_overlay = QtWidgets.QLabel(self.camera_view)
        self.sticker_overlay.setObjectName("stickerOverlay")
        self.sticker_overlay.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.sticker_overlay.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.sticker_overlay.setStyleSheet(
            "QLabel#stickerOverlay { background: rgba(20, 25, 40, 225); "
            "color: #ffffff; border: 2px solid #f4c95d; border-radius: 18px; "
            "padding: 8px 18px; font-family: 'Segoe UI Emoji', 'Segoe UI'; "
            "font-weight: 900; }"
        )
        self.sticker_overlay.hide()

        self.camera_hint = QtWidgets.QLabel(
            "Benda: letakkan objek di kotak tengah. Gestur/wajah: posisikan di depan kamera."
        )
        self.camera_hint.setObjectName("mutedText")
        self.camera_hint.setWordWrap(True)
        camera_layout.addWidget(self.camera_hint)
        content.addWidget(camera_card, 6)

        right_panel = QtWidgets.QFrame()
        right_panel.setObjectName("panel")
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(8)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_object_tab(), "Benda")
        self.tabs.addTab(self._build_gesture_tab(), "Gestur tangan")
        self.tabs.addTab(self._build_face_tab(), "Ekspresi wajah")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._refresh_counts()
        right_layout.addWidget(self.tabs, 1)
        content.addWidget(right_panel, 5)

        self.global_status = QtWidgets.QLabel(
            "Siap. Kumpulkan sampel per kelas, lalu latih model."
        )
        self.global_status.setObjectName("bottomStatus")
        main_layout.addWidget(self.global_status)

        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #101827; color: #e5edf7; font-family: Segoe UI; font-size: 10pt; }
            QFrame#panel { background: #182438; border: 1px solid #2a3b55; border-radius: 14px; }
            QLabel#mainTitle { color: #f3f7fc; font-size: 17pt; font-weight: 800; letter-spacing: 1px; }
            QLabel#subtitle, QLabel#mutedText { color: #a6b6ca; }
            QLabel#sectionTitle { color: #a9bdd6; font-weight: 700; letter-spacing: 1px; }
            QLabel#cameraView { background: #0a101b; border: 1px solid #34445d; border-radius: 10px; color: #91a4bd; }
            QLabel#statusPill { background: #23374c; color: #c8e8d7; border-radius: 12px; padding: 7px 12px; font-weight: 700; }
            QLabel#bottomStatus { color: #a9bdd6; padding: 5px 2px; }
            QTabWidget::pane { border: 1px solid #34445d; border-radius: 8px; top: -1px; }
            QTabBar::tab { background: #202f45; color: #b8c8da; padding: 10px 13px; margin-right: 4px; border-top-left-radius: 7px; border-top-right-radius: 7px; }
            QTabBar::tab:selected { background: #31577c; color: white; font-weight: 700; }
            QLineEdit { background: #0f1928; color: #f2f6fc; border: 1px solid #3b506d; border-radius: 7px; padding: 8px; }
            QPushButton { background: #2b425e; color: #edf5ff; border: 1px solid #49617f; border-radius: 8px; padding: 9px; font-weight: 650; }
            QPushButton:hover { background: #365779; }
            QPushButton:disabled { color: #7f91a7; background: #263448; border-color: #34445a; }
            QPushButton#primaryButton { background: #167d63; border-color: #269b7c; color: white; }
            QPushButton#primaryButton:hover { background: #1a9878; }
            QPushButton#captureButton { background: #344a68; }
            QLabel#resultLabel { background: #0d1725; border: 1px solid #31445e; border-radius: 9px; padding: 10px; color: #d8e8fa; font-size: 11pt; font-weight: 700; }
            QLabel#smallInfo { color: #b4c3d5; background: #132033; border-radius: 8px; padding: 8px; }
            QProgressBar { background: #0e1827; border: 1px solid #31445e; border-radius: 5px; height: 10px; text-align: center; }
            QProgressBar::chunk { background: #27a27e; border-radius: 4px; }
            QCheckBox { color: #d9e5f3; spacing: 7px; }
            """
        )

    def _make_label_field(self, default: str) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(default)
        edit.setPlaceholderText("Nama kelas")
        edit.setClearButtonEnabled(True)
        edit.textChanged.connect(self._refresh_counts)
        return edit

    def _build_object_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(8)
        intro = QtWidgets.QLabel(
            "Tambahkan kelas benda sebanyak yang diperlukan (minimal dua). "
            "Saat merekam, gerakkan objek perlahan di area kotak tengah."
        )
        intro.setWordWrap(True)
        intro.setObjectName("mutedText")
        layout.addWidget(intro)

        self.object_counts = self._info_label("Belum ada sampel benda.")
        layout.addWidget(self.object_counts)
        self._build_dynamic_class_area(
            mode="object",
            defaults=self.object_classifier.labels or ["Helm", "Botol minum"],
            parent_layout=layout,
            add_button_text="+ TAMBAH KELAS BENDA",
        )
        self.object_progress = self._progress_bar("Menunggu pengambilan sampel")
        layout.addWidget(self.object_progress)
        self.object_train_button = self._primary_button(
            "LATIH MODEL BENDA", self._train_object_model
        )
        layout.addWidget(self.object_train_button)
        self.object_result = self._result_label("Prediksi benda akan tampil di sini.")
        layout.addWidget(self.object_result)
        clear_button = QtWidgets.QPushButton("Hapus semua sampel dan model benda")
        clear_button.clicked.connect(self._clear_object_samples)
        layout.addWidget(clear_button)
        return page

    def _build_gesture_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(7)
        intro = QtWidgets.QLabel(
            "Tambahkan kelas gestur sebanyak yang diperlukan (minimal dua). "
            "Gunakan satu tangan dan pastikan terlihat utuh."
        )
        intro.setWordWrap(True)
        intro.setObjectName("mutedText")
        layout.addWidget(intro)

        self.hand_model_status = self._info_label("Model pelacak tangan belum diaktifkan.")
        layout.addWidget(self.hand_model_status)
        self.hand_enable_button = QtWidgets.QPushButton("AKTIFKAN PELACAK TANGAN")
        self.hand_enable_button.clicked.connect(self._enable_hand_tracker)
        layout.addWidget(self.hand_enable_button)

        self.gesture_counts = self._info_label("Belum ada sampel gestur.")
        layout.addWidget(self.gesture_counts)
        self._build_dynamic_class_area(
            mode="gesture",
            defaults=self.gesture_classifier.labels or ["Telapak terbuka", "Tangan mengepal"],
            parent_layout=layout,
            add_button_text="+ TAMBAH KELAS GESTUR",
        )
        self.gesture_progress = self._progress_bar("Menunggu pengambilan sampel")
        layout.addWidget(self.gesture_progress)
        self.gesture_train_button = self._primary_button(
            "LATIH MODEL GESTUR", self._train_gesture_model
        )
        layout.addWidget(self.gesture_train_button)
        self.gesture_result = self._result_label("Prediksi gestur akan tampil di sini.")
        layout.addWidget(self.gesture_result)

        self.feedback_checkbox = QtWidgets.QCheckBox(
            "Aktifkan stiker reaksi + efek suara saat gestur berubah"
        )
        self.feedback_checkbox.setChecked(True)
        layout.addWidget(self.feedback_checkbox)
        self.audio_status = QtWidgets.QLabel("Efek suara: aktif (SFX bawaan aplikasi)")
        self.audio_status.setObjectName("mutedText")
        self.audio_status.setWordWrap(True)
        layout.addWidget(self.audio_status)

        clear_button = QtWidgets.QPushButton("Hapus semua sampel dan model gestur")
        clear_button.clicked.connect(self._clear_gesture_samples)
        layout.addWidget(clear_button)
        return page

    def _build_face_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(6)
        intro = QtWidgets.QLabel(
            "Latih empat ekspresi dari volunteer: senang, sedih, terkejut, dan netral. "
            "Model belajar dari skor blendshape wajah yang terlihat kamera."
        )
        intro.setWordWrap(True)
        intro.setObjectName("mutedText")
        layout.addWidget(intro)

        self.face_model_status = self._info_label("Model pelacak wajah belum diaktifkan.")
        layout.addWidget(self.face_model_status)
        self.face_enable_button = QtWidgets.QPushButton("AKTIFKAN PELACAK WAJAH")
        self.face_enable_button.clicked.connect(self._enable_face_tracker)
        layout.addWidget(self.face_enable_button)

        self.face_label_edits: list[QtWidgets.QLineEdit] = []
        self.face_capture_buttons: list[QtWidgets.QPushButton] = []
        for index, default_label in enumerate(FACE_LABEL_DEFAULTS, start=1):
            layout.addWidget(self._class_heading(f"EKSPRESI {index}"))
            edit = self._make_label_field(default_label)
            button = self._capture_button(
                f"Rekam ekspresi {index} · 3 detik", "face", edit
            )
            self.face_label_edits.append(edit)
            self.face_capture_buttons.append(button)
            layout.addWidget(edit)
            layout.addWidget(button)

        self.face_counts = self._info_label("Belum ada sampel ekspresi wajah.")
        layout.addWidget(self.face_counts)
        self.face_progress = self._progress_bar("Menunggu pengambilan sampel")
        layout.addWidget(self.face_progress)
        self.face_train_button = self._primary_button(
            "LATIH MODEL EKSPRESI WAJAH", self._train_face_model
        )
        layout.addWidget(self.face_train_button)
        self.face_result = self._result_label("Prediksi ekspresi akan tampil di sini.")
        layout.addWidget(self.face_result)
        clear_button = QtWidgets.QPushButton("Hapus sampel dan model wajah")
        clear_button.clicked.connect(self._clear_face_samples)
        layout.addWidget(clear_button)
        note = QtWidgets.QLabel(
            "Catatan: aplikasi memperkirakan ekspresi visual, bukan memastikan perasaan seseorang."
        )
        note.setObjectName("mutedText")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        return page

    @staticmethod
    def _class_heading(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setStyleSheet(
            "color: #8fa9c5; font-size: 9pt; font-weight: 800; letter-spacing: 1px;"
        )
        return label

    @staticmethod
    def _info_label(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("smallInfo")
        label.setWordWrap(True)
        return label

    @staticmethod
    def _result_label(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("resultLabel")
        label.setWordWrap(True)
        return label

    @staticmethod
    def _progress_bar(message: str) -> QtWidgets.QProgressBar:
        bar = QtWidgets.QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setFormat(message)
        return bar

    @staticmethod
    def _primary_button(text: str, callback) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.setObjectName("primaryButton")
        button.clicked.connect(callback)
        return button

    def _capture_button(
        self, text: str, mode: str, label_edit: QtWidgets.QLineEdit
    ) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.setObjectName("captureButton")
        button.clicked.connect(lambda: self._start_capture(mode, label_edit.text()))
        return button

    # ------------------------- Dynamic class controls ----------------------
    def _build_dynamic_class_area(
        self,
        mode: str,
        defaults: list[str],
        parent_layout: QtWidgets.QVBoxLayout,
        add_button_text: str,
    ) -> None:
        rows: list[dict] = []
        setattr(self, f"{mode}_class_rows", rows)

        holder = QtWidgets.QWidget()
        class_layout = QtWidgets.QVBoxLayout(holder)
        class_layout.setContentsMargins(2, 2, 2, 2)
        class_layout.setSpacing(7)
        setattr(self, f"{mode}_class_layout", class_layout)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(205)
        scroll.setWidget(holder)
        parent_layout.addWidget(scroll, 1)

        add_button = QtWidgets.QPushButton(add_button_text)
        add_button.clicked.connect(lambda _checked=False, m=mode: self._add_dynamic_class(m))
        parent_layout.addWidget(add_button)
        setattr(self, f"{mode}_add_class_button", add_button)

        for label in defaults:
            self._add_class_row(mode, label)

    def _rows_for_mode(self, mode: str) -> list[dict]:
        return getattr(self, f"{mode}_class_rows")

    def _classifier_for_mode(self, mode: str) -> LiveClassifier:
        return self.object_classifier if mode == "object" else self.gesture_classifier

    def _model_path_for_mode(self, mode: str) -> Path:
        return OBJECT_MODEL_PATH if mode == "object" else GESTURE_MODEL_PATH

    def _invalidate_model(self, mode: str) -> None:
        classifier = self._classifier_for_mode(mode)
        classifier.pipeline = None
        classifier.labels = []
        self._model_path_for_mode(mode).unlink(missing_ok=True)
        if mode == "object":
            self.object_result.setText("Kelas atau sampel berubah. Latih ulang model benda.")
        else:
            self.gesture_result.setText("Kelas atau sampel berubah. Latih ulang model gestur.")
            self.last_reaction_label = None
            self.candidate_reaction_label = None
            self.candidate_reaction_frames = 0
            self.sticker_overlay.hide()

    def _add_class_row(self, mode: str, label: str) -> None:
        rows = self._rows_for_mode(mode)
        if len(rows) >= 12:
            QtWidgets.QMessageBox.information(
                self, "Batas kelas", "Versi ini mendukung maksimal 12 kelas per mode."
            )
            return
        if any(row["edit"].text().strip().casefold() == label.casefold() for row in rows):
            label = self._next_class_name(mode)

        card = QtWidgets.QFrame()
        card.setStyleSheet(
            "QFrame { background: #132033; border: 1px solid #344963; border-radius: 8px; }"
        )
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(9, 7, 9, 7)
        card_layout.setSpacing(6)

        top = QtWidgets.QHBoxLayout()
        heading = QtWidgets.QLabel(f"KELAS {len(rows) + 1}")
        heading.setStyleSheet("color: #8fa9c5; font-weight: 800; letter-spacing: 1px;")
        edit = QtWidgets.QLineEdit(label)
        edit.setPlaceholderText("Nama kelas")
        edit.setClearButtonEnabled(True)
        top.addWidget(heading)
        top.addWidget(edit, 1)
        card_layout.addLayout(top)

        bottom = QtWidgets.QHBoxLayout()
        capture = QtWidgets.QPushButton("REKAM SAMPEL · 3 DETIK")
        capture.setObjectName("captureButton")
        count = QtWidgets.QLabel("Sampel: 0")
        count.setObjectName("mutedText")
        remove = QtWidgets.QPushButton("Hapus kelas")
        remove.setMaximumWidth(105)
        bottom.addWidget(capture)
        bottom.addWidget(count, 1)
        bottom.addWidget(remove)
        card_layout.addLayout(bottom)

        row = {
            "widget": card,
            "heading": heading,
            "edit": edit,
            "capture": capture,
            "count": count,
            "remove": remove,
            "last_label": label,
        }
        rows.append(row)
        getattr(self, f"{mode}_class_layout").addWidget(card)

        capture.clicked.connect(
            lambda _checked=False, m=mode, r=row: self._start_capture_for_row(m, r)
        )
        remove.clicked.connect(
            lambda _checked=False, m=mode, r=row: self._remove_class_row(m, r)
        )
        edit.editingFinished.connect(lambda m=mode, r=row: self._commit_class_label(m, r))

        self._refresh_class_row_titles(mode)

    def _next_class_name(self, mode: str) -> str:
        base = "Benda" if mode == "object" else "Gestur"
        used = {row["edit"].text().strip().casefold() for row in self._rows_for_mode(mode)}
        number = 1
        while f"{base} {number}".casefold() in used:
            number += 1
        return f"{base} {number}"

    def _add_dynamic_class(self, mode: str) -> None:
        if self.capture_mode is not None:
            return
        if len(self._rows_for_mode(mode)) >= 12:
            QtWidgets.QMessageBox.information(
                self, "Batas kelas", "Maksimal 12 kelas per mode untuk menjaga penggunaan memori."
            )
            return
        self._add_class_row(mode, self._next_class_name(mode))
        self._invalidate_model(mode)
        self.global_status.setText("Kelas baru ditambahkan. Ambil sampel untuk semua kelas, lalu latih ulang.")

    def _refresh_class_row_titles(self, mode: str) -> None:
        rows = self._rows_for_mode(mode)
        for index, row in enumerate(rows, start=1):
            row["heading"].setText(f"KELAS {index}")
            row["remove"].setEnabled(len(rows) > 2 and self.capture_mode is None)

    def _commit_class_label(self, mode: str, row: dict) -> bool:
        new_label = row["edit"].text().strip()
        old_label = row["last_label"]
        if not new_label:
            QtWidgets.QMessageBox.warning(self, "Nama kelas kosong", "Nama kelas tidak boleh kosong.")
            row["edit"].setText(old_label)
            return False

        for other in self._rows_for_mode(mode):
            if other is not row and other["edit"].text().strip().casefold() == new_label.casefold():
                QtWidgets.QMessageBox.warning(
                    self, "Nama kelas duplikat", "Setiap kelas harus memiliki nama yang berbeda."
                )
                row["edit"].setText(old_label)
                return False

        if new_label != old_label:
            classifier = self._classifier_for_mode(mode)
            if old_label in classifier.samples:
                if new_label in classifier.samples:
                    QtWidgets.QMessageBox.warning(
                        self, "Nama kelas sudah digunakan", "Nama ini sudah memiliki sampel."
                    )
                    row["edit"].setText(old_label)
                    return False
                classifier.samples[new_label] = classifier.samples.pop(old_label)
            row["last_label"] = new_label
            self._invalidate_model(mode)
            self._refresh_counts()
            self.global_status.setText(
                f"Nama kelas diubah menjadi '{new_label}'. Latih ulang model setelah memastikan semua kelas memiliki sampel."
            )
        return True

    def _start_capture_for_row(self, mode: str, row: dict) -> None:
        if self._commit_class_label(mode, row):
            self._start_capture(mode, row["last_label"])

    def _remove_class_row(self, mode: str, row: dict) -> None:
        rows = self._rows_for_mode(mode)
        if self.capture_mode is not None:
            return
        if not self._commit_class_label(mode, row):
            return
        if len(rows) <= 2:
            QtWidgets.QMessageBox.information(
                self, "Minimal dua kelas", "Model klasifikasi membutuhkan minimal dua kelas."
            )
            return
        label = row["last_label"]
        answer = QtWidgets.QMessageBox.question(
            self,
            "Hapus kelas",
            f"Hapus kelas '{label}' beserta semua sampelnya?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._classifier_for_mode(mode).samples.pop(label, None)
        rows.remove(row)
        row["widget"].setParent(None)
        row["widget"].deleteLater()
        self._refresh_class_row_titles(mode)
        self._invalidate_model(mode)
        self._refresh_counts()
        self.global_status.setText(f"Kelas '{label}' dihapus. Latih ulang model dengan kelas yang tersisa.")

    # ----------------------------- Assets and audio -------------------------
    def _load_sound_effects(self) -> None:
        sound_files = {
            "pop": ASSETS_DIR / "sfx" / "pop.wav",
            "boing": ASSETS_DIR / "sfx" / "boing.wav",
            "sparkle": ASSETS_DIR / "sfx" / "sparkle.wav",
        }
        try:
            for name, path in sound_files.items():
                if not path.exists():
                    continue
                effect = QSoundEffect(self)
                effect.setSource(QtCore.QUrl.fromLocalFile(str(path)))
                effect.setLoopCount(1)
                effect.setVolume(0.42)
                self.sound_effects[name] = effect
            if len(self.sound_effects) != len(sound_files):
                self.audio_status.setText(
                    "Efek suara tidak lengkap. Jalankan aplikasi dari folder proyek agar file assets terbaca."
                )
        except Exception as exc:
            self.audio_status.setText(f"Audio tidak dapat disiapkan: {exc}")

    # ----------------------------- Camera ----------------------------------
    def _open_camera(self) -> None:
        if self.camera is not None:
            self.camera.release()
            self.camera = None

        backends = []
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(cv2.CAP_DSHOW)
        backends.append(cv2.CAP_ANY)

        for backend in backends:
            camera = cv2.VideoCapture(0, backend)
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                self.camera = camera
                self.camera_status.setText("KAMERA TERHUBUNG")
                self.camera_status.setStyleSheet(
                    "background: #174f43; color: #d2ffeb; border-radius: 12px; "
                    "padding: 7px 12px; font-weight: 700;"
                )
                self.global_status.setText("Kamera aktif. Pilih mode dan kumpulkan sampel.")
                return
            camera.release()

        self.camera_status.setText("KAMERA TIDAK TERDETEKSI")
        self.camera_status.setStyleSheet(
            "background: #65333a; color: #ffe1e3; border-radius: 12px; "
            "padding: 7px 12px; font-weight: 700;"
        )
        self.camera_view.setText(
            "Webcam tidak dapat dibuka.\nPeriksa koneksi kamera dan izin Camera Windows,\n"
            "kemudian klik Hubungkan kamera."
        )

    def _update_frame(self) -> None:
        if self.camera is None or not self.camera.isOpened():
            return
        ok, frame = self.camera.read()
        if not ok or frame is None:
            return

        # Mirror the preview; samples and inference use the same mirrored view.
        frame = cv2.flip(frame, 1)
        now = time.monotonic()
        active_tab = self.tabs.currentIndex()

        if active_tab == 0:
            self._draw_object_guide(frame)
            if self.capture_mode == "object":
                self.capture_tick += 1
                if self.capture_tick % 3 == 0:
                    self._collect_object_sample(frame)
            elif self.object_classifier.is_trained and now - self.last_object_prediction >= 0.25:
                self.last_object_prediction = now
                try:
                    prediction = self.object_classifier.predict_object_frame(frame)
                    self._show_prediction(self.object_result, prediction, "Benda")
                except Exception as exc:
                    self.object_result.setText(f"Prediksi gagal: {exc}")

        elif active_tab == 1:
            if self.hand_tracker is not None:
                try:
                    timestamp_ms = int(time.monotonic() * 1000)
                    feature, pixel_points = self.hand_tracker.process(frame, timestamp_ms)
                    if pixel_points:
                        self._draw_hand_landmarks(frame, pixel_points)
                    if self.capture_mode == "gesture":
                        self.capture_tick += 1
                        if self.capture_tick % 2 == 0 and feature is not None:
                            self.gesture_classifier.add_feature(
                                self.capture_label, feature, max_samples=400
                            )
                            self.captured_this_session += 1
                    elif (
                        self.gesture_classifier.is_trained
                        and feature is not None
                        and now - self.last_gesture_prediction >= 0.20
                    ):
                        self.last_gesture_prediction = now
                        prediction = self.gesture_classifier.predict_feature(feature)
                        self._show_prediction(self.gesture_result, prediction, "Gestur")
                        self._handle_gesture_reaction(prediction, now)
                    elif feature is None and self.gesture_classifier.is_trained:
                        self.gesture_result.setText(
                            "Tangan belum terdeteksi. Tunjukkan satu tangan ke kamera."
                        )
                except Exception as exc:
                    self.hand_model_status.setText(f"Pelacakan tangan bermasalah: {exc}")

        else:  # Face expression tab
            if self.face_tracker is not None:
                try:
                    timestamp_ms = int(time.monotonic() * 1000)
                    feature, face_points = self.face_tracker.process(frame, timestamp_ms)
                    if face_points:
                        self._draw_face_points(frame, face_points)
                    if self.capture_mode == "face":
                        self.capture_tick += 1
                        if self.capture_tick % 2 == 0 and feature is not None:
                            self.face_classifier.add_feature(
                                self.capture_label, feature, max_samples=500
                            )
                            self.captured_this_session += 1
                    elif (
                        self.face_classifier.is_trained
                        and feature is not None
                        and now - self.last_face_prediction >= 0.20
                    ):
                        self.last_face_prediction = now
                        prediction = self.face_classifier.predict_feature(feature)
                        self._show_prediction(self.face_result, prediction, "Ekspresi")
                    elif feature is None and self.face_classifier.is_trained:
                        self.face_result.setText(
                            "Wajah belum terdeteksi. Posisikan wajah di depan kamera."
                        )
                except Exception as exc:
                    self.face_model_status.setText(f"Pelacakan wajah bermasalah: {exc}")

        if self.capture_mode is not None:
            remaining = max(0.0, self.capture_end_time - now)
            progress_value = int((1.0 - remaining / 3.0) * 100)
            progress_value = max(0, min(100, progress_value))
            progress = self._progress_for_mode(self.capture_mode)
            progress.setValue(progress_value)
            progress.setFormat(f"Merekam... {remaining:.1f} detik")
            if now >= self.capture_end_time:
                self._finish_capture()

        if now >= self.sticker_expiry:
            self.sticker_overlay.hide()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = QtGui.QImage(
            rgb.data, width, height, int(rgb.strides[0]), QtGui.QImage.Format.Format_RGB888
        ).copy()
        pixmap = QtGui.QPixmap.fromImage(image)
        self.camera_view.setPixmap(
            pixmap.scaled(
                self.camera_view.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )
        if self.sticker_overlay.isVisible():
            self.sticker_overlay.adjustSize()
            self.sticker_overlay.move(
                max(8, (self.camera_view.width() - self.sticker_overlay.width()) // 2),
                max(8, self.camera_view.height() - self.sticker_overlay.height() - 22),
            )
            self.sticker_overlay.raise_()

    @staticmethod
    def _draw_object_guide(frame) -> None:
        height, width = frame.shape[:2]
        side = int(min(height, width) * 0.72)
        x1, y1 = (width - side) // 2, (height - side) // 2
        x2, y2 = x1 + side, y1 + side
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 160), 2)
        cv2.putText(
            frame,
            "PLACE OBJECT IN THIS AREA",
            (max(12, x1), max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (60, 220, 160),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_hand_landmarks(frame, points) -> None:
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
            (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
        ]
        for start, end in connections:
            cv2.line(frame, points[start], points[end], (70, 230, 170), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 4, (40, 120, 255), -1, cv2.LINE_AA)

    @staticmethod
    def _draw_face_points(frame, points) -> None:
        for point in points:
            cv2.circle(frame, point, 3, (90, 230, 170), -1, cv2.LINE_AA)

    # ----------------------------- Capture ---------------------------------
    def _start_capture(self, mode: str, label: str) -> None:
        label = label.strip()
        if not label:
            QtWidgets.QMessageBox.warning(
                self, "Nama kelas kosong", "Isi nama kelas terlebih dahulu."
            )
            return
        if self.camera is None or not self.camera.isOpened():
            QtWidgets.QMessageBox.warning(
                self, "Kamera belum aktif", "Hubungkan webcam terlebih dahulu."
            )
            return
        if self.capture_mode is not None:
            return
        if mode == "gesture" and self.hand_tracker is None:
            QtWidgets.QMessageBox.information(
                self,
                "Pelacak tangan belum aktif",
                "Klik AKTIFKAN PELACAK TANGAN terlebih dahulu.",
            )
            return
        if mode == "face" and self.face_tracker is None:
            QtWidgets.QMessageBox.information(
                self,
                "Pelacak wajah belum aktif",
                "Klik AKTIFKAN PELACAK WAJAH terlebih dahulu.",
            )
            return

        if mode in {"object", "gesture"}:
            self._invalidate_model(mode)

        self.capture_mode = mode
        self.capture_label = label
        self.capture_end_time = time.monotonic() + 3.0
        self.capture_tick = 0
        self.captured_this_session = 0
        self._set_capture_buttons_enabled(False)
        progress = self._progress_for_mode(mode)
        progress.setValue(0)
        progress.setFormat("Mulai merekam...")
        self.global_status.setText(f"Mengambil sampel untuk kelas '{label}'.")

    def _collect_object_sample(self, frame) -> None:
        try:
            self.object_classifier.add_object_frame(self.capture_label, frame)
            self.captured_this_session += 1
        except Exception as exc:
            self.global_status.setText(f"Gagal mengambil sampel benda: {exc}")

    def _finish_capture(self) -> None:
        mode = self.capture_mode
        label = self.capture_label
        count = self.captured_this_session
        self.capture_mode = None
        self.capture_label = ""
        self._set_capture_buttons_enabled(True)
        target = self._progress_for_mode(mode)
        target.setValue(100)
        target.setFormat(f"Selesai · {count} sampel baru")
        self._refresh_counts()

        tracker_missing = count == 0 and mode in {"gesture", "face"}
        if tracker_missing:
            subject = "Tangan" if mode == "gesture" else "Wajah"
            self.global_status.setText(
                f"{subject} tidak terdeteksi. Coba ulangi dengan posisi lebih jelas."
            )
        else:
            self.global_status.setText(
                f"Pengambilan sampel '{label}' selesai: {count} sampel baru."
            )

    def _set_capture_buttons_enabled(self, enabled: bool) -> None:
        for mode in ("object", "gesture"):
            add_button = getattr(self, f"{mode}_add_class_button")
            add_button.setEnabled(enabled)
            rows = self._rows_for_mode(mode)
            for row in rows:
                row["capture"].setEnabled(enabled)
                row["remove"].setEnabled(enabled and len(rows) > 2)
                row["edit"].setEnabled(enabled)
        for button in self.face_capture_buttons:
            button.setEnabled(enabled)

    def _progress_for_mode(self, mode: str) -> QtWidgets.QProgressBar:
        return {
            "object": self.object_progress,
            "gesture": self.gesture_progress,
            "face": self.face_progress,
        }[mode]

    # ----------------------------- Training --------------------------------
    def _validated_dynamic_labels(self, mode: str, subject: str) -> list[str] | None:
        rows = self._rows_for_mode(mode)
        labels: list[str] = []
        for row in rows:
            if not self._commit_class_label(mode, row):
                return None
            labels.append(row["last_label"])

        if len(labels) < 2:
            QtWidgets.QMessageBox.warning(
                self, "Kelas belum cukup", f"Tambahkan minimal dua kelas {subject}."
            )
            return None
        if len({label.casefold() for label in labels}) != len(labels):
            QtWidgets.QMessageBox.warning(
                self, "Nama kelas duplikat", "Setiap kelas harus memiliki nama yang berbeda."
            )
            return None

        classifier = self._classifier_for_mode(mode)
        if set(classifier.samples.keys()) != set(labels):
            missing = [label for label in labels if label not in classifier.samples]
            extra = [label for label in classifier.samples if label not in labels]
            details = []
            if missing:
                details.append("Belum ada sampel: " + ", ".join(missing))
            if extra:
                details.append("Sampel tidak terpakai: " + ", ".join(extra))
            QtWidgets.QMessageBox.warning(
                self,
                "Sampel belum lengkap",
                "Ambil sampel untuk setiap kelas dengan nama yang sesuai. " + " ".join(details),
            )
            return None
        return labels

    def _train_object_model(self) -> None:
        labels = self._validated_dynamic_labels("object", "benda")
        if labels is None:
            return
        try:
            self.object_classifier.train()
            self.object_classifier.save(OBJECT_MODEL_PATH)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Model benda belum siap", str(exc))
            return
        self.object_result.setText(
            f"Model benda berhasil dilatih dengan {len(labels)} kelas: {', '.join(labels)}."
        )
        self.global_status.setText(
            f"Model benda selesai dilatih dengan {len(labels)} kelas dan otomatis disimpan."
        )

    def _train_gesture_model(self) -> None:
        labels = self._validated_dynamic_labels("gesture", "gestur")
        if labels is None:
            return
        if self.hand_tracker is None:
            QtWidgets.QMessageBox.information(
                self, "Pelacak belum aktif", "Aktifkan pelacak tangan terlebih dahulu."
            )
            return
        try:
            self.gesture_classifier.train()
            self.gesture_classifier.save(GESTURE_MODEL_PATH)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Model gestur belum siap", str(exc))
            return
        self.gesture_result.setText(
            f"Model gestur berhasil dilatih dengan {len(labels)} kelas: {', '.join(labels)}."
        )
        self.global_status.setText(
            f"Model gestur selesai dilatih dengan {len(labels)} kelas dan otomatis disimpan."
        )

    def _train_face_model(self) -> None:
        labels = [edit.text().strip() for edit in self.face_label_edits]
        if not all(labels) or len(set(labels)) != 4:
            QtWidgets.QMessageBox.warning(
                self,
                "Periksa nama ekspresi",
                "Isi empat nama ekspresi yang berbeda dan tidak kosong.",
            )
            return
        if self.face_tracker is None:
            QtWidgets.QMessageBox.information(
                self, "Pelacak belum aktif", "Aktifkan pelacak wajah terlebih dahulu."
            )
            return
        if set(self.face_classifier.samples.keys()) != set(labels):
            QtWidgets.QMessageBox.warning(
                self,
                "Sampel belum sesuai",
                "Ambil sampel untuk keempat ekspresi. Nama kelas harus sama dengan nama "
                "yang dipakai saat merekam.",
            )
            return
        try:
            self.face_classifier.train()
            self.face_classifier.save(FACE_MODEL_PATH)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Model ekspresi belum siap", str(exc)
            )
            return
        self.face_result.setText(
            "Model ekspresi berhasil dilatih. Peragakan salah satu ekspresi di depan kamera."
        )
        self.global_status.setText("Model ekspresi wajah selesai dilatih dan disimpan lokal.")

    # ----------------------------- Trackers --------------------------------
    def _enable_hand_tracker(self) -> None:
        if not HAND_MODEL_PATH.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Model tangan belum tersedia",
                "File hand_landmarker.task belum ditemukan.\n\n"
                "Jalankan di terminal VS Code:\n"
                ".\\.venv\\Scripts\\python.exe download_models.py",
            )
            return
        try:
            if self.hand_tracker is not None:
                self.hand_tracker.close()
            self.hand_tracker = HandTracker(HAND_MODEL_PATH)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Gagal mengaktifkan pelacak tangan", str(exc))
            return
        self.hand_model_status.setText(
            "Pelacak tangan aktif. Tampilkan satu tangan di depan kamera."
        )
        self.hand_enable_button.setText("PELACAK TANGAN AKTIF")
        self.hand_enable_button.setEnabled(False)
        self.global_status.setText("Pelacakan tangan aktif. Tambahkan kelas gestur dan kumpulkan sampelnya.")

    def _enable_face_tracker(self) -> None:
        if not FACE_MODEL_PATH_TASK.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Model wajah belum tersedia",
                "File face_landmarker.task belum ditemukan.\n\n"
                "Jalankan di terminal VS Code:\n"
                ".\\.venv\\Scripts\\python.exe download_models.py",
            )
            return
        try:
            if self.face_tracker is not None:
                self.face_tracker.close()
            self.face_tracker = FaceTracker(FACE_MODEL_PATH_TASK)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Gagal mengaktifkan pelacak wajah", str(exc))
            return
        self.face_model_status.setText(
            "Pelacak wajah aktif. Posisikan satu wajah di depan kamera."
        )
        self.face_enable_button.setText("PELACAK WAJAH AKTIF")
        self.face_enable_button.setEnabled(False)
        self.global_status.setText(
            "Pelacakan wajah aktif. Kumpulkan sampel empat ekspresi secara bergantian."
        )

    # ----------------------------- Gesture reactions -----------------------
    @staticmethod
    def _reaction_for_label(label: str) -> tuple[str, str, str]:
        lowered = label.casefold()
        if any(word in lowered for word in ("telapak", "terbuka", "open", "wave")):
            return "👋", "HELLO!", "pop"
        if any(word in lowered for word in ("mengep", "kepal", "fist", "power")):
            return "😎", "POWER MODE!", "boing"
        if any(word in lowered for word in ("jempol", "thumb", "like", "bagus")):
            return "👍", "NICE!", "sparkle"
        if any(word in lowered for word in ("peace", "victory", "dua jari")):
            return "✌️", "PEACE!", "pop"
        return "🎉", "GESTURE DETECTED!", "sparkle"

    def _handle_gesture_reaction(self, prediction, now: float) -> None:
        if prediction is None:
            self.candidate_reaction_label = None
            self.candidate_reaction_frames = 0
            return
        label, confidence = prediction
        if confidence < 0.68:
            self.candidate_reaction_label = None
            self.candidate_reaction_frames = 0
            return
        if label == self.last_reaction_label:
            self.candidate_reaction_label = None
            self.candidate_reaction_frames = 0
            return  # One held gesture only triggers once.

        # Require three consistent prediction updates to avoid reaction flicker.
        if label == self.candidate_reaction_label:
            self.candidate_reaction_frames += 1
        else:
            self.candidate_reaction_label = label
            self.candidate_reaction_frames = 1
        if self.candidate_reaction_frames < 3:
            return
        if now - self.last_reaction_time < 1.25:
            return

        self.last_reaction_label = label
        self.last_reaction_time = now
        self.candidate_reaction_label = None
        self.candidate_reaction_frames = 0
        if not self.feedback_checkbox.isChecked():
            return

        emoji, headline, sound_name = self._reaction_for_label(label)
        self.sticker_overlay.setText(
            "<div align='center'>"
            f"<span style='font-size:38pt'>{emoji}</span><br>"
            f"<span style='font-size:17pt; letter-spacing:1px'>{headline}</span><br>"
            f"<span style='font-size:10pt; color:#d0d9e8'>AI: {label}</span>"
            "</div>"
        )
        self.sticker_overlay.adjustSize()
        self.sticker_overlay.show()
        self.sticker_expiry = now + 2.0
        self.sticker_overlay.raise_()

        sound = self.sound_effects.get(sound_name)
        if sound is not None:
            sound.stop()
            sound.play()

    # ----------------------------- Helpers ---------------------------------
    def _refresh_counts(self, *_args) -> None:
        self.object_counts.setText(
            self._format_counts(self.object_classifier.counts(), "Sampel benda")
        )
        self.gesture_counts.setText(
            self._format_counts(self.gesture_classifier.counts(), "Sampel gestur")
        )
        self.face_counts.setText(
            self._format_counts(self.face_classifier.counts(), "Sampel ekspresi")
        )
        for mode in ("object", "gesture"):
            counts = self._classifier_for_mode(mode).counts()
            for row in self._rows_for_mode(mode):
                label = row["last_label"]
                row["count"].setText(f"Sampel: {counts.get(label, 0)}")

    @staticmethod
    def _format_counts(counts: dict[str, int], title: str) -> str:
        if not counts:
            return f"{title}: belum ada sampel. Minimal 10 sampel per kelas."
        detail = "  |  ".join(f"{label}: {count}" for label, count in counts.items())
        return f"{title}: {detail}  ·  Minimal 10 per kelas"

    @staticmethod
    def _show_prediction(target: QtWidgets.QLabel, prediction, subject: str) -> None:
        if prediction is None:
            target.setText(f"Belum ada prediksi {subject.lower()}.")
            return
        label, confidence = prediction
        percent = confidence * 100
        if confidence < 0.55:
            target.setText(
                f"Belum yakin ({percent:.0f}%). Coba posisikan {subject.lower()} lebih jelas."
            )
        else:
            target.setText(f"{subject}: {label}\nSkor keyakinan model: {percent:.1f}%")

    def _clear_object_samples(self) -> None:
        self.object_classifier.clear_samples()
        OBJECT_MODEL_PATH.unlink(missing_ok=True)
        self.object_result.setText("Sampel dan model benda dihapus.")
        self.object_progress.setValue(0)
        self._refresh_counts()

    def _clear_gesture_samples(self) -> None:
        self.gesture_classifier.clear_samples()
        GESTURE_MODEL_PATH.unlink(missing_ok=True)
        self.gesture_result.setText("Sampel dan model gestur dihapus.")
        self.gesture_progress.setValue(0)
        self.last_reaction_label = None
        self.candidate_reaction_label = None
        self.candidate_reaction_frames = 0
        self.last_reaction_time = 0.0
        self.sticker_overlay.hide()
        self._refresh_counts()

    def _clear_face_samples(self) -> None:
        self.face_classifier.clear_samples()
        FACE_MODEL_PATH.unlink(missing_ok=True)
        self.face_result.setText("Sampel dan model ekspresi dihapus.")
        self.face_progress.setValue(0)
        self._refresh_counts()

    def _on_tab_changed(self, index: int) -> None:
        hints = {
            0: "Mode benda: letakkan satu objek di dalam kotak tengah.",
            1: "Mode gestur: tunjukkan satu tangan ke kamera; stiker dan SFX muncul saat kelas berubah.",
            2: "Mode wajah: posisikan satu wajah di tengah dan peragakan ekspresinya.",
        }
        self.camera_hint.setText(hints.get(index, "Pilih mode demonstrasi."))
        if index != 1:
            self.sticker_overlay.hide()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if hasattr(self, "timer"):
            self.timer.stop()
        if self.camera is not None:
            self.camera.release()
        if self.hand_tracker is not None:
            self.hand_tracker.close()
        if self.face_tracker is not None:
            self.face_tracker.close()
        for effect in self.sound_effects.values():
            effect.stop()
        event.accept()


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Teachable Machine Python")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
