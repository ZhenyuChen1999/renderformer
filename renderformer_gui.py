#!/usr/bin/env python3
import os
import sys
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QPushButton, QLineEdit, QLabel, 
                              QFileDialog, QTextEdit)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap, QImage

PYTHON = r"C:\Users\sh-gfx-perf\AppData\Local\Programs\Python\Python312\python.exe"
SCRIPT_DIR = Path(__file__).parent.absolute()
SCENE_PROCESSOR = SCRIPT_DIR / "scene_processor" / "convert_scene.py"

sys.path.insert(0, str(SCRIPT_DIR))
from infer import render_to_array


class RenderThread(QThread):
    status_update = pyqtSignal(str)
    image_ready = pyqtSignal(np.ndarray)
    render_time = pyqtSignal(float)
    error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.scene_path = None
        self.h5_path = None
        self.running = False
        self.rendering = False
    
    def set_scene(self, scene_path):
        self.scene_path = scene_path
        self.rendering = True
    
    def stop(self):
        self.running = False
        self.rendering = False
    
    def run(self):
        import time
        import traceback
        
        self.running = True
        self.status_update.emit("Thread ready")
        
        while self.running:
            if self.rendering and self.h5_path:
                try:
                    start_time = time.time()
                    
                    image_array = render_to_array(
                        h5_file=self.h5_path,
                        gamma=2.2,
                        resolution=512
                    )
                    
                    elapsed_time = time.time() - start_time
                    self.render_time.emit(elapsed_time)
                    self.image_ready.emit(image_array)
                    
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}"
                    self.error.emit(error_msg)
                    print(error_msg)
                    self.rendering = False
            else:
                time.sleep(0.1)


class RenderFormerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RenderFormer")
        self.setGeometry(100, 100, 1200, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        input_layout = QHBoxLayout()
        self.scene_input = QLineEdit()
        self.scene_input.setText("examples/cbox.json")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_scene)
        self.convert_btn = QPushButton("Convert")
        self.convert_btn.clicked.connect(self.convert_scene)
        
        input_layout.addWidget(QLabel("Scene JSON:"))
        input_layout.addWidget(self.scene_input)
        input_layout.addWidget(browse_btn)
        input_layout.addWidget(self.convert_btn)
        layout.addLayout(input_layout)
        
        render_control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Rendering")
        self.start_btn.clicked.connect(self.start_rendering)
        self.start_btn.setEnabled(False)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_rendering)
        self.stop_btn.setEnabled(False)
        
        render_control_layout.addWidget(self.start_btn)
        render_control_layout.addWidget(self.stop_btn)
        render_control_layout.addStretch()
        layout.addLayout(render_control_layout)
        
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        self.stats_label = QLabel()
        layout.addWidget(self.stats_label)
        
        self.image_label = QLabel()
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)
        
        self.render_thread = RenderThread()
        self.render_thread.status_update.connect(self.update_status)
        self.render_thread.image_ready.connect(self.display_image)
        self.render_thread.render_time.connect(self.update_stats)
        self.render_thread.error.connect(self.show_error)
        self.render_thread.start()
        
        self.frame_times = []
        self.max_frame_history = 10
    
    def browse_scene(self):
        examples_dir = str(SCRIPT_DIR / "examples")
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Scene JSON", examples_dir, "JSON Files (*.json)"
        )
        if filename:
            self.scene_input.setText(filename)
    
    def convert_scene(self):
        scene_path = self.scene_input.text()
        
        if not scene_path:
            self.status_label.setText("Error: Please enter a scene path")
            return
        
        check_path = Path(scene_path)
        if not check_path.is_absolute():
            check_path = SCRIPT_DIR / check_path
        
        if not check_path.exists():
            self.status_label.setText(f"Error: Scene file not found: {check_path}")
            return
        
        self.convert_btn.setEnabled(False)
        self.status_label.setText("Converting scene...")
        
        scene_name = check_path.stem
        h5_path = SCRIPT_DIR / "tmp" / scene_name / f"{scene_name}.h5"
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        
        result = subprocess.run(
            [PYTHON, str(SCENE_PROCESSOR), str(check_path), 
             "--output_h5_path", str(h5_path)],
            capture_output=True, text=True, cwd=str(SCRIPT_DIR)
        )
        
        if result.returncode != 0:
            self.status_label.setText(f"Conversion error: {result.stderr}")
            self.convert_btn.setEnabled(True)
            return
        
        self.render_thread.h5_path = str(h5_path)
        self.status_label.setText("Scene converted successfully")
        self.start_btn.setEnabled(True)
        self.convert_btn.setEnabled(True)
    
    def start_rendering(self):
        if not self.render_thread.h5_path:
            self.status_label.setText("Error: Please convert a scene first")
            return
        
        self.frame_times.clear()
        self.stats_label.clear()
        self.render_thread.rendering = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Rendering...")
    
    def stop_rendering(self):
        self.render_thread.rendering = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Rendering stopped")
    
    def update_status(self, status):
        self.status_label.setText(status)
    
    def update_stats(self, elapsed_time):
        self.frame_times.append(elapsed_time)
        if len(self.frame_times) > self.max_frame_history:
            self.frame_times.pop(0)
        
        avg_time = sum(self.frame_times) / len(self.frame_times)
        current_fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        self.stats_label.setText(
            f"Frame Time: {elapsed_time:.3f}s  |  "
            f"Current FPS: {current_fps:.2f}  |  "
            f"Avg FPS (last {len(self.frame_times)}): {avg_fps:.2f}"
        )
    
    def display_image(self, image_array):
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        
        image_array = np.ascontiguousarray(image_array)
        
        height, width, channel = image_array.shape
        bytes_per_line = channel * width
        
        q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image.copy())
        
        label_size = self.image_label.size()
        scaled_pixmap = pixmap.scaled(
            label_size.width(), 
            label_size.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled_pixmap)
    
    def show_error(self, error):
        from PyQt6.QtWidgets import QMessageBox
        self.status_label.setText("Error occurred - see details")
        print(f"ERROR: {error}")
        
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Render Error")
        msg_box.setText("An error occurred during rendering")
        msg_box.setDetailedText(error)
        msg_box.exec()
    
    def closeEvent(self, event):
        self.render_thread.stop()
        self.render_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RenderFormerGUI()
    window.show()
    sys.exit(app.exec())
