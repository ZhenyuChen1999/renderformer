#!/usr/bin/env python3
import os
import sys
import tempfile
import subprocess
import numpy as np
import math
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QPushButton, QLineEdit, QLabel, 
                              QFileDialog, QTextEdit)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage, QKeyEvent, QMouseEvent

PYTHON = r"C:\Users\sh-gfx-perf\AppData\Local\Programs\Python\Python312\python.exe"
SCRIPT_DIR = Path(__file__).parent.absolute()
SCENE_PROCESSOR = SCRIPT_DIR / "scene_processor" / "convert_scene.py"

sys.path.insert(0, str(SCRIPT_DIR))
from infer import render_to_array


def create_camera_matrix(position, yaw, pitch, up_mode='Y'):
    """Create camera-to-world matrix from position and rotation.
    Uses Blender coordinate system: -Z forward, +Y up, +X right (for Y-up mode)
    yaw: rotation around up axis (left/right)
    pitch: rotation around right axis (up/down)
    up_mode: 'Y' for Y-up or 'Z' for Z-up
    """
    # Convert degrees to radians
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    
    # World up based on mode
    world_up = np.array([0, 1, 0]) if up_mode == 'Y' else np.array([0, 0, 1])
    
    # Forward direction depends on up vector mode
    if up_mode == 'Y':
        # Y-up: standard Blender convention (-Z forward, +Y up)
        forward = np.array([
            math.sin(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
            -math.cos(yaw_rad) * math.cos(pitch_rad)
        ])
    else:
        # Z-up: X forward, Z up (common in CAD)
        forward = np.array([
            math.cos(yaw_rad) * math.cos(pitch_rad),
            math.sin(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad)
        ])
    
    # Calculate right direction (perpendicular to world_up and forward)
    right = np.cross(world_up, forward)
    right = right / (np.linalg.norm(right) + 1e-8)
    
    # Calculate up direction (perpendicular to forward and right)
    up = np.cross(forward, right)
    
    # Build rotation matrix
    rotation = np.eye(4)
    rotation[0, :3] = right
    rotation[1, :3] = up
    rotation[2, :3] = forward
    rotation[:3, 3] = position
    
    return rotation.astype(np.float32)


def extract_yaw_pitch_from_matrix(c2w):
    """Extract yaw and pitch from camera-to-world matrix.
    Returns yaw and pitch in degrees.
    """
    # Extract forward direction from the matrix (row 2 in Blender convention)
    forward = c2w[2, :3]
    
    # Calculate pitch (rotation around X axis)
    # pitch = arcsin(forward_y)
    pitch = math.degrees(math.asin(np.clip(forward[1], -1.0, 1.0)))
    
    # Calculate yaw (rotation around Y axis)
    # yaw = atan2(forward_x, -forward_z)
    yaw = math.degrees(math.atan2(forward[0], -forward[2]))
    
    return yaw, pitch


def forward_to_yaw_pitch(forward, up_mode='Y'):
    """Convert a forward direction vector to yaw/pitch angles.
    forward: normalized direction vector [x, y, z]
    up_mode: 'Y' or 'Z'
    Returns: (yaw, pitch) in degrees
    """
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    
    if up_mode == 'Y':
        # Y-up: forward = [sin(yaw)*cos(pitch), sin(pitch), -cos(yaw)*cos(pitch)]
        # pitch = arcsin(forward_y)
        pitch = math.degrees(math.asin(np.clip(forward[1], -1.0, 1.0)))
        # yaw = atan2(forward_x, -forward_z)
        yaw = math.degrees(math.atan2(forward[0], -forward[2]))
    else:
        # Z-up: forward = [cos(yaw)*cos(pitch), sin(yaw)*cos(pitch), sin(pitch)]
        # pitch = arcsin(forward_z)
        pitch = math.degrees(math.asin(np.clip(forward[2], -1.0, 1.0)))
        # yaw = atan2(forward_y, forward_x)
        yaw = math.degrees(math.atan2(forward[1], forward[0]))
    
    return yaw, pitch


class RenderThread(QThread):
    status_update = pyqtSignal(str)
    image_ready = pyqtSignal(np.ndarray)
    render_time = pyqtSignal(float)
    error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        import torch
        import h5py
        
        self.scene_path = None
        self.h5_path = None
        self.running = False
        self.rendering = False
        
        # Scene data
        self.scene_data = None
        
        # Camera parameters
        self.camera_position = np.array([0.0, 0.0, 3.0])
        self.camera_yaw = 0.0
        self.camera_pitch = 0.0
        self.camera_fov = 39.6
        self.up_mode = 'Y'
    
    def load_scene_data(self, h5_path):
        """Load scene data from H5 file and extract initial camera"""
        import torch
        import h5py
        
        with h5py.File(h5_path, 'r') as f:
            triangles = torch.from_numpy(np.array(f['triangles']).astype(np.float32))
            texture = torch.from_numpy(np.array(f['texture']).astype(np.float32))
            vn = torch.from_numpy(np.array(f['vn']).astype(np.float32))
            
            # Load initial camera if available
            if 'c2w' in f and 'fov' in f:
                c2w = np.array(f['c2w']).astype(np.float32)
                fov = np.array(f['fov']).astype(np.float32)
                # Extract position and orientation from first camera
                if len(c2w.shape) >= 2 and c2w.shape[0] > 0:
                    # Extract position
                    self.camera_position = c2w[0, :3, 3].copy()
                    # Extract yaw and pitch from rotation matrix
                    self.camera_yaw, self.camera_pitch = extract_yaw_pitch_from_matrix(c2w[0])
                    self.camera_fov = float(fov[0]) if fov.shape[0] > 0 else 39.6
                else:
                    # Fallback defaults
                    self.camera_position = np.array([0.0, 0.0, 3.0])
                    self.camera_yaw = 0.0
                    self.camera_pitch = 0.0
                    self.camera_fov = 39.6
            
            self.scene_data = {
                'triangles': triangles,
                'texture': texture,
                'vn': vn,
                'mask': torch.ones(triangles.shape[0], dtype=torch.bool)
            }
    
    def set_camera(self, position, yaw, pitch, up_mode='Y'):
        """Update camera parameters"""
        self.camera_position = position.copy()
        self.camera_yaw = yaw
        self.camera_pitch = pitch
        self.up_mode = up_mode
    
    def stop(self):
        self.running = False
        self.rendering = False
    
    def run(self):
        import time
        import traceback
        import torch
        
        self.running = True
        self.status_update.emit("Thread ready")
        
        # Load pipeline once
        pipeline = None
        device = None
        
        while self.running:
            if self.rendering and self.scene_data is not None:
                try:
                    start_time = time.time()
                    
                    # Initialize pipeline on first render
                    if pipeline is None:
                        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
                        from renderformer import RenderFormerRenderingPipeline
                        pipeline = RenderFormerRenderingPipeline.from_pretrained('microsoft/renderformer-v1.1-swin-large')
                        
                        if device == torch.device('cuda') and os.name == 'posix':
                            from renderformer_liger_kernel import apply_kernels
                            apply_kernels(pipeline.model)
                            torch.backends.cuda.matmul.allow_tf32 = True
                            torch.backends.cudnn.allow_tf32 = True
                        
                        pipeline.to(device)
                        self.status_update.emit("Pipeline loaded")
                    
                    # Create camera matrix from current camera state
                    c2w = create_camera_matrix(self.camera_position, self.camera_yaw, self.camera_pitch, self.up_mode)
                    c2w_tensor = torch.from_numpy(c2w).unsqueeze(0).unsqueeze(0).to(device)
                    fov_tensor = torch.tensor([[self.camera_fov]], dtype=torch.float32).unsqueeze(-1).to(device)
                    
                    # Prepare scene data
                    triangles = self.scene_data['triangles'].unsqueeze(0).to(device)
                    texture = self.scene_data['texture'].unsqueeze(0).to(device)
                    mask = self.scene_data['mask'].unsqueeze(0).to(device)
                    vn = self.scene_data['vn'].unsqueeze(0).to(device)
                    
                    # Render
                    rendered_imgs = pipeline(
                        triangles=triangles,
                        texture=texture,
                        mask=mask,
                        vn=vn,
                        c2w=c2w_tensor,
                        fov=fov_tensor,
                        resolution=512,
                        torch_dtype=torch.float16,
                    )
                    
                    # Tone map
                    hdr_img = rendered_imgs[0, 0].cpu().numpy().astype(np.float32)
                    ldr_img = np.clip(hdr_img, 0, 1)
                    ldr_img = np.power(ldr_img, 1.0 / 2.2)
                    ldr_img = (ldr_img * 255).astype(np.uint8)
                    
                    elapsed_time = time.time() - start_time
                    self.render_time.emit(elapsed_time)
                    self.image_ready.emit(ldr_img)
                    
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
        self.setWindowTitle("RenderFormer - Camera Control: WASD+QE move, Right-click drag to look, U to toggle up vector")
        self.setGeometry(100, 100, 1200, 900)
        
        # Camera state
        self.camera_position = np.array([0.0, 0.0, 3.0])
        self.camera_yaw = 0.0  # Rotation around Y axis
        self.camera_pitch = 0.0  # Rotation around X axis
        self.camera_speed = 0.05
        self.mouse_sensitivity = 0.2
        self.up_vector_mode = 'Y'  # 'Y' or 'Z'
        
        # Input state
        self.keys_pressed = set()
        self.mouse_dragging = False
        self.last_mouse_pos = None
        self.camera_control_active = False
        
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
        
        self.camera_label = QLabel("Camera: Position (0.00, 0.00, 0.00) | Yaw: 0.0° | Pitch: 0.0°")
        layout.addWidget(self.camera_label)
        
        self.up_vector_label = QLabel("Up Vector: +Y (Press U to toggle)")
        self.up_vector_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        layout.addWidget(self.up_vector_label)
        
        self.image_label = QLabel()
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMouseTracking(True)
        self.image_label.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.image_label.mousePressEvent = self.on_image_mouse_press
        self.image_label.mouseReleaseEvent = self.on_image_mouse_release
        self.image_label.mouseMoveEvent = self.on_image_mouse_move
        layout.addWidget(self.image_label)
        
        self.control_hint_label = QLabel("Click on the image to enable camera controls, press ESC to release, U to toggle up vector")
        self.control_hint_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.control_hint_label)
        
        self.render_thread = RenderThread()
        self.render_thread.status_update.connect(self.update_status)
        self.render_thread.image_ready.connect(self.display_image)
        self.render_thread.render_time.connect(self.update_stats)
        self.render_thread.error.connect(self.show_error)
        self.render_thread.start()
        
        self.frame_times = []
        self.max_frame_history = 10
        
        # Camera update timer
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera)
        self.camera_timer.start(16)  # ~60 FPS for camera updates
    
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
        self.render_thread.load_scene_data(str(h5_path))
        
        # Initialize camera from scene (position, yaw, pitch)
        self.camera_position = self.render_thread.camera_position.copy()
        self.camera_yaw = self.render_thread.camera_yaw
        self.camera_pitch = self.render_thread.camera_pitch
        
        # Update camera label to show initial position
        self.camera_label.setText(
            f"Camera: Position ({self.camera_position[0]:.2f}, {self.camera_position[1]:.2f}, {self.camera_position[2]:.2f}) | "
            f"Yaw: {self.camera_yaw:.1f}° | Pitch: {self.camera_pitch:.1f}°"
        )
        
        self.status_label.setText("Scene loaded - Click image to enable camera controls")
        self.start_btn.setEnabled(True)
        self.convert_btn.setEnabled(True)
    
    def start_rendering(self):
        if not self.render_thread.h5_path:
            self.status_label.setText("Error: Please convert a scene first")
            return
        
        self.frame_times.clear()
        self.stats_label.clear()
        
        # Sync camera state to render thread
        self.render_thread.set_camera(self.camera_position, self.camera_yaw, self.camera_pitch, self.up_vector_mode)
        
        self.render_thread.rendering = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Rendering...")
    
    def stop_rendering(self):
        self.render_thread.rendering = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Rendering stopped")
        if self.camera_control_active:
            self.release_camera_control()
    
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
    
    def on_image_mouse_press(self, event):
        """Handle mouse press on the rendered image"""
        if not self.render_thread.rendering:
            return
        
        # Left click activates camera controls
        if event.button() == Qt.MouseButton.LeftButton:
            self.camera_control_active = True
            self.image_label.setStyleSheet("border: 3px solid #4CAF50;")
            self.control_hint_label.setText("🎮 Camera controls ACTIVE - Use WASD+QE to move, right-click drag to look, U to toggle up vector, ESC to release")
            self.control_hint_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self.setFocus()
        
        # Right click for camera rotation
        elif self.camera_control_active and event.button() == Qt.MouseButton.RightButton:
            self.mouse_dragging = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.CursorShape.BlankCursor)
    
    def on_image_mouse_release(self, event):
        """Handle mouse release on the rendered image"""
        if self.camera_control_active and event.button() == Qt.MouseButton.RightButton:
            self.mouse_dragging = False
            self.last_mouse_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def on_image_mouse_move(self, event):
        """Handle mouse move on the rendered image for camera rotation"""
        if self.camera_control_active and self.mouse_dragging and self.last_mouse_pos is not None:
            delta = event.pos() - self.last_mouse_pos
            
            # Update camera rotation
            self.camera_yaw += delta.x() * self.mouse_sensitivity
            self.camera_pitch -= delta.y() * self.mouse_sensitivity
            
            # Clamp pitch to avoid gimbal lock
            self.camera_pitch = max(-89.0, min(89.0, self.camera_pitch))
            
            # Update render thread camera
            if self.render_thread.rendering:
                self.render_thread.set_camera(self.camera_position, self.camera_yaw, self.camera_pitch, self.up_vector_mode)
            
            # Update camera label
            self.camera_label.setText(
                f"Camera: Position ({self.camera_position[0]:.2f}, {self.camera_position[1]:.2f}, {self.camera_position[2]:.2f}) | "
                f"Yaw: {self.camera_yaw:.1f}° | Pitch: {self.camera_pitch:.1f}°"
            )
            
            self.last_mouse_pos = event.pos()
    
    def release_camera_control(self):
        """Release camera controls and return focus to UI"""
        self.camera_control_active = False
        self.keys_pressed.clear()
        self.mouse_dragging = False
        self.image_label.setStyleSheet("")
        self.control_hint_label.setText("Click on the image to enable camera controls, press ESC to release, U to toggle up vector")
        self.control_hint_label.setStyleSheet("color: gray; font-style: italic;")
        self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def update_camera(self):
        """Update camera position based on input state"""
        if not self.camera_control_active or not self.keys_pressed or not self.render_thread.rendering:
            return
        
        # Calculate movement directions
        yaw_rad = math.radians(self.camera_yaw)
        pitch_rad = math.radians(self.camera_pitch)
        
        # World up based on selected mode
        world_up = np.array([0, 1, 0]) if self.up_vector_mode == 'Y' else np.array([0, 0, 1])
        
        # Forward direction depends on up vector mode
        if self.up_vector_mode == 'Y':
            # Y-up: standard Blender convention (-Z forward, +Y up)
            forward = np.array([
                math.sin(yaw_rad) * math.cos(pitch_rad),
                math.sin(pitch_rad),
                -math.cos(yaw_rad) * math.cos(pitch_rad)
            ])
        else:
            # Z-up: X forward, Z up (common in CAD)
            forward = np.array([
                math.cos(yaw_rad) * math.cos(pitch_rad),
                math.sin(yaw_rad) * math.cos(pitch_rad),
                math.sin(pitch_rad)
            ])
        
        # Right direction (perpendicular to world_up and forward)
        right = np.cross(world_up, forward)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        # Camera's local up direction (perpendicular to forward and right)
        up = np.cross(forward, right)
        up = up / (np.linalg.norm(up) + 1e-8)
        
        # Apply movement
        movement = np.array([0.0, 0.0, 0.0])
        
        if Qt.Key.Key_W in self.keys_pressed:
            movement += forward * self.camera_speed
        if Qt.Key.Key_S in self.keys_pressed:
            movement -= forward * self.camera_speed
        if Qt.Key.Key_D in self.keys_pressed:
            movement += right * self.camera_speed
        if Qt.Key.Key_A in self.keys_pressed:
            movement -= right * self.camera_speed
        if Qt.Key.Key_E in self.keys_pressed:
            movement += up * self.camera_speed
        if Qt.Key.Key_Q in self.keys_pressed:
            movement -= up * self.camera_speed
        
        self.camera_position += movement
        
        # Update render thread camera
        self.render_thread.set_camera(self.camera_position, self.camera_yaw, self.camera_pitch, self.up_vector_mode)
        
        # Update camera label
        self.camera_label.setText(
            f"Camera: Position ({self.camera_position[0]:.2f}, {self.camera_position[1]:.2f}, {self.camera_position[2]:.2f}) | "
            f"Yaw: {self.camera_yaw:.1f}° | Pitch: {self.camera_pitch:.1f}°"
        )
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events"""
        if event.key() == Qt.Key.Key_Escape:
            if self.camera_control_active:
                self.release_camera_control()
            return
        
        if event.key() == Qt.Key.Key_U:
            # Calculate current forward direction in current coordinate system
            yaw_rad = math.radians(self.camera_yaw)
            pitch_rad = math.radians(self.camera_pitch)
            
            if self.up_vector_mode == 'Y':
                # Current Y-up forward direction
                forward = np.array([
                    math.sin(yaw_rad) * math.cos(pitch_rad),
                    math.sin(pitch_rad),
                    -math.cos(yaw_rad) * math.cos(pitch_rad)
                ])
            else:
                # Current Z-up forward direction
                forward = np.array([
                    math.cos(yaw_rad) * math.cos(pitch_rad),
                    math.sin(yaw_rad) * math.cos(pitch_rad),
                    math.sin(pitch_rad)
                ])
            
            # Toggle up vector mode
            new_mode = 'Z' if self.up_vector_mode == 'Y' else 'Y'
            self.up_vector_mode = new_mode
            
            # Convert forward direction to new yaw/pitch in new coordinate system
            self.camera_yaw, self.camera_pitch = forward_to_yaw_pitch(forward, new_mode)
            
            # Update UI
            self.up_vector_label.setText(f"Up Vector: +{self.up_vector_mode} (Press U to toggle)")
            self.camera_label.setText(
                f"Camera: Position ({self.camera_position[0]:.2f}, {self.camera_position[1]:.2f}, {self.camera_position[2]:.2f}) | "
                f"Yaw: {self.camera_yaw:.1f}° | Pitch: {self.camera_pitch:.1f}°"
            )
            
            # Sync to render thread if rendering
            if self.render_thread.rendering:
                self.render_thread.set_camera(self.camera_position, self.camera_yaw, self.camera_pitch, self.up_vector_mode)
            return
        
        if self.camera_control_active:
            self.keys_pressed.add(event.key())
        else:
            super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event: QKeyEvent):
        """Handle key release events"""
        if self.camera_control_active:
            self.keys_pressed.discard(event.key())
        else:
            super().keyReleaseEvent(event)
    
    
    def closeEvent(self, event):
        self.camera_timer.stop()
        self.render_thread.stop()
        self.render_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RenderFormerGUI()
    window.show()
    sys.exit(app.exec())
