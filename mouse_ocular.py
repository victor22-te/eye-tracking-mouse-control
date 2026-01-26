"""
Mouse Ocular - Control del mouse con seguimiento ocular
========================================================
Este programa utiliza visión artificial para:
- Mover el mouse siguiendo la posición de la cabeza/ojos
- Click izquierdo: guiñar ojo izquierdo
- Click derecho: guiñar ojo derecho
- Doble click: doble parpadeo
- Screenshot: cerrar ambos ojos por 1 segundo

Usa MediaPipe Face Landmarker con 478 puntos para detección precisa de ojos.

Dependencias: opencv-python, mediapipe, pyautogui, numpy, screeninfo, pillow
"""

import cv2
import numpy as np
import pyautogui
import time
import os
import urllib.request
from screeninfo import get_monitors

# ===== CONFIGURACIÓN DE MEDIAPIPE FACE LANDMARKER =====
FACE_DETECTOR_AVAILABLE = False
face_detector = None

def download_face_model():
    """Descarga el modelo de detección facial si no existe."""
    model_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        print("📥 Descargando modelo de detección facial (478 puntos)...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        try:
            urllib.request.urlretrieve(url, model_path)
            print("✅ Modelo facial descargado")
            return model_path
        except Exception as e:
            print(f"❌ Error descargando modelo: {e}")
            return None
    return model_path

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    model_path = download_face_model()
    if model_path and os.path.exists(model_path):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,  # Esto nos da el EAR directamente!
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        face_detector = vision.FaceLandmarker.create_from_options(options)
        FACE_DETECTOR_AVAILABLE = True
        print("✅ MediaPipe Face Landmarker inicializado (478 puntos)")
    
except ImportError as e:
    print(f"⚠️ MediaPipe no disponible: {e}")
except Exception as e:
    print(f"⚠️ Error inicializando MediaPipe: {e}")

# Configuración de pyautogui
try:
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0
    _ = pyautogui.position()
    PYAUTOGUI_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Advertencia: pyautogui no disponible ({e})")
    PYAUTOGUI_AVAILABLE = False


class MouseOcular:
    """
    Control del mouse mediante seguimiento ocular preciso.
    Utiliza MediaPipe Face Landmarker con 478 puntos faciales.
    Usa Face Blendshapes para detectar guiños con precisión.
    """
    
    # Índices de landmarks para los ojos (MediaPipe Face Mesh)
    # Ojo derecho (desde perspectiva de la persona)
    RIGHT_EYE_UPPER = [159, 158, 157, 173, 133]
    RIGHT_EYE_LOWER = [145, 144, 163, 7, 33]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    
    # Ojo izquierdo (desde perspectiva de la persona)
    LEFT_EYE_UPPER = [386, 385, 384, 398, 362]
    LEFT_EYE_LOWER = [374, 373, 390, 249, 263]
    LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
    
    # Puntos específicos para calcular EAR (Eye Aspect Ratio)
    RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]  # p1, p2, p3, p4, p5, p6
    LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]  # p1, p2, p3, p4, p5, p6
    
    # Índices del iris para seguimiento de mirada
    RIGHT_IRIS = [468, 469, 470, 471, 472]
    LEFT_IRIS = [473, 474, 475, 476, 477]
    
    def __init__(self):
        print("🔧 Inicializando Mouse Ocular...")
        
        # Obtener información de la pantalla
        try:
            monitor = get_monitors()[0]
            self.screen_width = monitor.width
            self.screen_height = monitor.height
        except:
            self.screen_width = 1920
            self.screen_height = 1080
        
        print(f"📺 Pantalla: {self.screen_width}x{self.screen_height}")
        
        # Configuración de la cámara
        # Índice 1 para usar la cámara USB externa (USB Camera)
        # Si no funciona, probar con índice 2 o 3
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("No se puede abrir la cámara")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📷 Cámara: {self.cam_width}x{self.cam_height}")
        
        # Referencia al detector facial
        self.face_detector = face_detector
        
        # Fallback a Haar Cascades si MediaPipe no está disponible
        if not FACE_DETECTOR_AVAILABLE:
            print("⚠️ Usando Haar Cascades como fallback")
            haar_path = cv2.data.haarcascades
            self.face_cascade = cv2.CascadeClassifier(haar_path + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(haar_path + 'haarcascade_eye.xml')
        
        # Para suavizado del movimiento del mouse
        self.smooth_x = self.screen_width // 2
        self.smooth_y = self.screen_height // 2
        self.SMOOTHING_FACTOR = 0.15
        
        # Valores de EAR (Eye Aspect Ratio)
        self.left_ear = 0.3
        self.right_ear = 0.3
        self.EAR_THRESHOLD = 0.20  # Umbral para considerar ojo cerrado
        self.WINK_DIFF_THRESHOLD = 0.08  # Diferencia mínima entre ojos para guiño
        
        # Historial de EAR para suavizado
        self.left_ear_history = []
        self.right_ear_history = []
        self.EAR_HISTORY_SIZE = 4
        
        # Contadores para confirmar guiños
        self.left_wink_counter = 0
        self.right_wink_counter = 0
        self.WINK_CONFIRM_FRAMES = 2
        
        # Cooldown para evitar clicks repetidos
        self.last_click_time = 0
        self.CLICK_COOLDOWN = 0.5
        
        # Para doble click
        self.last_blink_time = 0
        self.blink_count = 0
        self.DOUBLE_BLINK_THRESHOLD = 0.6
        
        # Para screenshot
        self.both_closed_start = None
        self.SCREENSHOT_HOLD_TIME = 1.0
        self.screenshot_done = False
        
        # Calibración
        self.calibrated = False
        self.center_x = 0.5
        self.center_y = 0.5
        
        # Sensibilidad del movimiento
        self.sensitivity_x = 3.5
        self.sensitivity_y = 3.0
        
        # Historial de posiciones
        self.position_history = []
        self.HISTORY_SIZE = 4
        
        # Estado
        self.no_face_counter = 0
        self.MAX_NO_FACE_FRAMES = 10

    def calculate_ear(self, landmarks, eye_indices, w, h):
        """
        Calcula el Eye Aspect Ratio (EAR) para un ojo.
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """
        try:
            # Obtener puntos
            p1 = np.array([landmarks[eye_indices[0]].x * w, landmarks[eye_indices[0]].y * h])
            p2 = np.array([landmarks[eye_indices[1]].x * w, landmarks[eye_indices[1]].y * h])
            p3 = np.array([landmarks[eye_indices[2]].x * w, landmarks[eye_indices[2]].y * h])
            p4 = np.array([landmarks[eye_indices[3]].x * w, landmarks[eye_indices[3]].y * h])
            p5 = np.array([landmarks[eye_indices[4]].x * w, landmarks[eye_indices[4]].y * h])
            p6 = np.array([landmarks[eye_indices[5]].x * w, landmarks[eye_indices[5]].y * h])
            
            # Calcular distancias
            vertical1 = np.linalg.norm(p2 - p6)
            vertical2 = np.linalg.norm(p3 - p5)
            horizontal = np.linalg.norm(p1 - p4)
            
            if horizontal == 0:
                return 0.3
            
            ear = (vertical1 + vertical2) / (2.0 * horizontal)
            return ear
        except:
            return 0.3

    def get_blendshape_values(self, face_blendshapes):
        """
        Obtiene los valores de blendshapes para los ojos.
        Estos valores ya vienen calculados por MediaPipe y son muy precisos.
        """
        left_blink = 0.0
        right_blink = 0.0
        
        if face_blendshapes is None or len(face_blendshapes) == 0:
            return left_blink, right_blink
        
        blendshapes = face_blendshapes[0]
        
        for bs in blendshapes:
            if bs.category_name == 'eyeBlinkLeft':
                left_blink = bs.score
            elif bs.category_name == 'eyeBlinkRight':
                right_blink = bs.score
        
        return left_blink, right_blink

    def smooth_ear_values(self, left_ear, right_ear):
        """Aplica suavizado temporal a los valores de EAR."""
        self.left_ear_history.append(left_ear)
        self.right_ear_history.append(right_ear)
        
        if len(self.left_ear_history) > self.EAR_HISTORY_SIZE:
            self.left_ear_history.pop(0)
        if len(self.right_ear_history) > self.EAR_HISTORY_SIZE:
            self.right_ear_history.pop(0)
        
        avg_left = sum(self.left_ear_history) / len(self.left_ear_history)
        avg_right = sum(self.right_ear_history) / len(self.right_ear_history)
        
        return avg_left, avg_right

    def detect_wink(self, left_blink, right_blink):
        """
        Detecta guiños usando los valores de blendshape.
        left_blink y right_blink son valores de 0-1 donde 1 = completamente cerrado.
        """
        # Suavizar valores
        left_smooth, right_smooth = self.smooth_ear_values(left_blink, right_blink)
        
        # Calcular diferencia entre ojos
        diff = abs(left_smooth - right_smooth)
        
        left_wink = False
        right_wink = False
        both_closed = False
        
        # Si hay una diferencia significativa, es un guiño
        if diff > self.WINK_DIFF_THRESHOLD:
            if left_smooth > right_smooth + self.WINK_DIFF_THRESHOLD:
                # Ojo izquierdo más cerrado -> Guiño izquierdo
                if left_smooth > 0.4:  # Umbral de confianza
                    left_wink = True
            elif right_smooth > left_smooth + self.WINK_DIFF_THRESHOLD:
                # Ojo derecho más cerrado -> Guiño derecho
                if right_smooth > 0.4:
                    right_wink = True
        
        # Ambos cerrados
        if left_smooth > 0.5 and right_smooth > 0.5:
            both_closed = True
            left_wink = False
            right_wink = False
        
        return left_wink, right_wink, both_closed, left_smooth, right_smooth

    def get_face_center(self, landmarks, w, h):
        """Obtiene el centro del rostro basado en landmarks."""
        # Usar la nariz como punto central
        nose = landmarks[1]  # Punta de la nariz
        
        # También podemos usar el promedio de algunos puntos clave
        center_x = nose.x
        center_y = nose.y
        
        return center_x, center_y

    def handle_gestures(self, current_time, left_wink, right_wink, both_closed):
        """Maneja los gestos de los ojos para generar acciones."""
        can_click = (current_time - self.last_click_time) > self.CLICK_COOLDOWN
        
        # SCREENSHOT: Ambos ojos cerrados por tiempo prolongado
        if both_closed:
            if self.both_closed_start is None:
                self.both_closed_start = current_time
                self.screenshot_done = False
            elif (current_time - self.both_closed_start) >= self.SCREENSHOT_HOLD_TIME:
                if not self.screenshot_done:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.png"
                    try:
                        pyautogui.screenshot(filename)
                        print(f"📸 Screenshot guardado: {filename}")
                    except Exception as e:
                        print(f"❌ Error al tomar screenshot: {e}")
                    self.screenshot_done = True
                    self.last_click_time = current_time
            return "screenshot_pending"
        
        else:
            # Se abrieron los ojos después de estar cerrados
            if self.both_closed_start is not None:
                hold_time = current_time - self.both_closed_start
                self.both_closed_start = None
                
                # Si fue un parpadeo corto, verificar doble parpadeo
                if hold_time < self.SCREENSHOT_HOLD_TIME and not self.screenshot_done:
                    if (current_time - self.last_blink_time) < self.DOUBLE_BLINK_THRESHOLD:
                        self.blink_count += 1
                        if self.blink_count >= 2 and can_click:
                            try:
                                pyautogui.doubleClick()
                                print("👆👆 Doble click!")
                            except Exception as e:
                                print(f"❌ Error: {e}")
                            self.blink_count = 0
                            self.last_click_time = current_time
                            return "double_click"
                    else:
                        self.blink_count = 1
                    self.last_blink_time = current_time
            
            # GUIÑO IZQUIERDO -> Click izquierdo
            if left_wink and not right_wink:
                self.left_wink_counter += 1
                self.right_wink_counter = 0
                if self.left_wink_counter >= self.WINK_CONFIRM_FRAMES and can_click:
                    try:
                        pyautogui.click(button='left')
                        print("👆 Click izquierdo!")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                    self.last_click_time = current_time
                    self.left_wink_counter = 0
                    return "left_click"
            else:
                self.left_wink_counter = max(0, self.left_wink_counter - 1)
            
            # GUIÑO DERECHO -> Click derecho
            if right_wink and not left_wink:
                self.right_wink_counter += 1
                self.left_wink_counter = 0
                if self.right_wink_counter >= self.WINK_CONFIRM_FRAMES and can_click:
                    try:
                        pyautogui.click(button='right')
                        print("👆 Click derecho!")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                    self.last_click_time = current_time
                    self.right_wink_counter = 0
                    return "right_click"
            else:
                self.right_wink_counter = max(0, self.right_wink_counter - 1)
        
        return None

    def calculate_smoothed_position(self, target_x, target_y):
        """Calcula la posición suavizada del mouse."""
        self.position_history.append((target_x, target_y))
        if len(self.position_history) > self.HISTORY_SIZE:
            self.position_history.pop(0)
        
        if len(self.position_history) > 0:
            avg_x = sum(p[0] for p in self.position_history) / len(self.position_history)
            avg_y = sum(p[1] for p in self.position_history) / len(self.position_history)
        else:
            avg_x, avg_y = target_x, target_y
        
        self.smooth_x += (avg_x - self.smooth_x) * self.SMOOTHING_FACTOR
        self.smooth_y += (avg_y - self.smooth_y) * self.SMOOTHING_FACTOR
        
        return int(self.smooth_x), int(self.smooth_y)

    def process_frame(self, frame):
        """Procesa un frame de video."""
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        current_time = time.time()
        
        left_blink = 0.0
        right_blink = 0.0
        face_detected = False
        face_center_x = 0.5
        face_center_y = 0.5
        
        if FACE_DETECTOR_AVAILABLE and self.face_detector is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = self.face_detector.detect(mp_image)
                
                if result.face_landmarks and len(result.face_landmarks) > 0:
                    face_detected = True
                    landmarks = result.face_landmarks[0]
                    
                    # Obtener valores de parpadeo de los blendshapes
                    left_blink, right_blink = self.get_blendshape_values(result.face_blendshapes)
                    
                    # Si no tenemos blendshapes, calcular EAR manualmente
                    if left_blink == 0 and right_blink == 0:
                        left_ear = self.calculate_ear(landmarks, self.LEFT_EYE_EAR, w, h)
                        right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE_EAR, w, h)
                        # Convertir EAR a valor de "parpadeo" (inverso)
                        left_blink = max(0, 1.0 - (left_ear / 0.3))
                        right_blink = max(0, 1.0 - (right_ear / 0.3))
                    
                    # Obtener centro del rostro
                    face_center_x, face_center_y = self.get_face_center(landmarks, w, h)
                    
                    # Dibujar landmarks de los ojos
                    for idx in self.LEFT_EYE:
                        x = int(landmarks[idx].x * w)
                        y = int(landmarks[idx].y * h)
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    
                    for idx in self.RIGHT_EYE:
                        x = int(landmarks[idx].x * w)
                        y = int(landmarks[idx].y * h)
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    
                    # Dibujar iris
                    for idx in self.LEFT_IRIS:
                        if idx < len(landmarks):
                            x = int(landmarks[idx].x * w)
                            y = int(landmarks[idx].y * h)
                            cv2.circle(frame, (x, y), 2, (255, 0, 255), -1)
                    
                    for idx in self.RIGHT_IRIS:
                        if idx < len(landmarks):
                            x = int(landmarks[idx].x * w)
                            y = int(landmarks[idx].y * h)
                            cv2.circle(frame, (x, y), 2, (255, 0, 255), -1)
                    
                    # Dibujar centro de la nariz
                    nose_x = int(landmarks[1].x * w)
                    nose_y = int(landmarks[1].y * h)
                    cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 255), -1)
                    
            except Exception as e:
                pass
        
        if face_detected:
            self.no_face_counter = 0
            
            # Detectar guiños
            left_wink, right_wink, both_closed, left_smooth, right_smooth = self.detect_wink(left_blink, right_blink)
            
            # Guardar valores para mostrar
            self.left_ear = left_smooth
            self.right_ear = right_smooth
            
            # Calibrar
            if not self.calibrated:
                self.center_x = face_center_x
                self.center_y = face_center_y
                self.calibrated = True
                print("🎯 Calibrado!")
            
            # Mover el mouse - DIRECCIÓN CORREGIDA
            if not both_closed:
                # Calcular desplazamiento
                offset_x = (face_center_x - self.center_x) * self.sensitivity_x
                offset_y = (face_center_y - self.center_y) * self.sensitivity_y
                
                # Mapear a pantalla (izquierda es izquierda, derecha es derecha)
                target_x = self.screen_width / 2 + (offset_x * self.screen_width)
                target_y = self.screen_height / 2 + (offset_y * self.screen_height)
                
                # Limitar a la pantalla
                target_x = max(0, min(self.screen_width - 1, target_x))
                target_y = max(0, min(self.screen_height - 1, target_y))
                
                # Suavizar
                smooth_x, smooth_y = self.calculate_smoothed_position(target_x, target_y)
                
                try:
                    pyautogui.moveTo(smooth_x, smooth_y)
                except:
                    pass
            
            # Manejar gestos
            self.handle_gestures(current_time, left_wink, right_wink, both_closed)
        
        else:
            self.no_face_counter += 1
            if self.no_face_counter > self.MAX_NO_FACE_FRAMES:
                self.left_ear_history = []
                self.right_ear_history = []
        
        # Dibujar información
        self.draw_info(frame, current_time, face_detected)
        
        return frame

    def draw_info(self, frame, current_time, face_detected):
        """Dibuja información en el frame."""
        h, w = frame.shape[:2]
        
        # Fondo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (300, 160), (0, 0, 0), -1)
        cv2.rectangle(overlay, (w - 300, 5), (w - 5, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Estado de detección
        det_status = "ROSTRO DETECTADO" if face_detected else "BUSCANDO ROSTRO..."
        det_color = (0, 255, 0) if face_detected else (0, 0, 255)
        cv2.putText(frame, det_status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, det_color, 2)
        
        # Barras de parpadeo (invertidas: más lleno = más cerrado)
        cv2.putText(frame, "Ojo Izq:", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        bar_width = int(self.left_ear * 150)
        bar_color = (0, 0, 255) if self.left_ear > 0.4 else (0, 255, 0)
        cv2.rectangle(frame, (90, 40), (90 + bar_width, 55), bar_color, -1)
        cv2.rectangle(frame, (90, 40), (240, 55), (255, 255, 255), 1)
        cv2.putText(frame, f"{self.left_ear:.2f}", (250, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(frame, "Ojo Der:", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        bar_width = int(self.right_ear * 150)
        bar_color = (0, 0, 255) if self.right_ear > 0.4 else (0, 255, 0)
        cv2.rectangle(frame, (90, 65), (90 + bar_width, 80), bar_color, -1)
        cv2.rectangle(frame, (90, 65), (240, 80), (255, 255, 255), 1)
        cv2.putText(frame, f"{self.right_ear:.2f}", (250, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Estado de calibración
        cal_status = "CALIBRADO" if self.calibrated else "CALIBRANDO..."
        cal_color = (0, 255, 0) if self.calibrated else (0, 255, 255)
        cv2.putText(frame, cal_status, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cal_color, 2)
        
        # Tecnología en uso
        tech = "MediaPipe 478pts" if FACE_DETECTOR_AVAILABLE else "Haar Cascades"
        cv2.putText(frame, f"Motor: {tech}", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Barra de progreso para screenshot
        if self.both_closed_start is not None:
            progress = (current_time - self.both_closed_start) / self.SCREENSHOT_HOLD_TIME
            progress = min(1.0, progress)
            bar_width = int(280 * progress)
            cv2.rectangle(frame, (10, 135), (10 + bar_width, 155), (0, 165, 255), -1)
            cv2.rectangle(frame, (10, 135), (290, 155), (255, 255, 255), 2)
            cv2.putText(frame, "SCREENSHOT", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instrucciones
        instructions = [
            "CONTROLES:",
            "Mover cabeza -> Mouse",
            "Guino ojo izq -> Click izq",
            "Guino ojo der -> Click der",
            "Parpadeo x2 -> Doble click",
            "Ojos cerrados 1s -> Screenshot",
            "",
            "C = Recalibrar",
            "ESC = Salir pantalla completa",
            "Q = Salir"
        ]
        
        for i, text in enumerate(instructions):
            color = (255, 255, 0) if i == 0 else (255, 255, 255)
            cv2.putText(frame, text, (w - 290, 25 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def recalibrate(self):
        """Recalibra el centro de la posición."""
        self.calibrated = False
        self.position_history = []
        self.smooth_x = self.screen_width // 2
        self.smooth_y = self.screen_height // 2
        self.left_ear_history = []
        self.right_ear_history = []
        print("🎯 Recalibrando... Mira al centro de la pantalla")

    def run(self):
        """Bucle principal de la aplicación."""
        print("\n" + "=" * 60)
        print("   🖱️  MOUSE OCULAR - Control con Visión Artificial")
        print("=" * 60)
        print("\n📌 CONTROLES:")
        print("   🔄 Mover la cabeza -------> Mover el mouse")
        print("   😉 Guiñar ojo izquierdo --> Click izquierdo")
        print("   😉 Guiñar ojo derecho ----> Click derecho")
        print("   😑 Doble parpadeo --------> Doble click")
        print("   😴 Ojos cerrados 1 seg ---> Screenshot")
        print("\n⌨️  TECLAS:")
        print("   C -> Recalibrar posición")
        print("   ESC -> Salir de pantalla completa")
        print("   Q -> Salir")
        print("\n⚠️  Mira al centro de la pantalla para calibrar...")
        print("=" * 60 + "\n")
        
        # Crear ventana en pantalla completa
        window_name = 'Mouse Ocular - Control con Vision Artificial'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error: No se puede leer de la cámara")
                    break
                
                processed_frame = self.process_frame(frame)
                cv2.imshow(window_name, processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('c') or key == ord('C'):
                    self.recalibrate()
                elif key == 27:  # ESC
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupción del usuario")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("\n👋 Mouse Ocular cerrado. ¡Hasta luego!")


def main():
    """Función principal."""
    try:
        mouse_ocular = MouseOcular()
        mouse_ocular.run()
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        input("\nPresiona Enter para salir...")


if __name__ == "__main__":
    main()
