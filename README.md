# 🖱️ Mouse Ocular - Control de Mouse con Seguimiento de Ojos

Sistema de control de mouse utilizando visión artificial y seguimiento ocular. Permite mover el cursor con los movimientos de la cabeza y realizar clicks mediante gestos de parpadeo.

## ✨ Características

| Gesto | Acción |
|-------|--------|
|  Mover la cabeza | Mover el cursor del mouse |
|  Guiñar ojo izquierdo | Click izquierdo |
|  Guiñar ojo derecho | Click derecho |
|  Doble parpadeo rápido | Doble click |
|  Cerrar ambos ojos por 1 segundo | Captura de pantalla (Screenshot) |

## 📋 Requisitos

- Python 3.8 o superior
- Webcam funcional
- Windows 10/11

## 🚀 Instalación

### 1. Crear y activar entorno virtual

```powershell
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual (Windows PowerShell)
.\venv\Scripts\activate

# Activar entorno virtual (Windows CMD)
venv\Scripts\activate.bat

# Activar entorno virtual (Linux/Mac)
source venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 🎮 Uso

### Ejecutar el programa

```powershell
# Asegúrate de tener el entorno virtual activado
.\venv\Scripts\activate

# Ejecutar
python mouse_ocular.py
```

### Controles de teclado

| Tecla | Función |
|-------|---------|
| `C` | Recalibrar posición (mira al centro de la pantalla) |
| `Q` | Salir del programa |

## ⚙️ Configuración

Puedes ajustar los siguientes parámetros en el archivo `mouse_ocular.py`:

```python
# Sensibilidad del movimiento (mayor = más rápido)
self.sensitivity_x = 4.0
self.sensitivity_y = 3.5

# Suavizado del movimiento (menor = más suave pero más lento)
self.SMOOTHING_FACTOR = 0.08

# Frames necesarios para confirmar cierre de ojo (más = menos falsos positivos)
self.BLINK_THRESHOLD = 4

# Tiempo para screenshot con ambos ojos cerrados
self.SCREENSHOT_HOLD_TIME = 1.2  # segundos

# Tiempo máximo entre parpadeos para considerar doble click
self.DOUBLE_BLINK_THRESHOLD = 0.7  # segundos

# Cooldown entre clicks para evitar repeticiones
self.CLICK_COOLDOWN = 0.5  # segundos
```

## 🔧 Solución de Problemas

### El cursor se mueve erráticamente
- Recalibra presionando `C` mientras miras al centro de la pantalla
- Aumenta el `SMOOTHING_FACTOR` para movimientos más suaves (ej: 0.05)
- Asegúrate de tener buena iluminación frontal

### No detecta los guiños
- Reduce el valor de `BLINK_THRESHOLD` (ej: 3)
- Verifica que la cámara esté a la altura de los ojos
- Asegúrate de que hay buena iluminación

### Clicks falsos o accidentales
- Aumenta el valor de `BLINK_THRESHOLD` (ej: 5 o 6)
- Aumenta `CLICK_COOLDOWN` para mayor espacio entre clicks (ej: 0.7)

### El programa no detecta mi rostro
- Acércate más a la cámara (distancia recomendada: 50-80cm)
- Mejora la iluminación del ambiente
- Evita contraluz (luz detrás de ti)

### Los screenshots no funcionan
- Verifica que tienes permisos de escritura en la carpeta del proyecto
- Los screenshots se guardan en el mismo directorio que `mouse_ocular.py`

## 📁 Estructura del Proyecto

```
Proyecto Mouse Ocular/
├── venv/                  # Entorno virtual de Python
├── mouse_ocular.py        # Programa principal
├── requirements.txt       # Dependencias
├── README.md              # Este archivo
└── screenshot_*.png       # Screenshots capturados (se generan al usar la función)
```

## 🛠️ Tecnologías Utilizadas

- **OpenCV**: Captura de video y detección de rostro/ojos con Haar Cascades
- **PyAutoGUI**: Control del mouse y capturas de pantalla
- **NumPy**: Cálculos numéricos para suavizado
- **Screeninfo**: Obtención de dimensiones de la pantalla

## 📝 Cómo funciona

1. **Detección de rostro**: Se usa un clasificador Haar Cascade para detectar el rostro en cada frame.
2. **Detección de ojos**: Se divide la región del rostro y se buscan los ojos usando clasificadores especializados.
3. **Seguimiento**: El centro del rostro se mapea a coordenadas de pantalla con suavizado.
4. **Gestos**: Se detectan transiciones en el estado de los ojos (abierto→cerrado) para generar acciones.

## 💡 Tips para mejor funcionamiento

1. **Iluminación**: Asegúrate de tener luz frontal, evita contraluz
2. **Posición**: Siéntate frente a la cámara con la cara bien visible
3. **Calibración**: Al iniciar, mira al centro de la pantalla para una mejor calibración
4. **Recalibrar**: Si el cursor no sigue bien, presiona `C` para recalibrar

## Autores

* ### **Victor Francisco Villafaña Hernández**
* ### **Edgar Cortes Garcia**

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Siéntete libre de abrir issues o pull requests.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.
