# Video Meme Cam Detector
## tarea-02

- **Yurineth Vargas Salazar**

- Asignatura: Dispositivos Periféricos y Plataformas para la Interacción Digital **DIS9087**

Proyecto de reconocimiento de gestos, utilizando Python y MediaPipe. Realizado tomando como referencia este repositorio:

- <https://github.com/catherpiee/meowmeowcatcam>


Apunta tu cámara web hacia ti, realiza un gesto facial o de manos, y reproduce en tiempo real el meme en video correspondiente.

Dos paneles lado a lado:
- **Cámara** — Flujo de tu webcam con landmarks de manos y rostro dibujados, más un HUD de telemetría y depuración en vivo.
- **Meme** — Video en bucle (`.mp4`) del meme correspondiente al gesto que estás realizando.

## Jerarquía de Decisión de Gestos (NewMemes)

Los gestos se evalúan en este orden de precedencia:

| # | Nombre | Como se activa | Video (`memes/NewMemes/`) |
|---|---|---|---|
| **1** | **DefaultCat** | Estado base sin gestos o reposo (fallback) | `DefaultCat.mp4` |
| **2** | **ClapClap** | Ambas manos abiertas, a la altura del pecho, con las palmas enfrentadas (distancia entre manos < 1.4× escala) | `ClapClap.mp4` |
| **3** | **Sad** | Cabeza inclinada hacia abajo entre 35° y 45° respecto a la vertical, con la barbilla aproximándose al pecho | `Sad.mp4` |
| **4** | **Muejeje** | Ambas manos presentes, todos dedos extendidos tocándose (distancia entre puntas < 1.4× escala) | `Muejeje.mp4` |
| **5** | **Hiii** | 1 mano levantada al lado de rostro/mejillas, con la palma orientada hacia la cámara | `Hiii.mp4` |
| **6** | **Coquette** | Una mano ubicada lateralmente junto a la cabeza, con la muñeca y/o palma próxima a la región de una de las orejas | `Coquette.mp4` |
| **7** | **SpeedLaugh** | Ojos cerrados, labios fruncidos, cejas fruncidas y mueve cabeza ligeramente hacia abajo entre 10° y 20° respecto a la vertical | `SpeedLaugh.mp4` |
| **8** | **EwwCover** | 1 mano ubicada frente a la región central inferior del rostro, cubriendo o aproximándose a la zona de nariz y boca | `EwwCover.mp4` |

---

- [carpeta de imágenes](./dppi-tarea-02/memes/NewMemes)

- [video](./)

## Ejecución en Escritorio (Python)

Requiere Python 3 y cámara web.

```bash
cd Gatush-main/ddpi-tarea-02
python -m venv .venv

# En Windows (PowerShell):
.venv\Scripts\activate

# En macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python gesture_meme.py
```

Presiona `q` o `Esc` en la ventana de la cámara para salir.

---

## Ejecución en Navegador (Web)

No requiere instalación local. Debe servirse sobre HTTP local para que el navegador conceda permisos de cámara:

```bash
cd Gatush-main/ddpi-tarea-02
python -m http.server 8000
```

Abre `http://localhost:8000` en tu navegador y autoriza el acceso a la cámara.

---

## HUD de Depuración en Vivo

La pantalla muestra en la esquina superior izquierda valores en tiempo real:
- **Gesto activo**: Nombre del gesto detectado.
- **Pitch**: Inclinación vertical de la cabeza (en grados) para calibrar `Sad` (35°-45°) vs `SpeedLaugh` (10°-20°).
- **Yaw**: Giro lateral de la cabeza.
- **SpeedLaugh**: Valores normalizados de parpadeo/ojos cerrados, labios fruncidos y cejas fruncidas.
