"""
Webcam gesture -> meme detector (desktop version).
Linked to 'memes/NewMemes' videos.

DEBUG MODE ACTIVADO:
  - DEBUG = True al inicio para activar/desactivar logs detallados.
  - No altera ninguna lógica ni umbral existente.

Gestures Hierarchy:
  1 - DefaultCat   -> memes/NewMemes/DefaultCat.mp4 (Base / neutral state)
  2 - ClapClap     -> memes/NewMemes/ClapClap.mp4   (2 open hands at chest level, palms facing each other, dist < 1.4x scale)
  3 - Sad          -> memes/NewMemes/Sad.mp4        (Head tilted down 35°-45° relative to vertical, chin to chest)
  4 - Muejeje      -> memes/NewMemes/Muejeje.mp4    (2 hands, all fingers extended, fingertips touching < 1.4x scale)
  5 - Hiii         -> memes/NewMemes/Hiii.mp4       (1 hand raised extended beside face/cheek, palm facing camera)
  6 - Coquette     -> memes/NewMemes/Coquette.mp4   (1 hand beside head, wrist/palm near ear region)
  7 - SpeedLaugh   -> memes/NewMemes/SpeedLaugh.mp4 (Eyes closed, puckered lips, furrowed brows, head down 10°-20°)
  8 - EwwCover     -> memes/NewMemes/EwwCover.mp4   (1 hand covers nose)

Press q or ESC to quit.
"""

import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from mediapipe import Image, ImageFormat

# ==============================================================================
# CONFIGURACION DE DEBUG
# ==============================================================================
DEBUG = True
DEBUG_FRAME_INTERVAL = 6  # Imprime cada 6 frames (~5 veces por seg a 30fps) para no saturar la consola

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
MEMES = ROOT / "memes" / "NewMemes"

GESTURE_MEMES = {
    "default": ["DefaultCat.mp4"],
    "ClapClap": ["ClapClap.mp4"],
    "Sad": ["Sad.mp4"],
    "Muejeje": ["Muejeje.mp4"],
    "Hiii": ["Hiii.mp4"],
    "Coquette": ["Coquette.mp4"],
    "SpeedLaugh": ["SpeedLaugh.mp4"],
    "EwwCover": ["EwwCover.mp4"],
}

STABLE_FRAMES_REQUIRED = 4
DEFAULT_FALLBACK_MS = 600
FACE_STALE_MS = 1200

# Angular thresholds (degrees)
SAD_PITCH_MIN_DEG = 30.0
SAD_PITCH_MAX_DEG = 55.0
SPEEDLAUGH_PITCH_MIN_DEG = 8.0
SPEEDLAUGH_PITCH_MAX_DEG = 28.0
EWW_COVER_NOSE_DIST_THR = 0.85

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ---- Geometry Helpers ----------------------------------------------------
def p3(lm):
    return np.array([lm.x, lm.y, getattr(lm, "z", 0.0)])


def dist(a, b):
    return float(np.linalg.norm(a - b))


def angle_deg(v1, v2):
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 < 1e-9 or m2 < 1e-9:
        return 180.0
    cos_a = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def finger_extended(pts, mcp, pip, tip):
    v1 = pts[pip] - pts[mcp]
    v2 = pts[tip] - pts[pip]
    return angle_deg(v1, v2) < 45


def head_pose_from_transform_matrix(matrix):
    r = np.asarray(matrix)[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy < 1e-6:
        return 0.0, 0.0
    yaw = math.atan2(-r[2, 0], sy)
    pitch = math.atan2(r[2, 1], r[2, 2])
    return math.degrees(yaw), math.degrees(pitch)


def classify_hand(landmarks):
    pts = [p3(lm) for lm in landmarks]
    hand_scale = dist(pts[0], pts[9]) or 1e-6

    index_up = finger_extended(pts, 5, 6, 8)
    middle_up = finger_extended(pts, 9, 10, 12)
    ring_up = finger_extended(pts, 13, 14, 16)
    pinky_up = finger_extended(pts, 17, 18, 20)

    thumb_pinky_spread = dist(pts[4], pts[17]) / hand_scale
    thumb_out = thumb_pinky_spread > 1.05

    curled_count = sum(1 for v in (index_up, middle_up, ring_up, pinky_up) if not v)

    # Check if fingers are extended upright (tips above respective PIP joints)
    fingers_upright = (
        pts[8][1] < pts[6][1] and pts[12][1] < pts[10][1] and pts[16][1] < pts[14][1]
    )

    return {
        "indexUp": index_up,
        "middleUp": middle_up,
        "ringUp": ring_up,
        "pinkyUp": pinky_up,
        "thumbOut": thumb_out,
        "curledCount": curled_count,
        "handScale": hand_scale,
        "fingersUpright": fingers_upright,
        "wrist": pts[0],
        "thumbTip": pts[4],
        "indexTip": pts[8],
        "middleTip": pts[12],
        "ringTip": pts[16],
        "pinkyTip": pts[20],
        "palmCenter": pts[9],
    }


class GestureState:
    def __init__(self):
        self.last_face = None
        self.face_seen_this_frame = False
        self.last_yaw_debug = 0.0
        self.last_pitch_debug = 0.0
        self.last_blink_debug = 0.0
        self.last_pucker_debug = 0.0
        self.last_brow_debug = 0.0

    def update_face(self, face_result):
        now = time.time() * 1000
        saw_face = bool(face_result.face_landmarks)

        if saw_face:
            f = face_result.face_landmarks[0]
            pts = [p3(lm) for lm in f]

            upper_lip, lower_lip = pts[13], pts[14]
            mouth_corner_r, mouth_corner_l = pts[61], pts[291]
            right_cheek, left_cheek = pts[234], pts[454]
            right_ear, left_ear = pts[127], pts[356]
            right_temple, left_temple = pts[103], pts[332]
            nose_tip = pts[1]
            chin = pts[152]
            forehead = pts[10]

            r_eye_top, r_eye_bot = pts[159], pts[145]
            r_eye_in, r_eye_out = pts[133], pts[33]
            l_eye_top, l_eye_bot = pts[386], pts[374]
            l_eye_in, l_eye_out = pts[362], pts[263]

            r_brow = pts[70]
            l_brow = pts[300]

            mouth_center = (upper_lip + lower_lip) / 2
            mouth_width = dist(mouth_corner_r, mouth_corner_l)
            mouth_open = dist(upper_lip, lower_lip)
            face_width = dist(right_cheek, left_cheek) or 1e-6
            face_center = (right_cheek + left_cheek) / 2
            lower_face_center = (nose_tip + mouth_center) / 2

            yaw_deg, pitch_deg = 0.0, 0.0
            if face_result.facial_transformation_matrixes:
                yaw_deg, pitch_deg = head_pose_from_transform_matrix(
                    face_result.facial_transformation_matrixes[0]
                )

            # Geometric fallback metrics
            r_eye_open = dist(r_eye_top, r_eye_bot) / (dist(r_eye_in, r_eye_out) or 1e-6)
            l_eye_open = dist(l_eye_top, l_eye_bot) / (dist(l_eye_in, l_eye_out) or 1e-6)
            avg_eye_open = (r_eye_open + l_eye_open) / 2

            r_brow_dist = dist(r_brow, r_eye_top) / face_width
            l_brow_dist = dist(l_brow, l_eye_top) / face_width
            avg_brow_height = (r_brow_dist + l_brow_dist) / 2

            geom_pitch_ratio = (nose_tip[1] - (r_eye_top[1] + l_eye_top[1]) / 2) / (
                (chin[1] - (r_eye_top[1] + l_eye_top[1]) / 2) or 1e-6
            )

            # Blendshapes extraction
            blink_score = 0.0
            pucker_score = 0.0
            brow_score = 0.0
            brow_furrow_score = 0.0

            if face_result.face_blendshapes:
                categories = {c.category_name: c.score for c in face_result.face_blendshapes[0]}
                blink_score = max(
                    categories.get("eyeBlinkLeft", 0.0),
                    categories.get("eyeBlinkRight", 0.0),
                    categories.get("eyeSquintLeft", 0.0),
                    categories.get("eyeSquintRight", 0.0),
                )
                pucker_score = max(
                    categories.get("mouthPucker", 0.0),
                    categories.get("mouthFunnel", 0.0),
                    (categories.get("mouthPressLeft", 0.0) + categories.get("mouthPressRight", 0.0)) / 2,
                )
                brow_score = max(
                    categories.get("browInnerUp", 0.0),
                    categories.get("browOuterUpLeft", 0.0),
                    categories.get("browOuterUpRight", 0.0),
                )
                brow_furrow_score = max(
                    categories.get("browDownLeft", 0.0),
                    categories.get("browDownRight", 0.0),
                    categories.get("browInnerUp", 0.0),
                )

            # Final blended features
            is_eyes_closed = blink_score > 0.38 or avg_eye_open < 0.18
            is_puckered = pucker_score > 0.30 or (mouth_width / face_width < 0.33 and mouth_open / face_width < 0.15)
            is_brows_active = brow_score > 0.30 or brow_furrow_score > 0.30 or avg_brow_height > 0.16

            # Pitch down checks:
            # Sad: between 30° and 55° (approx 35°-45° range)
            is_head_down_sad = (
                (SAD_PITCH_MIN_DEG <= abs(pitch_deg) <= SAD_PITCH_MAX_DEG) or geom_pitch_ratio > 0.65
            )
            # SpeedLaugh: slight pitch down between 8° and 28° (approx 10°-20° range)
            is_head_down_speedlaugh = (
                (SPEEDLAUGH_PITCH_MIN_DEG <= abs(pitch_deg) <= SPEEDLAUGH_PITCH_MAX_DEG) or (0.54 <= geom_pitch_ratio <= 0.64)
            )

            self.last_face = {
                "mouthCenter": mouth_center,
                "faceWidth": face_width,
                "faceCenter": face_center,
                "lowerFaceCenter": lower_face_center,
                "noseTip": nose_tip,
                "rightCheek": right_cheek,
                "leftCheek": left_cheek,
                "rightEar": right_ear,
                "leftEar": left_ear,
                "rightTemple": right_temple,
                "leftTemple": left_temple,
                "chin": chin,
                "forehead": forehead,
                "yawDeg": yaw_deg,
                "pitchDeg": pitch_deg,
                "isEyesClosed": is_eyes_closed,
                "isPuckered": is_puckered,
                "isBrowsActive": is_brows_active,
                "isHeadDownSad": is_head_down_sad,
                "isHeadDownSpeedLaugh": is_head_down_speedlaugh,
                "geomPitchRatio": geom_pitch_ratio,
                "t": now,
            }
            self.last_yaw_debug = yaw_deg
            self.last_pitch_debug = pitch_deg
            self.last_blink_debug = blink_score if blink_score > 0 else (1.0 - avg_eye_open * 4)
            self.last_pucker_debug = pucker_score if pucker_score > 0 else (0.4 - mouth_width / face_width)
            self.last_brow_debug = max(brow_score, brow_furrow_score) if max(brow_score, brow_furrow_score) > 0 else avg_brow_height

        self.face_seen_this_frame = saw_face

    def decide(self, hand_result):
        now = time.time() * 1000
        face_is_fresh = self.last_face is not None and (now - self.last_face["t"] < FACE_STALE_MS)

        hands = [classify_hand(lm) for lm in hand_result.hand_landmarks] if hand_result.hand_landmarks else []

        # Estructura de telemetría detallada para modo DEBUG
        debug_info = {
            "face_is_fresh": face_is_fresh,
            "saw_face": self.face_seen_this_frame,
            "num_hands": len(hands),
            "hands": hands,
            "face": self.last_face,
            "eval_clap": None,
            "eval_muejeje": None,
            "eval_eww": None,
            "eval_coquette": None,
            "eval_hiii": None,
            "eval_speedlaugh": None,
            "eval_sad": None,
            "chosen_gesture": "default",
        }

        # ------------------------------------------------------------------
        # 1. Check Two-Hand Gestures (ClapClap, Muejeje)
        # ------------------------------------------------------------------
        if len(hands) >= 2:
            h1, h2 = hands[0], hands[1]
            avg_scale = (h1["handScale"] + h2["handScale"]) / 2
            d_palm = dist(h1["palmCenter"], h2["palmCenter"]) / avg_scale
            d_wrist = dist(h1["wrist"], h2["wrist"]) / avg_scale
            d_index_tip = dist(h1["indexTip"], h2["indexTip"]) / avg_scale
            d_middle_tip = dist(h1["middleTip"], h2["middleTip"]) / avg_scale

            both_open = h1["curledCount"] <= 1 and h2["curledCount"] <= 1
            is_chest_level = False
            if face_is_fresh:
                is_chest_level = (h1["palmCenter"][1] > self.last_face["mouthCenter"][1] + self.last_face["faceWidth"] * 0.2)

            is_clap = bool(both_open and d_palm < 1.40 and d_wrist < 1.65 and (not face_is_fresh or is_chest_level or d_palm < 1.15))
            debug_info["eval_clap"] = {
                "both_open": both_open,
                "d_palm": d_palm,
                "d_wrist": d_wrist,
                "is_chest_level": is_chest_level,
                "result": is_clap,
            }

            # 2. ClapClap
            if is_clap:
                debug_info["chosen_gesture"] = "ClapClap"
                return "ClapClap", debug_info

            # 4. Muejeje: Ambas manos presentes, todos dedos extendidos tocándose en puntas (< 1.4x escala)
            fingers_extended = h1["curledCount"] <= 1 and h2["curledCount"] <= 1
            is_muejeje = bool(fingers_extended and d_index_tip < 1.40 and (d_middle_tip < 1.55 or d_palm > 0.65))
            debug_info["eval_muejeje"] = {
                "fingers_extended": fingers_extended,
                "d_index_tip": d_index_tip,
                "d_middle_tip": d_middle_tip,
                "d_palm": d_palm,
                "result": is_muejeje,
            }

            if is_muejeje:
                debug_info["chosen_gesture"] = "Muejeje"
                return "Muejeje", debug_info
        else:
            debug_info["eval_clap"] = {"result": False, "reason": f"Requiere 2 manos (detectadas: {len(hands)})"}
            debug_info["eval_muejeje"] = {"result": False, "reason": f"Requiere 2 manos (detectadas: {len(hands)})"}

        # ------------------------------------------------------------------
        # 2. Check One-Hand Face/Head Gestures (EwwCover, Coquette, Hiii)
        # ------------------------------------------------------------------
        if face_is_fresh and len(hands) >= 1:
            lf = self.last_face
            mouth_center = lf["mouthCenter"]
            face_width = lf["faceWidth"]
            face_center = lf["faceCenter"]
            lower_face_center = lf["lowerFaceCenter"]
            nose_tip = lf["noseTip"]

            eww_candidates = []
            coquette_candidates = []
            hiii_candidates = []

            for idx, h in enumerate(hands):
                # Distances to nose and lower face
                d_nose = dist(h["palmCenter"], nose_tip) / face_width
                d_index_nose = dist(h["indexTip"], nose_tip) / face_width
                d_middle_nose = dist(h["middleTip"], nose_tip) / face_width
                d_wrist_nose = dist(h["wrist"], nose_tip) / face_width
                d_lower_face = dist(h["palmCenter"], lower_face_center) / face_width
                d_mouth = dist(h["palmCenter"], mouth_center) / face_width
                dx_face = abs(h["palmCenter"][0] - face_center[0]) / face_width
                is_centered = dx_face < 0.50
                min_dist_nose = min(d_nose, d_index_nose, d_middle_nose, d_wrist_nose)

                # 8. EwwCover: 1 mano cubre nariz
                cond_nose_close = min_dist_nose < EWW_COVER_NOSE_DIST_THR
                cond_lower_face_centered = (is_centered and d_lower_face < 0.75)
                is_covering_nose = bool(cond_nose_close or cond_lower_face_centered)

                eww_candidates.append({
                    "hand_idx": idx,
                    "d_nose_palm": d_nose,
                    "d_index_nose": d_index_nose,
                    "d_middle_nose": d_middle_nose,
                    "d_wrist_nose": d_wrist_nose,
                    "min_dist_nose": min_dist_nose,
                    "d_lower_face": d_lower_face,
                    "d_mouth": d_mouth,
                    "dx_face": dx_face,
                    "is_centered": is_centered,
                    f"cond_nose_close (<{EWW_COVER_NOSE_DIST_THR:.2f})": cond_nose_close,
                    "cond_lower_face_centered (<0.75 & dx<0.50)": cond_lower_face_centered,
                    "result": is_covering_nose,
                })

                if is_covering_nose:
                    debug_info["eval_eww"] = eww_candidates
                    debug_info["chosen_gesture"] = "EwwCover"
                    return "EwwCover", debug_info

                # 6. Coquette: Una mano ubicada lateralmente junto a la cabeza, muñeca/palma próxima a una de las orejas
                d_r_ear = dist(h["palmCenter"], lf["rightEar"]) / face_width
                d_l_ear = dist(h["palmCenter"], lf["leftEar"]) / face_width
                d_r_temple = dist(h["palmCenter"], lf["rightTemple"]) / face_width
                d_l_temple = dist(h["palmCenter"], lf["leftTemple"]) / face_width
                d_r_ear_wrist = dist(h["wrist"], lf["rightEar"]) / face_width
                d_l_ear_wrist = dist(h["wrist"], lf["leftEar"]) / face_width
                min_ear_dist = min(d_r_ear, d_l_ear, d_r_temple, d_l_temple, d_r_ear_wrist, d_l_ear_wrist)
                is_coquette_height = h["palmCenter"][1] < (mouth_center[1] - face_width * 0.1)
                is_coquette = bool(min_ear_dist < 0.95 and is_coquette_height)

                coquette_candidates.append({
                    "hand_idx": idx,
                    "min_ear_dist": min_ear_dist,
                    "is_coquette_height": is_coquette_height,
                    "result": is_coquette,
                })

                if is_coquette:
                    debug_info["eval_eww"] = eww_candidates
                    debug_info["eval_coquette"] = coquette_candidates
                    debug_info["chosen_gesture"] = "Coquette"
                    return "Coquette", debug_info

                # 5. Hiii: 1 mano levantada extendida al lado de rostro/mejillas, con la palma orientada hacia la cámara
                is_open = h["curledCount"] <= 1
                is_beside_face = dx_face > 0.45
                d_cheek = min(dist(h["palmCenter"], lf["rightCheek"]), dist(h["palmCenter"], lf["leftCheek"])) / face_width
                is_hiii = bool(is_open and is_beside_face and (d_mouth < 2.5 and d_cheek < 1.6) and h["fingersUpright"])

                hiii_candidates.append({
                    "hand_idx": idx,
                    "is_open": is_open,
                    "is_beside_face": is_beside_face,
                    "d_cheek": d_cheek,
                    "d_mouth": d_mouth,
                    "fingers_upright": h["fingersUpright"],
                    "result": is_hiii,
                })

                if is_hiii:
                    debug_info["eval_eww"] = eww_candidates
                    debug_info["eval_coquette"] = coquette_candidates
                    debug_info["eval_hiii"] = hiii_candidates
                    debug_info["chosen_gesture"] = "Hiii"
                    return "Hiii", debug_info

            debug_info["eval_eww"] = eww_candidates
            debug_info["eval_coquette"] = coquette_candidates
            debug_info["eval_hiii"] = hiii_candidates
        else:
            reason = "Rostro no detectado/stale" if not face_is_fresh else "No hay manos detectadas"
            debug_info["eval_eww"] = {"result": False, "reason": reason}
            debug_info["eval_coquette"] = {"result": False, "reason": reason}
            debug_info["eval_hiii"] = {"result": False, "reason": reason}

        # ------------------------------------------------------------------
        # 3. Check SpeedLaugh (Composite Face Expression + slight pitch 10°-20°)
        # 7. SpeedLaugh: Ojos cerrados, labios fruncidos, cejas fruncidas y mueve cabeza ligeramente hacia abajo entre 10° y 20°
        # ------------------------------------------------------------------
        if face_is_fresh:
            lf = self.last_face
            speed_laugh_features = [
                lf["isEyesClosed"],
                lf["isPuckered"],
                lf["isBrowsActive"],
                lf["isHeadDownSpeedLaugh"],
            ]
            is_speed_laugh = bool(sum(1 for feat in speed_laugh_features if feat) >= 3 and (lf["isEyesClosed"] and lf["isPuckered"]))
            debug_info["eval_speedlaugh"] = {
                "is_eyes_closed": lf["isEyesClosed"],
                "is_puckered": lf["isPuckered"],
                "is_brows_active": lf["isBrowsActive"],
                "is_head_down_speedlaugh": lf["isHeadDownSpeedLaugh"],
                "result": is_speed_laugh,
            }
            if is_speed_laugh:
                debug_info["chosen_gesture"] = "SpeedLaugh"
                return "SpeedLaugh", debug_info

            # ------------------------------------------------------------------
            # 4. Check Sad (Cabeza abajo 35°-45°)
            # 3. Sad: Cabeza inclinada hacia abajo entre 35° y 45° respecto a la vertical, barbilla al pecho
            # ------------------------------------------------------------------
            is_sad = bool(lf["isHeadDownSad"])
            debug_info["eval_sad"] = {
                "is_head_down_sad": is_sad,
                "pitch_deg": lf["pitchDeg"],
                "geom_pitch_ratio": lf["geomPitchRatio"],
                "result": is_sad,
            }
            if is_sad:
                debug_info["chosen_gesture"] = "Sad"
                return "Sad", debug_info
        else:
            debug_info["eval_speedlaugh"] = {"result": False, "reason": "Rostro no detectado/stale"}
            debug_info["eval_sad"] = {"result": False, "reason": "Rostro no detectado/stale"}

        # ------------------------------------------------------------------
        # 1. DefaultCat - Estado base sin gestos o reposo
        # ------------------------------------------------------------------
        debug_info["chosen_gesture"] = "default"
        return "default", debug_info


def print_debug_log(frame_idx, state, hand_result, debug_info, gesture, candidate_gesture, candidate_streak, current_gesture):
    """Imprime un bloque detallado de telemetría para diagnosticar EwwCover y todos los gestos."""
    print("\n" + "=" * 70)
    print(f"[DEBUG LOG - FRAME #{frame_idx}]")
    print("=" * 70)

    # 1. Estado del Rostro
    lf = debug_info.get("face")
    if lf:
        print("[ROSTRO]")
        print(f"  * Detectado en este frame: {'SI' if debug_info['saw_face'] else 'NO'} | Estado fresco: {'SI' if debug_info['face_is_fresh'] else 'NO'}")
        print(f"  * Posicion Nariz (lm[1]):      X={lf['noseTip'][0]:.4f}, Y={lf['noseTip'][1]:.4f}, Z={lf['noseTip'][2]:.4f}")
        print(f"  * Posicion Boca Central:       X={lf['mouthCenter'][0]:.4f}, Y={lf['mouthCenter'][1]:.4f}, Z={lf['mouthCenter'][2]:.4f}")
        print(f"  * Posicion Centro Facial:      X={lf['faceCenter'][0]:.4f}, Y={lf['faceCenter'][1]:.4f}, Z={lf['faceCenter'][2]:.4f}")
        print(f"  * Posicion Cara Inferior:      X={lf['lowerFaceCenter'][0]:.4f}, Y={lf['lowerFaceCenter'][1]:.4f}, Z={lf['lowerFaceCenter'][2]:.4f}")
        print(f"  * Ancho Facial (face_width):   {lf['faceWidth']:.4f}")
        print(f"  * Orientacion Cabeza:          Pitch={lf['pitchDeg']:+.1f}° | Yaw={lf['yawDeg']:+.1f}° | Ratio Nariz/Barbilla={lf['geomPitchRatio']:.3f}")
        print(f"  * Expresiones Faciales:        Ojos Cerrados={'SI' if lf['isEyesClosed'] else 'NO'} | Labios Fruncidos={'SI' if lf['isPuckered'] else 'NO'} | Cejas Activas={'SI' if lf['isBrowsActive'] else 'NO'}")
    else:
        print("[ROSTRO] NO DETECTADO")

    # 2. Estado de Manos
    num_hands = debug_info.get("num_hands", 0)
    print(f"\n[MANOS] Cantidad detectada = {num_hands}")
    for idx, h in enumerate(debug_info.get("hands", [])):
        print(f"  * Mano #{idx}:")
        print(f"    - Muneca (lm[0]):          X={h['wrist'][0]:.4f}, Y={h['wrist'][1]:.4f}, Z={h['wrist'][2]:.4f}")
        print(f"    - Centro Palma (lm[9]):    X={h['palmCenter'][0]:.4f}, Y={h['palmCenter'][1]:.4f}, Z={h['palmCenter'][2]:.4f}")
        print(f"    - Punta Indice (lm[8]):    X={h['indexTip'][0]:.4f}, Y={h['indexTip'][1]:.4f}, Z={h['indexTip'][2]:.4f}")
        print(f"    - Punta Medio (lm[12]):    X={h['middleTip'][0]:.4f}, Y={h['middleTip'][1]:.4f}, Z={h['middleTip'][2]:.4f}")
        print(f"    - Escala de Mano:          {h['handScale']:.4f}")
        print(f"    - Dedos Flexionados:       {h['curledCount']} de 4 | Erguida hacia arriba={'SI' if h['fingersUpright'] else 'NO'}")

    # 3. Diagnóstico Especial EwwCover
    print("\n[DIAGNOSTICO ESPECIAL: EwwCover (1 mano cubre nariz)]")
    eval_eww = debug_info.get("eval_eww")
    if isinstance(eval_eww, list) and len(eval_eww) > 0:
        for item in eval_eww:
            h_idx = item["hand_idx"]
            res = item["result"]
            print(f"  * Evaluacion para Mano #{h_idx}:")
            print(f"    - Distancia Palma -> Nariz (norm):   {item['d_nose_palm']:.3f}  (Umbral < {EWW_COVER_NOSE_DIST_THR:.2f} -> {'CUMPLE' if item['d_nose_palm'] < EWW_COVER_NOSE_DIST_THR else 'NO CUMPLE'})")
            print(f"    - Distancia Indice -> Nariz (norm):  {item['d_index_nose']:.3f}  (Umbral < {EWW_COVER_NOSE_DIST_THR:.2f} -> {'CUMPLE' if item['d_index_nose'] < EWW_COVER_NOSE_DIST_THR else 'NO CUMPLE'})")
            print(f"    - Distancia Medio -> Nariz (norm):   {item['d_middle_nose']:.3f}  (Umbral < {EWW_COVER_NOSE_DIST_THR:.2f} -> {'CUMPLE' if item['d_middle_nose'] < EWW_COVER_NOSE_DIST_THR else 'NO CUMPLE'})")
            print(f"    - Distancia Muneca -> Nariz (norm):  {item['d_wrist_nose']:.3f}  (Umbral < {EWW_COVER_NOSE_DIST_THR:.2f} -> {'CUMPLE' if item['d_wrist_nose'] < EWW_COVER_NOSE_DIST_THR else 'NO CUMPLE'})")
            print(f"    - Minima distancia a la Nariz:       {item['min_dist_nose']:.3f}  (Umbral < {EWW_COVER_NOSE_DIST_THR:.2f} -> {'CUMPLE' if item[f'cond_nose_close (<{EWW_COVER_NOSE_DIST_THR:.2f})'] else 'NO CUMPLE'})")
            print(f"    - Distancia a Cara Inferior:         {item['d_lower_face']:.3f}  (Umbral < 0.75 -> {'CUMPLE' if item['d_lower_face'] < 0.75 else 'NO CUMPLE'})")
            print(f"    - Desviacion Horizontal X (|dx|/w):  {item['dx_face']:.3f}  (Umbral < 0.50 [centrado] -> {'CUMPLE' if item['is_centered'] else 'NO CUMPLE'})")
            print(f"    ----------------------------------------------------------")
            print(f"    => EwwCover = {'TRUE (DETECTADO)' if res else 'FALSE (NO DETECTADO)'}")
            if not res:
                reasons = []
                if not item[f"cond_nose_close (<{EWW_COVER_NOSE_DIST_THR:.2f})"]:
                    reasons.append(f"Mínima dist a nariz {item['min_dist_nose']:.3f} >= {EWW_COVER_NOSE_DIST_THR:.2f}")
                if not item["is_centered"]:
                    reasons.append(f"Mano no centrada (|dx|/w={item['dx_face']:.3f} >= 0.50)")
                if item["d_lower_face"] >= 0.75:
                    reasons.append(f"Dist a cara inferior {item['d_lower_face']:.3f} >= 0.75")
                print(f"    => CONDICION FALLIDA: {' | '.join(reasons)}")
    else:
        reason = eval_eww.get("reason", "Desconocido") if isinstance(eval_eww, dict) else "No evaluado"
        print(f"  => EwwCover = FALSE ({reason})")

    # 4. Evaluación de Otros Gestos en la Jerarquía
    print("\n[EVALUACION DE OTROS GESTOS EN JERARQUIA]")
    print(f"  1. ClapClap:    {debug_info.get('eval_clap')}")
    print(f"  2. Muejeje:     {debug_info.get('eval_muejeje')}")
    print(f"  3. Coquette:    {debug_info.get('eval_coquette')}")
    print(f"  4. Hiii:        {debug_info.get('eval_hiii')}")
    print(f"  5. SpeedLaugh:  {debug_info.get('eval_speedlaugh')}")
    print(f"  6. Sad:         {debug_info.get('eval_sad')}")

    # 5. Estado y Transición
    print("\n[ESTADO FINAL DE DETECCION]")
    print(f"  * Gesto instantaneo evaluado: {gesture}")
    print(f"  * Gesto candidato:            {candidate_gesture}")
    print(f"  * Racha de estabilidad:       {candidate_streak} / {STABLE_FRAMES_REQUIRED} frames requeridos")
    print(f"  * Gesto activo en pantalla:   {current_gesture} (Video: {GESTURE_MEMES.get(current_gesture, [''])[0]})")
    print("=" * 70 + "\n")


class VideoMemePlayer:
    def __init__(self, memes_dir, gesture_memes):
        self.memes_dir = memes_dir
        self.gesture_memes = gesture_memes
        self.caps = {}
        for gesture, files in gesture_memes.items():
            video_path = str(memes_dir / files[0])
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Advertencia: No se pudo abrir el video {video_path}")
            self.caps[gesture] = cap
        self.current_gesture = "default"

    def set_gesture(self, gesture):
        if gesture != self.current_gesture:
            self.current_gesture = gesture
            if gesture in self.caps and self.caps[gesture].isOpened():
                self.caps[gesture].set(cv2.CAP_PROP_POS_FRAMES, 0)

    def get_frame(self, target_height):
        cap = self.caps.get(self.current_gesture)
        if cap is None or not cap.isOpened():
            return np.zeros((target_height, int(target_height * 4 / 3), 3), dtype=np.uint8)
        ok, vframe = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, vframe = cap.read()
        if not ok or vframe is None:
            return np.zeros((target_height, int(target_height * 4 / 3), 3), dtype=np.uint8)
        return fit_to_height(vframe, target_height)

    def release(self):
        for cap in self.caps.values():
            if cap.isOpened():
                cap.release()


def draw_debug_hud(frame, state, gesture):
    lines = [
        f"Gesto activo: {gesture}",
        f"Pitch (Inclinacion): {state.last_pitch_debug:+.1f} deg (Sad: 35-45 deg | SpeedLaugh: 10-20 deg)",
        f"Yaw (Giro cabeza): {state.last_yaw_debug:+.1f} deg",
        f"SpeedLaugh: Ojos={state.last_blink_debug:.2f}, Labios={state.last_pucker_debug:.2f}, Cejas={state.last_brow_debug:.2f}",
        f"DEBUG LOGS: {'ON (Consola)' if DEBUG else 'OFF'}",
    ]
    for i, line in enumerate(lines):
        y = 26 + i * 22
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1, cv2.LINE_AA)


def draw_landmarks(frame, hand_result):
    h, w = frame.shape[:2]
    for hand in hand_result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (80, 220, 120), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (60, 140, 255), -1)


def fit_to_height(img, height):
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w * scale), height))


def main():
    hand_landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
        )
    )
    face_landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
    )

    meme_player = VideoMemePlayer(MEMES, GESTURE_MEMES)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la camara web (indice 0)")

    cv2.namedWindow("Camera")
    cv2.namedWindow("Meme")
    cv2.moveWindow("Camera", 40, 80)
    cv2.moveWindow("Meme", 720, 80)

    state = GestureState()
    current_gesture = "default"
    candidate_gesture = "default"
    candidate_streak = 0
    last_non_default_at = time.time() * 1000

    frame_idx = 0
    start_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            frame = cv2.flip(frame, 1)  # Modo espejo

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - start_time) * 1000)

            hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)
            face_result = face_landmarker.detect_for_video(mp_image, ts_ms)
            state.update_face(face_result)

            gesture, debug_info = state.decide(hand_result)

            now = time.time() * 1000
            if gesture == candidate_gesture:
                candidate_streak += 1
            else:
                candidate_gesture = gesture
                candidate_streak = 1

            if candidate_streak >= STABLE_FRAMES_REQUIRED and gesture != current_gesture:
                current_gesture = gesture
                meme_player.set_gesture(current_gesture)

            if gesture != "default":
                last_non_default_at = now
            elif now - last_non_default_at > DEFAULT_FALLBACK_MS and current_gesture != "default":
                current_gesture = "default"
                meme_player.set_gesture("default")

            # Impresión de logs en consola en modo DEBUG
            if DEBUG and (frame_idx % DEBUG_FRAME_INTERVAL == 0 or gesture != candidate_gesture):
                print_debug_log(
                    frame_idx=frame_idx,
                    state=state,
                    hand_result=hand_result,
                    debug_info=debug_info,
                    gesture=gesture,
                    candidate_gesture=candidate_gesture,
                    candidate_streak=candidate_streak,
                    current_gesture=current_gesture,
                )

            draw_landmarks(frame, hand_result)
            draw_debug_hud(frame, state, current_gesture)

            meme_view = meme_player.get_frame(frame.shape[0])

            cv2.imshow("Camera", frame)
            cv2.imshow("Meme", meme_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        meme_player.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        face_landmarker.close()


if __name__ == "__main__":
    main()
