import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ---- Meme mapping (NewMemes folder) ------------------------------------
const GESTURE_MEMES = {
  default: ["memes/NewMemes/DefaultCat.mp4"],
  ClapClap: ["memes/NewMemes/ClapClap.mp4"],
  Sad: ["memes/NewMemes/Sad.mp4"],
  Muejeje: ["memes/NewMemes/Muejeje.mp4"],
  Hiii: ["memes/NewMemes/Hiii.mp4"],
  Coquette: ["memes/NewMemes/Coquette.mp4"],
  SpeedLaugh: ["memes/NewMemes/SpeedLaugh.mp4"],
  EwwCover: ["memes/NewMemes/EwwCover.mp4"],
};

const STABLE_FRAMES_REQUIRED = 4;
const DEFAULT_FALLBACK_MS = 600;
const FACE_STALE_MS = 1200;

// Angular thresholds (degrees)
const SAD_PITCH_MIN_DEG = 30.0;
const SAD_PITCH_MAX_DEG = 55.0;
const SPEEDLAUGH_PITCH_MIN_DEG = 8.0;
const SPEEDLAUGH_PITCH_MAX_DEG = 28.0;
const EWW_COVER_NOSE_DIST_THR = 0.85;

const video = document.getElementById("video");
const memeVideo = document.getElementById("memeVideo");
const debugHud = document.getElementById("debugHud");

let handLandmarker, faceLandmarker;
let lastVideoTime = -1;
let currentGesture = "default";
let candidateGesture = "default";
let candidateStreak = 0;
let lastNonDefaultAt = performance.now();

let lastFace = null;
let lastFaceSeenThisFrame = false;
let lastYawDebug = 0;
let lastPitchDebug = 0;
let lastBlinkDebug = 0;
let lastPuckerDebug = 0;
let lastBrowDebug = 0;

async function init() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });

  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  });

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  requestAnimationFrame(loop);
}

// ---- 3D Geometry Helpers -----------------------------------------------
function vec(a, b) {
  return { x: b.x - a.x, y: b.y - a.y, z: (b.z || 0) - (a.z || 0) };
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

function angleDeg(v1, v2) {
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const m1 = Math.hypot(v1.x, v1.y, v1.z);
  const m2 = Math.hypot(v2.x, v2.y, v2.z);
  if (m1 < 1e-9 || m2 < 1e-9) return 180;
  return (Math.acos(Math.min(1, Math.max(-1, dot / (m1 * m2)))) * 180) / Math.PI;
}

function fingerExtended(lm, mcp, pip, tip) {
  const angle = angleDeg(vec(lm[mcp], lm[pip]), vec(lm[pip], lm[tip]));
  return angle < 45;
}

function headPoseFromTransformMatrix(matrixData) {
  const r00 = matrixData[0];
  const r10 = matrixData[4];
  const r20 = matrixData[8];
  const r21 = matrixData[9];
  const r22 = matrixData[10];

  const sy = Math.hypot(r00, r10);
  if (sy < 1e-6) return { yaw: 0, pitch: 0 };

  const yaw = (Math.atan2(-r20, sy) * 180) / Math.PI;
  const pitch = (Math.atan2(r21, r22) * 180) / Math.PI;
  return { yaw, pitch };
}

function classifyHand(lm) {
  const handScale = dist(lm[0], lm[9]) || 1e-6;

  const indexUp = fingerExtended(lm, 5, 6, 8);
  const middleUp = fingerExtended(lm, 9, 10, 12);
  const ringUp = fingerExtended(lm, 13, 14, 16);
  const pinkyUp = fingerExtended(lm, 17, 18, 20);

  const thumbPinkySpread = dist(lm[4], lm[17]) / handScale;
  const thumbOut = thumbPinkySpread > 1.05;

  const curledCount = [indexUp, middleUp, ringUp, pinkyUp].filter((v) => !v).length;
  const fingersUpright = lm[8].y < lm[6].y && lm[12].y < lm[10].y && lm[16].y < lm[14].y;

  return {
    indexUp,
    middleUp,
    ringUp,
    pinkyUp,
    thumbOut,
    curledCount,
    handScale,
    fingersUpright,
    wrist: lm[0],
    thumbTip: lm[4],
    indexTip: lm[8],
    middleTip: lm[12],
    ringTip: lm[16],
    pinkyTip: lm[20],
    palmCenter: lm[9],
  };
}

function updateFace(faceResult) {
  const now = performance.now();
  const sawFace = !!(faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0);

  if (sawFace) {
    const f = faceResult.faceLandmarks[0];
    const upperLip = f[13];
    const lowerLip = f[14];
    const mouthCornerR = f[61];
    const mouthCornerL = f[291];
    const rightCheek = f[234];
    const leftCheek = f[454];
    const rightEar = f[127];
    const leftEar = f[356];
    const rightTemple = f[103];
    const leftTemple = f[332];
    const noseTip = f[1];
    const chin = f[152];
    const forehead = f[10];

    const rEyeTop = f[159], rEyeBot = f[145];
    const rEyeIn = f[133], rEyeOut = f[33];
    const lEyeTop = f[386], lEyeBot = f[374];
    const lEyeIn = f[362], lEyeOut = f[263];

    const rBrow = f[70];
    const lBrow = f[300];

    const mouthCenter = {
      x: (upperLip.x + lowerLip.x) / 2,
      y: (upperLip.y + lowerLip.y) / 2,
      z: ((upperLip.z || 0) + (lowerLip.z || 0)) / 2,
    };
    const lowerFaceCenter = {
      x: (noseTip.x + mouthCenter.x) / 2,
      y: (noseTip.y + mouthCenter.y) / 2,
      z: ((noseTip.z || 0) + (mouthCenter.z || 0)) / 2,
    };
    const mouthWidth = dist(mouthCornerR, mouthCornerL);
    const mouthOpen = dist(upperLip, lowerLip);
    const faceWidth = dist(rightCheek, leftCheek) || 1e-6;
    const faceCenter = {
      x: (rightCheek.x + leftCheek.x) / 2,
      y: (rightCheek.y + leftCheek.y) / 2,
      z: ((rightCheek.z || 0) + (leftCheek.z || 0)) / 2,
    };

    let yawDeg = 0, pitchDeg = 0;
    if (faceResult.facialTransformationMatrixes && faceResult.facialTransformationMatrixes.length > 0) {
      const pose = headPoseFromTransformMatrix(faceResult.facialTransformationMatrixes[0].data);
      yawDeg = pose.yaw;
      pitchDeg = pose.pitch;
    }

    const rEyeOpen = dist(rEyeTop, rEyeBot) / (dist(rEyeIn, rEyeOut) || 1e-6);
    const lEyeOpen = dist(lEyeTop, lEyeBot) / (dist(lEyeIn, lEyeOut) || 1e-6);
    const avgEyeOpen = (rEyeOpen + lEyeOpen) / 2;

    const rBrowDist = dist(rBrow, rEyeTop) / faceWidth;
    const lBrowDist = dist(lBrow, lEyeTop) / faceWidth;
    const avgBrowHeight = (rBrowDist + lBrowDist) / 2;

    const geomPitchRatio = (noseTip.y - (rEyeTop.y + lEyeTop.y) / 2) / (
      (chin.y - (rEyeTop.y + lEyeTop.y) / 2) || 1e-6
    );

    let blinkScore = 0, puckerScore = 0, browScore = 0, browFurrowScore = 0;
    if (faceResult.faceBlendshapes && faceResult.faceBlendshapes.length > 0) {
      const categories = {};
      for (const cat of faceResult.faceBlendshapes[0].categories) {
        categories[cat.categoryName] = cat.score;
      }
      blinkScore = Math.max(
        categories["eyeBlinkLeft"] || 0,
        categories["eyeBlinkRight"] || 0,
        categories["eyeSquintLeft"] || 0,
        categories["eyeSquintRight"] || 0
      );
      puckerScore = Math.max(
        categories["mouthPucker"] || 0,
        categories["mouthFunnel"] || 0,
        ((categories["mouthPressLeft"] || 0) + (categories["mouthPressRight"] || 0)) / 2
      );
      browScore = Math.max(
        categories["browInnerUp"] || 0,
        categories["browOuterUpLeft"] || 0,
        categories["browOuterUpRight"] || 0
      );
      browFurrowScore = Math.max(
        categories["browDownLeft"] || 0,
        categories["browDownRight"] || 0,
        categories["browInnerUp"] || 0
      );
    }

    const isEyesClosed = blinkScore > 0.38 || avgEyeOpen < 0.18;
    const isPuckered = puckerScore > 0.30 || (mouthWidth / faceWidth < 0.33 && mouthOpen / faceWidth < 0.15);
    const isBrowsActive = browScore > 0.30 || browFurrowScore > 0.30 || avgBrowHeight > 0.16;

    const isHeadDownSad = (
      (Math.abs(pitchDeg) >= SAD_PITCH_MIN_DEG && Math.abs(pitchDeg) <= SAD_PITCH_MAX_DEG) || geomPitchRatio > 0.65
    );
    const isHeadDownSpeedLaugh = (
      (Math.abs(pitchDeg) >= SPEEDLAUGH_PITCH_MIN_DEG && Math.abs(pitchDeg) <= SPEEDLAUGH_PITCH_MAX_DEG) || (geomPitchRatio >= 0.54 && geomPitchRatio <= 0.64)
    );

    lastFace = {
      mouthCenter,
      lowerFaceCenter,
      noseTip,
      faceWidth,
      faceCenter,
      rightCheek,
      leftCheek,
      rightEar,
      leftEar,
      rightTemple,
      leftTemple,
      chin,
      forehead,
      yawDeg,
      pitchDeg,
      isEyesClosed,
      isPuckered,
      isBrowsActive,
      isHeadDownSad,
      isHeadDownSpeedLaugh,
      t: now,
    };
    lastYawDebug = yawDeg;
    lastPitchDebug = pitchDeg;
    lastBlinkDebug = blinkScore > 0 ? blinkScore : (1.0 - avgEyeOpen * 4);
    lastPuckerDebug = puckerScore > 0 ? puckerScore : (0.4 - mouthWidth / faceWidth);
    lastBrowDebug = Math.max(browScore, browFurrowScore) > 0 ? Math.max(browScore, browFurrowScore) : avgBrowHeight;
  }
  lastFaceSeenThisFrame = sawFace;
}

function decideGesture(handResult) {
  const now = performance.now();
  const faceIsFresh = !!lastFace && now - lastFace.t < FACE_STALE_MS;

  const hands = (handResult.landmarks && handResult.landmarks.length > 0)
    ? handResult.landmarks.map(classifyHand)
    : [];

  // -----------------------------------------------------------------------
  // 1. Two-Hand Gestures (ClapClap, Muejeje)
  // -----------------------------------------------------------------------
  if (hands.length >= 2) {
    const h1 = hands[0];
    const h2 = hands[1];
    const avgScale = (h1.handScale + h2.handScale) / 2;
    const dPalm = dist(h1.palmCenter, h2.palmCenter) / avgScale;
    const dWrist = dist(h1.wrist, h2.wrist) / avgScale;
    const dIndexTip = dist(h1.indexTip, h2.indexTip) / avgScale;
    const dMiddleTip = dist(h1.middleTip, h2.middleTip) / avgScale;

    // 2. ClapClap: Ambas manos abiertas, a la altura del pecho, palmas enfrentadas (distancia < 1.4x escala)
    const bothOpen = h1.curledCount <= 1 && h2.curledCount <= 1;
    if (bothOpen && dPalm < 1.40 && dWrist < 1.65) {
      if (faceIsFresh) {
        const isChestLevel = (h1.palmCenter.y > lastFace.mouthCenter.y + lastFace.faceWidth * 0.2);
        if (isChestLevel || dPalm < 1.15) {
          return "ClapClap";
        }
      } else {
        return "ClapClap";
      }
    }

    // 4. Muejeje: Ambas manos presentes, todos dedos extendidos tocándose en puntas (< 1.4x escala)
    const fingersExtended = h1.curledCount <= 1 && h2.curledCount <= 1;
    if (fingersExtended && dIndexTip < 1.40 && (dMiddleTip < 1.55 || dPalm > 0.65)) {
      return "Muejeje";
    }
  }

  // -----------------------------------------------------------------------
  // 2. One-Hand Face/Head Gestures (EwwCover, Coquette, Hiii)
  // -----------------------------------------------------------------------
  if (faceIsFresh && hands.length >= 1) {
    const lf = lastFace;
    for (const h of hands) {
      const dNose = dist(h.palmCenter, lf.noseTip) / lf.faceWidth;
      const dIndexNose = dist(h.indexTip, lf.noseTip) / lf.faceWidth;
      const dMiddleNose = dist(h.middleTip, lf.noseTip) / lf.faceWidth;
      const dWristNose = dist(h.wrist, lf.noseTip) / lf.faceWidth;
      const dLowerFace = dist(h.palmCenter, lf.lowerFaceCenter) / lf.faceWidth;
      const dMouth = dist(h.palmCenter, lf.mouthCenter) / lf.faceWidth;
      const isCentered = Math.abs(h.palmCenter.x - lf.faceCenter.x) < (lf.faceWidth * 0.50);

      // 8. EwwCover: 1 mano cubre nariz
      const isCoveringNose = (
        Math.min(dNose, dIndexNose, dMiddleNose, dWristNose) < EWW_COVER_NOSE_DIST_THR ||
        (isCentered && dLowerFace < 0.75)
      );
      if (isCoveringNose) {
        return "EwwCover";
      }

      // 6. Coquette: Una mano ubicada lateralmente junto a la cabeza, muñeca/palma próxima a una oreja
      const dREar = dist(h.palmCenter, lf.rightEar) / lf.faceWidth;
      const dLEar = dist(h.palmCenter, lf.leftEar) / lf.faceWidth;
      const dRTemple = dist(h.palmCenter, lf.rightTemple) / lf.faceWidth;
      const dLTemple = dist(h.palmCenter, lf.leftTemple) / lf.faceWidth;
      const dREarWrist = dist(h.wrist, lf.rightEar) / lf.faceWidth;
      const dLEarWrist = dist(h.wrist, lf.leftEar) / lf.faceWidth;
      const minEarDist = Math.min(dREar, dLEar, dRTemple, dLTemple, dREarWrist, dLEarWrist);

      if (minEarDist < 0.95 && h.palmCenter.y < (lf.mouthCenter.y - lf.faceWidth * 0.1)) {
        return "Coquette";
      }

      // 5. Hiii: 1 mano levantada extendida al lado de rostro/mejillas, con la palma orientada hacia la cámara
      const isOpen = h.curledCount <= 1;
      const isBesideFace = Math.abs(h.palmCenter.x - lf.faceCenter.x) > (lf.faceWidth * 0.45);
      const dCheek = Math.min(dist(h.palmCenter, lf.rightCheek), dist(h.palmCenter, lf.leftCheek)) / lf.faceWidth;

      if (isOpen && isBesideFace && (dMouth < 2.5 && dCheek < 1.6) && h.fingersUpright) {
        return "Hiii";
      }
    }
  }

  // -----------------------------------------------------------------------
  // 3. SpeedLaugh (Ojos cerrados, labios fruncidos, cejas fruncidas, cabeza 10°-20° abajo)
  // -----------------------------------------------------------------------
  if (faceIsFresh) {
    const lf = lastFace;
    const speedLaughFeatures = [
      lf.isEyesClosed,
      lf.isPuckered,
      lf.isBrowsActive,
      lf.isHeadDownSpeedLaugh,
    ].filter(Boolean).length;

    if (speedLaughFeatures >= 3 && lf.isEyesClosed && lf.isPuckered) {
      return "SpeedLaugh";
    }
  }

  // -----------------------------------------------------------------------
  // 4. Sad: Cabeza inclinada hacia abajo entre 35° y 45° respecto a vertical
  // -----------------------------------------------------------------------
  if (faceIsFresh && lastFace.isHeadDownSad) {
    return "Sad";
  }

  // -----------------------------------------------------------------------
  // 1. DefaultCat - Estado base
  // -----------------------------------------------------------------------
  return "default";
}

function applyGesture(gesture) {
  if (gesture === currentGesture) return;
  currentGesture = gesture;
  const targetSrc = GESTURE_MEMES[gesture][0];
  if (!memeVideo.src.endsWith(targetSrc)) {
    memeVideo.src = targetSrc;
    memeVideo.currentTime = 0;
    memeVideo.play().catch(() => {});
  }
}

function loop() {
  const now = performance.now();
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const ts = performance.now();

    const handResult = handLandmarker.detectForVideo(video, ts);
    const faceResult = faceLandmarker.detectForVideo(video, ts);
    updateFace(faceResult);

    const gesture = decideGesture(handResult);

    // Debounce
    if (gesture === candidateGesture) {
      candidateStreak++;
    } else {
      candidateGesture = gesture;
      candidateStreak = 1;
    }

    if (candidateStreak >= STABLE_FRAMES_REQUIRED) {
      applyGesture(gesture);
    }

    if (gesture !== "default") lastNonDefaultAt = now;
    if (now - lastNonDefaultAt > DEFAULT_FALLBACK_MS && currentGesture !== "default") {
      applyGesture("default");
    }

    updateDebugHud();
  }
  requestAnimationFrame(loop);
}

function updateDebugHud() {
  if (!debugHud) return;
  debugHud.textContent =
    `Gesto activo: ${currentGesture}\n` +
    `Pitch (Inclinacion): ${lastPitchDebug >= 0 ? "+" : ""}${lastPitchDebug.toFixed(1)} deg (Sad: 35-45 deg | SpeedLaugh: 10-20 deg)\n` +
    `Yaw (Giro cabeza): ${lastYawDebug >= 0 ? "+" : ""}${lastYawDebug.toFixed(1)} deg\n` +
    `SpeedLaugh: Ojos=${lastBlinkDebug.toFixed(2)}, Labios=${lastPuckerDebug.toFixed(2)}, Cejas=${lastBrowDebug.toFixed(2)}`;
}

init().catch((err) => console.error(err));
