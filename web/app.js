import '@mediapipe/tasks-vision';

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const anglesEl = document.getElementById('angles');
const modelSel = document.getElementById('model');
const resSel = document.getElementById('res');

let pose;
let running = false;

async function loadPose(model) {
  const files = {
    lite: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
    full: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
    heavy: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
  };
  const { PoseLandmarker, FilesetResolver } = window;
  const resolver = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
  );
  pose = await PoseLandmarker.createFromOptions(resolver, {
    baseOptions: {
      modelAssetPath: files[model] || files.full,
    },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPoseTrackingConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

async function setupCamera(w, h) {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: w, height: h } });
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

function drawLine(a, b, color, width = 6) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}

function drawCircle(p, r, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
  ctx.fill();
}

function degBetween(a, b, c) {
  const v1x = a.x - b.x, v1y = a.y - b.y;
  const v2x = c.x - b.x, v2y = c.y - b.y;
  const n1 = Math.hypot(v1x, v1y);
  const n2 = Math.hypot(v2x, v2y);
  if (n1 === 0 || n2 === 0) return 0;
  let cos = (v1x * v2x + v1y * v2y) / (n1 * n2);
  cos = Math.max(-1, Math.min(1, cos));
  return (Math.acos(cos) * 180) / Math.PI;
}

function colorFor(angle, safe, caution) {
  const inRange = (x, r) => x >= r[0] && x <= r[1];
  if (inRange(angle, safe)) return '#00ff00';
  const arr = Array.isArray(caution[0]) ? caution : [caution];
  if (arr.some(r => inRange(angle, r))) return '#ffff00';
  return '#ff0000';
}

function toPx(lm) {
  return { x: lm.x * canvas.width, y: lm.y * canvas.height };
}

async function loop() {
  running = true;
  const now = performance.now();
  const results = await pose.detectForVideo(video, now);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (results.landmarks && results.landmarks[0]) {
    const lm = results.landmarks[0];
    const idx = window.PoseLandmarker.PoseLandmark;

    const shoulderL = toPx(lm[idx.LEFT_SHOULDER]);
    const shoulderR = toPx(lm[idx.RIGHT_SHOULDER]);
    const hip = toPx(lm[idx.LEFT_HIP]);
    const knee = toPx(lm[idx.LEFT_KNEE]);
    const ankle = toPx(lm[idx.LEFT_ANKLE]);

    const torsoCenter = { x: (shoulderL.x + shoulderR.x) / 2, y: (shoulderL.y + shoulderR.y) / 2 };
    const torsoRef = { x: torsoCenter.x, y: torsoCenter.y - 50 };

    const kneeAngle = degBetween(hip, knee, ankle);
    const hipAngle = degBetween(torsoCenter, hip, knee);
    const torsoAngle = degBetween(torsoRef, torsoCenter, hip);

    const kneeColor = colorFor(kneeAngle, [90, 180], [[80, 90]]);
    const hipColor = colorFor(hipAngle, [160, 180], [[140, 160]]);
    const torsoColor = colorFor(torsoAngle, [160, 180], [[140, 160]]);

    drawCircle(knee, 8, kneeColor);
    drawCircle(hip, 8, hipColor);
    drawCircle(torsoCenter, 8, torsoColor);
    drawLine(hip, knee, kneeColor);
    drawLine(knee, ankle, kneeColor);
    drawLine(torsoCenter, hip, torsoColor);
    drawLine(torsoRef, torsoCenter, torsoColor, 4);

    ctx.strokeStyle = '#3c3c3c';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(torsoCenter.x, 0);
    ctx.lineTo(torsoCenter.x, canvas.height);
    ctx.stroke();

    const lvl = c => (c === '#00ff00' ? 0 : c === '#ffff00' ? 1 : 2);
    const worst = Math.max(lvl(kneeColor), lvl(hipColor), lvl(torsoColor));
    const status = worst === 0 ? 'GOOD' : worst === 1 ? 'CAUTION' : 'UNSAFE';
    statusEl.textContent = status;
    anglesEl.textContent = `knee ${kneeAngle|0}°, hip ${hipAngle|0}°, torso ${torsoAngle|0}°`;

    // HUD banner
    ctx.fillStyle = 'rgba(30,30,30,0.9)';
    ctx.fillRect(20, 20, 360, 70);
    ctx.fillStyle = worst === 0 ? '#00c800' : worst === 1 ? '#00c8c8' : '#0000c8';
    ctx.font = '28px system-ui';
    ctx.fillText(status, 30, 66);
  }

  if (running) requestAnimationFrame(loop);
}

async function main() {
  const [w, h] = (resSel.value || '640x360').split('x').map(Number);
  await setupCamera(w, h);
  await loadPose(modelSel.value || 'full');
  loop();
}

main();


