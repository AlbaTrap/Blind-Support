<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>YOLO Web App</title>
  <style>
    body { font-family: sans-serif; margin: 0; padding: 0; display: flex; flex-direction: column; align-items: center; }
    #container { position: relative; width: 640px; }
    video, canvas { width: 640px; height: 480px; }
    #overlay { position: absolute; left: 0; top: 0; }
    #status { margin: 10px; }
    #controls { margin: 10px; display: flex; gap: 10px; }
  </style>
</head>
<body>
  <h3>YOLO Web App</h3>
  <div id="controls">
    <input id="wsUrl" type="text" size="40" placeholder="ws://localhost:8000/ws" />
    <button id="connectBtn">Connect</button>
    <button id="disconnectBtn">Disconnect</button>
    <label>Interval (ms): <input id="interval" type="number" value="200" /></label>
    <label>TTS every (s): <input id="ttsInterval" type="number" value="3" /></label>
  </div>

  <div id="container">
    <video id="video" autoplay playsinline muted></video>
    <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
    <canvas id="overlay" width="640" height="480"></canvas>
  </div>

  <div id="status">Disconnected</div>

  <script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const overlay = document.getElementById('overlay');
    const ctx = canvas.getContext('2d');
    const octx = overlay.getContext('2d');
    const wsUrlInput = document.getElementById('wsUrl');
    const statusEl = document.getElementById('status');
    const connectBtn = document.getElementById('connectBtn');
    const disconnectBtn = document.getElementById('disconnectBtn');
    const intervalInput = document.getElementById('interval');
    const ttsIntervalInput = document.getElementById('ttsInterval');

    let ws = null;
    let timer = null;
    let lastTtsTime = 0;

    // Kamera
    async function startCamera() {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
      video.srcObject = stream;
      await video.play();
    }

    function drawDetections(detections) {
      octx.clearRect(0, 0, overlay.width, overlay.height);
      detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;
        octx.strokeStyle = 'magenta';
        octx.lineWidth = 2;
        octx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        octx.fillStyle = 'rgba(0,0,0,0.6)';
        octx.fillRect(x1, y1 - 20, 180, 20);
        octx.fillStyle = 'white';
        octx.font = '14px sans-serif';
        const label = `${det.class} (${det.position}) ${det.confidence}`;
        octx.fillText(label, x1 + 4, y1 - 6);
      });
    }

    function speakPhrase(text) {
      const utter = new SpeechSynthesisUtterance(text);
      utter.rate = 1.0; // możesz dopasować
      speechSynthesis.speak(utter);
    }

    function speakDetections(detections) {
      const now = Date.now();
      const ttsIntervalMs = Number(ttsIntervalInput.value) * 1000;
      if (now - lastTtsTime < ttsIntervalMs) return;
      lastTtsTime = now;

      // zbuduj krótką, skondensowaną frazę z kilku detekcji
      const phrases = detections.slice(0, 3).map(d => `${d.class} ${d.position}`);
      if (phrases.length > 0) {
        speakPhrase(phrases.join(', '));
      }
    }

    function sendFrame() {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      // rysujemy klatkę do canvas, eksportujemy jako JPEG base64
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.6); // kompresja dla mniejszego transferu
      ws.send(JSON.stringify({ image: dataUrl }));
    }

    connectBtn.onclick = () => {
      const url = wsUrlInput.value.trim();
      if (!url) return alert('Podaj ws URL serwera (np. ws://localhost:8000/ws)');
      ws = new WebSocket(url);
      ws.onopen = () => {
        statusEl.textContent = 'Connected';
        const iv = Number(intervalInput.value) || 200;
        timer = setInterval(sendFrame, iv);
      };
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data.error) {
            console.warn('Server error:', data.error);
            return;
          }
          drawDetections(data.detections || []);
          speakDetections(data.detections || []);
        } catch (e) {
          console.error('parse error', e);
        }
      };
      ws.onclose = () => {
        statusEl.textContent = 'Disconnected';
        if (timer) { clearInterval(timer); timer = null; }
      };
      ws.onerror = (e) => {
        console.error('ws error', e);
      };
    };

    disconnectBtn.onclick = () => {
      if (ws) ws.close();
    };

    startCamera().catch(err => {
      alert('Nie udało się uruchomić kamery: ' + err.message);
    });
  </script>
</body>
</html>
