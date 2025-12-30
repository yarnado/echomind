/* =========================
   ECOMIND FRONTEND LOGIC
   ========================= */

// ---------- NAVIGATION ----------
function showSection(id) {
  document.querySelectorAll(".section").forEach(section => {
    section.classList.remove("active");
  });
  document.getElementById(id).classList.add("active");
}

// ---------- CHATBOT LOGIC ----------
const chatBox = document.getElementById("chatBox");
const chatInput = document.getElementById("chatInput");

// Simulated emotional intelligence responses
function generateBotResponse(userText) {
  const text = userText.toLowerCase();

  if (text.includes("sad") || text.includes("upset")) {
    return "I’m really sorry you’re feeling this way. Want to tell me what happened?";
  }
  if (text.includes("angry") || text.includes("mad")) {
    return "Those feelings are valid. Let’s slow down and breathe together.";
  }
  if (text.includes("anxious") || text.includes("scared")) {
    return "It sounds overwhelming. You’re safe here. I’m listening.";
  }
  return "Thank you for sharing. Your emotions matter, and I’m here with you.";
}

function sendMessage() {
  const message = chatInput.value.trim();
  if (!message) return;

  // User message
  chatBox.innerHTML += `<p><b>You:</b> ${message}</p>`;

  // Bot response
  const response = generateBotResponse(message);
  setTimeout(() => {
    chatBox.innerHTML += `<p><b>Ecomind:</b> ${response}</p>`;
    chatBox.scrollTop = chatBox.scrollHeight;
  }, 600);

  chatInput.value = "";
}

// ---------- EMOTION DETECTION (CAMERA PIPELINE) ----------
const video = document.getElementById("video");
const emotionResult = document.getElementById("emotionResult");

const emotions = ["Happy", "Sad", "Anxious", "Neutral", "Stressed"];

function simulateEmotionDetection() {
  const detectedEmotion = emotions[Math.floor(Math.random() * emotions.length)];
  emotionResult.innerText = `Detected Emotion: ${detectedEmotion}`;
  storeEmotion(detectedEmotion);
}

// Camera access
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
    setInterval(simulateEmotionDetection, 5000);
  })
  .catch(() => {
    emotionResult.innerText = "Camera access denied.";
  });

// ---------- INSIGHTS LOGIC ----------
let emotionHistory = [];

function storeEmotion(emotion) {
  emotionHistory.push({
    emotion: emotion,
    time: new Date().toLocaleTimeString()
  });
  updateInsights();
}

function updateInsights() {
  const insightsSection = document.getElementById("insights");
  let html = "<h2>Emotional Insights</h2>";

  if (emotionHistory.length === 0) {
    html += "<p>No emotional data yet.</p>";
  } else {
    html += "<ul>";
    emotionHistory.slice(-5).forEach(e => {
      html += `<li>${e.time} — ${e.emotion}</li>`;
    });
    html += "</ul>";
  }

  insightsSection.innerHTML = html;
}
