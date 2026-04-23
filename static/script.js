async function send() {
  const inputEl = document.getElementById("input");
  const roleEl = document.getElementById("role");
  const chatEl = document.getElementById("chat");

  const text = inputEl.value.trim();
  const role = roleEl.value;

  if (!text) return;

  appendMessage(chatEl, text, "user");
  inputEl.value = "";

  // 🔥 Loading message
  const loadingMsg = appendMessage(chatEl, "Thinking...", "bot");

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        question: text,
        role: role
      })
    });

    const data = await res.json();

    // Replace loading text
    loadingMsg.textContent = data.answer || "No response";

  } catch (err) {
    loadingMsg.textContent = "⚠️ Server error. Try again.";
  }
}

// Create message
function appendMessage(container, text, type) {
  const div = document.createElement("div");
  div.className = `message ${type}`;
  div.textContent = text;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

// Enter key support
document.addEventListener("DOMContentLoaded", () => {
  const inputEl = document.getElementById("input");

  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      send();
    }
  });
});