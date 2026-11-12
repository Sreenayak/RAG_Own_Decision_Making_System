const form = document.getElementById("chat-form");
const input = document.getElementById("question");
const sendButton = document.getElementById("send-button");
const welcome = document.getElementById("welcome");
const messages = document.getElementById("messages");
const chatScroll = document.getElementById("chat-scroll");
const sidebar = document.getElementById("sidebar");
let waitingForAnswer = false;

function scrollToLatest() {
  requestAnimationFrame(() => {
    chatScroll.scrollTop = chatScroll.scrollHeight;
  });
}

function addMessage(role, text = "") {
  const row = document.createElement("article");
  row.className = `message ${role}`;

  if (role === "user") {
    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = text;
    row.append(content);

    const avatar = document.createElement("span");
    avatar.className = "message-avatar user-avatar";
    avatar.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="8" r="3.5"/><path d="M5 21c.7-4.1 3-6 7-6s6.3 1.9 7 6"/></svg>';
    row.append(avatar);
  } else {
    const avatar = document.createElement("span");
    avatar.className = "message-avatar assistant-avatar";
    avatar.innerHTML = '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M17.9 3.5 7.7 17h6.9l-1.1 11.5L24.3 14h-7.1l.7-10.5Z"/></svg>';
    row.append(avatar);

    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = text;
    row.append(content);
  }

  messages.append(row);
  scrollToLatest();
  return row;
}

function showTyping() {
  const row = addMessage("assistant");
  const indicator = document.createElement("span");
  indicator.className = "typing";
  indicator.innerHTML = "<i></i><i></i><i></i>";
  row.querySelector(".message-content").append(indicator);
  return row;
}

async function askSmashAI(question) {
  if (!question || waitingForAnswer) return;

  welcome.hidden = true;
  addMessage("user", question);
  input.value = "";
  input.style.height = "auto";
  waitingForAnswer = true;
  sendButton.disabled = true;
  const responseRow = showTyping();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Smash AI could not answer right now.");
    }
    responseRow.querySelector(".message-content").textContent = payload.answer;
  } catch (error) {
    responseRow.classList.add("error");
    responseRow.querySelector(".message-content").textContent = error.message;
  } finally {
    waitingForAnswer = false;
    sendButton.disabled = false;
    input.focus();
    scrollToLatest();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  askSmashAI(input.value.trim());
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt;
    askSmashAI(input.value.trim());
  });
});

document.getElementById("new-chat").addEventListener("click", () => {
  messages.replaceChildren();
  welcome.hidden = false;
  input.value = "";
  input.focus();
  sidebar.classList.remove("open");
});

document.getElementById("mobile-menu").addEventListener("click", () => {
  sidebar.classList.toggle("open");
});

document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    input.focus();
  }
  if (event.key === "Escape") {
    sidebar.classList.remove("open");
  }
});
