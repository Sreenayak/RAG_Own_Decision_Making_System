const storageKey = "smash-ai-conversations-v1";
const themeKey = "smash-ai-theme";
const form = document.getElementById("chat-form");
const input = document.getElementById("question");
const sendButton = document.getElementById("send-button");
const welcome = document.getElementById("welcome");
const messages = document.getElementById("messages");
const chatScroll = document.getElementById("chat-scroll");
const sidebar = document.getElementById("sidebar");
const chatList = document.getElementById("chat-list");
const historyEmpty = document.getElementById("history-empty");
const searchInput = document.getElementById("chat-search");
const chatTitle = document.getElementById("chat-title");
const exportButton = document.getElementById("export-chat");
const themeButton = document.getElementById("theme-toggle");
const apiStatus = document.getElementById("api-status");
const setupDialog = document.getElementById("setup-dialog");
const toast = document.getElementById("toast");
let conversations = loadConversations();
let activeConversationId = conversations[0]?.id || null;
let waitingForAnswer = false;
let toastTimer;

function loadConversations() {
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || "[]");
    if (!Array.isArray(saved)) return [];
    return saved
      .filter((item) =>
        item &&
        typeof item.id === "string" &&
        typeof item.title === "string" &&
        Array.isArray(item.messages)
      )
      .map((item) => ({
        ...item,
        messages: item.messages.filter((message) =>
          message &&
          (message.role === "user" || message.role === "assistant") &&
          typeof message.text === "string"
        ),
      }));
  } catch {
    return [];
  }
}

function saveConversations() {
  try {
    localStorage.setItem(storageKey, JSON.stringify(conversations));
  } catch {
    showToast("Browser storage is full. Export or delete old chats.");
  }
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("visible"), 2300);
}

async function updateApiStatus() {
  const label = apiStatus.querySelector(".status-label");
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error("Status request failed.");
    const status = await response.json();
    apiStatus.classList.toggle("needs-key", !status.api_key_configured);
    label.textContent = status.api_key_configured ? "Ready" : "Key needed";
    apiStatus.title = status.api_key_configured
      ? "Groq is configured"
      : "Click for API key setup instructions";
  } catch {
    label.textContent = "Status unavailable";
    apiStatus.title = "Could not check API setup";
  }
}

function activeConversation() {
  return conversations.find((item) => item.id === activeConversationId) || null;
}

function makeConversation(question) {
  const conversation = {
    id: crypto.randomUUID(),
    title: question.replace(/\s+/g, " ").trim().slice(0, 48),
    updatedAt: Date.now(),
    messages: [],
  };
  conversations.unshift(conversation);
  activeConversationId = conversation.id;
  return conversation;
}

function updateConversationList() {
  const query = searchInput.value.trim().toLocaleLowerCase();
  const sorted = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);
  const filtered = sorted.filter((conversation) =>
    !query ||
    conversation.title.toLocaleLowerCase().includes(query) ||
    conversation.messages.some((message) =>
      (message.text || "").toLocaleLowerCase().includes(query)
    )
  );

  chatList.replaceChildren();
  for (const conversation of filtered) {
    const row = document.createElement("div");
    row.className = "history-row";
    if (conversation.id === activeConversationId) row.classList.add("active");

    const selectButton = document.createElement("button");
    selectButton.className = "history-entry";
    selectButton.type = "button";
    selectButton.title = conversation.title;
    selectButton.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 5.5A2.5 2.5 0 0 1 7.5 3H20v15H7.5A2.5 2.5 0 0 0 5 20.5v-15Z"/><path d="M5 18.5A2.5 2.5 0 0 1 7.5 16H20"/></svg>';
    const title = document.createElement("span");
    title.className = "history-title";
    title.textContent = conversation.title;
    selectButton.append(title);
    selectButton.addEventListener("click", () => openConversation(conversation.id));

    const deleteButton = document.createElement("button");
    deleteButton.className = "history-delete";
    deleteButton.type = "button";
    deleteButton.title = "Delete conversation";
    deleteButton.setAttribute("aria-label", `Delete ${conversation.title}`);
    deleteButton.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M10 11v6m4-6v6M6 7l1 14h10l1-14M9 7V4h6v3"/></svg>';
    deleteButton.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteConversation(conversation.id);
    });

    row.append(selectButton, deleteButton);
    chatList.append(row);
  }

  historyEmpty.hidden = filtered.length > 0;
  historyEmpty.textContent = query
    ? "No conversations match your search."
    : "Your conversations will appear here.";
}

function avatar(role) {
  const element = document.createElement("span");
  element.className = `message-avatar ${role === "user" ? "user-avatar" : "assistant-avatar"}`;
  if (role === "user") {
    element.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="8" r="3.5"/><path d="M5 21c.7-4.1 3-6 7-6s6.3 1.9 7 6"/></svg>';
  } else {
    element.innerHTML = '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M16 2.8 19.2 12.8 29.2 16l-10 3.2L16 29.2l-3.2-10L2.8 16l10-3.2L16 2.8Z"/><circle cx="16" cy="16" r="2.8"/></svg>';
  }
  return element;
}

function addAnswerTools(row, message) {
  const tools = document.createElement("div");
  tools.className = "message-tools";

  const copy = document.createElement("button");
  copy.className = "message-tool";
  copy.type = "button";
  copy.title = "Copy answer";
  copy.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="13" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h2"/></svg><span>Copy</span>';
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(message.text);
      showToast("Answer copied.");
    } catch {
      showToast("Could not access the clipboard.");
    }
  });

  const feedbackButtons = [
    { value: "up", label: "Helpful", icon: '<path d="M7 10v11H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3Zm0 0 5-8a3 3 0 0 1 2 3v3h5a3 3 0 0 1 3 3l-2 7a3 3 0 0 1-3 3H7"/>' },
    { value: "down", label: "Not helpful", icon: '<path d="M7 14V3H4a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h3Zm0 0 5 8a3 3 0 0 0 2-3v-3h5a3 3 0 0 0 3-3l-2-7a3 3 0 0 0-3-3H7"/>' },
  ];

  tools.append(copy);
  for (const item of feedbackButtons) {
    const button = document.createElement("button");
    button.className = "message-tool";
    if (message.feedback === item.value) button.classList.add("selected");
    button.type = "button";
    button.title = item.label;
    button.setAttribute("aria-label", item.label);
    button.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${item.icon}</svg>`;
    button.addEventListener("click", () => {
      message.feedback = message.feedback === item.value ? null : item.value;
      saveConversations();
      renderConversation();
      showToast(message.feedback ? "Thanks for your feedback." : "Feedback removed.");
    });
    tools.append(button);
  }
  row.querySelector(".message-content").append(tools);
}

function renderMessage(message, typing = false) {
  const row = document.createElement("article");
  row.className = `message ${message.role}${message.isError ? " error" : ""}`;
  row.append(avatar(message.role));

  const content = document.createElement("div");
  content.className = "message-content";
  content.textContent = message.text;
  row.append(content);

  if (typing) {
    const indicator = document.createElement("span");
    indicator.className = "typing";
    indicator.innerHTML = "<i></i><i></i><i></i>";
    content.append(indicator);
  } else if (message.role === "assistant" && !message.isError) {
    addAnswerTools(row, message);
  }

  messages.append(row);
  return row;
}

function renderConversation() {
  const conversation = activeConversation();
  messages.replaceChildren();
  if (!conversation) {
    chatTitle.textContent = "New chat";
    welcome.hidden = false;
    exportButton.disabled = true;
  } else {
    chatTitle.textContent = conversation.title;
    welcome.hidden = conversation.messages.length > 0;
    exportButton.disabled = conversation.messages.length === 0;
    for (const message of conversation.messages) renderMessage(message);
  }
  updateConversationList();
  if (conversation?.messages.length) {
    scrollToLatest();
  } else {
    chatScroll.scrollTop = 0;
  }
}

function scrollToLatest() {
  requestAnimationFrame(() => {
    chatScroll.scrollTop = chatScroll.scrollHeight;
  });
}

function openConversation(id) {
  activeConversationId = id;
  renderConversation();
  sidebar.classList.remove("open");
}

function startNewChat() {
  activeConversationId = null;
  input.value = "";
  input.style.height = "auto";
  renderConversation();
  input.focus();
  sidebar.classList.remove("open");
}

function deleteConversation(id) {
  conversations = conversations.filter((conversation) => conversation.id !== id);
  if (activeConversationId === id) activeConversationId = null;
  saveConversations();
  renderConversation();
  showToast("Conversation deleted.");
}

async function askSmashAI(question) {
  if (!question || waitingForAnswer) return;

  let conversation = activeConversation();
  if (!conversation) conversation = makeConversation(question);
  const history = conversation.messages
    .filter((message) => (message.role === "user" || message.role === "assistant") && !message.isError)
    .slice(-8)
    .map(({ role, text }) => ({ role, content: text.slice(-3500) }));

  conversation.messages.push({ role: "user", text: question, createdAt: Date.now() });
  conversation.updatedAt = Date.now();
  saveConversations();
  welcome.hidden = true;
  input.value = "";
  input.style.height = "auto";
  waitingForAnswer = true;
  sendButton.disabled = true;
  renderConversation();
  const responseRow = renderMessage({ role: "assistant", text: "" }, true);

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Smash AI could not answer right now.");

    const answer = { role: "assistant", text: payload.answer, createdAt: Date.now() };
    conversation.messages.push(answer);
    conversation.updatedAt = Date.now();
    saveConversations();
    responseRow.remove();
    renderMessage(answer);
    renderConversation();
  } catch (error) {
    const answer = {
      role: "assistant",
      text: error.message,
      isError: true,
      createdAt: Date.now(),
    };
    conversation.messages.push(answer);
    conversation.updatedAt = Date.now();
    saveConversations();
    responseRow.remove();
    renderMessage(answer);
    renderConversation();
    if (error.message.includes("Groq API key is not configured")) {
      apiStatus.classList.add("needs-key");
      apiStatus.querySelector(".status-label").textContent = "Key needed";
    }
  } finally {
    waitingForAnswer = false;
    sendButton.disabled = false;
    input.focus();
    scrollToLatest();
  }
}

function exportConversation() {
  const conversation = activeConversation();
  if (!conversation || conversation.messages.length === 0) return;
  const markdown = [
    `# ${conversation.title}`,
    "",
    ...conversation.messages.flatMap((message) => [
      `## ${message.role === "user" ? "You" : "Smash AI"}`,
      "",
      message.text,
      "",
    ]),
  ].join("\n");
  const file = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(file);
  link.download = `${conversation.title.replace(/[<>:"/\\|?*]+/g, "-").slice(0, 48) || "smash-ai-chat"}.md`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  showToast("Conversation exported.");
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  document.body.dataset.theme = theme;
  themeButton.setAttribute(
    "aria-label",
    theme === "dark" ? "Switch to light theme" : "Switch to dark theme"
  );
  try {
    localStorage.setItem(themeKey, theme);
  } catch {
    showToast("Theme will reset when this tab closes.");
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

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => askSmashAI(button.dataset.prompt));
});

document.getElementById("new-chat").addEventListener("click", startNewChat);
searchInput.addEventListener("input", updateConversationList);
exportButton.addEventListener("click", exportConversation);
themeButton.addEventListener("click", () => {
  const nextTheme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  setTheme(nextTheme);
});
apiStatus.addEventListener("click", () => setupDialog.showModal());
document.getElementById("close-setup").addEventListener("click", () => setupDialog.close());
document.getElementById("setup-done").addEventListener("click", () => setupDialog.close());
setupDialog.addEventListener("click", (event) => {
  if (event.target === setupDialog) setupDialog.close();
});
document.getElementById("mobile-menu").addEventListener("click", () => {
  sidebar.classList.toggle("open");
});

document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    searchInput.focus();
  }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "n") {
    event.preventDefault();
    startNewChat();
  }
  if (event.key === "Escape") sidebar.classList.remove("open");
});

try {
  setTheme(localStorage.getItem(themeKey) || "light");
} catch {
  document.documentElement.dataset.theme = "light";
}
renderConversation();
updateApiStatus();
