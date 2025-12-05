const STORAGE_KEY = "aiRoadmapProgress_v1";

let roadmap = null;
let currentStageId = null;
let selectedTopicId = null;
let progressMap = {};

function loadProgress() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch (e) {
    return {};
  }
}

function saveProgress(map) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
  } catch (e) {
    // ignore
  }
}

function $(id) {
  return document.getElementById(id);
}

async function loadRoadmap() {
  const res = await fetch("topics.json");
  roadmap = await res.json();
}

function getAllTopics() {
  if (!roadmap) return [];
  const topics = [];
  roadmap.stages.forEach(stage => {
    stage.modules.forEach(mod => {
      mod.topics.forEach(topic => {
        topics.push({
          ...topic,
          module: mod.name,
          stageId: stage.id,
          stageTitle: stage.title
        });
      });
    });
  });
  return topics;
}

function getStageTopics(stageId) {
  const stage = roadmap.stages.find(s => s.id === stageId);
  if (!stage) return [];
  const topics = [];
  stage.modules.forEach(mod => {
    mod.topics.forEach(topic => {
      topics.push({
        ...topic,
        module: mod.name,
        stageId: stage.id,
        stageTitle: stage.title
      });
    });
  });
  return topics;
}

function initStageTabs() {
  const stageSwitcher = $("stageSwitcher");
  stageSwitcher.innerHTML = "";

  roadmap.stages.forEach((stage, index) => {
    const btn = document.createElement("button");
    btn.className = "stage-tab";
    btn.dataset.stageId = stage.id;
    btn.innerHTML = `<span class="dot"></span>${stage.shortTitle || stage.title}`;
    if (index === 0) {
      btn.classList.add("active");
      currentStageId = stage.id;
    }
    btn.addEventListener("click", () => {
      setActiveStage(stage.id);
    });
    stageSwitcher.appendChild(btn);
  });
}

function setActiveStage(stageId) {
  currentStageId = stageId;
  document.querySelectorAll(".stage-tab").forEach(tab => {
    tab.classList.toggle("active", tab.dataset.stageId === stageId);
  });

  const stage = roadmap.stages.find(s => s.id === stageId);
  if (!stage) return;

  $("stageTitle").textContent = stage.title;
  $("stageSummary").textContent = stage.summary;

  const stageTopics = getStageTopics(stageId);
  $("stageCountLabel").textContent = `${stageTopics.length} topics`;

  createNodes(stageTopics);
  computeProgress();
  resetDetailsPanel();
}

function createNodes(stageTopics) {
  const nodesLayer = $("nodesLayer");
  nodesLayer.innerHTML = "";

  const total = stageTopics.length;
  stageTopics.forEach((topic, index) => {
    const node = document.createElement("div");
    node.className = "node";
    node.dataset.topicId = topic.id;

    const x = 8 + (index / Math.max(total - 1, 1)) * 84;
    const y = index % 2 === 0 ? 38 : 64;
    node.style.left = x + "%";
    node.style.top = y + "%";

    if (progressMap[topic.id]) {
      node.classList.add("completed");
    }

    const label = document.createElement("div");
    label.className = "node-label";
    label.textContent = topic.title;
    node.appendChild(label);

    node.addEventListener("click", () => {
      setActiveTopic(topic.id);
    });

    nodesLayer.appendChild(node);
  });

  const firstIncomplete = stageTopics.find(t => !progressMap[t.id]) || stageTopics[0];
  if (firstIncomplete) {
    setActiveTopic(firstIncomplete.id, false);
  }
}

function setActiveTopic(topicId, scrollIntoView = true) {
  selectedTopicId = topicId;

  document.querySelectorAll(".node").forEach(node => {
    node.classList.toggle("active", node.dataset.topicId === topicId);
    if (scrollIntoView && node.dataset.topicId === topicId) {
      node.scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });
    }
  });

  const allTopics = getAllTopics();
  const topic = allTopics.find(t => t.id === topicId);
  if (!topic) return;

  $("detailsTitle").textContent = topic.title;
  $("detailsStagePill").textContent = topic.stageTitle;
  $("detailsModulePill").textContent = topic.module;
  $("detailsIdPill").textContent = topic.id;

  $("detailsBody").innerHTML = `
    <p>${topic.description}</p>
    <p>Later, replace this placeholder with a full chapter loaded from a JSON file for this topic: theory, examples, exercises, and quiz. You can generate that chapter using Google AI Studio and save it as <code>chapters/${topic.id}.json</code>.</p>
  `;

  $("toggleDoneButton").disabled = false;
  updateDoneButtonUI(!!progressMap[topic.id]);
}

function updateDoneButtonUI(isCompleted) {
  const btn = $("toggleDoneButton");
  if (isCompleted) {
    btn.textContent = "Mark as not done";
    btn.classList.add("completed");
  } else {
    btn.textContent = "Mark as done";
    btn.classList.remove("completed");
  }
}

function resetDetailsPanel() {
  $("detailsTitle").textContent = "Select a topic";
  $("detailsStagePill").textContent = "Stage";
  $("detailsModulePill").textContent = "Module";
  $("detailsIdPill").textContent = "Topic ID";
  $("detailsBody").innerHTML = `
    <p>Tap a node on the roadmap to see details here.</p>
    <p>This panel is designed to show full chapter content later. For now, it's a placeholder while you focus on roadmap structure and UI.</p>
  `;
  $("toggleDoneButton").disabled = true;
  updateDoneButtonUI(false);
}

function computeProgress() {
  const allTopics = getAllTopics();
  const total = allTopics.length;
  const completed = allTopics.filter(t => progressMap[t.id]).length;

  $("overallProgressText").textContent = `${completed} / ${total} topics completed`;

  if (currentStageId) {
    const stageTopics = getStageTopics(currentStageId);
    const stageTotal = stageTopics.length;
    const stageCompleted = stageTopics.filter(t => progressMap[t.id]).length;
    $("miniProgressText").textContent = `${stageCompleted} / ${stageTotal} topics in this stage`;
  } else {
    $("miniProgressText").textContent = "";
  }
}

function attachDoneButtonHandler() {
  $("toggleDoneButton").addEventListener("click", () => {
    if (!selectedTopicId) return;

    const current = !!progressMap[selectedTopicId];
    const newState = !current;
    progressMap[selectedTopicId] = newState;
    saveProgress(progressMap);

    document.querySelectorAll(".node").forEach(node => {
      if (node.dataset.topicId === selectedTopicId) {
        node.classList.toggle("completed", newState);
      }
    });

    updateDoneButtonUI(newState);
    computeProgress();
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  progressMap = loadProgress();

  await loadRoadmap();
  initStageTabs();
  setActiveStage(roadmap.stages[0].id);
  attachDoneButtonHandler();
  computeProgress();
});