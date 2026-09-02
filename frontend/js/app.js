/**
 * GraphRAG Frontend Application
 * Interacts with FastAPI backend, manages chat, uploads documents, and renders vis-network graph.
 */

// ── State Management ───────────────────────────────────────────────────
const state = {
  activeMode: 'graph', // 'graph' or 'vector'
  documents: [],
  network: null,
  networkData: { nodes: new vis.DataSet([]), edges: new vis.DataSet([]) },
  physicsEnabled: true,
  health: { neo4j: false, qdrant: false, sqlite: false, llm: '' },
  currentGraphFacts: [],
};

// ── Entity Color Mapping ───────────────────────────────────────────────
const ENTITY_COLORS = {
  Person: '#f43f5e',
  Organization: '#8a74f9',
  Concept: '#00f2fe',
  Event: '#f59e0b',
  Location: '#10b981',
  Technology: '#ff7043',
  Date: '#a855f7',
  Document: '#94a3b8',
  Metric: '#22d3ee',
  default: '#38bdf8',
};

function getEntityColor(type) {
  return ENTITY_COLORS[type] || ENTITY_COLORS.default;
}

// ── DOM Elements ───────────────────────────────────────────────────────
const elements = {
  btnModeGraph: document.getElementById('btn-mode-graph'),
  btnModeVector: document.getElementById('btn-mode-vector'),
  activeModeBadge: document.getElementById('active-mode-badge'),
  dotNeo4j: document.getElementById('dot-neo4j'),
  dotQdrant: document.getElementById('dot-qdrant'),
  dotSqlite: document.getElementById('dot-sqlite'),
  dotLlm: document.getElementById('dot-llm'),
  lblLlm: document.getElementById('lbl-llm'),
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('file-input'),
  docList: document.getElementById('doc-list'),
  docCount: document.getElementById('doc-count'),
  chatThread: document.getElementById('chat-thread'),
  chatForm: document.getElementById('chat-form'),
  chatInput: document.getElementById('chat-input'),
  sendBtn: document.getElementById('send-btn'),
  graphNetwork: document.getElementById('graph-network'),
  graphEmpty: document.getElementById('graph-empty'),
  btnPhysics: document.getElementById('btn-physics'),
  btnFit: document.getElementById('btn-fit'),
  btnExpandLayout: document.getElementById('btn-expand-layout'),
  mainWorkspace: document.querySelector('.main-workspace'),
  docSubgraphSelect: document.getElementById('doc-subgraph-select'),
  entityInspector: document.getElementById('entity-inspector'),
  inspectorName: document.getElementById('inspector-name'),
  inspectorType: document.getElementById('inspector-type'),
  inspectorDesc: document.getElementById('inspector-desc'),
  toastContainer: document.getElementById('toast-container'),
};

// ── Toast Notifications ────────────────────────────────────────────────
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span>${type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️'}</span>
    <span>${message}</span>
  `;
  elements.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(20px)';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// ── Initialization ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initGraphNetwork();
  setupEventListeners();
  checkHealth();
  loadDocuments();

  // Poll system health and active document processing every 4 seconds
  setInterval(() => {
    checkHealth();
    pollProcessingDocs();
  }, 4000);
});

// ── Event Listeners ────────────────────────────────────────────────────
function setupEventListeners() {
  // Mode Switcher
  elements.btnModeGraph.addEventListener('click', () => setRetrievalMode('graph'));
  elements.btnModeVector.addEventListener('click', () => setRetrievalMode('vector'));

  // Drag & Drop Upload
  elements.dropzone.addEventListener('click', () => elements.fileInput.click());
  elements.fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) uploadFile(e.target.files[0]);
  });

  elements.dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.dropzone.classList.add('dragover');
  });

  elements.dropzone.addEventListener('dragleave', () => {
    elements.dropzone.classList.remove('dragover');
  });

  elements.dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });

  // Chat Form Submission
  elements.chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const query = elements.chatInput.value.trim();
    if (query) submitQuery(query);
  });

  // Graph Controls
  elements.btnPhysics.addEventListener('click', togglePhysics);
  elements.btnFit.addEventListener('click', () => {
    if (state.network) state.network.fit({ animation: { duration: 600 } });
  });
  if (elements.btnExpandLayout) {
    elements.btnExpandLayout.addEventListener('click', toggleExpandedLayout);
  }

  // Subgraph Selector
  elements.docSubgraphSelect.addEventListener('change', (e) => {
    const val = e.target.value;
    if (val === 'all') {
      renderGraphFacts(state.currentGraphFacts);
    } else {
      loadDocumentSubgraph(val);
    }
  });

  // Document Library Actions (Event Delegation)
  elements.docList.addEventListener('click', async (e) => {
    const btn = e.target.closest('.doc-btn-icon');
    if (!btn) return;
    const docId = btn.getAttribute('data-doc-id');
    if (!docId) return;
    e.preventDefault();
    e.stopPropagation();
    await deleteDocument(docId);
  });

  // Close entity inspector on outside click
  document.addEventListener('click', (e) => {
    if (!elements.entityInspector.contains(e.target) && !e.target.closest('#graph-network')) {
      elements.entityInspector.classList.remove('active');
    }
  });
}

// ── Retrieval Mode ─────────────────────────────────────────────────────
function setRetrievalMode(mode) {
  state.activeMode = mode;
  if (mode === 'graph') {
    elements.btnModeGraph.classList.add('active');
    elements.btnModeVector.classList.remove('active');
    elements.activeModeBadge.textContent = 'MODE: GRAPH-ENHANCED';
    elements.activeModeBadge.style.color = 'var(--accent-cyan)';
  } else {
    elements.btnModeVector.classList.add('active');
    elements.btnModeGraph.classList.remove('active');
    elements.activeModeBadge.textContent = 'MODE: VECTOR-ONLY';
    elements.activeModeBadge.style.color = 'var(--accent-purple)';
  }
}

// ── System Health ──────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error('Health check failed');
    const data = await res.json();

    // Neo4j AuraDB status
    updateDot(elements.dotNeo4j, data.neo4j === 'connected');
    // Local Qdrant status
    updateDot(elements.dotQdrant, data.qdrant === 'connected');
    // SQLite status
    updateDot(elements.dotSqlite, data.sqlite === 'connected');
    // LLM status
    const hasLlm = !!data.llm_provider;
    updateDot(elements.dotLlm, hasLlm);
    elements.lblLlm.textContent = data.llm_provider ? data.llm_provider.toUpperCase() : 'LLM';
  } catch (err) {
    updateDot(elements.dotNeo4j, false);
    updateDot(elements.dotQdrant, false);
    updateDot(elements.dotSqlite, false);
    updateDot(elements.dotLlm, false);
  }
}

function updateDot(dotEl, isConnected) {
  dotEl.className = 'status-dot ' + (isConnected ? 'connected' : 'disconnected');
}

// ── Document Management ────────────────────────────────────────────────
async function loadDocuments() {
  try {
    const res = await fetch('/documents');
    if (!res.ok) throw new Error('Failed to load documents');
    const docs = await res.json();
    state.documents = docs;
    renderDocumentList(docs);
    updateSubgraphDropdown(docs);
  } catch (err) {
    console.error(err);
  }
}

function renderDocumentList(docs) {
  elements.docCount.textContent = docs.length;
  elements.docList.innerHTML = '';

  if (docs.length === 0) {
    elements.docList.innerHTML = `
      <div style="text-align: center; color: var(--text-muted); font-size: 0.78rem; padding: 12px 0;">
        No documents uploaded yet
      </div>`;
    return;
  }

  docs.forEach((doc) => {
    const item = document.createElement('div');
    item.className = 'doc-item';
    
    let progressStr = '';
    if (doc.status === 'processing' || doc.status === 'extracting') {
      if (doc.total_chunks > 0) {
        progressStr = ` (${doc.processed_chunks}/${doc.total_chunks} chunks)`;
      }
    }

    const statusEmojis = {
      ready: '✅ Ready',
      processing: '⏳ Parsing',
      extracting: '🔍 Extracting KG',
      failed: '❌ Failed',
    };

    item.innerHTML = `
      <div class="doc-info">
        <div class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
        <div class="doc-status ${doc.status}">
          ${statusEmojis[doc.status] || doc.status}${progressStr}
        </div>
      </div>
      <div class="doc-actions">
        <button class="doc-btn-icon" title="Delete Document" data-doc-id="${doc.id}">
          🗑️
        </button>
      </div>
    `;
    elements.docList.appendChild(item);
  });
}

function updateSubgraphDropdown(docs) {
  const readyDocs = docs.filter((d) => d.status === 'ready');
  const currentVal = elements.docSubgraphSelect.value;
  elements.docSubgraphSelect.innerHTML = '<option value="all">Active Answer Graph</option>';

  readyDocs.forEach((d) => {
    const opt = document.createElement('option');
    opt.value = d.id;
    opt.textContent = `📄 ${d.filename}`;
    elements.docSubgraphSelect.appendChild(opt);
  });

  if (currentVal && elements.docSubgraphSelect.querySelector(`option[value="${currentVal}"]`)) {
    elements.docSubgraphSelect.value = currentVal;
  }
}

async function pollProcessingDocs() {
  const hasProcessing = state.documents.some(
    (d) => d.status === 'processing' || d.status === 'extracting'
  );
  if (hasProcessing) {
    await loadDocuments();
  }
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  showToast(`Uploading ${file.name}...`, 'info');

  try {
    const res = await fetch('/upload', {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Upload failed');
    }
    const data = await res.json();
    showToast(`Uploaded ${data.filename}! Processing started...`, 'success');
    elements.fileInput.value = '';
    await loadDocuments();
  } catch (err) {
    showToast(`Upload error: ${err.message}`, 'error');
  }
}

async function deleteDocument(docId) {
  try {
    showToast('Deleting document and graph nodes...', 'info');
    const res = await fetch(`/documents/${docId}`, { method: 'DELETE' });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Deletion failed');
    }
    showToast('Document deleted successfully', 'success');
    await loadDocuments();
    if (elements.docSubgraphSelect.value === docId) {
      elements.docSubgraphSelect.value = 'all';
      renderGraphFacts(state.currentGraphFacts);
    }
  } catch (err) {
    showToast(`Delete error: ${err.message}`, 'error');
  }
}

// Expose to window for backwards compatibility
window.deleteDocument = deleteDocument;

// ── Chat & Query Processing ────────────────────────────────────────────
async function submitQuery(question) {
  // Append user message to thread
  appendUserMessage(question);
  elements.chatInput.value = '';
  elements.sendBtn.disabled = true;

  // Placeholder assistant message
  const assistantBubble = appendAssistantLoading();

  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: question,
        mode: state.activeMode,
        top_k: 5,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Query failed');
    }

    const data = await res.json();
    renderAssistantResponse(assistantBubble, data);

    // Update Knowledge Graph on the right canvas
    if (data.graph_facts && data.graph_facts.length > 0) {
      state.currentGraphFacts = data.graph_facts;
      if (elements.docSubgraphSelect.value === 'all') {
        renderGraphFacts(data.graph_facts);
      }
    }
  } catch (err) {
    assistantBubble.innerHTML = `<p style="color: var(--accent-pink);">⚠️ Error: ${err.message}</p>`;
  } finally {
    elements.sendBtn.disabled = false;
    elements.chatThread.scrollTop = elements.chatThread.scrollHeight;
  }
}

function appendUserMessage(text) {
  const msgDiv = document.createElement('div');
  msgDiv.className = 'chat-message user';
  msgDiv.innerHTML = `
    <div class="chat-avatar">U</div>
    <div class="chat-bubble">
      <p>${escapeHtml(text)}</p>
    </div>
  `;
  elements.chatThread.appendChild(msgDiv);
  elements.chatThread.scrollTop = elements.chatThread.scrollHeight;
}

function appendAssistantLoading() {
  const msgDiv = document.createElement('div');
  msgDiv.className = 'chat-message assistant';
  msgDiv.innerHTML = `
    <div class="chat-avatar">AI</div>
    <div class="chat-bubble">
      <p style="color: var(--text-secondary); display: flex; align-items: center; gap: 8px;">
        <span style="display: inline-block; animation: spin 1s linear infinite;">⏳</span>
        Reasoning over knowledge graph and passages...
      </p>
    </div>
  `;
  elements.chatThread.appendChild(msgDiv);
  elements.chatThread.scrollTop = elements.chatThread.scrollHeight;
  return msgDiv.querySelector('.chat-bubble');
}

function renderAssistantResponse(bubbleEl, data) {
  // Render Markdown answer
  const rawHtml = marked.parse(data.answer || 'No response generated.');
  let contentHtml = rawHtml;

  // Render Source Citations Accordion
  if (data.sources && data.sources.length > 0) {
    const citationsHtml = data.sources
      .map((s, idx) => `
        <div class="citation-card">
          <div class="citation-meta">
            <span>📄 ${s.filename || 'Source ' + (idx + 1)} (Page ${s.page_number ?? '?'})</span>
            ${s.relevance_score ? `<span>Score: ${(s.relevance_score * 100).toFixed(1)}%</span>` : ''}
          </div>
          <div class="citation-text">${escapeHtml(s.text || '')}</div>
        </div>
      `)
      .join('');

    contentHtml += `
      <div class="accordion-box" onclick="this.classList.toggle('open')">
        <div class="accordion-header">
          <span>📖 Source Passages & Citations (${data.sources.length})</span>
          <span>▼</span>
        </div>
        <div class="accordion-content">
          ${citationsHtml}
        </div>
      </div>
    `;
  }

  // Render Graph Reasoning Facts Accordion
  if (data.graph_facts && data.graph_facts.length > 0) {
    const factsHtml = data.graph_facts
      .map((f) => `
        <div class="graph-fact-pill">
          <strong>${escapeHtml(f.source_entity)}</strong>
          <span class="graph-fact-rel">—[${escapeHtml(f.relation)}]→</span>
          <strong>${escapeHtml(f.target_entity)}</strong>
        </div>
      `)
      .join('');

    contentHtml += `
      <div class="accordion-box" onclick="this.classList.toggle('open')">
        <div class="accordion-header" style="color: var(--accent-cyan);">
          <span>🔗 Extracted Graph Facts (${data.graph_facts.length})</span>
          <span>▼</span>
        </div>
        <div class="accordion-content">
          ${factsHtml}
        </div>
      </div>
    `;
  }

  bubbleEl.innerHTML = contentHtml;
}

// ── vis-network Graph Visualizer ───────────────────────────────────────
function initGraphNetwork() {
  const container = elements.graphNetwork;

  const options = {
    nodes: {
      shape: 'dot',
      size: 18,
      font: {
        face: 'JetBrains Mono',
        size: 12,
        color: '#f5f5f5',
        strokeWidth: 2,
        strokeColor: '#121212',
      },
      borderWidth: 2,
      borderWidthSelected: 3,
      shadow: false,
    },
    edges: {
      width: 1.5,
      color: {
        color: '#383838',
        highlight: '#ffffff',
        hover: '#00f0ff',
      },
      font: {
        face: 'JetBrains Mono',
        size: 10,
        color: '#888888',
        strokeWidth: 0,
        align: 'middle',
      },
      arrows: { to: { enabled: true, scaleFactor: 0.5 } },
      smooth: false,
      shadow: false,
    },
    physics: {
      enabled: true,
      solver: 'barnesHut',
      barnesHut: {
        gravitationalConstant: -3200, // Balanced node repulsion to prevent collisions
        centralGravity: 0.35,         // Strong central gravity: Keeps unconnected nodes clustered and prevents them from drifting away
        springLength: 100,            // Ideal link distance
        springConstant: 0.05,         // Elastic spring force: Dragging any node pulls connected nodes smoothly with full physical tension
        damping: 0.15,                // Fluid damping for bouncy, organic tactile movement
        avoidOverlap: 0.25,
      },
      maxVelocity: 45,
      minVelocity: 0.1,
      timestep: 0.5,
      adaptiveTimestep: true,
      stabilization: {
        enabled: true,
        iterations: 120,
        updateInterval: 25,
        fit: true,
      },
    },
    interaction: {
      hover: true,
      tooltipDelay: 150,
      zoomView: true,
      dragView: true,
      dragNodes: true,
      selectable: true,
    },
  };

  state.network = new vis.Network(container, state.networkData, options);

  // Node Selection Handler
  state.network.on('click', (params) => {
    if (params.nodes.length > 0) {
      const nodeId = params.nodes[0];
      const nodeObj = state.networkData.nodes.get(nodeId);
      if (nodeObj) {
        showEntityInspector(nodeObj);
      }
    } else {
      elements.entityInspector.classList.remove('active');
    }
  });
}

function renderGraphFacts(facts) {
  if (!facts || facts.length === 0) {
    elements.graphEmpty.style.display = 'block';
    state.networkData.nodes.clear();
    state.networkData.edges.clear();
    return;
  }

  elements.graphEmpty.style.display = 'none';

  const nodesMap = new Map();
  const edgesList = [];

  facts.forEach((fact, idx) => {
    const src = fact.source_entity;
    const tgt = fact.target_entity;
    const rel = fact.relation;

    if (!nodesMap.has(src)) {
      nodesMap.set(src, {
        id: src,
        label: src,
        color: { background: getEntityColor('Concept'), border: '#000000' },
        entityType: 'Concept',
        description: `Source entity involved in relationship: ${rel}`,
      });
    }

    if (!nodesMap.has(tgt)) {
      nodesMap.set(tgt, {
        id: tgt,
        label: tgt,
        color: { background: getEntityColor('Organization'), border: '#000000' },
        entityType: 'Entity',
        description: `Target entity connected via ${rel}`,
      });
    }

    edgesList.push({
      id: `e_${idx}`,
      from: src,
      to: tgt,
      label: rel,
    });
  });

  state.networkData.nodes.clear();
  state.networkData.edges.clear();
  state.networkData.nodes.add(Array.from(nodesMap.values()));
  state.networkData.edges.add(edgesList);

  if (state.network) {
    state.network.fit({ animation: { duration: 600 } });
  }
}

async function loadDocumentSubgraph(docId) {
  try {
    showToast('Loading document subgraph...', 'info');
    const res = await fetch(`/graph/document/${docId}`);
    if (!res.ok) throw new Error('Failed to load subgraph');
    const data = await res.json();

    if (!data.nodes || data.nodes.length === 0) {
      elements.graphEmpty.style.display = 'block';
      state.networkData.nodes.clear();
      state.networkData.edges.clear();
      showToast('No entities extracted for this document yet.', 'info');
      return;
    }

    elements.graphEmpty.style.display = 'none';

    const visNodes = data.nodes.map((n) => ({
      id: n.name,
      label: n.name,
      color: { background: getEntityColor(n.type), border: '#000000' },
      entityType: n.type,
      description: n.description || 'No description available',
    }));

    const visEdges = data.edges.map((e, idx) => ({
      id: `edge_${idx}`,
      from: e.source,
      to: e.target,
      label: e.relation,
    }));

    state.networkData.nodes.clear();
    state.networkData.edges.clear();
    state.networkData.nodes.add(visNodes);
    state.networkData.edges.add(visEdges);

    if (state.network) {
      state.network.fit({ animation: { duration: 600 } });
    }
  } catch (err) {
    showToast(`Subgraph error: ${err.message}`, 'error');
  }
}

function showEntityInspector(node) {
  elements.inspectorName.textContent = node.label;
  elements.inspectorType.textContent = node.entityType || 'Entity';
  elements.inspectorType.style.background = `${getEntityColor(node.entityType)}25`;
  elements.inspectorType.style.color = getEntityColor(node.entityType);
  elements.inspectorDesc.textContent = node.description || 'No additional description available.';
  elements.entityInspector.classList.add('active');
}

function togglePhysics() {
  state.physicsEnabled = !state.physicsEnabled;
  if (state.network) {
    state.network.setOptions({ physics: { enabled: state.physicsEnabled } });
  }
  elements.btnPhysics.textContent = `Physics: ${state.physicsEnabled ? 'ON' : 'OFF'}`;
  elements.btnPhysics.style.color = state.physicsEnabled ? 'var(--text-primary)' : 'var(--text-muted)';
}

let isExpandedLayout = false;

function toggleExpandedLayout() {
  isExpandedLayout = !isExpandedLayout;
  if (elements.mainWorkspace) {
    elements.mainWorkspace.classList.toggle('expanded-layout', isExpandedLayout);
  }
  if (elements.btnExpandLayout) {
    elements.btnExpandLayout.innerHTML = isExpandedLayout ? '🗗 Split' : '⛶ Expand';
    elements.btnExpandLayout.title = isExpandedLayout ? 'Restore Split Screen View' : 'Expand Assistant & Move Graph Below';
    elements.btnExpandLayout.classList.toggle('active', isExpandedLayout);
  }
  // Allow DOM layout transition then redraw and fit graph to full width
  setTimeout(() => {
    if (state.network) {
      state.network.redraw();
      state.network.fit({ animation: { duration: 400 } });
    }
  }, 120);
}

// ── Helpers ────────────────────────────────────────────────────────────
function escapeHtml(unsafe) {
  return (unsafe || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
