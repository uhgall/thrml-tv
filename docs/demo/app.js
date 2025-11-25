(function () {
  const replayConfig =
    typeof window !== "undefined" && window.THRML_REPLAY_DATA
      ? window.THRML_REPLAY_DATA
      : Object.freeze({ mode: "server" });
  const replayMode = replayConfig.mode || "server";
  const isOfflineReplay = replayMode === "offline";

  const state = {
    graph: null,
    history: [],
    currentIndex: -1,
    isPlaying: true,
    playTimer: null,
    socket: null,
    reconnectTimer: null,
    pendingLambdaConflict: null,
    layoutMode: "geographic",
    fdgLayout: null,
  };

  const controls = {};
  const logEntries = [];
  const maxLogEntries = 200;
  const visuals = {};
  let lambdaSliderSendTimer = null;
  let suppressLambdaSliderEmit = false;
  let cloneFallbackWarningShown = false;

  function lambdaAdjustmentEnabled() {
    return !isOfflineReplay && Boolean(state.graph?.lambda_conflict_adjustable);
  }

  const palette = d3
    .schemeTableau10.concat(d3.schemeSet3 ?? [])
    .concat(d3.schemeCategory10 ?? []);
  const colourScale = d3.scaleOrdinal(palette);
  const layoutModes = {
    geographic: "geographic",
    fdg: "fdg",
  };

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function cloneData(value) {
    if (!cloneFallbackWarningShown && typeof structuredClone === "function") {
      try {
        return structuredClone(value);
      } catch (error) {
        console.warn("structuredClone failed; falling back to JSON cloning.", error);
        cloneFallbackWarningShown = true;
      }
    }
    return JSON.parse(JSON.stringify(value));
  }

  document.addEventListener("DOMContentLoaded", init);

  function log(message, { level = "info", data } = {}) {
    const timestamp = new Date().toISOString();
    const formatted = `[${timestamp}] [${level.toUpperCase()}] ${message}${
      data !== undefined ? ` ${JSON.stringify(data)}` : ""
    }`;

    // Browser console
    switch (level) {
      case "error":
        console.error(formatted);
        break;
      case "warn":
        console.warn(formatted);
        break;
      default:
        console.log(formatted);
    }

    const messageLower = String(message).toLowerCase();
    const shouldHide = messageLower.includes("web client connected");
    if (shouldHide) {
      return;
    }

    const uiEntry = [message, data !== undefined ? JSON.stringify(data) : ""]
      .filter(Boolean)
      .join(" ")
      .trim();

    if (!uiEntry) {
      return;
    }

    logEntries.push(uiEntry);
    if (logEntries.length > maxLogEntries) {
      logEntries.shift();
    }
    const output = document.getElementById("log-output");
    if (output) {
      output.textContent = logEntries.join("\n");
      output.scrollTop = output.scrollHeight;
    }
  }

  function configureLambdaSlider() {
    if (!controls.lambdaSlider || !lambdaAdjustmentEnabled()) {
      return;
    }
    const base = state.graph?.lambda_conflict ?? 1;
    const slider = controls.lambdaSlider;
    slider.min = "0";
    slider.max = String(Math.max(base * 2, base + 10));
    slider.step = "0.1";
    updateLambdaSliderUI(base, { updateGraph: true });
  }

  function updateLambdaSliderUI(value, { updateGraph = true } = {}) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return;
    }

    if (updateGraph && state.graph) {
      state.graph.lambda_conflict = numeric;
      updateMetadata();
    }

    if (!controls.lambdaSlider || !lambdaAdjustmentEnabled()) {
      return;
    }

    const slider = controls.lambdaSlider;
    const display = controls.lambdaSliderValue;
    const currentMax = Number(slider.max);
    if (!Number.isFinite(currentMax) || numeric > currentMax) {
      slider.max = String(Math.max(numeric * 1.25, numeric + 1));
    }

    suppressLambdaSliderEmit = true;
    slider.value = String(numeric);
    suppressLambdaSliderEmit = false;

    if (display) {
      display.textContent = numeric.toFixed(2);
    }

  }

  function queueLambdaSliderSend(value) {
    if (!lambdaAdjustmentEnabled()) {
      return;
    }
    if (lambdaSliderSendTimer) {
      clearTimeout(lambdaSliderSendTimer);
    }
    lambdaSliderSendTimer = setTimeout(() => {
      sendLambdaSliderUpdate(value);
    }, 150);
  }

  function sendLambdaSliderUpdate(value) {
    if (!lambdaAdjustmentEnabled()) {
      return;
    }
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return;
    }
    if (!state.socket || state.socket.readyState !== WebSocket.OPEN) {
      state.pendingLambdaConflict = numeric;
      lambdaSliderSendTimer = null;
      return;
    }
    state.socket.send(
      JSON.stringify({
        type: "set_lambda_conflict",
        value: numeric,
      })
    );
    state.pendingLambdaConflict = null;
    lambdaSliderSendTimer = null;
  }

  async function init() {
    log("Initialising dashboard…");
    try {
      if (isOfflineReplay) {
        log("Offline replay detected; loading embedded graph metadata…");
        loadOfflineGraph();
      } else {
        log("Fetching graph metadata…");
        await fetchGraph();
        log("Graph metadata loaded.");
      }
    } catch (error) {
      log("Failed to load graph metadata", { level: "error", data: String(error) });
      return;
    }

    if (!state.graph) {
      log("Graph metadata unavailable; aborting startup.", { level: "error" });
      return;
    }

    if (isOfflineReplay) {
      state.layoutMode = layoutModes.fdg;
      state.isPlaying = false;
    }

    setupControls();
    setupVisuals();
    log("Controls and visuals initialised.");

    try {
      if (isOfflineReplay) {
        log("Loading embedded history…");
        loadOfflineHistory();
      } else {
        log("Fetching history file…");
        await fetchHistory();
      }
    } catch (error) {
      log("History load failed", { level: "warn", data: String(error) });
    }

    if (isOfflineReplay) {
      log("Offline replay ready. Skipping WebSocket connection.");
    } else {
      log("Connecting WebSocket…");
      connectWebSocket();
    }

    if (state.history.length) {
      if (isOfflineReplay) {
        log(`Loaded ${state.history.length} history entries. Jumping to first.`);
        setCurrentIndex(0);
        if (state.layoutMode === layoutModes.fdg) {
          const fdg = createForceLayout();
          if (fdg?.simulation) {
            fdg.simulation.alpha(1).alphaTarget(0.2).restart();
          }
        }
      } else {
        log(`Loaded ${state.history.length} history entries. Jumping to latest.`);
        setCurrentIndex(state.history.length - 1);
      }
    } else {
      log("No history entries available yet.");
      updateControlsAvailability();
    }
    log("Dashboard ready.");
  }

  async function fetchGraph() {
    const response = await fetch("/graph");
    if (!response.ok) {
      throw new Error(`Graph endpoint responded with ${response.status}`);
    }
    state.graph = await response.json();
    if (state.fdgLayout?.simulation) {
      state.fdgLayout.simulation.stop();
    }
    state.layoutMode = layoutModes.geographic;
    state.fdgLayout = null;
    updateLayoutToggleUI();
    updateMetadata();
  }

  function loadOfflineGraph() {
    if (!replayConfig?.graph) {
      throw new Error("Offline replay bundle is missing graph metadata.");
    }
    state.graph = cloneData(replayConfig.graph);
    if (state.fdgLayout?.simulation) {
      state.fdgLayout.simulation.stop();
    }
    state.layoutMode = layoutModes.geographic;
    state.fdgLayout = null;
    updateLayoutToggleUI();
    updateMetadata();
  }

  async function fetchHistory() {
    const response = await fetch("/history");
    if (!response.ok) {
      return;
    }

    const body = await response.text();
    if (!body.trim()) {
      return;
    }

    body
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .forEach((line) => {
        try {
          const entry = JSON.parse(line);
          if (entry.type === "state") {
            pushState(entry, { replace: true });
          }
        } catch (error) {
          console.warn("Could not parse history line", error);
        }
      });
  }

  function loadOfflineHistory() {
    const entries = Array.isArray(replayConfig?.history) ? replayConfig.history : [];
    state.history = [];
    state.currentIndex = -1;
    entries.forEach((entry) => {
      if (!entry || typeof entry !== "object") {
        return;
      }
      if (entry.type === "state") {
        const cloned = cloneData(entry);
        pushState(cloned, { replace: true });
      } else if (entry.type === "log" && typeof entry.message === "string") {
        log(entry.message, {
          level: entry.level ?? "info",
          data: entry.extra ?? undefined,
        });
      }
    });
  }

  function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const url = `${protocol}://${window.location.host}/ws/state`;
    const socket = new WebSocket(url);
    state.socket = socket;
    log(`WebSocket opening: ${url}`);

    socket.addEventListener("message", (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === "state") {
          pushState(message);
        } else if (message.type === "lambda_conflict") {
          updateLambdaSliderUI(message.value, { updateGraph: true });
        } else if (message.type === "log") {
          log(message.message ?? "", {
            level: message.level ?? "info",
            data: message.extra ?? undefined,
          });
        } else {
          log(`Unknown message type "${message.type}"`, {
            level: "warn",
            data: message,
          });
        }
      } catch (error) {
        log("Received malformed websocket payload", { level: "warn", data: String(error) });
      }
    });

    socket.addEventListener("close", () => {
      log("WebSocket closed; scheduling reconnect.", { level: "warn" });
      state.socket = null;
      if (state.reconnectTimer) {
        clearTimeout(state.reconnectTimer);
      }
      state.reconnectTimer = setTimeout(connectWebSocket, 2000);
    });

    socket.addEventListener("open", () => {
      log("WebSocket connected.");
      if (lambdaAdjustmentEnabled()) {
        if (state.pendingLambdaConflict != null) {
          sendLambdaSliderUpdate(state.pendingLambdaConflict);
        } else if (controls.lambdaSlider) {
          const value = Number(controls.lambdaSlider.value);
          if (Number.isFinite(value)) {
            sendLambdaSliderUpdate(value);
          }
        }
      }
    });

    socket.addEventListener("error", (event) => {
      log("WebSocket error", { level: "error", data: String(event) });
    });
  }

  function setupControls() {
    controls.playToggle = document.getElementById("play-toggle");
    controls.prev = document.getElementById("prev-step");
    controls.next = document.getElementById("next-step");
    controls.first = document.getElementById("first-step");
    controls.last = document.getElementById("last-step");
    controls.slider = document.getElementById("step-slider");
    controls.stepLabel = document.getElementById("step-label");
    controls.lambdaSlider = document.getElementById("lambda-slider");
    controls.lambdaSliderValue = document.getElementById("lambda-slider-value");
    controls.logOutput = document.getElementById("log-output");
    controls.layoutToggle = document.getElementById("layout-toggle");
    controls.layoutButtons = {
      [layoutModes.geographic]: document.getElementById("layout-geographic"),
      [layoutModes.fdg]: document.getElementById("layout-fdg"),
    };

    controls.playToggle.addEventListener("click", () => togglePlay());
    controls.prev.addEventListener("click", () => stepBy(-1, { user: true }));
    controls.next.addEventListener("click", () => stepBy(1, { user: true }));
    controls.first.addEventListener("click", () => setCurrentIndex(0));
    controls.last.addEventListener("click", () => setCurrentIndex(state.history.length - 1));

    controls.slider.addEventListener("input", (event) => {
      const index = Number(event.target.value);
      togglePlay(false);
      setCurrentIndex(index);
    });

    document.addEventListener("keydown", handleKeydown);
    controls.playToggle.textContent = state.isPlaying ? "Pause" : "Play";
    log("Controls wired.");

    if (isOfflineReplay && controls.logOutput) {
      const eventLogSection = controls.logOutput.closest("#event-log");
      if (eventLogSection) {
        eventLogSection.style.display = "none";
      } else {
        controls.logOutput.style.display = "none";
      }
    }

    if (controls.layoutButtons) {
      Object.entries(controls.layoutButtons).forEach(([mode, button]) => {
        if (!button) {
          return;
        }
        button.addEventListener("click", () => setLayoutMode(mode));
      });
    }
    updateLayoutToggleUI();

    const lambdaAdjustable = lambdaAdjustmentEnabled();
    const sliderElement = controls.lambdaSlider;
    const sliderValue = controls.lambdaSliderValue;

    if (!lambdaAdjustable && sliderElement) {
      const container = sliderElement.closest(".slider-label");
      if (container) {
        container.style.display = "none";
      } else {
        sliderElement.style.display = "none";
      }
      sliderElement.disabled = true;
      if (sliderValue) {
        sliderValue.style.display = "none";
      }
      controls.lambdaSlider = null;
      controls.lambdaSliderValue = null;
    } else if (lambdaAdjustable && sliderElement) {
      configureLambdaSlider();
      sliderElement.addEventListener("input", (event) => {
        const value = Number(event.target.value);
        if (!Number.isFinite(value)) {
          return;
        }
        if (sliderValue) {
          sliderValue.textContent = value.toFixed(2);
        }
        if (state.graph) {
          state.graph.lambda_conflict = value;
          updateMetadata();
        }
        if (suppressLambdaSliderEmit) {
          return;
        }
        queueLambdaSliderSend(value);
      });
      sliderElement.addEventListener("change", (event) => {
        const value = Number(event.target.value);
        if (!Number.isFinite(value)) {
          return;
        }
        queueLambdaSliderSend(value);
      });
    }
  }

  function handleKeydown(event) {
    const tag = event.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA") {
      return;
    }
    switch (event.key) {
      case "ArrowLeft":
        event.preventDefault();
        stepBy(-1, { user: true });
        break;
      case "ArrowRight":
        event.preventDefault();
        stepBy(1, { user: true });
        break;
      case "Home":
        event.preventDefault();
        setCurrentIndex(0);
        break;
      case "End":
        event.preventDefault();
        setCurrentIndex(state.history.length - 1);
        break;
      case " ":
        event.preventDefault();
        togglePlay();
        break;
      default:
        break;
    }
  }

  function setupVisuals() {
    setupScatter();
    setupEnergy();
  }

  function setupScatter() {
    const container = d3.select("#scatter-container");
    const width = container.node().clientWidth || 640;
    const height = container.node().clientHeight || 520;
    const margin = { top: 24, right: 24, bottom: 36, left: 48 };
    const labelOffsetX = 8;
    const labelOffsetY = -6;
    const baseNodeRadius = 2.8;

    const svg = container
      .append("svg")
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet");

    const lonRangeRaw = state.graph.lon_range ?? [0, 1];
    const lonRange = lonRangeRaw.map((value) => -value).sort((a, b) => a - b);
    const latRange = state.graph.lat_range ?? [0, 1];

    const xScale = d3.scaleLinear().domain(lonRange).range([margin.left, width - margin.right]);
    const yScale = d3.scaleLinear().domain(latRange).range([height - margin.bottom, margin.top]);

    const stations = state.graph.stations;

    const edgeData = (state.graph.edges || []).map((pair, idx) => ({
      index: idx,
      sourceIndex: pair[0],
      targetIndex: pair[1],
      source: stations[pair[0]],
      target: stations[pair[1]],
    }));

    const viewport = svg.append("g").attr("class", "scatter-viewport");

    const edgesGroup = viewport.append("g").attr("class", "edges");
    const edgeSelection = edgesGroup
      .selectAll("line")
      .data(edgeData, (d) => d.index)
      .join("line")
      .attr("x1", (d) => xScale(-d.source.lon))
      .attr("y1", (d) => yScale(d.source.lat))
      .attr("x2", (d) => xScale(-d.target.lon))
      .attr("y2", (d) => yScale(d.target.lat));

    const nodesGroup = viewport.append("g").attr("class", "nodes");
    const nodeSelection = nodesGroup
      .selectAll("circle")
      .data(stations, (d) => d.index)
      .join("circle")
      .attr("cx", (d) => xScale(-d.lon))
      .attr("cy", (d) => yScale(d.lat))
      .attr("r", baseNodeRadius);

    const labelGroup = viewport.append("g").attr("class", "labels");
    const labelSelection = labelGroup
      .selectAll("text")
      .data(stations, (d) => d.index)
      .join("text")
      .attr("x", (d) => xScale(-d.lon) + labelOffsetX)
      .attr("y", (d) => yScale(d.lat) + labelOffsetY)
      .text((d) => d.city)
      .attr("display", "none");

    const tooltip = d3
      .select("body")
      .selectAll(".scatter-tooltip")
      .data([null])
      .join("div")
      .attr("class", "scatter-tooltip")
      .style("opacity", 0)
      .style("pointer-events", "none");

    function positionTooltip(event) {
      tooltip
        .style("left", `${event.pageX + 12}px`)
        .style("top", `${event.pageY + 12}px`);
    }

    nodeSelection
      .on("mouseenter", (event, d) => {
        const cityLabel = d.city && d.city !== "" ? d.city : "Station";
        tooltip.style("opacity", 1).text(`${cityLabel}: ${d.station_id}`);
        positionTooltip(event);
      })
      .on("mousemove", (event) => {
        positionTooltip(event);
      })
      .on("mouseleave", () => {
        tooltip.style("opacity", 0);
      });

    const firstLabel = labelSelection.node();
    const baseLabelFontSize =
      firstLabel && typeof window !== "undefined"
        ? Number.parseFloat(window.getComputedStyle(firstLabel).fontSize) || 12
        : 12;

    const zoomBehavior = d3
      .zoom()
      .scaleExtent([0.5, 8])
      .on("zoom", (event) => {
        if (visuals.scatter) {
          visuals.scatter.zoomTransform = event.transform;
        }
        viewport.attr("transform", event.transform);
        updateZoomAdjustments();
        applyLayoutPositions();
      });

    visuals.scatter = {
      svg,
      width,
      height,
      margin,
      stations,
      xScale,
      yScale,
      viewport,
      edgeSelection,
      nodeSelection,
      labelSelection,
      edgeData,
      zoomBehavior,
      tooltip,
      zoomTransform: d3.zoomIdentity,
      baseNodeRadius,
      baseLabelFontSize,
      labelOffsetX,
      labelOffsetY,
    };

    svg.call(zoomBehavior);
    updateZoomAdjustments();
    applyLayoutPositions();
    if (state.layoutMode === layoutModes.fdg) {
      const fdg = createForceLayout();
      if (fdg?.simulation) {
        fdg.simulation.alpha(1).alphaTarget(0.25).restart();
      }
    }
  }

  function createForceLayout() {
    if (!state.graph) {
      return null;
    }
    const scatter = visuals.scatter;
    if (!scatter) {
      return null;
    }
    if (state.fdgLayout?.simulation) {
      state.fdgLayout.simulation.stop();
    }

    const centerX = scatter.width / 2;
    const centerY = scatter.height / 2;
    const initialRadius = Math.min(scatter.width, scatter.height) * 0.28;

    const nodes = state.graph.stations.map((station, idx) => ({
      index: idx,
      id: station.station_id ?? idx,
      x: centerX + (Math.random() - 0.5) * initialRadius,
      y: centerY + (Math.random() - 0.5) * initialRadius,
    }));

    const links = (state.graph.edges || []).map((pair) => ({
      source: pair[0],
      target: pair[1],
    }));

    const simulation = d3
      .forceSimulation(nodes)
      .force(
        "link",
        d3
          .forceLink(links)
          .id((d) => d.index)
          .distance(48)
          .strength(0.08)
      )
      .force("charge", d3.forceManyBody().strength(-65))
      .force("center", d3.forceCenter(scatter.width / 2, scatter.height / 2))
      .force("collision", d3.forceCollide().radius(14))
      .force("x", d3.forceX(scatter.width / 2).strength(0.045))
      .force("y", d3.forceY(scatter.height / 2).strength(0.045))
      .alphaDecay(0.045);

    simulation.on("tick", () => {
      if (state.layoutMode === layoutModes.fdg) {
        applyLayoutPositions(layoutModes.fdg);
      }
    });

    state.fdgLayout = {
      nodes,
      simulation,
    };
    applyLayoutPositions(layoutModes.fdg);
    return state.fdgLayout;
  }

  function setLayoutMode(mode) {
    const normalised = layoutModes[mode] || mode || layoutModes.geographic;
    const targetMode =
      normalised === layoutModes.fdg ? layoutModes.fdg : layoutModes.geographic;
    if (state.layoutMode === targetMode) {
      if (targetMode === layoutModes.fdg) {
        const fdg = createForceLayout();
        if (fdg?.simulation) {
          fdg.simulation.alpha(1).alphaTarget(0.25).restart();
        }
      }
      updateLayoutToggleUI();
      return;
    }

    state.layoutMode = targetMode;

    if (targetMode === layoutModes.fdg) {
      const fdg = createForceLayout();
      if (fdg?.simulation) {
        fdg.simulation.alpha(1).alphaTarget(0.25).restart();
      }
    } else if (state.fdgLayout?.simulation) {
      state.fdgLayout.simulation.alphaTarget(0);
    }

    updateLayoutToggleUI();
    applyLayoutPositions(targetMode);
  }

  function updateLayoutToggleUI() {
    if (!controls.layoutButtons) {
      return;
    }
    Object.entries(controls.layoutButtons).forEach(([mode, button]) => {
      if (!button) {
        return;
      }
      const isActive = state.layoutMode === mode;
      button.classList.toggle("active", isActive);
      button.setAttribute("aria-pressed", String(isActive));
    });
  }

  function updateZoomAdjustments() {
    const scatter = visuals.scatter;
    if (!scatter) {
      return;
    }
    const scale = scatter.zoomTransform?.k ?? 1;
    scatter.nodeSelection.attr("r", scatter.baseNodeRadius / scale);
    if (Number.isFinite(scatter.baseLabelFontSize)) {
      scatter.labelSelection.style("font-size", `${scatter.baseLabelFontSize / scale}px`);
    }
  }

  function applyLayoutPositions(mode = state.layoutMode) {
    const scatter = visuals.scatter;
    if (!scatter || !scatter.stations) {
      return;
    }

    const effectiveMode =
      mode === layoutModes.fdg && state.fdgLayout?.nodes ? layoutModes.fdg : layoutModes.geographic;
    const scale = scatter.zoomTransform?.k ?? 1;
    const { margin, width, height, labelOffsetX, labelOffsetY } = scatter;
    const fdgPadding =
      effectiveMode === layoutModes.fdg ? Math.min(width, height) * 0.8 : 0;
    const clampXMin = margin.left - fdgPadding;
    const clampXMax = width - margin.right + fdgPadding;
    const clampYMin = margin.top - fdgPadding;
    const clampYMax = height - margin.bottom + fdgPadding;
    const stations = scatter.stations;
    const positions = new Array(stations.length);

    function resolvePosition(index) {
      if (positions[index]) {
        return positions[index];
      }
      let x;
      let y;
      if (effectiveMode === layoutModes.fdg && state.fdgLayout?.nodes) {
        const node = state.fdgLayout.nodes[index];
        if (node && Number.isFinite(node.x) && Number.isFinite(node.y)) {
          x = clamp(node.x, clampXMin, clampXMax);
          y = clamp(node.y, clampYMin, clampYMax);
        }
      }
      if (x === undefined || y === undefined) {
        const station = stations[index];
        if (!station) {
          positions[index] = { x: 0, y: 0 };
          return positions[index];
        }
        x = scatter.xScale(-station.lon);
        y = scatter.yScale(station.lat);
      }
      positions[index] = { x, y };
      return positions[index];
    }

    scatter.nodeSelection.each(function (_, idx) {
      const pos = resolvePosition(idx);
      this.setAttribute("cx", pos.x);
      this.setAttribute("cy", pos.y);
    });

    scatter.edgeSelection.each(function (d) {
      const source = resolvePosition(d.sourceIndex);
      const target = resolvePosition(d.targetIndex);
      this.setAttribute("x1", source.x);
      this.setAttribute("y1", source.y);
      this.setAttribute("x2", target.x);
      this.setAttribute("y2", target.y);
    });

    scatter.labelSelection.each(function (_, idx) {
      const pos = resolvePosition(idx);
      this.setAttribute("x", pos.x + labelOffsetX / scale);
      this.setAttribute("y", pos.y + labelOffsetY / scale);
    });
  }

  function setupEnergy() {
    const container = d3.select("#energy-container");
    const width = container.node().clientWidth || 420;
    const height = container.node().clientHeight || 260;
    const margin = { top: 24, right: 24, bottom: 40, left: 56 };

    const svg = container
      .append("svg")
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet");

    const xScale = d3.scaleLinear().range([margin.left, width - margin.right]);
    const yScale = d3.scaleLinear().range([height - margin.bottom, margin.top]);

    const xAxisGroup = svg
      .append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${height - margin.bottom})`);
    const yAxisGroup = svg
      .append("g")
      .attr("class", "y-axis")
      .attr("transform", `translate(${margin.left},0)`);

    svg
      .append("text")
      .attr("class", "axis-title")
      .attr("x", width / 2)
      .attr("y", height - 6)
      .attr("text-anchor", "middle")
      .text("Step");

    svg
      .append("text")
      .attr("class", "axis-title")
      .attr("transform", "rotate(-90)")
      .attr("x", -height / 2)
      .attr("y", 16)
      .attr("text-anchor", "middle")
      .text("Edge Violations");

    const line = d3
      .line()
      .x((d) => xScale(d.step))
      .y((d) => yScale(d.edge_violation_count ?? 0))
      .curve(d3.curveMonotoneX);

    const path = svg.append("path").attr("class", "energy-line").attr("fill", "none").attr("stroke", "#2563eb").attr("stroke-width", 1.8);

    const marker = svg
      .append("circle")
      .attr("class", "energy-marker")
      .attr("r", 5)
      .attr("fill", "#ef4444")
      .attr("stroke", "#ffffff")
      .attr("stroke-width", 1.5)
      .attr("display", "none");

    visuals.energy = {
      svg,
      xScale,
      yScale,
      xAxisGroup,
      yAxisGroup,
      line,
      path,
      marker,
    };
  }

  function pushState(entry, options = {}) {
    const history = state.history;
    const last = history[history.length - 1];
    if (options.replace && last && last.step === entry.step) {
      history[history.length - 1] = entry;
    } else if (!last || last.step !== entry.step) {
      history.push(entry);
    } else {
      history[history.length - 1] = entry;
    }

    controls.slider.max = String(Math.max(0, history.length - 1));
    updateControlsAvailability();
    updateEnergySeries();

    const shouldFollow = state.isPlaying || state.currentIndex === -1;

    if (shouldFollow) {
      setCurrentIndex(history.length - 1);
    } else {
      render();
    }
  }

  function setCurrentIndex(index) {
    if (!state.history.length) {
      return;
    }
    const clamped = Math.max(0, Math.min(index, state.history.length - 1));
    state.currentIndex = clamped;
    controls.slider.value = String(clamped);
    render();
  }

  function stepBy(delta, opts = {}) {
    if (!state.history.length) {
      return;
    }
    const nextIndex = Math.max(0, Math.min(state.history.length - 1, state.currentIndex + delta));
    if (opts.user) {
      togglePlay(false);
    }
    setCurrentIndex(nextIndex);
  }

  function togglePlay(force) {
    const shouldPlay = typeof force === "boolean" ? force : !state.isPlaying;
    if (state.isPlaying === shouldPlay) {
      return;
    }
    state.isPlaying = shouldPlay;
    if (controls.playToggle) {
      controls.playToggle.textContent = shouldPlay ? "Pause" : "Play";
    }

    if (state.playTimer) {
      clearInterval(state.playTimer);
      state.playTimer = null;
    }

    if (shouldPlay && state.history.length > 0) {
      state.playTimer = setInterval(() => {
        if (state.currentIndex >= state.history.length - 1) {
          return;
        }
        setCurrentIndex(state.currentIndex + 1);
      }, 700);
    }
  }

  function render() {
    const entry = state.history[state.currentIndex];
    if (!entry) {
      controls.stepLabel.textContent = "Awaiting data…";
      return;
    }

    controls.stepLabel.textContent = formatStepSummary(entry);
    renderScatter(entry);
    renderEnergyMarker(entry);
  }

  function renderScatter(entry) {
    const scatter = visuals.scatter;
    if (!scatter) {
      return;
    }
    applyLayoutPositions();
    const assignment = entry.assignment;
    const domainMask = entry.domain_violation_mask.map(Boolean);
    const edgeMask = entry.edge_violation_mask.map(Boolean);

    const incidentEdgeViolation = new Array(state.graph.station_count).fill(false);
    scatter.edgeSelection.classed("edge-violation", (d) => {
      const violated = !!edgeMask[d.index];
      if (violated) {
        incidentEdgeViolation[d.sourceIndex] = true;
        incidentEdgeViolation[d.targetIndex] = true;
      }
      return violated;
    });

    if (edgeMask.some(Boolean)) {
      scatter.edgeSelection.filter((d) => edgeMask[d.index]).raise();
    }

    scatter.nodeSelection
      .attr("fill", (_, idx) => colourScale(assignment[idx] ?? 0))
      .classed("domain-violation", (_, idx) => domainMask[idx])
      .classed(
        "edge-violation",
        (_, idx) => incidentEdgeViolation[idx] && !domainMask[idx]
      );

    scatter.labelSelection.classed(
      "violation",
      (_, idx) => domainMask[idx] || incidentEdgeViolation[idx]
    );
  }

  function updateEnergySeries() {
    const energy = visuals.energy;
    if (!energy || !state.history.length) {
      return;
    }
    const steps = state.history.map((d) => d.step);
    const counts = state.history.map((d) => Number(d.edge_violation_count ?? 0));
    const minStep = d3.min(steps) ?? 0;
    const maxStep = d3.max(steps) ?? minStep + 1;
    const minCount = d3.min(counts) ?? 0;
    const maxCount = d3.max(counts) ?? minCount;

    energy.xScale.domain([minStep, maxStep === minStep ? minStep + 1 : maxStep]);
    const upper =
      maxCount === minCount ? maxCount + 1 : maxCount * 1.05 + (maxCount >= 0 ? 0 : 1);
    const lower = Math.min(0, minCount * 0.95);
    energy.yScale.domain([lower, upper]);

    energy.path.datum(state.history).attr("d", energy.line);
    energy.xAxisGroup.call(d3.axisBottom(energy.xScale).ticks(6).tickFormat(d3.format(",d")));
    energy.yAxisGroup.call(d3.axisLeft(energy.yScale).ticks(6).tickFormat(d3.format(",d")));
  }

  function renderEnergyMarker(entry) {
    const energy = visuals.energy;
    if (!energy) {
      return;
    }
    if (!entry) {
      energy.marker.attr("display", "none");
      return;
    }
    energy.marker
      .attr("display", null)
      .attr("cx", energy.xScale(entry.step))
      .attr("cy", energy.yScale(entry.edge_violation_count ?? 0));
  }

  function formatStepSummary(entry) {
    const energyFormatter = d3.format(".4~g");
    const violations = [
      `${entry.domain_violation_count} domain`,
      `${entry.edge_violation_count} edge`,
    ].join(", ");
    return `Step ${entry.step.toLocaleString()} • Energy ${energyFormatter(entry.energy)} • Violations ${violations}`;
  }

  function updateMetadata() {
    const meta = document.getElementById("run-metadata");
    if (!state.graph || !meta) {
      return;
    }
    const edgeCount = state.graph.edges ? state.graph.edges.length : 0;
    const entries = [
      state.graph.run_name ?? "Unnamed run",
      `${state.graph.station_count} stations`,
      `${edgeCount} edges`,
    ];
    if (typeof state.graph.lambda_domain === "number") {
      entries.push(`λ_domain ${state.graph.lambda_domain}`);
    }
    if (typeof state.graph.lambda_conflict === "number") {
      entries.push(`λ_conflict ${state.graph.lambda_conflict}`);
    }
    meta.textContent = entries.join(" • ");
  }

  function updateControlsAvailability() {
    const hasData = state.history.length > 0;
    controls.playToggle.disabled = !hasData;
    controls.prev.disabled = !hasData;
    controls.next.disabled = !hasData;
    controls.first.disabled = !hasData;
    controls.last.disabled = !hasData;
    controls.slider.disabled = !hasData;
    if (!hasData) {
      controls.stepLabel.textContent = "Waiting for sampler…";
    }
  }
})();


