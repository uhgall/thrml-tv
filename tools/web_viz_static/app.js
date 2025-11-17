(function () {
  const state = {
    graph: null,
    history: [],
    currentIndex: -1,
    isPlaying: false,
    playTimer: null,
    socket: null,
    reconnectTimer: null,
    pendingLambdaConflict: null,
  };

  const controls = {};
  const logEntries = [];
  const maxLogEntries = 200;
  const visuals = {};
  let lambdaSliderSendTimer = null;
  let suppressLambdaSliderEmit = false;

  const palette = d3
    .schemeTableau10.concat(d3.schemeSet3 ?? [])
    .concat(d3.schemeCategory10 ?? []);
  const colourScale = d3.scaleOrdinal(palette);

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
    if (!controls.lambdaSlider) {
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
    if (!controls.lambdaSlider) {
      return;
    }
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
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

    if (updateGraph && state.graph) {
      state.graph.lambda_conflict = numeric;
      updateMetadata();
    }
  }

  function queueLambdaSliderSend(value) {
    if (lambdaSliderSendTimer) {
      clearTimeout(lambdaSliderSendTimer);
    }
    lambdaSliderSendTimer = setTimeout(() => {
      sendLambdaSliderUpdate(value);
    }, 150);
  }

  function sendLambdaSliderUpdate(value) {
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
      log("Fetching graph metadata…");
      await fetchGraph();
    } catch (error) {
      log("Failed to load graph metadata", { level: "error", data: String(error) });
      return;
    }
    log("Graph metadata loaded.");

    setupControls();
    setupVisuals();
    log("Controls and visuals initialised.");

    try {
      log("Fetching history file…");
      await fetchHistory();
    } catch (error) {
      log("History fetch failed", { level: "warn", data: String(error) });
    }

    log("Connecting WebSocket…");
    connectWebSocket();

    if (state.history.length) {
      log(`Loaded ${state.history.length} history entries. Jumping to latest.`);
      setCurrentIndex(state.history.length - 1);
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
      if (state.pendingLambdaConflict != null) {
        sendLambdaSliderUpdate(state.pendingLambdaConflict);
      } else if (controls.lambdaSlider) {
        const value = Number(controls.lambdaSlider.value);
        if (Number.isFinite(value)) {
          sendLambdaSliderUpdate(value);
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
    log("Controls wired.");

    if (controls.lambdaSlider) {
      configureLambdaSlider();
      controls.lambdaSlider.addEventListener("input", (event) => {
        const value = Number(event.target.value);
        if (!Number.isFinite(value)) {
          return;
        }
        if (controls.lambdaSliderValue) {
          controls.lambdaSliderValue.textContent = value.toFixed(2);
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
      controls.lambdaSlider.addEventListener("change", (event) => {
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
      .attr("r", 2.8);

    const labelGroup = viewport.append("g").attr("class", "labels");
    const labelSelection = labelGroup
      .selectAll("text")
      .data(stations, (d) => d.index)
      .join("text")
      .attr("x", (d) => xScale(-d.lon) + 8)
      .attr("y", (d) => yScale(d.lat) - 6)
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

    const zoomBehavior = d3
      .zoom()
      .scaleExtent([0.5, 8])
      .on("zoom", (event) => {
        viewport.attr("transform", event.transform);
      });

    svg.call(zoomBehavior);

    visuals.scatter = {
      svg,
      xScale,
      yScale,
      viewport,
      edgeSelection,
      nodeSelection,
      labelSelection,
      edgeData,
      zoomBehavior,
      tooltip,
    };
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

    const shouldFollow =
      state.isPlaying ||
      state.currentIndex === -1 ||
      state.currentIndex === history.length - 2;

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
    if (!state.history.length) {
      return;
    }
    const shouldPlay = typeof force === "boolean" ? force : !state.isPlaying;
    if (state.isPlaying === shouldPlay) {
      return;
    }
    state.isPlaying = shouldPlay;
    controls.playToggle.textContent = shouldPlay ? "Pause" : "Play";

    if (state.playTimer) {
      clearInterval(state.playTimer);
      state.playTimer = null;
    }

    if (shouldPlay) {
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


