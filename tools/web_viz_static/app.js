(function () {
  const state = {
    graph: null,
    history: [],
    currentIndex: -1,
    isPlaying: false,
    playTimer: null,
    socket: null,
    reconnectTimer: null,
  };

  const controls = {};
  const visuals = {};

  const palette = d3
    .schemeTableau10.concat(d3.schemeSet3 ?? [])
    .concat(d3.schemeCategory10 ?? []);
  const colourScale = d3.scaleOrdinal(palette);

  document.addEventListener("DOMContentLoaded", init);

  async function init() {
    try {
      await fetchGraph();
    } catch (error) {
      console.error("Failed to load graph metadata", error);
      return;
    }

    setupControls();
    setupVisuals();

    try {
      await fetchHistory();
    } catch (error) {
      console.warn("History fetch failed", error);
    }

    connectWebSocket();

    if (state.history.length) {
      setCurrentIndex(state.history.length - 1);
    } else {
      updateControlsAvailability();
    }
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

    socket.addEventListener("message", (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === "state") {
          pushState(message);
        }
      } catch (error) {
        console.warn("Received malformed websocket payload", error);
      }
    });

    socket.addEventListener("close", () => {
      state.socket = null;
      if (state.reconnectTimer) {
        clearTimeout(state.reconnectTimer);
      }
      state.reconnectTimer = setTimeout(connectWebSocket, 2000);
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

    const lonRange = state.graph.lon_range ?? [0, 1];
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

    const edgesGroup = svg.append("g").attr("class", "edges");
    const edgeSelection = edgesGroup
      .selectAll("line")
      .data(edgeData, (d) => d.index)
      .join("line")
      .attr("x1", (d) => xScale(d.source.lon))
      .attr("y1", (d) => yScale(d.source.lat))
      .attr("x2", (d) => xScale(d.target.lon))
      .attr("y2", (d) => yScale(d.target.lat));

    const nodesGroup = svg.append("g").attr("class", "nodes");
    const nodeSelection = nodesGroup
      .selectAll("circle")
      .data(stations, (d) => d.index)
      .join("circle")
      .attr("cx", (d) => xScale(d.lon))
      .attr("cy", (d) => yScale(d.lat))
      .attr("r", 5.5);

    const labelGroup = svg.append("g").attr("class", "labels");
    const labelSelection = labelGroup
      .selectAll("text")
      .data(stations, (d) => d.index)
      .join("text")
      .attr("x", (d) => xScale(d.lon) + 8)
      .attr("y", (d) => yScale(d.lat) - 6)
      .text((d) => d.city);

    visuals.scatter = {
      svg,
      xScale,
      yScale,
      edgeSelection,
      nodeSelection,
      labelSelection,
      edgeData,
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
    const yScale = d3.scaleLog().range([height - margin.bottom, margin.top]);

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
      .text("Energy (log)");

    const line = d3
      .line()
      .x((d) => xScale(d.step))
      .y((d) => yScale(Math.max(1e-6, d.energy)))
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
    const energies = state.history.map((d) => Math.max(1e-6, d.energy));
    const minStep = d3.min(steps) ?? 0;
    const maxStep = d3.max(steps) ?? minStep + 1;
    const minEnergy = d3.min(energies) ?? 1e-6;
    const maxEnergy = d3.max(energies) ?? minEnergy;

    energy.xScale.domain([minStep, maxStep === minStep ? minStep + 1 : maxStep]);
    const lower = Math.max(1e-6, minEnergy * 0.95);
    const upper = Math.max(1e-6, maxEnergy * 1.05);
    energy.yScale.domain([lower, upper]);

    energy.path.datum(state.history).attr("d", energy.line);
    energy.xAxisGroup.call(d3.axisBottom(energy.xScale).ticks(6).tickFormat(d3.format(",d")));
    energy.yAxisGroup.call(d3.axisLeft(energy.yScale).ticks(6, "~g"));
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
      .attr("cy", energy.yScale(Math.max(1e-6, entry.energy)));
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
    meta.textContent = [
      state.graph.run_name ?? "Unnamed run",
      `${state.graph.station_count} stations`,
      `${edgeCount} edges`,
    ].join(" • ");
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


