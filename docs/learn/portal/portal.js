/* ============================================================================
   Learn to Build an LLM Learning Portal — engine (portal.js)

   Holds three things every module reuses:
     1. Portal registry  — modules register themselves; this builds the tabs.
     2. Net math          — forward / loss / backprop / update for a tiny network
                            (the SAME logic as the Python XOR demo, nothing hidden).
     3. Renderers/helpers — draw a network as SVG, draw a loss chart, set explanation.

   No frameworks; plain functions on a global `Portal` object so it opens from file://.
   ============================================================================ */
(function () {
  "use strict";

  // ---- tiny DOM helpers -----------------------------------------------------
  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === "class") node.className = attrs[k];
      else if (k === "html") node.innerHTML = attrs[k];
      else if (k.startsWith("on") && typeof attrs[k] === "function")
        node.addEventListener(k.slice(2), attrs[k]);
      else node.setAttribute(k, attrs[k]);
    }
    (children || []).forEach(c => node.appendChild(typeof c === "string" ? document.createTextNode(c) : c));
    return node;
  }
  function svgEl(tag, attrs) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    if (attrs) for (const k in attrs) node.setAttribute(k, attrs[k]);
    return node;
  }
  function fmt(x, d) { return (Math.round(x * Math.pow(10, d || 2)) / Math.pow(10, d || 2)).toFixed(d || 2); }

  // ==========================================================================
  // 1) NET MATH  — a generic tiny fully-connected network with sigmoid neurons
  // ==========================================================================
  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  // A seeded pseudo-random generator so "Reset" is reproducible.
  function makeRng(seed) {
    let s = seed >>> 0 || 1;
    return function () { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  }

  // Build a net from a shape like [2, 2, 1] (inputs, hidden, output).
  // Each neuron: { w:[...], b, z, a, blame, dW:[...], db }
  function makeNet(shape, seed) {
    const rng = makeRng(seed);
    const rand = () => rng() * 2 - 1;            // in [-1, 1]
    const layers = [];
    for (let l = 1; l < shape.length; l++) {
      const nIn = shape[l - 1], nOut = shape[l];
      const neurons = [];
      for (let j = 0; j < nOut; j++) {
        neurons.push({
          w: Array.from({ length: nIn }, rand),
          b: rand(),
          z: 0, a: 0, blame: 0,
          dW: Array.from({ length: nIn }, () => 0), db: 0
        });
      }
      layers.push({ neurons });
    }
    return { shape: shape.slice(), layers };
  }

  // Forward pass: fills z & a for every neuron. Returns the per-layer input vectors
  // (layerInputs[0] = x, layerInputs[l] = activations feeding layer l) and the prediction.
  function forward(net, x) {
    const layerInputs = [x.slice()];
    let inp = x;
    for (let l = 0; l < net.layers.length; l++) {
      const out = [];
      net.layers[l].neurons.forEach(n => {
        let z = n.b;
        for (let i = 0; i < n.w.length; i++) z += n.w[i] * inp[i];
        n.z = z; n.a = sigmoid(z);
        out.push(n.a);
      });
      layerInputs.push(out);
      inp = out;
    }
    return { layerInputs, pred: inp[0] };       // single output assumed for our demos
  }

  // Binary cross-entropy loss for one example.
  function loss(pred, y) {
    const eps = 1e-9;
    return -(y * Math.log(pred + eps) + (1 - y) * Math.log(1 - pred + eps));
  }

  // Backprop: fills blame (pre-activation slope) and dW/db for every neuron.
  // Assumes forward(net, x) already ran. Output layer uses the tidy (a - y).
  function backprop(net, x, y) {
    const { layerInputs } = forward(net, x);   // re-run so a/z are current
    const L = net.layers.length;

    // output layer (single neuron): blame = a - y
    const outLayer = net.layers[L - 1];
    outLayer.neurons.forEach(n => { n.blame = n.a - y; });

    // hidden layers, walking backward, reusing the next layer's blame (the baton)
    for (let l = L - 2; l >= 0; l--) {
      const layer = net.layers[l], next = net.layers[l + 1];
      layer.neurons.forEach((n, j) => {
        let s = 0;
        next.neurons.forEach(nn => { s += nn.blame * nn.w[j]; });
        n.blame = s * n.a * (1 - n.a);          // through this neuron's sigmoid bend
      });
    }

    // turn blame into weight/bias slopes
    for (let l = 0; l < L; l++) {
      const layer = net.layers[l], inp = layerInputs[l];
      layer.neurons.forEach(n => {
        for (let i = 0; i < n.w.length; i++) n.dW[i] = n.blame * inp[i];
        n.db = n.blame;
      });
    }
  }

  function applyUpdate(net, lr) {
    net.layers.forEach(layer => layer.neurons.forEach(n => {
      for (let i = 0; i < n.w.length; i++) n.w[i] -= lr * n.dW[i];
      n.b -= lr * n.db;
    }));
  }

  // One full training pass over a dataset (returns the average loss). Used by "Run epoch".
  function trainEpoch(net, data, lr) {
    let total = 0;
    data.forEach(([x, y]) => {
      const { pred } = forward(net, x);
      total += loss(pred, y);
      backprop(net, x, y);
      applyUpdate(net, lr);
    });
    return total / data.length;
  }

  // ==========================================================================
  // 2) RENDERERS  — draw a network (SVG) and a loss chart (canvas)
  // ==========================================================================

  // colour for a weight value: green positive, red negative, brighter = bigger
  function weightColor(w) {
    const mag = Math.min(1, Math.abs(w) / 3);
    return w >= 0
      ? `rgba(70, 211, 154, ${0.25 + 0.6 * mag})`
      : `rgba(255, 107, 107, ${0.25 + 0.6 * mag})`;
  }
  // brightness fill for a neuron given its activation 0..1
  function activationFill(a) {
    const t = Math.max(0, Math.min(1, a));
    const c = Math.round(30 + t * 200);
    return `rgb(${Math.round(c * 0.35)}, ${Math.round(c * 0.7)}, ${c})`;
  }

  /* Draw the network. opts:
       inputs:     array of input values (left column)
       inputNames: labels for inputs
       highlight:  { layer: idx, dir: 'fwd'|'bwd' }  (optional, to emphasise a phase)
       showBlame:  if true, show each neuron's blame instead of activation in the sublabel
  */
  function drawNetwork(container, net, opts) {
    opts = opts || {};
    const inputs = opts.inputs || [];
    const cols = [inputs.length, ...net.shape.slice(1)];   // node counts per column
    const W = 640, H = 360, padX = 70, padY = 40;
    const colX = (c) => padX + (W - 2 * padX) * (cols.length === 1 ? 0.5 : c / (cols.length - 1));
    const nodeY = (count, i) => count === 1 ? H / 2 : padY + (H - 2 * padY) * (i / (count - 1));
    const R = 22;

    const svg = svgEl("svg", { class: "net", viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "xMidYMid meet" });

    // give a human name to the node in column c, row idx (for hover tooltips)
    const lastCol = cols.length - 1;
    function nodeName(c, idx) {
      if (c === 0) return '"' + ((opts.inputNames && opts.inputNames[idx]) || ("x" + (idx + 1))) + '"';
      if (c === lastCol) return "the output neuron";
      return "hidden neuron H" + (idx + 1);
    }
    // attach an SVG <title> (native hover tooltip) to a node
    function tip(node, text) { const t = svgEl("title"); t.textContent = text; node.appendChild(t); return node; }

    // --- connections (weights) drawn first so nodes sit on top ---
    for (let l = 0; l < net.layers.length; l++) {
      const fromCount = cols[l], toCount = cols[l + 1];
      const hot = opts.highlight && opts.highlight.layer === l;
      net.layers[l].neurons.forEach((n, j) => {
        for (let i = 0; i < n.w.length; i++) {
          const x1 = colX(l), y1 = nodeY(fromCount, i);
          const x2 = colX(l + 1), y2 = nodeY(toCount, j);
          const line = svgEl("line", {
            x1, y1, x2, y2,
            stroke: weightColor(n.w[i]),
            "stroke-width": (1 + Math.min(6, Math.abs(n.w[i]) * 2)),
            opacity: hot ? 1 : 0.85
          });
          const wTip = `This number is the WEIGHT from ${nodeName(l, i)} to ${nodeName(l + 1, j)} = ${fmt(n.w[i])}. ` +
            (n.w[i] >= 0 ? "Positive (green): it pushes the next neuron up." : "Negative (red): it pushes the next neuron down.") +
            " Bigger = stronger connection (thicker line). This is one of the dials training tunes — it changes as the net learns.";
          tip(line, wTip);
          svg.appendChild(line);
          // weight value label near the midpoint (same tooltip on the number itself)
          const lbl = svgEl("text", {
            x: (x1 + x2) / 2, y: (y1 + y2) / 2 - 3, class: "net-weight-label",
            "text-anchor": "middle"
          });
          lbl.textContent = fmt(n.w[i], 2);
          tip(lbl, wTip);
          svg.appendChild(lbl);
        }
      });
    }

    // --- input nodes ---
    for (let i = 0; i < cols[0]; i++) {
      const x = colX(0), y = nodeY(cols[0], i);
      const nameI = (opts.inputNames && opts.inputNames[i]) || ("x" + (i + 1));
      const iTip = `This number is the value of input ${nameI} = ${fmt(inputs[i] || 0)}. Inputs come from the data (they describe the example on stage) — they are NOT learned.`;
      const circ = svgEl("circle", { cx: x, cy: y, r: R, fill: activationFill(inputs[i] || 0), stroke: "#33404f" });
      tip(circ, iTip);
      svg.appendChild(circ);
      const v = svgEl("text", { x, y: y + 4, class: "net-node-label", "text-anchor": "middle" });
      v.textContent = fmt(inputs[i] || 0, 2);
      tip(v, iTip);
      svg.appendChild(v);
      const nm = svgEl("text", { x, y: y - R - 6, class: "net-sub-label", "text-anchor": "middle" });
      nm.textContent = nameI;
      tip(nm, iTip);
      svg.appendChild(nm);
    }

    // --- neuron nodes (hidden + output) ---
    for (let l = 0; l < net.layers.length; l++) {
      const count = cols[l + 1], hot = opts.highlight && opts.highlight.layer === l;
      const isOut = (l === net.layers.length - 1);
      net.layers[l].neurons.forEach((n, j) => {
        const x = colX(l + 1), y = nodeY(count, j);
        const circ = svgEl("circle", {
          cx: x, cy: y, r: R, fill: activationFill(n.a),
          stroke: hot ? (opts.highlight.dir === "bwd" ? "#ffd24e" : "#4ea1ff") : "#33404f",
          "stroke-width": hot ? 3 : 1
        });
        const role = isOut
          ? `This number is the OUTPUT neuron's activation = ${fmt(n.a)} — the network's prediction (0–1; round for yes/no). It comes from: weighted-sum the two hidden values (z = ${fmt(n.z)}), then the sigmoid bend.`
          : `This number is hidden neuron H${j + 1}'s activation = ${fmt(n.a)}. It comes from: weighted-sum the inputs (z = ${fmt(n.z)}), then the sigmoid bend. The two hidden numbers are the network's re-description of the input.`;
        const nTip = role + (opts.showBlame ? ` Blame = ${fmt(n.blame, 3)} (how much this neuron is responsible for the loss).` : "");
        tip(circ, nTip);
        svg.appendChild(circ);
        const v = svgEl("text", { x, y: y + 4, class: "net-node-label", "text-anchor": "middle" });
        v.textContent = fmt(n.a, 2);
        tip(v, nTip);
        svg.appendChild(v);
        // sublabel: either activation detail or blame (during backprop)
        const sub = svgEl("text", { x, y: y - R - 6, class: "net-sub-label", "text-anchor": "middle" });
        sub.textContent = opts.showBlame
          ? "blame " + fmt(n.blame, 3)
          : (isOut ? "output" : "h" + (j + 1));
        tip(sub, opts.showBlame
          ? `BLAME on ${isOut ? "the output neuron" : "hidden neuron H" + (j + 1)} = ${fmt(n.blame, 3)} — how much this neuron is responsible for the loss. Backprop computes it so each weight knows which way to nudge.`
          : nTip);
        svg.appendChild(sub);
      });
    }

    container.innerHTML = "";
    container.appendChild(svg);
  }

  // Loss chart: simple line over the recorded history.
  function drawLossChart(canvas, history) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 400, h = canvas.clientHeight || 160;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    // axes
    ctx.strokeStyle = "#33404f"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(34, 8); ctx.lineTo(34, h - 20); ctx.lineTo(w - 8, h - 20); ctx.stroke();
    ctx.fillStyle = "#9aa7b4"; ctx.font = "10px Segoe UI";
    if (!history.length) { ctx.fillText("loss will appear here as you train", 44, h / 2); return; }
    const maxL = Math.max(...history, 0.001);
    const x0 = 34, x1 = w - 8, y0 = h - 20, y1 = 8;
    ctx.fillText(fmt(maxL, 2), 4, y1 + 8);
    ctx.fillText("0", 22, y0);
    ctx.strokeStyle = "#4ea1ff"; ctx.lineWidth = 2; ctx.beginPath();
    history.forEach((L, i) => {
      const x = x0 + (x1 - x0) * (history.length === 1 ? 0 : i / (history.length - 1));
      const y = y0 - (y0 - y1) * (L / maxL);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = "#9aa7b4";
    ctx.fillText("epochs →", w - 60, h - 6);
  }

  // ==========================================================================
  // 3) PORTAL REGISTRY  — tabs + module rendering
  // ==========================================================================
  const modules = [];
  let activeId = null;

  function register(mod) { modules.push(mod); }

  function renderTabs() {
    const bar = document.getElementById("tabbar");
    bar.innerHTML = "";
    modules.forEach(m => {
      bar.appendChild(el("button", {
        class: "tab" + (m.id === activeId ? " active" : ""),
        onclick: () => selectModule(m.id)
      }, [m.tab || m.title]));
    });
  }

  function selectModule(id) {
    activeId = id;
    renderTabs();
    const mod = modules.find(m => m.id === id);
    const content = document.getElementById("content");
    content.innerHTML = "";
    if (mod.intro) content.appendChild(el("p", { class: "module-intro" }, [mod.intro]));
    const mount = el("div", {});
    content.appendChild(mount);
    mod.render(mount, Portal);            // hand the module the engine API
  }

  function boot() {
    if (!modules.length) return;
    renderTabs();
    selectModule(modules[0].id);
  }

  // public API handed to modules
  const Portal = {
    register, boot,
    // math
    makeNet, forward, loss, backprop, applyUpdate, trainEpoch, sigmoid,
    // dom + render helpers
    el, svgEl, fmt, drawNetwork, drawLossChart, weightColor, activationFill
  };
  window.Portal = Portal;
})();
