/* ============================================================================
   Module 2 tab — Neural Networks (interactive demos)

   Three demos, chosen from a sub-menu:
     1. Single neuron        — sliders; watch the weighted sum + sigmoid live.
     2. XOR network          — the 2-2-1 net, stepped phase-by-phase with narration.
     3. "Should I go outside?" — a single neuron on meaningful, named inputs.

   Registers itself with the portal engine via Portal.register({...}).
   ============================================================================ */
(function () {
  "use strict";

  Portal.register({
    id: "module2",
    tab: "Module 2 — Neural Networks",
    title: "Module 2 — Neural Networks",
    intro: "Watch a network think. Pick a demo below — start with one neuron, then see the full " +
           "XOR network learn step by step, with every number explained as it happens.",
    render: function (mount, P) {
      // sub-menu state
      const demos = [
        { id: "neuron", name: "1 · Single neuron", build: buildNeuronDemo },
        { id: "xor",    name: "2 · XOR network (step-by-step)", build: buildXorDemo },
        { id: "outside",name: "3 · “Should I go outside?”", build: buildOutsideDemo }
      ];
      let current = "neuron";   // open on demo 1 (the single neuron)

      const switcher = P.el("div", { class: "demo-switch" });
      const host = P.el("div", {});
      function renderSwitch() {
        switcher.innerHTML = "";
        demos.forEach(d => switcher.appendChild(P.el("button", {
          class: "demo-chip" + (d.id === current ? " active" : ""),
          onclick: () => { current = d.id; renderSwitch(); host.innerHTML = ""; demos.find(x => x.id === current).build(host, P); }
        }, [d.name])));
      }
      renderSwitch();
      mount.appendChild(switcher);
      mount.appendChild(host);
      demos.find(d => d.id === current).build(host, P);
    }
  });

  // =========================================================================
  // DEMO 1 — single neuron
  // =========================================================================
  function buildNeuronDemo(host, P) {
    const net = P.makeNet([2, 1], 7);
    // make it a clear illustrative neuron
    net.layers[0].neurons[0].w = [0.8, -0.6];
    net.layers[0].neurons[0].b = 0.1;
    let inputs = [1.0, 0.5];

    const panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["A single neuron"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["Five stages, start to finish: inputs → × weights → sum → + bias → the activation bend. " +
       "Drag anything and watch each stage react."]));

    const guideN = P.el("details", { class: "guide" });
    guideN.innerHTML =
      '<summary>What am I looking at?</summary>' +
      '<div class="guide-body">' +
        'A neuron is the smallest unit of a network. It turns its inputs into one output number in <b>five steps</b>:<br>' +
        '<b>1 · Inputs</b> arrive (from the data).<br>' +
        '<b>2 · × weights</b>: multiply each input by its own weight. The weight sets how much that input matters — ' +
        '<code>0</code> = ignore it, a <b>negative</b> weight = it pushes the output the other way.<br>' +
        '<b>3 · Sum</b>: add the weighted inputs into one number — the <b>weighted sum</b> (a dot product).<br>' +
        '<b>4 · + bias</b>: add the bias — a single shift up or down. This gives <code>z</code>.<br>' +
        '<b>5 · Activation bend</b>: pass <code>z</code> through the activation (here the <b>sigmoid</b>; ReLU is another ' +
        'common one) → the final output, squeezed to 0–1 (near 1 = “yes”, near 0 = “no”, 0.5 = “unsure”).<br><br>' +
        '<b>The key split:</b> steps 2–4 are the <b>straight-line</b> part — multiplying, summing, and shifting can only ever ' +
        'make a straight-line relationship (the <b>bias slides</b> the line; it never bends it). <b>Step 5 is the only bend.</b> ' +
        'That’s why every neuron needs an activation: stack a thousand straight lines and you still just get a straight line — ' +
        'the bends are what let a network learn curvy, real-world patterns.<br><br>' +
        '<b>Inputs</b> come from the data (they change per situation). <b>Weights & bias</b> are the neuron’s own dials — ' +
        'in a real network these are learned by training; here you set them by hand to feel how they work. ' +
        'The two left circles are the inputs; the right circle is the neuron’s output. Drag the sliders and watch each stage in the panel.' +
      '</div>';
    panel.appendChild(guideN);

    const layout = P.el("div", { class: "layout" });
    const stage = P.el("div", {});
    const side = P.el("div", {});
    layout.appendChild(stage); layout.appendChild(side);
    panel.appendChild(layout);

    const netBox = P.el("div", {});
    stage.appendChild(netBox);

    // sliders
    function slider(name, val, min, max, step, onInput, tip) {
      const row = P.el("div", { class: "slider-row", title: tip || "" });
      const valEl = P.el("span", { class: "val" }, [P.fmt(val, 2)]);
      const input = P.el("input", { type: "range", min, max, step, value: val, title: tip || "" });
      input.addEventListener("input", () => { valEl.textContent = P.fmt(parseFloat(input.value), 2); onInput(parseFloat(input.value)); });
      row.appendChild(P.el("span", { class: "name", title: tip || "" }, [name]));
      row.appendChild(input);
      row.appendChild(valEl);
      return row;
    }
    const n = net.layers[0].neurons[0];
    side.appendChild(P.el("div", { class: "explain" }, [P.el("div", { class: "phase-tag" }, ["inputs (data)"])]));
    side.appendChild(slider("input x1", inputs[0], -2, 2, 0.1, v => { inputs[0] = v; redraw(); }, "Input 1: part of the data fed to the neuron. Changes the weighted sum."));
    side.appendChild(slider("input x2", inputs[1], -2, 2, 0.1, v => { inputs[1] = v; redraw(); }, "Input 2: part of the data fed to the neuron."));
    side.appendChild(P.el("div", { class: "hint" }, ["weights (the neuron's learned opinion):"]));
    side.appendChild(slider("weight w1", n.w[0], -3, 3, 0.1, v => { n.w[0] = v; redraw(); }, "Weight on input 1: how much input 1 matters. Positive = pushes output up, negative = down. In a real net this is LEARNED."));
    side.appendChild(slider("weight w2", n.w[1], -3, 3, 0.1, v => { n.w[1] = v; redraw(); }, "Weight on input 2: how much input 2 matters."));
    side.appendChild(slider("bias b", n.b, -3, 3, 0.1, v => { n.b = v; redraw(); }, "Bias: a single number added to the weighted sum no matter what the inputs are — the neuron's built-in lean. Positive = fires more easily, negative = harder to fire. It only SLIDES the result; it never bends it. A learned dial, like the weights."));

    const explain = P.el("div", { class: "explain" });
    side.appendChild(explain);

    function redraw() {
      const r = P.forward(net, inputs);
      P.drawNetwork(netBox, net, { inputs, inputNames: ["x1", "x2"] });
      const p1 = inputs[0] * n.w[0], p2 = inputs[1] * n.w[1];   // each input × its weight
      const wsum = p1 + p2;                                      // the weighted sum (no bias yet)
      const z = n.z;                                             // wsum + bias
      explain.innerHTML =
        '<div class="phase-tag">forward · the five stages</div>' +
        '<div class="stage-list">' +
          `<div><b>1 · inputs</b> &nbsp; <code>x1 = ${P.fmt(inputs[0])}, &nbsp; x2 = ${P.fmt(inputs[1])}</code></div>` +
          `<div><b>2 · × weights</b> &nbsp; <code>x1×w1 = ${P.fmt(inputs[0])}×${P.fmt(n.w[0])} = ${P.fmt(p1)}</code>, ` +
            `<code>x2×w2 = ${P.fmt(inputs[1])}×${P.fmt(n.w[1])} = ${P.fmt(p2)}</code></div>` +
          `<div><b>3 · sum</b> &nbsp; <code>${P.fmt(p1)} + ${P.fmt(p2)} = ${P.fmt(wsum)}</code> &nbsp;<span class="hint">(the weighted sum)</span></div>` +
          `<div title="BIAS — a single number the neuron adds to the weighted sum, no matter what the inputs are. It is the neuron's built-in lean: a positive bias makes it fire more easily, a negative bias makes it harder to fire. Adding a number can only SLIDE the result up or down — it never bends the line. Like weights, the bias is a learned dial."><b>4 · + bias</b> &nbsp; <code>${P.fmt(wsum)} + ${P.fmt(n.b)} = ${P.fmt(z)}</code> &nbsp;<span class="hint">(still a straight line — bias just shifts it)</span></div>` +
          `<div title="ACTIVATION BEND — the function applied last, to the number z. Here it is the sigmoid σ, which squeezes any z into the 0–1 range (big z → near 1, very negative z → near 0, z=0 → 0.5). It is the ONLY step that bends the straight line into a curve — the non-linearity. Without it, stacking neurons would still only make straight lines. ReLU is another common activation."><b>5 · activation bend</b> &nbsp; <code>σ(${P.fmt(z)}) = ${P.fmt(r.pred)}</code> &nbsp;<span class="hint">(the only bend → final output)</span></div>` +
        '</div>' +
        `<span class="hint">Steps 2–4 can only make a straight line; step 5 is what curves it. ` +
        `A big positive z → output near 1 (“yes”); big negative → near 0 (“no”); z = 0 → 0.5 (“unsure”).</span>`;
    }
    host.appendChild(panel);
    redraw();
  }

  // =========================================================================
  // DEMO 2 — XOR network, stepped phase by phase
  // =========================================================================
  function buildXorDemo(host, P) {
    const DATA = [
      [[0, 0], 0], [[0, 1], 1], [[1, 0], 1], [[1, 1], 0]
    ];
    const LR = 1.0;
    // seeds that reliably escape XOR's local minima at this learning rate
    const GOOD_SEEDS = [5, 7, 10, 12, 13, 2, 16, 18, 19];
    let seedPtr = 0;
    let net = P.makeNet([2, 2, 1], GOOD_SEEDS[seedPtr]);
    let exampleIdx = 0;                 // start on example 1 = (0,0)
    let lossHistory = [];
    let epochCount = 0;
    let autoTimer = null;

    // phase walk for ONE example
    const PHASES = ["ready", "fwd-hidden", "fwd-output", "loss", "bwd-output", "bwd-hidden", "update"];
    let phase = 0;

    const panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["The XOR network — learning, one step at a time"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["2 inputs → 2 hidden neurons → 1 output. XOR = “output 1 only when the inputs differ.” " +
       "A single layer can’t do this — the hidden layer’s bend is what makes it possible."]));

    // -- in-portal guide: what am I looking at? --
    const guide = P.el("details", { class: "guide" });
    guide.innerHTML =
      '<summary>What am I looking at?  (read me first)</summary>' +
      '<div class="guide-body">' +
        '<h4>The task: XOR</h4>' +
        'XOR is a rule about <b>two inputs</b>, each either <code>0</code> or <code>1</code>. The rule: ' +
        '<b>output 1 when the two inputs are different, 0 when they are the same.</b> ' +
        'There are only four possible inputs in the whole world — that is the entire dataset:' +
        '<table><tr><th>inputs</th><th>same or different?</th><th>correct output</th></tr>' +
          '<tr><td><code>(0, 0)</code></td><td>same (both 0)</td><td>0</td></tr>' +
          '<tr><td><code>(0, 1)</code></td><td>different</td><td>1</td></tr>' +
          '<tr><td><code>(1, 0)</code></td><td>different</td><td>1</td></tr>' +
          '<tr><td><code>(1, 1)</code></td><td>same (both 1)</td><td>0</td></tr></table>' +
        'The four buttons below are these four examples. <code>(0,1)→1</code> means “inputs 0 and 1, ' +
        'the correct answer is 1.” The network sees <b>one example at a time</b> — the one “on stage.”' +
        '<h4>What the picture means</h4>' +
        '<b>Left column</b> = the two inputs of the example on stage. <b>Middle</b> = the 2 hidden ' +
        'neurons (the network’s scratch work). <b>Right</b> = the output neuron — its number is the ' +
        'network’s prediction (a confidence between 0 and 1; round it to get a yes/no).' +
        '<h4>How the network learns</h4>' +
        'Each press of <b>Next step ▶</b> walks one example through four stages: ' +
        '<code>forward</code> (compute a guess) → <code>loss</code> (how wrong) → ' +
        '<code>backward</code> (blame flows right→left) → <code>update</code> (weights nudge). ' +
        'One pass through all four examples = one <b>epoch</b>. The buttons <b>+100 / +500 epochs</b> ' +
        'do many passes fast — watch the loss chart fall and the predictions table flip to all-correct.' +
      '</div>';
    panel.appendChild(guide);

    // example chooser
    const exRow = P.el("div", { class: "controls" });
    exRow.appendChild(P.el("span", { class: "name" }, ["Example on stage:"]));
    DATA.forEach((d, i) => {
      const same = d[0][0] === d[0][1];
      exRow.appendChild(P.el("button", {
        class: "demo-chip" + (i === exampleIdx ? " active" : ""),
        "data-ex": i,
        title: `inputs ${d[0][0]} and ${d[0][1]} are ${same ? "the SAME" : "DIFFERENT"} → correct answer ${d[1]}`,
        onclick: () => { exampleIdx = i; phase = 0; refreshExChips(); render(); }
      }, [`(${d[0][0]},${d[0][1]})→${d[1]}`]));
    });
    panel.appendChild(exRow);
    panel.appendChild(P.el("p", { class: "caption" },
      ["Each button is one of XOR’s four cases: the two numbers are the inputs, the number after “→” is the correct answer. Click one to load it into the network."]));
    function refreshExChips() {
      [...exRow.querySelectorAll("[data-ex]")].forEach(b =>
        b.classList.toggle("active", parseInt(b.getAttribute("data-ex")) === exampleIdx));
    }

    const layout = P.el("div", { class: "layout" });
    const stage = P.el("div", {});
    const side = P.el("div", {});
    layout.appendChild(stage); layout.appendChild(side);
    panel.appendChild(layout);

    const netBox = P.el("div", {});
    stage.appendChild(netBox);
    netBox.title = "The network. Hover any neuron (circle) or weight (line) for what it is and its current value.";
    const legend = P.el("div", { class: "legend" });
    legend.innerHTML =
      '<span class="item"><span class="swatch" style="background:#cfe6ff"></span> bright neuron = high activation (~1)</span>' +
      '<span class="item"><span class="swatch" style="background:#19324a"></span> dark neuron = low (~0)</span>' +
      '<span class="item"><span class="swatch line" style="background:#46d39a"></span> green line = positive weight</span>' +
      '<span class="item"><span class="swatch line" style="background:#ff6b6b"></span> red line = negative weight</span>' +
      '<span class="item">thicker line = bigger weight. Numbers on nodes/lines are their current values.</span>';
    stage.appendChild(legend);
    const lossCanvas = P.el("canvas", { class: "loss", title: "Average loss across all four examples, one point per epoch. It should fall toward 0 as the network learns." });
    stage.appendChild(P.el("div", { class: "hint" }, ["Average loss per epoch (lower = better):"]));
    stage.appendChild(lossCanvas);

    // step controls
    const ctr = P.el("div", { class: "controls" });
    const btnStep = P.el("button", { class: "btn primary", title: "Advance ONE phase (forward → loss → backward → update) for the example on stage, with narration.", onclick: nextStep }, ["Next step ▶"]);
    const btnAuto = P.el("button", { class: "btn", title: "Play the steps automatically at the chosen speed. Click again to pause.", onclick: toggleAuto }, ["Auto ▶▶"]);
    const btnEpoch = P.el("button", { class: "btn", title: "Train for 1 full pass over all four examples (no animation), then update the loss chart.", onclick: () => { runEpochs(1); } }, ["+1 epoch"]);
    const btnEpoch100 = P.el("button", { class: "btn", title: "Train 100 epochs fast — watch the loss chart start to fall.", onclick: () => { runEpochs(100); } }, ["+100 epochs"]);
    const btnEpoch500 = P.el("button", { class: "btn", title: "Train 500 epochs — usually enough to solve XOR (predictions flip to all-correct).", onclick: () => { runEpochs(500); } }, ["+500 epochs"]);
    const btnReset = P.el("button", { class: "btn", title: "Start over from a fresh (known-good) random set of weights and clear the loss chart.", onclick: reset }, ["Reset (new start)"]);
    const speed = P.el("input", { type: "range", min: 120, max: 1200, step: 60, value: 600, title: "How fast Auto plays the steps." });
    ctr.appendChild(btnStep); ctr.appendChild(btnAuto);
    ctr.appendChild(btnEpoch); ctr.appendChild(btnEpoch100); ctr.appendChild(btnEpoch500); ctr.appendChild(btnReset);
    ctr.appendChild(P.el("label", { title: "Drag to set the Auto-play speed." }, ["speed", speed]));
    side.appendChild(ctr);

    const explain = P.el("div", { class: "explain" });
    side.appendChild(explain);

    // live stats
    const stats = P.el("div", { class: "stats" });
    const statEpoch = stat("epoch"); const statLoss = stat("avg loss"); const statPhase = stat("phase");
    stats.appendChild(statEpoch.box); stats.appendChild(statLoss.box); stats.appendChild(statPhase.box);
    side.appendChild(stats);
    function stat(k) { const v = P.el("div", { class: "v" }, ["—"]); return { box: P.el("div", { class: "stat" }, [P.el("div", { class: "k" }, [k]), v]), v }; }

    // predictions table
    const predWrap = P.el("div", {});
    side.appendChild(predWrap);

    panel.appendChild(P.el("div", { class: "hint" },
      ["Tip: press “Next step” to walk forward → loss → backward → update for the example on stage. " +
       "“Run 10 epochs” trains fast and updates the loss chart."]));
    host.appendChild(panel);

    // ---- step machine -------------------------------------------------------
    function nextStep() {
      phase++;
      if (phase >= PHASES.length) {
        // finished this example's update → advance to next example, count epoch after the 4th
        exampleIdx = (exampleIdx + 1) % DATA.length;
        if (exampleIdx === 0) recordEpoch();
        phase = 1; // straight into forward of the next example
        refreshExChips();
      }
      const name = PHASES[phase];
      const [x, y] = DATA[exampleIdx];

      if (name === "fwd-hidden" || name === "fwd-output") P.forward(net, x);
      if (name === "bwd-output" || name === "bwd-hidden") P.backprop(net, x, y);
      if (name === "update") { P.applyUpdate(net, LR); }
      render();
    }

    function recordEpoch() {
      // compute the true average loss across all 4 examples at current weights
      let tot = 0; DATA.forEach(([x, y]) => { tot += P.loss(P.forward(net, x).pred, y); });
      lossHistory.push(tot / DATA.length); epochCount++;
    }

    function runEpochs(k) {
      stopAuto();
      for (let e = 0; e < k; e++) { P.trainEpoch(net, DATA, LR); recordEpoch(); }
      phase = 0;
      render();
    }

    function reset() {
      stopAuto();
      seedPtr = (seedPtr + 1) % GOOD_SEEDS.length;   // cycle known-good starting points
      net = P.makeNet([2, 2, 1], GOOD_SEEDS[seedPtr]);
      lossHistory = []; epochCount = 0; exampleIdx = 0; phase = 0;
      refreshExChips(); render();
    }

    function toggleAuto() {
      if (autoTimer) { stopAuto(); return; }
      btnAuto.textContent = "Pause ⏸";
      const tick = () => { nextStep(); autoTimer = setTimeout(tick, 1320 - parseInt(speed.value)); };
      autoTimer = setTimeout(tick, 1320 - parseInt(speed.value));
    }
    function stopAuto() { if (autoTimer) { clearTimeout(autoTimer); autoTimer = null; btnAuto.textContent = "Auto ▶▶"; } }

    // ---- rendering ----------------------------------------------------------
    function render() {
      const name = PHASES[phase];
      const [x, y] = DATA[exampleIdx];
      const isBwd = name.startsWith("bwd");
      let highlight = null;
      if (name === "fwd-hidden") highlight = { layer: 0, dir: "fwd" };
      if (name === "fwd-output") highlight = { layer: 1, dir: "fwd" };
      if (name === "bwd-output") highlight = { layer: 1, dir: "bwd" };
      if (name === "bwd-hidden") highlight = { layer: 0, dir: "bwd" };

      // make sure activations reflect the staged example
      if (name === "ready") { net.layers.forEach(l => l.neurons.forEach(nn => { nn.a = 0; })); }
      else P.forward(net, x);

      P.drawNetwork(netBox, net, { inputs: x, inputNames: ["x1", "x2"], highlight, showBlame: isBwd });
      P.drawLossChart(lossCanvas, lossHistory);

      statEpoch.v.textContent = epochCount;
      statLoss.v.textContent = lossHistory.length ? P.fmt(lossHistory[lossHistory.length - 1], 4) : "—";
      statPhase.v.textContent = prettyPhase(name);

      explain.innerHTML = explainFor(name, net, x, y);
      renderPreds();
    }

    function prettyPhase(name) {
      return ({ "ready": "ready", "fwd-hidden": "forward", "fwd-output": "forward", "loss": "loss",
                "bwd-output": "backward", "bwd-hidden": "backward", "update": "update" })[name];
    }

    function explainFor(name, net, x, y) {
      const tag = (t) => `<div class="phase-tag">${t}</div>`;
      const h = net.layers[0].neurons, o = net.layers[1].neurons[0];
      const pred = o.a;
      if (name === "ready") {
        const same = x[0] === x[1];
        return tag("ready") +
          `On stage: inputs <code>(${x[0]}, ${x[1]})</code>. These two are <b>${same ? "the same" : "different"}</b>, ` +
          `so by the XOR rule the correct answer is <code>${y}</code>. ` +
          `Press <b>Next step ▶</b> to push this example through the network and see what it currently guesses.`;
      }
      if (name === "fwd-hidden")
        return tag("forward · hidden layer") +
          `Each hidden neuron does a weighted sum of the inputs, then the sigmoid bend:<br>` +
          `H1: <code>σ(${P.fmt(h[0].z)}) = ${P.fmt(h[0].a)}</code> &nbsp; H2: <code>σ(${P.fmt(h[1].z)}) = ${P.fmt(h[1].a)}</code><br>` +
          `<span class="hint">These two numbers are the network’s re-description of the input.</span>`;
      if (name === "fwd-output")
        return tag("forward · output") +
          `The output neuron takes the two hidden numbers (<code>${P.fmt(h[0].a)}, ${P.fmt(h[1].a)}</code>), ` +
          `weighted-sums them (+bias) and bends:<br><code>prediction = σ(${P.fmt(o.z)}) = ${P.fmt(pred)}</code>`;
      if (name === "loss")
        return tag("loss") +
          `How wrong was it? True answer <code>${y}</code>, predicted <code>${P.fmt(pred)}</code>.<br>` +
          `cross-entropy loss = <code>${P.fmt(P.loss(pred, y), 4)}</code> ` +
          `<span class="hint">(0 = perfect; bigger = worse).</span>`;
      if (name === "bwd-output")
        return tag("backward · output") +
          `Backprop starts at the end. The output’s blame is simply <code>pred − true = ${P.fmt(pred)} − ${y} = ${P.fmt(o.blame, 3)}</code>. ` +
          `<span class="hint">Positive blame = output was too high; it will be pulled down.</span>`;
      if (name === "bwd-hidden")
        return tag("backward · hidden") +
          `The blame is passed <b>back</b> to the hidden neurons (each gets output-blame × its weight, through its own bend):<br>` +
          `H1 blame <code>${P.fmt(h[0].blame, 3)}</code> &nbsp; H2 blame <code>${P.fmt(h[1].blame, 3)}</code><br>` +
          `<span class="hint">This is the “baton” of blame flowing right → left.</span>`;
      if (name === "update")
        return tag("update") +
          `Every weight steps a little opposite its slope: <code>w ← w − ${LR} × slope</code>. ` +
          `The weights just nudged toward a lower loss. <span class="hint">One tiny step — repeat thousands of times.</span>`;
      return "";
    }

    function renderPreds() {
      const rows = DATA.map(([x, y]) => {
        const p = P.forward(net, x).pred;
        const ok = Math.round(p) === y;
        return `<tr><td>(${x[0]}, ${x[1]})</td><td>${y}</td><td>${P.fmt(p, 3)}</td>` +
               `<td class="${ok ? "ok" : "bad"}">${ok ? "OK" : "off"}</td></tr>`;
      }).join("");
      predWrap.innerHTML =
        `<div class="hint" style="margin-top:14px">Current predictions (want ~0/1/1/0):</div>` +
        `<table class="preds"><tr><th>input</th><th>true</th><th>pred</th><th></th></tr>${rows}</table>`;
    }

    render();
  }

  // =========================================================================
  // DEMO 3 — "Should I go outside?" neuron (named, meaningful inputs)
  // =========================================================================
  function buildOutsideDemo(host, P) {
    // a single neuron with hand-set, interpretable weights
    const net = P.makeNet([3, 1], 3);
    net.layers[0].neurons[0].w = [0.9, -3.0, 1.2];  // temp(+), raining(--), weekend(+)
    net.layers[0].neurons[0].b = -0.5;
    const n = net.layers[0].neurons[0];
    // inputs: temperature (scaled 0..1 ~ cold..hot), raining(0/1), weekend(0/1)
    let temp = 0.7, raining = 0, weekend = 1;

    const panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["A real decision: “Should I go outside?”"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["One neuron, but the inputs mean something. The weights encode an opinion: warmth helps a little, " +
       "rain kills it, the weekend helps. Change the situation and watch the confidence move."]));

    const guideO = P.el("details", { class: "guide" });
    guideO.innerHTML =
      '<summary>What am I looking at?</summary>' +
      '<div class="guide-body">' +
        'This is the same single neuron as demo 1, but the inputs have <b>real meanings</b>:<br>' +
        '<b>temperature</b> (0 = freezing, 1 = hot), <b>raining</b> (0 = no, 1 = yes), <b>weekend</b> (0 = no, 1 = yes).<br>' +
        'The neuron’s <b>weights</b> are its opinion: temperature <code>+0.9</code> (warmth helps a little), ' +
        'raining <code>−3.0</code> (rain strongly hurts), weekend <code>+1.2</code> (weekend helps). ' +
        'It weighted-sums the situation, bends it with the sigmoid, and the output is its <b>confidence</b> that you should go ' +
        'outside (over 0.5 = yes). Drag the sliders — flipping <b>raining</b> on its own can flip the whole decision, because its weight is so negative.' +
      '</div>';
    panel.appendChild(guideO);

    const layout = P.el("div", { class: "layout" });
    const stage = P.el("div", {}); const side = P.el("div", {});
    layout.appendChild(stage); layout.appendChild(side);
    panel.appendChild(layout);
    const netBox = P.el("div", {}); stage.appendChild(netBox);

    function slider(name, val, min, max, step, onInput, tip) {
      const row = P.el("div", { class: "slider-row", title: tip || "" });
      const valEl = P.el("span", { class: "val" }, [P.fmt(val, 2)]);
      const input = P.el("input", { type: "range", min, max, step, value: val, title: tip || "" });
      input.addEventListener("input", () => { valEl.textContent = P.fmt(parseFloat(input.value), 2); onInput(parseFloat(input.value)); });
      row.appendChild(P.el("span", { class: "name", title: tip || "" }, [name]));
      row.appendChild(input); row.appendChild(valEl);
      return row;
    }
    side.appendChild(P.el("div", { class: "hint" }, ["The situation (inputs):"]));
    side.appendChild(slider("temperature", temp, 0, 1, 0.05, v => { temp = v; redraw(); }, "How warm it is: 0 = freezing, 1 = hot. Weight +0.9 — warmth helps a little."));
    side.appendChild(slider("raining (0/1)", raining, 0, 1, 1, v => { raining = v; redraw(); }, "Is it raining? 0 = no, 1 = yes. Weight −3.0 — rain strongly discourages going out."));
    side.appendChild(slider("weekend (0/1)", weekend, 0, 1, 1, v => { weekend = v; redraw(); }, "Is it the weekend? 0 = no, 1 = yes. Weight +1.2 — the weekend encourages going out."));
    const explain = P.el("div", { class: "explain" });
    side.appendChild(explain);

    function redraw() {
      const x = [temp, raining, weekend];
      const r = P.forward(net, x);
      P.drawNetwork(netBox, net, { inputs: x, inputNames: ["temp", "rain", "wknd"] });
      const verdict = r.pred > 0.5 ? "YES, go outside" : "no, stay in";
      explain.innerHTML =
        '<div class="phase-tag">forward</div>' +
        `weighted sum: <code>z = (${P.fmt(temp)}×${P.fmt(n.w[0])}) + (${raining}×${P.fmt(n.w[1])}) + (${weekend}×${P.fmt(n.w[2])}) + ${P.fmt(n.b)} = ${P.fmt(n.z)}</code><br>` +
        `confidence: <code>σ(${P.fmt(n.z)}) = ${P.fmt(r.pred)}</code> → <b>${verdict}</b><br>` +
        `<span class="hint">Notice rain’s big negative weight (${P.fmt(n.w[1])}) — turning rain on alone can flip the decision.</span>`;
    }
    host.appendChild(panel);
    redraw();
  }
})();
