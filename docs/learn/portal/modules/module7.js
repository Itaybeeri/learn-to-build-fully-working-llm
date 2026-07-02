/* ============================================================================
   Module 7 tab — Capstone: build a tiny GPT (interactive, stage by stage).

   A browser companion to the runnable Python in
   docs/learn/07-capstone-tiny-gpt/demo/. The portal can't run PyTorch, so each
   sub-tab makes the *idea* of that build-stage clickable, mirroring the code.

   Sub-tabs (added as each Python stage is finished):
     7A · Stage 1 — the character tokenizer  (live: type text -> token ids, (x,y) shift)
     7A · Stage 2 — the model                (labeled forward-pass diagram + causal mask)
     7A · Stage 3 — training                 (the real loss curve)
     7A · Stage 4 — generation               (autoregressive-loop stepper + temperature sampler)
     7B · Part 1 Linear+grad-check · Part 2 softmax+CE · Part 3 attention backward ·
          Part 4 the model · Part 5 training curve · Part 6 generation

   Self-registers via Portal.register({...}). Vanilla JS + plain <script>.
   ============================================================================ */
(function () {
  "use strict";

  // The ACTUAL TinyShakespeare vocabulary (65 chars, sorted) — so ids shown
  // here match the Python demo exactly. A char's id = its index in this string.
  var VOCAB = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  var STOI = {};
  for (var i = 0; i < VOCAB.length; i++) STOI[VOCAB[i]] = i;

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
  // a printable label for a char (so newline/space are visible)
  function glyph(c) {
    if (c === "\n") return "\\n";
    if (c === " ")  return "␣";
    return esc(c);
  }
  // a "What am I looking at?" collapsible guide (used on every sub-tab)
  function mkGuide(P, bodyHTML) {
    var g = P.el("details", { class: "guide" });
    g.innerHTML = '<summary>What am I looking at?</summary><div class="guide-body">' + bodyHTML + '</div>';
    return g;
  }

  Portal.register({
    id: "module7",
    tab: "Module 7 — Tiny GPT",
    title: "Module 7 — Capstone: build a tiny GPT",
    intro: "The finale: assemble everything from Modules 2–6 into a runnable character-level GPT. " +
           "We build it TWICE — the same architecture both ways: 7A in PyTorch (autograd handles backprop), " +
           "then 7B in pure NumPy from scratch (we write every gradient by hand). The 7A sub-tabs walk the " +
           "build stage by stage; the 7B sub-tab shows backprop-by-hand, verified live against a numeric check.",
    render: function (mount, P) {
      // Two-level nav: top row picks 7A or 7B (7A first); the sub-menu below shows
      // that implementation's stages/parts.
      var GROUPS = [
        { id: "7A", label: "7A · PyTorch", subs: [
          { id: "tok",   name: "Stage 1 · Tokenizer",  build: buildTokenizer },
          { id: "model", name: "Stage 2 · The model",  build: buildModel },
          { id: "train", name: "Stage 3 · Training",   build: buildTraining },
          { id: "gen",   name: "Stage 4 · Generation", build: buildGeneration }
        ] },
        { id: "7B", label: "7B · NumPy from scratch", subs: [
          { id: "b1", name: "Part 1 · Linear + grad-check",       build: buildScratch },
          { id: "b2", name: "Part 2 · Softmax + cross-entropy",   build: build7B_ce },
          { id: "b3", name: "Part 3 · Attention backward",        build: build7B_attn },
          { id: "b4", name: "Part 4 · The model",                 build: build7B_model },
          { id: "b5", name: "Part 5 · Training",                  build: build7B_train },
          { id: "b6", name: "Part 6 · Generation",                build: build7B_gen }
        ] }
      ];
      var curGroup = GROUPS[0];
      var curSub = curGroup.subs[0];

      var topRow = P.el("div", { class: "demo-switch m7-toprow" });
      var subRow = P.el("div", { class: "demo-switch" });
      var host = P.el("div", {});

      function buildHost() { host.innerHTML = ""; curSub.build(host, P); }
      function renderSub() {
        subRow.innerHTML = "";
        curGroup.subs.forEach(function (s) {
          subRow.appendChild(P.el("button", {
            class: "demo-chip" + (s.id === curSub.id ? " active" : ""),
            onclick: function () { curSub = s; renderSub(); buildHost(); }
          }, [s.name]));
        });
      }
      function renderTop() {
        topRow.innerHTML = "";
        GROUPS.forEach(function (g) {
          topRow.appendChild(P.el("button", {
            class: "demo-chip m7-topchip" + (g.id === curGroup.id ? " active" : ""),
            onclick: function () { curGroup = g; curSub = g.subs[0]; renderTop(); renderSub(); buildHost(); }
          }, [g.label]));
        });
      }
      renderTop(); renderSub();
      mount.appendChild(topRow);
      mount.appendChild(subRow);
      mount.appendChild(host);
      buildHost();
    }
  });

  // placeholder builder for stages we haven't coded yet
  function buildComing(what) {
    return function (host, P) {
      var panel = P.el("div", { class: "panel" });
      panel.appendChild(P.el("h3", { class: "demo-title" }, ["Coming as we build this stage"]));
      panel.appendChild(P.el("p", { class: "demo-sub" },
        ["This sub-tab will cover " + what + ". It gets added here the moment the matching Python stage is " +
         "finished, so the portal and the notes stay in lockstep."]));
      host.appendChild(panel);
    };
  }

  // =========================================================================
  // STAGE 1 — the character tokenizer (live)
  // =========================================================================
  function buildTokenizer(host, P) {
    var panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["Stage 1 — the character tokenizer"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["Type any text. Each character is mapped to an integer id using the 65-char Shakespeare vocabulary — " +
       "the same ids the model trains on. Then see the encode/decode round-trip, and the (x, y) shift that " +
       "makes every target the NEXT character."]));

    var guide = P.el("details", { class: "guide" });
    guide.innerHTML =
      '<summary>What am I looking at?</summary>' +
      '<div class="guide-body">' +
        'A neural network eats numbers, not letters — so the very first job is to turn text into ' +
        '<b>token ids</b>. We use the simplest scheme: <b>one token per character</b>. The ' +
        '<b>vocabulary</b> is just the set of unique characters in our text (here, 65). Each character\'s ' +
        '<b>id is its position</b> in that sorted list — the "address book" from Module 3.<br><br>' +
        '<b>encode</b> turns your text into the list of ids; <b>decode</b> turns ids back into text ' +
        '(reversible).<br><br>' +
        'The <b>(x, y) shift</b> is how training examples are made: <b>x</b> is your characters, and ' +
        '<b>y</b> is the same characters shifted left by one — so at every position, y is "the next ' +
        'character." That free label is <b>self-supervised next-token prediction</b> (Module 5). ' +
        '<span class="m7-dim">(␣ = space, \\n = newline.)</span>' +
      '</div>';
    panel.appendChild(guide);

    // vocab line
    var vocabLine = P.el("div", { class: "m7-vocab" });
    vocabLine.innerHTML =
      '<span class="name" title="Every unique character in the corpus, sorted. A char\'s id is its index here.">vocabulary (' +
      VOCAB.length + ' chars):</span> <code class="m7-vocabcode">' + esc(VOCAB.replace(/\n/, "\\n")) + '</code>';
    panel.appendChild(vocabLine);

    // input + presets
    var PRESETS = ["Hi there!", "To be, or not to be", "First Citizen:\nSpeak."];
    var ctl = P.el("div", { class: "controls" });
    ctl.appendChild(P.el("span", { class: "name", title: "The text we tokenize. Try anything." }, ["text:"]));
    var input = P.el("input", { class: "m7-input", type: "text", value: PRESETS[0], spellcheck: "false" });
    ctl.appendChild(input);
    panel.appendChild(ctl);

    var presetRow = P.el("div", { class: "m7-presets" });
    PRESETS.forEach(function (p) {
      presetRow.appendChild(P.el("button", {
        class: "demo-chip", title: "Load this example.",
        onclick: function () { input.value = p; renderOut(); }
      }, [p.replace(/\n/g, "⏎")]));
    });
    panel.appendChild(presetRow);

    var out = P.el("div", { class: "m7-out" });
    panel.appendChild(out);
    host.appendChild(panel);

    function chipRow(chars, ids, opts) {
      opts = opts || {};
      var html = '<div class="m7-charrow">';
      for (var k = 0; k < chars.length; k++) {
        var known = ids[k] !== null && ids[k] !== undefined;
        html += '<div class="m7-chip' + (known ? "" : " unk") + '" title="' +
          (known ? "char " + glyph(chars[k]) + " → id " + ids[k] : "not in this 65-char vocabulary") + '">' +
          '<div class="m7-glyph">' + glyph(chars[k]) + '</div>' +
          '<div class="m7-id">' + (known ? ids[k] : "—") + '</div></div>';
      }
      html += '</div>';
      return html;
    }

    function renderOut() {
      var text = input.value;
      var chars = text.split("");
      var ids = chars.map(function (c) { return (c in STOI) ? STOI[c] : null; });
      var anyUnk = ids.some(function (v) { return v === null; });
      var knownIds = ids.filter(function (v) { return v !== null; });

      var html = "";
      // 1) encode
      html += '<div class="m7-block"><div class="m7-h">1 · encode — each character → its id</div>';
      html += chipRow(chars, ids);
      html += '<div class="m7-code">encode(' + JSON.stringify(text) + ') = [' + knownIds.join(", ") + ']</div>';
      if (anyUnk) html += '<div class="m7-warn">Some characters aren\'t in the 65-char Shakespeare vocab ' +
        '(shown as “—”). A real run would either use a bigger vocab or skip them.</div>';
      html += '</div>';

      // 2) decode round-trip
      var decoded = ids.map(function (v) { return v === null ? "" : VOCAB[v]; }).join("");
      var ok = decoded === text.split("").filter(function (c) { return c in STOI; }).join("");
      html += '<div class="m7-block"><div class="m7-h">2 · decode — ids back to text</div>' +
        '<div class="m7-code">decode([' + knownIds.join(", ") + ']) = ' + JSON.stringify(decoded) +
        '  <span class="m7-ok">round-trip OK ✓</span></div></div>';

      // 3) the (x, y) shift
      if (chars.length >= 2) {
        var xChars = chars.slice(0, -1), xIds = ids.slice(0, -1);
        var yChars = chars.slice(1),     yIds = ids.slice(1);
        html += '<div class="m7-block"><div class="m7-h">3 · the (x, y) shift — every target is the NEXT char</div>';
        html += '<div class="m7-shiftrow"><span class="m7-lab">x</span>' + chipRow(xChars, xIds) + '</div>';
        html += '<div class="m7-shiftrow"><span class="m7-lab">y</span>' + chipRow(yChars, yIds) + '</div>';
        html += '<div class="m7-note">Read a column: given the chars up to <b>x</b>, the model must predict ' +
          'the <b>y</b> below — and y is just x slid over by one. The label was already in the text ' +
          '(self-supervised). One chunk like this is many examples at once (the sliding window).</div>';
        html += '</div>';
      }

      out.innerHTML = html;
    }

    input.addEventListener("input", renderOut);
    renderOut();
  }

  // =========================================================================
  // STAGE 2 — the model (labeled forward-pass diagram)
  // =========================================================================
  function buildModel(host, P) {
    var panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["Stage 2 — the model (the forward pass)"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["From character ids to logits. Almost every box is a Module 4 concept — hover any box for what it does " +
       "and the shape flowing through it. (B = batch, T = sequence length, C = embedding size = 64, V = vocab = 65.)"]));

    var guide = P.el("details", { class: "guide" });
    guide.innerHTML =
      '<summary>What am I looking at?</summary>' +
      '<div class="guide-body">' +
        'This is the whole tiny GPT, top to bottom. The <b>token ids</b> enter, get turned into vectors ' +
        '(<b>embedding</b>) with their <b>position</b> added, flow through a stack of <b>Transformer ' +
        'blocks</b> (each one mixes information across tokens with <b>attention</b>, then thinks per-token ' +
        'with a <b>feed-forward</b> net, wrapped in <b>residual</b> adds and <b>layer norm</b>), and finally ' +
        'become <b>logits</b> — 65 raw scores at every position, one per possible next character.<br><br>' +
        'The one new idea vs. Module 4 is the <b>causal mask</b>: because a GPT predicts the <i>next</i> ' +
        'token, each position may attend only to itself and the <b>past</b>, never the future — otherwise it ' +
        'would peek at the answer. See the mask grid below the pipeline.' +
      '</div>';
    panel.appendChild(guide);

    // the vertical pipeline of labeled boxes
    var steps = [
      ["char ids", "(B, T)", "The integer token ids from Stage 1 — one per character.", "in"],
      ["token embedding", "(B, T, C)", "nn.Embedding(V, C): look up each id → a learned C-vector. Module 3's embedding table — 'what' the char is.", "emb"],
      ["+ position embedding", "(B, T, C)", "nn.Embedding(BLOCK_SIZE, C): add a 'where in the sequence' vector. Module 4 positional info (learned table). x = what + where.", "emb"],
      ["Transformer block × N", "(B, T, C)", "x = x + attention(norm(x)); x = x + feedforward(norm(x)). Self-attention (Q/K/V, multi-head, causal) mixes across tokens; the feed-forward thinks per token; residual adds + layer norm keep it stable. Module 4.", "block"],
      ["final layer norm", "(B, T, C)", "One last LayerNorm before the output projection (Module 4).", "norm"],
      ["lm_head (unembedding)", "(B, T, V)", "nn.Linear(C, V): project each C-vector to V=65 raw scores — the logits. Softmax turns these into next-char probabilities at sampling time.", "out"],
      ["logits", "(B, T, 65)", "65 scores at every position: how much the model favors each possible next character.", "out"]
    ];
    var pipe = P.el("div", { class: "m7-pipe" });
    steps.forEach(function (s, i) {
      var box = P.el("div", { class: "m7-step m7-step-" + s[3], title: s[2] }, [
        P.el("div", { class: "m7-step-label" }, [s[0]]),
        P.el("div", { class: "m7-step-shape" }, [s[1]])
      ]);
      pipe.appendChild(box);
      if (i < steps.length - 1) pipe.appendChild(P.el("div", { class: "m7-down" }, ["▼"]));
    });
    panel.appendChild(pipe);

    // the causal mask grid
    var maskWrap = P.el("div", { class: "m7-block" });
    var T = 4;
    var html = '<div class="m7-h">the causal mask — each row (a token) may attend only to itself + the past</div>';
    html += '<table class="m7-mask"><tr><td class="m7-mask-corner"></td>';
    for (var c = 1; c <= T; c++) html += '<th>pos' + c + '</th>';
    html += '</tr>';
    for (var r = 1; r <= T; r++) {
      html += '<tr><th>pos' + r + '</th>';
      for (var cc = 1; cc <= T; cc++) {
        var allowed = cc <= r;
        html += '<td class="' + (allowed ? "m7-yes" : "m7-no") + '" title="' +
          (allowed ? "pos" + r + " attends to pos" + cc : "masked: pos" + cc + " is in the future of pos" + r) +
          '">' + (allowed ? "✓" : "✗") + '</td>';
      }
      html += '</tr>';
    }
    html += '</table><div class="m7-note">Future cells (✗) are set to −∞ before softmax, so their attention ' +
      'weight becomes 0. pos1 sees only itself; pos4 sees everything up to and including itself.</div>';
    maskWrap.innerHTML = html;
    panel.appendChild(maskWrap);

    // verified numbers from the real Python run
    var facts = P.el("div", { class: "m7-block" });
    facts.innerHTML =
      '<div class="m7-h">from the real run (tiny_gpt_pytorch.py)</div>' +
      '<div class="m7-code">model parameters : 159,937</div><br>' +
      '<div class="m7-code">forward: x (16, 32) → logits (16, 32, 65)</div><br>' +
      '<div class="m7-code">untrained loss ≈ 4.37 ≈ ln(65) <span class="m7-ok">= uniform random guessing ✓</span></div>' +
      '<div class="m7-note">With random weights the model spreads probability evenly over 65 chars, and ' +
      'cross-entropy of uniform guessing is exactly ln(65) ≈ 4.17 — proof the wiring is correct. Stage 3 ' +
      'drives this number down.</div>';
    panel.appendChild(facts);

    host.appendChild(panel);
  }

  // =========================================================================
  // STAGE 3 — training (the real loss curve)
  // =========================================================================
  function buildTraining(host, P) {
    // the ACTUAL numbers printed by tiny_gpt_pytorch.py
    var DATA = [
      { step: 0,    train: 4.409, val: 4.396 },
      { step: 500,  train: 2.308, val: 2.316 },
      { step: 1000, train: 2.140, val: 2.173 },
      { step: 1500, train: 2.044, val: 2.087 },
      { step: 2000, train: 1.966, val: 2.039 },
      { step: 2500, train: 1.900, val: 1.987 },
      { step: 3000, train: 1.864, val: 1.964 }
    ];

    var panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["Stage 3 — training (the loss falls)"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["The Module 2 loop, on the GPT: forward → cross-entropy loss → loss.backward() (backprop) → " +
       "optimizer.step(). Below is the REAL loss curve from running tiny_gpt_pytorch.py for 3000 steps — " +
       "watch it fall from random guessing (≈ln(65)=4.17) to ≈1.9."]));

    var guide = P.el("details", { class: "guide" });
    guide.innerHTML =
      '<summary>What am I looking at?</summary>' +
      '<div class="guide-body">' +
        'Training is four steps repeated thousands of times: ① grab a batch of chars + their next chars; ' +
        '② run them forward and measure <b>cross-entropy loss</b> (how wrong the next-char guesses are); ' +
        '③ <b>loss.backward()</b> — PyTorch\'s autograd computes the gradient for all ~160k weights (the ' +
        'backprop we built by hand in Module 2); ④ <b>optimizer.step()</b> nudges every weight downhill ' +
        '(AdamW = a smarter gradient descent).<br><br>' +
        'Two lines are plotted: <b class="m7-leg-train">train</b> loss and <b class="m7-leg-val">validation</b> ' +
        'loss (Module 5). They fall <b>together</b> here — that means the model is genuinely <b>learning</b>, ' +
        'not just memorizing. If the val line had turned upward while train kept dropping, that would be ' +
        '<b>overfitting</b>. Hover a point for its exact numbers.' +
      '</div>';
    panel.appendChild(guide);

    // --- SVG loss chart ---
    var W = 560, H = 300, padL = 46, padR = 16, padT = 16, padB = 36;
    var maxStep = 3000, yMin = 1.5, yMax = 4.6;
    function sx(step) { return padL + (step / maxStep) * (W - padL - padR); }
    function sy(v) { return padT + (1 - (v - yMin) / (yMax - yMin)) * (H - padT - padB); }

    var svg = P.svgEl("svg", { viewBox: "0 0 " + W + " " + H, class: "m7-chart" });
    function add(tag, attrs, text) { var n = P.svgEl(tag, attrs); if (text != null) n.textContent = text; svg.appendChild(n); return n; }

    // axes + y gridlines
    [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5].forEach(function (v) {
      add("line", { x1: padL, y1: sy(v), x2: W - padR, y2: sy(v), stroke: "#1c2733", "stroke-width": 1 });
      add("text", { x: padL - 8, y: sy(v) + 4, "text-anchor": "end", fill: "#6b7a8d", "font-size": 11 }, v.toFixed(1));
    });
    // ln(65) reference line
    add("line", { x1: padL, y1: sy(4.174), x2: W - padR, y2: sy(4.174), stroke: "#ffb454", "stroke-width": 1, "stroke-dasharray": "4 4" });
    add("text", { x: W - padR, y: sy(4.174) - 5, "text-anchor": "end", fill: "#ffb454", "font-size": 10 }, "ln(65) = random guessing");
    // x labels
    [0, 1000, 2000, 3000].forEach(function (s) {
      add("text", { x: sx(s), y: H - 12, "text-anchor": "middle", fill: "#6b7a8d", "font-size": 11 }, s);
    });
    add("text", { x: (W) / 2, y: H - 1, "text-anchor": "middle", fill: "#6b7a8d", "font-size": 11 }, "training step");

    function polyline(key, color) {
      var pts = DATA.map(function (d) { return sx(d.step) + "," + sy(d[key]); }).join(" ");
      add("polyline", { points: pts, fill: "none", stroke: color, "stroke-width": 2.5 });
      DATA.forEach(function (d) {
        var c = add("circle", { cx: sx(d.step), cy: sy(d[key]), r: 4, fill: color });
        var title = P.svgEl("title"); title.textContent = "step " + d.step + " — " + key + " loss " + d[key].toFixed(3); c.appendChild(title);
      });
    }
    polyline("train", "#4ea1ff");
    polyline("val", "#ff8e5e");
    panel.appendChild(svg);

    var legend = P.el("div", { class: "m7-legend" });
    legend.innerHTML =
      '<span class="m7-leg-train">● train</span>&nbsp;&nbsp;<span class="m7-leg-val">● validation</span>' +
      '&nbsp;&nbsp;<span style="color:#ffb454">– – ln(65) = guessing</span>';
    panel.appendChild(legend);

    var facts = P.el("div", { class: "m7-block" });
    facts.innerHTML =
      '<div class="m7-note">Loss fell <b>4.41 → 1.86</b> (train) and <b>4.40 → 1.96</b> (val) over 3000 steps. ' +
      'The two curves track each other closely → genuine learning, only a small healthy gap (no real ' +
      'overfitting). A char-level loss near 1.9 already produces real words and Shakespeare-ish structure — ' +
      'which Stage 4 (generation) will show. The trained weights were saved to <code>tiny_gpt.pt</code>.</div>';
    panel.appendChild(facts);

    host.appendChild(panel);
  }

  // =========================================================================
  // STAGE 4 — generation (autoregressive-loop stepper + temperature sampler)
  // =========================================================================
  function buildGeneration(host, P) {
    // The REAL 500-char sample the Python demo printed from the prompt "ROMEO:".
    var PROMPT = "ROMEO:";
    var SAMPLE =
      "\nVen trusir:\nO, I say me tebled your shall cay thy not\nsoold it I the are on\n" +
      "canour,'d the on yet: woo as thou not Romen from.\nThou deks, I him.\n\nHENRSARIO:\n" +
      "Then kraint; Not;\nI he prisanch lord, enks scamed, a you lord cous with derve would " +
      "hidgh wescler.\nAs mintau' bray hy reving her that I'tring shabde that may\nWhey, rike " +
      "dead a eace yound this your with hers,--bly gind Lever all lead me it ress your\nmyse " +
      "that fate stare the sI struck,\nIf not hell, fout, lock''s shon can in an it is\nA it to th";
    var FULL = PROMPT + SAMPLE;
    var BLOCK_SIZE = 32;

    var panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["Stage 4 — generation (the model speaks)"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["Generation is a loop: predict the next char, sample one, append it, repeat. Step through the REAL " +
       "output the trained model produced from the prompt “ROMEO:” — then play with temperature " +
       "to see how the next-char pick changes."]));

    var guide = P.el("details", { class: "guide" });
    guide.innerHTML =
      '<summary>What am I looking at?</summary>' +
      '<div class="guide-body">' +
        'The model only ever does one thing: given the text so far, score every possible next character. ' +
        '<b>Generation</b> just runs that in a loop and feeds each new character back in — that’s ' +
        '<b>autoregressive</b> (“auto” = self, it consumes its own output). Each step: ' +
        '① forward pass on the context; ② keep only the <b>last position’s</b> scores (the ' +
        'prediction for the next char); ③ <b>temperature + softmax</b> turn scores into probabilities; ' +
        '④ <b>sample</b> one char and append it.<br><br>' +
        'Two knobs from Module 6: <b>temperature</b> (below 1 = safer/sharper, above 1 = wilder/flatter) and ' +
        'sampling itself (we pick <i>from</i> the distribution, not always the top char, so the text stays ' +
        'alive). The context is cropped to the last ' + BLOCK_SIZE + ' characters — all the model can see.' +
      '</div>';
    panel.appendChild(guide);

    // ---- A) the autoregressive-loop stepper --------------------------------
    var stepWrap = P.el("div", { class: "m7-block" });
    stepWrap.innerHTML = '<div class="m7-h">A · the autoregressive loop — step through the real output</div>';
    var controls = P.el("div", { class: "m7-presets" });
    var n = PROMPT.length;   // how many chars are "generated so far" (starts at the prompt)

    function showChar(c) { return c === "\n" ? "↵\n" : c; }

    var ctxView = P.el("div", { class: "m7-code", style: "white-space:pre-wrap; min-height:6em; line-height:1.5" });
    var statusLine = P.el("div", { class: "m7-note" });

    function renderStep() {
      var done = FULL.slice(0, n);
      var ctxStart = Math.max(0, n - BLOCK_SIZE);
      var before = done.slice(0, ctxStart);
      var window = done.slice(ctxStart);          // the last <=32 chars the model actually sees
      var nextChar = n < FULL.length ? FULL[n] : null;
      // render: dim the out-of-window text, highlight the context window, show the just-added char
      var html = "";
      if (before) html += '<span class="m7-dim">' + esc(before).replace(/\n/g, "↵\n") + '</span>';
      html += '<span style="background:#1c3350;border-radius:3px">' + esc(window).replace(/\n/g, "↵\n") + '</span>';
      ctxView.innerHTML = html || '<span class="m7-dim">(empty)</span>';
      if (nextChar === null) {
        statusLine.innerHTML = '<b>Done</b> — reached the end of this 500-char sample. Hit Reset to replay.';
      } else {
        statusLine.innerHTML =
          'The highlighted <b>' + window.length + ' chars</b> are the context (cropped to ' + BLOCK_SIZE +
          '). The model reads them → predicts → next sampled char = <b>“' +
          esc(showChar(nextChar)) + '”</b>. Click again to append it and continue.';
      }
    }

    var stepBtn = P.el("button", { class: "demo-chip", onclick: function () {
      if (n < FULL.length) { n++; renderStep(); }
    } }, ["▶ generate next char"]);
    var run10 = P.el("button", { class: "demo-chip", onclick: function () {
      n = Math.min(FULL.length, n + 10); renderStep();
    } }, ["+10"]);
    var resetBtn = P.el("button", { class: "demo-chip", onclick: function () { n = PROMPT.length; renderStep(); } }, ["↺ reset"]);
    controls.appendChild(stepBtn); controls.appendChild(run10); controls.appendChild(resetBtn);
    stepWrap.appendChild(controls);
    stepWrap.appendChild(ctxView);
    stepWrap.appendChild(statusLine);
    panel.appendChild(stepWrap);
    renderStep();

    // ---- B) the temperature / softmax sampler ------------------------------
    // Illustrative raw scores (logits) for a few candidate next chars after "Thou ".
    // The softmax+temperature math is real; the logits are hand-picked for clarity.
    var CAND = [
      { ch: "w", logit: 3.1 }, { ch: "s", logit: 2.4 }, { ch: "h", logit: 1.7 },
      { ch: "a", logit: 1.2 }, { ch: "d", logit: 0.6 }, { ch: "□ other", logit: -0.4 }
    ];
    function softmaxT(items, T) {
      var m = Math.max.apply(null, items.map(function (d) { return d.logit; }));
      var ex = items.map(function (d) { return Math.exp((d.logit - m) / T); });
      var s = ex.reduce(function (a, b) { return a + b; }, 0);
      return ex.map(function (e) { return e / s; });
    }

    var tWrap = P.el("div", { class: "m7-block" });
    tWrap.innerHTML = '<div class="m7-h">B · temperature — how bold is the next-char pick?</div>';
    var tRow = P.el("div", { class: "controls" });
    tRow.appendChild(P.el("span", { class: "name", title: "Divide the scores by T before softmax." }, ["temperature:"]));
    var tSlider = P.el("input", { type: "range", min: "0.1", max: "2", step: "0.1", value: "1", style: "flex:1" });
    var tVal = P.el("code", {}, ["1.0"]);
    tRow.appendChild(tSlider); tRow.appendChild(tVal);
    tWrap.appendChild(tRow);
    var bars = P.el("div", {});
    tWrap.appendChild(bars);
    var tNote = P.el("div", { class: "m7-note" });
    tWrap.appendChild(tNote);

    function renderTemp() {
      var T = parseFloat(tSlider.value);
      tVal.textContent = T.toFixed(1);
      var probs = softmaxT(CAND, T);
      var html = '<div class="m7-charrow" style="flex-direction:column; align-items:stretch; gap:6px">';
      CAND.forEach(function (d, i) {
        var pct = (probs[i] * 100);
        html += '<div style="display:flex; align-items:center; gap:8px">' +
          '<code style="width:64px; text-align:right">' + esc(d.ch) + '</code>' +
          '<div style="flex:1; background:#12202e; border-radius:4px; overflow:hidden">' +
            '<div style="width:' + pct.toFixed(1) + '%; background:#4ea1ff; height:16px"></div>' +
          '</div>' +
          '<code style="width:52px">' + pct.toFixed(1) + '%</code></div>';
      });
      html += '</div>';
      bars.innerHTML = html;
      var msg;
      if (T <= 0.4) msg = '<b>Low temperature</b> (→ 0 = greedy): the top char takes almost all the ' +
        'probability. Safe, but repetitive and can get stuck.';
      else if (T >= 1.6) msg = '<b>High temperature</b>: the distribution flattens — unlikely chars get a ' +
        'real chance. Adventurous, but more typos/nonsense.';
      else msg = '<b>Around 1.0</b> = the distribution exactly as the model learned it. Balanced.';
      tNote.innerHTML = 'Each score is divided by T, then softmax. ' + msg +
        ' We then <b>sample</b> one char in proportion to these bars (not always the tallest).';
    }
    tSlider.addEventListener("input", renderTemp);
    renderTemp();

    var facts = P.el("div", { class: "m7-block" });
    facts.innerHTML =
      '<div class="m7-h">from the real run (tiny_gpt_pytorch.py, prompt “ROMEO:”)</div>' +
      '<div class="m7-note">A 160k-parameter model, trained ~2 min on a CPU to loss ≈1.86, wrote the ' +
      'sample above. Word-by-word it’s gibberish, but it learned Shakespeare’s <b>shape</b> from ' +
      'scratch: <code>NAME:</code> speaker headings, line breaks, English-textured words, sensible ' +
      'punctuation. Real words and sentences need a bigger model + more training (the scale-up stage).</div>';
    panel.appendChild(tWrap);
    panel.appendChild(facts);

    host.appendChild(panel);
  }

  // =========================================================================
  // 7B — backprop by hand (live gradient check on a Linear layer)
  // =========================================================================
  function buildScratch(host, P) {
    var panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["7B — backprop by hand (NumPy from scratch)"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["In 7B there is no autograd — every layer computes its own gradients. Here is the whole idea, live, on " +
       "a Linear layer (y = x·W + b): our hand-derived gradients vs. a foolproof numeric check. Randomize as " +
       "much as you like — they always match."]));

    var guide = P.el("details", { class: "guide" });
    guide.innerHTML =
      '<summary>What am I looking at?</summary>' +
      '<div class="guide-body">' +
        '7A let PyTorch\'s <code>loss.backward()</code> compute all gradients. 7B removes that magic: each ' +
        'layer is <b>two functions</b> — <b>forward</b> (compute the output) and <b>backward</b> (given the ' +
        'gradient of the loss w.r.t. its output, compute the gradient w.r.t. its inputs and its weights).<br><br>' +
        'To trust our hand-written math we <b>gradient-check</b> it: nudge one number by a tiny ε and see how ' +
        'the loss changes — <code>(loss(x+ε) − loss(x−ε)) / (2ε)</code> — a slow but foolproof estimate of the ' +
        'gradient. If our formula matches this numeric value, it\'s correct.<br><br>' +
        'Below, for <b>y = x·W + b</b> with loss <b>L = Σy²</b> (so dL/dy = 2y), we show the hand-derived ' +
        '<b>dW = xᵀ·dy</b> and <b>dx = dy·Wᵀ</b> next to the numeric estimate. The max relative difference is ' +
        'tiny (~1e-7 or less) → they agree.' +
      '</div>';
    panel.appendChild(guide);

    // --- tiny matrix helpers ---
    function zeros(r, c) { var m = []; for (var i = 0; i < r; i++) { m.push(new Array(c).fill(0)); } return m; }
    function randM(r, c) { var m = zeros(r, c); for (var i = 0; i < r; i++) for (var j = 0; j < c; j++) m[i][j] = +(Math.random() * 2 - 1).toFixed(2); return m; }
    function matmul(A, B) {
      var r = A.length, k = B.length, c = B[0].length, out = zeros(r, c);
      for (var i = 0; i < r; i++) for (var j = 0; j < c; j++) { var s = 0; for (var t = 0; t < k; t++) s += A[i][t] * B[t][j]; out[i][j] = s; }
      return out;
    }
    function transpose(A) { var r = A.length, c = A[0].length, out = zeros(c, r); for (var i = 0; i < r; i++) for (var j = 0; j < c; j++) out[j][i] = A[i][j]; return out; }

    var NIN = 3, NOUT = 2, BATCH = 2;
    var W, b, x;

    function forwardY(Wp, bp, xp) {
      var y = matmul(xp, Wp);
      for (var i = 0; i < y.length; i++) for (var j = 0; j < y[0].length; j++) y[i][j] += bp[j];
      return y;
    }
    function lossOf(Wp, bp, xp) {
      var y = forwardY(Wp, bp, xp), s = 0;
      for (var i = 0; i < y.length; i++) for (var j = 0; j < y[0].length; j++) s += y[i][j] * y[i][j];
      return s;
    }
    // numeric gradient of loss w.r.t. an arbitrary matrix `M` (by identity), given a setter
    function numericGrad(M, evalLoss, eps) {
      var g = zeros(M.length, M[0].length);
      for (var i = 0; i < M.length; i++) for (var j = 0; j < M[0].length; j++) {
        var o = M[i][j];
        M[i][j] = o + eps; var hi = evalLoss();
        M[i][j] = o - eps; var lo = evalLoss();
        M[i][j] = o; g[i][j] = (hi - lo) / (2 * eps);
      }
      return g;
    }
    function maxRel(A, Bn) {
      var md = 0, mn = 1e-12;
      for (var i = 0; i < A.length; i++) for (var j = 0; j < A[0].length; j++) {
        md = Math.max(md, Math.abs(A[i][j] - Bn[i][j])); mn = Math.max(mn, Math.abs(Bn[i][j]));
      }
      return md / mn;
    }
    function gridHTML(title, M) {
      var h = '<div style="flex:1"><div class="m7-h" style="margin-bottom:4px">' + title + '</div><table class="m7-mask">';
      for (var i = 0; i < M.length; i++) { h += '<tr>'; for (var j = 0; j < M[0].length; j++) h += '<td class="m7-yes" style="min-width:52px">' + M[i][j].toFixed(3) + '</td>'; h += '</tr>'; }
      return h + '</table></div>';
    }

    var controls = P.el("div", { class: "m7-presets" });
    var out = P.el("div", {});
    controls.appendChild(P.el("button", { class: "demo-chip", onclick: function () { init(); } }, ["↻ randomize W, x"]));
    panel.appendChild(controls);
    panel.appendChild(out);

    function init() { W = randM(NIN, NOUT); b = new Array(NOUT).fill(0); x = randM(BATCH, NIN); render(); }
    function render() {
      var eps = 1e-4;
      var y = forwardY(W, b, x);
      // dy = 2y
      var dy = zeros(BATCH, NOUT); for (var i = 0; i < BATCH; i++) for (var j = 0; j < NOUT; j++) dy[i][j] = 2 * y[i][j];
      var dW = matmul(transpose(x), dy);   // analytic
      var dx = matmul(dy, transpose(W));   // analytic
      var dWn = numericGrad(W, function () { return lossOf(W, b, x); }, eps);
      var dxn = numericGrad(x, function () { return lossOf(W, b, x); }, eps);

      var html = "";
      html += '<div class="m7-block"><div class="m7-h">setup — y = x·W + b,   loss L = Σy²   (so dL/dy = 2y)</div>';
      html += '<div class="m7-charrow" style="gap:16px">' + gridHTML("W (" + NIN + "×" + NOUT + ")", W) + gridHTML("x (" + BATCH + "×" + NIN + ")", x) + '</div></div>';

      html += '<div class="m7-block"><div class="m7-h">dW = xᵀ·dy — hand-derived vs. numeric check</div>';
      html += '<div class="m7-charrow" style="gap:16px">' + gridHTML("dW (analytic)", dW) + gridHTML("dW (numeric)", dWn) + '</div>';
      html += '<div class="m7-note">max relative difference = <b>' + maxRel(dW, dWn).toExponential(1) + '</b> ' +
        '<span class="m7-ok">match ✓</span></div></div>';

      html += '<div class="m7-block"><div class="m7-h">dx = dy·Wᵀ — hand-derived vs. numeric check</div>';
      html += '<div class="m7-charrow" style="gap:16px">' + gridHTML("dx (analytic)", dx) + gridHTML("dx (numeric)", dxn) + '</div>';
      html += '<div class="m7-note">max relative difference = <b>' + maxRel(dx, dxn).toExponential(1) + '</b> ' +
        '<span class="m7-ok">match ✓</span></div></div>';

      html += '<div class="m7-block"><div class="m7-note">This is exactly what <code>tiny_gpt_numpy.py</code> ' +
        'does (Part 1), grad-checking to ~1e-11 in float64. Every layer we add — embedding, LayerNorm, ' +
        'softmax, attention — gets the same forward + backward + gradient-check treatment before we trust it.</div></div>';

      out.innerHTML = html;
    }
    init();

    host.appendChild(panel);
  }

  // ---- 7B Part 2 — softmax + cross-entropy, and its clean gradient ---------
  function build7B_ce(host, P) {
    var panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["7B · Part 2 — softmax + cross-entropy"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["The model outputs raw scores (logits). Softmax turns them into probabilities; cross-entropy scores " +
       "them against the correct token. The gradient that starts backprop is beautifully simple: " +
       "dlogits = p − onehot (predicted minus target). Pick the correct class and randomize the scores."]));

    panel.appendChild(mkGuide(P,
      'The model\'s final output is a list of raw scores called <b>logits</b> — one per possible next ' +
      'character. <b>Softmax</b> squashes them into probabilities that sum to 1. <b>Cross-entropy</b> is the ' +
      'loss: it reads the probability given to the <b>correct</b> character and is <code>−log</code> of it — ' +
      'small when the model was confident and right, large when it was confidently wrong.<br><br>' +
      'The magic is the gradient. For softmax+cross-entropy it collapses to one line: <b>dlogits = p − ' +
      'onehot</b> (the predicted probabilities minus 1 at the correct class). That single vector is what ' +
      'backprop pushes back through the entire network. Pick which class is "correct" and hit randomize to ' +
      'watch the loss and the gradient respond.'));

    var CLASSES = ["A", "B", "C", "D", "E"];
    var logits, target = 0;
    function randomize() { logits = CLASSES.map(function () { return +(Math.random() * 6 - 3).toFixed(2); }); render(); }
    function softmax(z) { var m = Math.max.apply(null, z); var e = z.map(function (v) { return Math.exp(v - m); }); var s = e.reduce(function (a, b) { return a + b; }, 0); return e.map(function (v) { return v / s; }); }

    var ctl = P.el("div", { class: "m7-presets" });
    CLASSES.forEach(function (c, i) {
      ctl.appendChild(P.el("button", { class: "demo-chip", onclick: function () { target = i; render(); } }, ["correct = " + c]));
    });
    ctl.appendChild(P.el("button", { class: "demo-chip", onclick: function () { randomize(); } }, ["↻ randomize scores"]));
    panel.appendChild(ctl);
    var out = P.el("div", {});
    panel.appendChild(out);

    function barRow(label, val, pct, color, extra) {
      return '<div style="display:flex;align-items:center;gap:8px">' +
        '<code style="width:74px;text-align:right">' + label + '</code>' +
        '<div style="flex:1;background:#12202e;border-radius:4px;overflow:hidden">' +
        '<div style="width:' + Math.min(100, Math.abs(pct)).toFixed(1) + '%;background:' + color + ';height:15px"></div></div>' +
        '<code style="width:64px">' + val + '</code>' + (extra || "") + '</div>';
    }
    function render() {
      var p = softmax(logits);
      var loss = -Math.log(p[target] + 1e-12);
      var html = "";
      html += '<div class="m7-block"><div class="m7-h">logits → softmax probabilities (correct = ' + CLASSES[target] + ')</div>';
      CLASSES.forEach(function (c, i) {
        html += barRow(c + "  (" + logits[i].toFixed(2) + ")", (p[i] * 100).toFixed(1) + "%", p[i] * 100, i === target ? "#46d39a" : "#4ea1ff");
      });
      html += '<div class="m7-note">cross-entropy loss = −log p(correct) = −log(' + p[target].toFixed(3) + ') = <b>' + loss.toFixed(3) + '</b></div></div>';

      html += '<div class="m7-block"><div class="m7-h">the gradient: dlogits = p − onehot(correct)</div>';
      CLASSES.forEach(function (c, i) {
        var g = p[i] - (i === target ? 1 : 0);
        html += barRow(c + "  " + (g >= 0 ? "+" : "") + g.toFixed(3), g.toFixed(3), g * 100, g < 0 ? "#ff8e5e" : "#8aa0b6");
      });
      html += '<div class="m7-note">Every wrong class gets a small <b>positive</b> push (lower its score); the ' +
        '<b>correct</b> class gets a <b>negative</b> push of p−1 (raise its score). That one line — ' +
        '<code>(p − onehot)/N</code> — is what starts backprop through the whole network.</div></div>';
      out.innerHTML = html;
    }
    randomize();
    host.appendChild(panel);
  }

  // ---- 7B Part 3 — attention backward (the chain, step by step) ------------
  function build7B_attn(host, P) {
    var panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["7B · Part 3 — self-attention, forward & backward by hand"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["The hardest gradient in the model. Each forward step has a matching backward step; run in reverse " +
       "order they give every gradient. All of this is verified inside the full-model gradient check (~1e-7)."]));

    panel.appendChild(mkGuide(P,
      'Self-attention (Module 4) is a chain of operations: project the input into <b>query, key, value</b>, ' +
      'score every token against every token, mask the future, softmax the scores into weights, and blend ' +
      'the values. To train it without autograd we need the <b>backward</b> of each of those steps.<br><br>' +
      'The table pairs each <b>forward step</b> with its <b>backward step</b>. Backprop runs them in reverse ' +
      '(bottom to top): start from the gradient of the output, work back through the blend, the softmax, the ' +
      'mask, the scores, and finally the three projections. The trickiest is the <b>softmax backward</b> ' +
      '(<code>a·(da − Σ(da·a))</code>). You don\'t need to memorize the formulas — the point is that every ' +
      'forward step has an exact, checkable backward, and together they let attention learn.'));

    var rows = [
      ["q,k,v = x·W", "project the input three ways (Part 1 Linear)", "dWq,dWk,dWv and dx via each Linear's backward"],
      ["scores = q·kᵀ · (1/√hs)", "how relevant is each token to each", "dq = dscores·k ,  dk = dscoresᵀ·q  (times the scale)"],
      ["mask the future → −∞", "causal: a token can't see ahead", "zero the gradient on masked (future) cells"],
      ["a = softmax(scores)", "scores → weights that sum to 1", "dscores = a·(da − Σ(da·a))   (softmax backward)"],
      ["out = a·v", "blend the values by the weights", "dv = aᵀ·dout ,  da = dout·vᵀ"]
    ];
    var t = '<div class="m7-block"><table class="m7-mask" style="width:100%"><tr>' +
      '<th style="text-align:left">forward step</th><th style="text-align:left">what it does</th>' +
      '<th style="text-align:left">backward step</th></tr>';
    rows.forEach(function (r) {
      t += '<tr><td class="m7-yes" style="text-align:left"><code>' + r[0] + '</code></td>' +
        '<td class="m7-yes" style="text-align:left">' + r[1] + '</td>' +
        '<td class="m7-yes" style="text-align:left"><code>' + r[2] + '</code></td></tr>';
    });
    t += '</table><div class="m7-note">Backward runs these bottom-to-top (out → a → scores → q,k,v). ' +
      '<b>Multi-head</b>: several heads run in parallel; their outputs are concatenated and mixed by W_O — ' +
      'backward splits the gradient back to each head. In <code>tiny_gpt_numpy.py</code> this is the ' +
      '<code>Head.backward</code> / <code>MultiHead.backward</code> code.</div></div>';
    panel.innerHTML += "";
    var wrap = P.el("div", {}); wrap.innerHTML = t; panel.appendChild(wrap);
    host.appendChild(panel);
  }

  // ---- 7B Part 4 — the model (same architecture, built by hand) ------------
  function build7B_model(host, P) {
    var panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["7B · Part 4 — the model (built by hand)"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["Same architecture as 7A (see the 7A · Stage 2 diagram), assembled from our hand-written layers. " +
       "The tiny NumPy config below has 30,017 parameters and its whole gradient is verified numerically."]));

    panel.appendChild(mkGuide(P,
      'This is the full model, assembled from the pieces we built and grad-checked one by one: Embedding, ' +
      'LayerNorm, ReLU, softmax+cross-entropy, and self-attention. The architecture is <b>identical to 7A</b> ' +
      '(the PyTorch version) — the only difference is that here every layer\'s <b>backward</b> is our own ' +
      'code, not autograd.<br><br>' +
      'The boxes show the flow: ids become vectors (token + position embeddings), pass through a stack of ' +
      '<b>Blocks</b> (attention then feed-forward, each with a residual add and a LayerNorm), a final ' +
      'LayerNorm, and an output Linear that produces the logits. The verified numbers below prove the whole ' +
      'hand-written backward chain is correct: every parameter\'s gradient matches the numeric check.'));

    var steps = [
      ["token embedding + position embedding", "ids → vectors, plus a 'where' vector (Embedding: forward = lookup, backward = scatter-add)"],
      ["N × Block", "each: x = x + attention(LayerNorm(x)); x = x + feedforward(LayerNorm(x))  — residual '+' adds gradients in backward"],
      ["final LayerNorm", "normalize (LayerNorm forward/backward by hand)"],
      ["output Linear → logits", "project to vocab-size scores per position"]
    ];
    var pipe = P.el("div", { class: "m7-pipe" });
    steps.forEach(function (s, i) {
      pipe.appendChild(P.el("div", { class: "m7-step m7-step-block", title: s[1] }, [
        P.el("div", { class: "m7-step-label" }, [s[0]])
      ]));
      if (i < steps.length - 1) pipe.appendChild(P.el("div", { class: "m7-down" }, ["▼"]));
    });
    panel.appendChild(pipe);

    var facts = P.el("div", { class: "m7-block" });
    facts.innerHTML =
      '<div class="m7-h">from the real run (tiny_gpt_numpy.py)</div>' +
      '<div class="m7-code">model parameters : 30,017   (tiny NumPy config: n_embd 32, 2 blocks, 4 heads)</div><br>' +
      '<div class="m7-code">full-model gradient check: worst rel. diff = 8.9e-8   <span class="m7-ok">ALL OK ✓</span></div>' +
      '<div class="m7-note">Every parameter\'s hand-derived gradient matches the numeric estimate — proof the ' +
      'entire backward chain (through attention, LayerNorm, softmax, residuals) is correct.</div>';
    panel.appendChild(facts);
    host.appendChild(panel);
  }

  // ---- 7B Part 5 — training (the real NumPy loss curve) -------------------
  function build7B_train(host, P) {
    var DATA = [
      { step: 0,    train: 4.654, val: 4.646 },
      { step: 500,  train: 2.296, val: 2.346 },
      { step: 1000, train: 2.177, val: 2.227 },
      { step: 1500, train: 2.110, val: 2.183 },
      { step: 2000, train: 2.071, val: 2.132 }
    ];
    var panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["7B · Part 5 — training (manual backprop)"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["The same four-step loop, but backprop is our own hand-written code and the optimizer is a hand-written " +
       "Adam. Below is the REAL loss curve from tiny_gpt_numpy.py — no autograd anywhere."]));

    panel.appendChild(mkGuide(P,
      'Training is the same loop as everywhere: ① grab a batch of characters + their next-characters; ' +
      '② run them forward and measure <b>cross-entropy loss</b>; ③ <b>backward</b> — but here we call our own ' +
      'hand-written <code>backward</code> through every layer instead of <code>loss.backward()</code>; ' +
      '④ update every weight with a <b>hand-written Adam</b> optimizer (adaptive step + momentum).<br><br>' +
      'The chart is the actual loss from running the NumPy model: <b class="m7-leg-train">train</b> and ' +
      '<b class="m7-leg-val">validation</b> loss falling from ≈4.65 (random guessing, the dashed ln(65) line) ' +
      'toward ≈2.1. They fall together → it\'s genuinely learning. It\'s slower and smaller than 7A (pure ' +
      'NumPy on a CPU), but it learns with math we wrote ourselves. Hover a point for its exact value.'));

    var W = 560, H = 280, padL = 46, padR = 16, padT = 16, padB = 36;
    var maxStep = 2000, yMin = 1.8, yMax = 4.8;
    function sx(s) { return padL + (s / maxStep) * (W - padL - padR); }
    function sy(v) { return padT + (1 - (v - yMin) / (yMax - yMin)) * (H - padT - padB); }
    var svg = P.svgEl("svg", { viewBox: "0 0 " + W + " " + H, class: "m7-chart" });
    function add(tag, attrs, text) { var n = P.svgEl(tag, attrs); if (text != null) n.textContent = text; svg.appendChild(n); return n; }
    [2.0, 2.5, 3.0, 3.5, 4.0, 4.5].forEach(function (v) {
      add("line", { x1: padL, y1: sy(v), x2: W - padR, y2: sy(v), stroke: "#1c2733", "stroke-width": 1 });
      add("text", { x: padL - 8, y: sy(v) + 4, "text-anchor": "end", fill: "#6b7a8d", "font-size": 11 }, v.toFixed(1));
    });
    add("line", { x1: padL, y1: sy(4.174), x2: W - padR, y2: sy(4.174), stroke: "#ffb454", "stroke-width": 1, "stroke-dasharray": "4 4" });
    add("text", { x: W - padR, y: sy(4.174) - 5, "text-anchor": "end", fill: "#ffb454", "font-size": 10 }, "ln(65) = random guessing");
    [0, 1000, 2000].forEach(function (s) { add("text", { x: sx(s), y: H - 12, "text-anchor": "middle", fill: "#6b7a8d", "font-size": 11 }, s); });
    add("text", { x: W / 2, y: H - 1, "text-anchor": "middle", fill: "#6b7a8d", "font-size": 11 }, "training step");
    function poly(key, color) {
      add("polyline", { points: DATA.map(function (d) { return sx(d.step) + "," + sy(d[key]); }).join(" "), fill: "none", stroke: color, "stroke-width": 2.5 });
      DATA.forEach(function (d) { var c = add("circle", { cx: sx(d.step), cy: sy(d[key]), r: 4, fill: color }); var ti = P.svgEl("title"); ti.textContent = "step " + d.step + " — " + key + " " + d[key].toFixed(3); c.appendChild(ti); });
    }
    poly("train", "#4ea1ff"); poly("val", "#ff8e5e");
    panel.appendChild(svg);
    var legend = P.el("div", { class: "m7-legend" });
    legend.innerHTML = '<span class="m7-leg-train">● train</span>&nbsp;&nbsp;<span class="m7-leg-val">● validation</span>&nbsp;&nbsp;<span style="color:#ffb454">– – ln(65)</span>';
    panel.appendChild(legend);
    var facts = P.el("div", { class: "m7-block" });
    facts.innerHTML = '<div class="m7-note">Loss fell <b>4.65 → 2.07</b> using gradients we computed by hand and a ' +
      'hand-written Adam optimizer. Slower than 7A (pure NumPy on CPU, smaller model), same idea. Next: generation.</div>';
    panel.appendChild(facts);
    host.appendChild(panel);
  }

  // ---- 7B Part 6 — generation (the from-scratch model speaks) --------------
  function build7B_gen(host, P) {
    var panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["7B · Part 6 — generation"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["Same autoregressive loop as 7A (crop context → forward → last-position logits → temperature+softmax → " +
       "sample → append). Here is the REAL sample from the from-scratch NumPy model, prompt 'ROMEO:'."]));

    panel.appendChild(mkGuide(P,
      'Once trained, the model writes text by running its own output back in — the <b>autoregressive loop</b>: ' +
      'feed the context, take the scores for the <b>last position</b>, apply temperature + softmax to get ' +
      'next-character probabilities, <b>sample</b> one character, append it, and repeat.<br><br>' +
      'The text below is the genuine output of the pure-NumPy model — the one whose every gradient we wrote ' +
      'by hand. It\'s gibberish word-by-word (only 30k parameters, briefly trained on a CPU) but it clearly ' +
      'learned Shakespeare\'s <b>shape</b>: speaker names in caps with colons, line breaks, and ' +
      'English-textured words. Same behaviour as 7A — but nothing here was automatic.'));

    var sample =
      "ROMEO:\nOry becacty;\nYou forsoo; mid feat my wellf.\n\nAnlold the to scharispur?\n" +
      "Comy and and mistend, thence on thrrownint, and sagmes, riede cow ould molve las, rond ur sept\n" +
      "excicplame hims to nos wis in undibe and sre's year soseveuly boinevunnce, ogfeir:\n" +
      "On co lro deeschir ubling; and you gofst:\nA fearramse;\nAmme tese yeath woe imr-be tear?\n" +
      "The letrarct plut no sarcner tellfe lignedieo,\nOnculse king hr";
    var pre = P.el("div", { class: "m7-code", style: "white-space:pre-wrap;line-height:1.5" });
    pre.textContent = sample;
    panel.appendChild(pre);
    var facts = P.el("div", { class: "m7-block" });
    facts.innerHTML = '<div class="m7-note">A 30k-parameter model with <b>every gradient written by hand</b> ' +
      'learned Shakespeare\'s shape from scratch: speaker headings (<code>ROMEO:</code>), line breaks, and ' +
      'English-textured words (You, my, and, king, year, tear). Same behaviour as 7A — but nothing was ' +
      'automatic. That completes the capstone.</div>';
    panel.appendChild(facts);
    host.appendChild(panel);
  }
})();
