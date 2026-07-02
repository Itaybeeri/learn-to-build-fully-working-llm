/* ============================================================================
   Module 5 tab — Training LLMs (interactive demos)

   Demo 1 · Sliding window — next-token prediction made visible:
     Type (or pick) a short sentence. The demo splits it into tokens and walks a
     SLIDING WINDOW along it: at each position the input is "everything before"
     and the label is "the next word". One sentence yields many free examples, and
     the text labels itself — that's self-supervised learning. A Play / Next / Prev
     stepper walks position by position; the full list of (context → next word)
     examples is shown with a "1 sentence → N examples" counter.

   Demo 2 · Scaling laws + emergence — datasets & scale made visible:
     One "scale" knob grows data + parameters + compute together. As scale rises,
     the LOSS falls along a smooth, predictable downward curve (a real power-law-ish
     function of scale) — that smoothness is why labs can PREDICT a bigger model's
     gain before building it. Beside it, a toy EMERGENT ABILITY ("can do 2-step
     arithmetic") reads ~0% while scale is small, then switches on SHARPLY past a
     threshold (a real steep sigmoid) — sudden, not gradual.

   Demo 3 · Data cleaning — quality made visible:
     A toy "raw web crawl" of ~10 snippets mixes good text with junk (spam,
     gibberish, exact duplicates, a boilerplate menu, a toxic-ish line). Toggle the
     three cleaning steps — Quality filter, Deduplicate, Safety filter — and watch
     the junk drop out while a counter shows snippets/tokens before → after.
     Teaching point: garbage in, garbage out; quality often beats raw quantity.

   Demo 4 · Overfitting — train vs. validation loss made visible:
     Drag a TRAINING-TIME knob (epochs) and watch two loss curves: training loss
     slides down toward zero, while VALIDATION loss (a held-out set the model never
     trains on) falls, bottoms out, then turns back UP — the curves SPLIT, which is
     overfitting. A star marks the validation minimum (the best model / where early
     stopping would halt). Toggle model size, data amount, and regularization to see
     each cause/remedy reshape the gap.

   Registers itself with the portal engine via Portal.register({...}).
   ============================================================================ */
(function () {
  "use strict";

  Portal.register({
    id: "module5",
    tab: "Module 5 — Training LLMs",
    title: "Module 5 — Training LLMs",
    intro: "How does an LLM actually learn from text? These four demos show the engine of training: " +
           "the sliding window that turns one sentence into many free examples (next-token prediction), " +
           "the scaling laws that make bigger models predictably better — plus the sudden emergent " +
           "abilities scale unlocks — the data cleaning that decides what the model becomes, and the " +
           "train-vs-validation loss curves that reveal overfitting.",
    render: function (mount, P) {
      const demos = [
        { id: "window",  name: "1 · Sliding window",          build: buildSlidingWindow },
        { id: "scaling", name: "2 · Scaling laws + emergence", build: buildScaling },
        { id: "clean",   name: "3 · Data cleaning",            build: buildCleaning },
        { id: "overfit", name: "4 · Overfitting",              build: buildOverfitting }
      ];
      let current = "window";
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

  // small helper: split a sentence into word tokens (toy tokenizer — words, no sub-words)
  function tokenize(s) {
    return s.trim().split(/\s+/).filter(Boolean);
  }
  // escape a string for safe insertion into innerHTML
  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  // =========================================================================
  // DEMO 1 — Sliding window (next-token prediction)
  // =========================================================================
  function buildSlidingWindow(host, P) {
    const panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["Sliding window — one sentence, many free examples"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["Pick or type a short sentence. We split it into tokens and slide a window along it. At each " +
       "position the INPUT is everything before and the LABEL is the next word — and that label is " +
       "free, because it's already in the text. That's self-supervised next-token prediction."]));

    const guide = P.el("details", { class: "guide" });
    guide.innerHTML =
      '<summary>What am I looking at?</summary>' +
      '<div class="guide-body">' +
        'An LLM learns by playing one game: <b>guess the next word</b>. To train any model you normally ' +
        'need <b>labeled examples</b> — an input paired with the correct answer — and a human usually has ' +
        'to write those answers by hand. Here there is no human: <b>the text labels itself</b>.<br><br>' +
        '<b>One cut = one example.</b> Take a sentence and cut it anywhere: the words before the cut are ' +
        'the <b>input (context)</b>, and the very next word is the <b>label (correct answer)</b>. Nobody ' +
        'tagged it — the next word was already sitting there. This is <b>self-supervised learning</b>.<br>' +
        '<b>Slide for many.</b> Don’t cut once — make a cut at <i>every</i> position. Walk the window ' +
        'left to right; each position is its own (context → next word) example. So one short sentence ' +
        'becomes several examples, and a whole book becomes millions.<br><br>' +
        'Below, the <b>highlighted</b> tokens are the growing context the model gets to see, and the ' +
        '<b>dashed</b> token is the next word it must predict (the answer hidden from it). Press ' +
        '<b>Play ▶</b> to walk the window, or step with Next / Prev. Hover any token or example row.' +
      '</div>';
    panel.appendChild(guide);

    // --- sentence input + presets ---
    const PRESETS = [
      "the cat sat on the mat",
      "the capital of France is Paris",
      "all cats are mammals so Tom is a mammal"
    ];
    let sentence = PRESETS[0];
    let pos = 0;          // current window position: input = tokens[0..pos], label = tokens[pos+1]
    let playTimer = null;

    const ctl = P.el("div", { class: "controls" });
    ctl.appendChild(P.el("span", { class: "name", title: "The sentence we turn into training examples." }, ["sentence:"]));
    const input = P.el("input", {
      type: "text", value: sentence,
      title: "Type any short sentence, then it re-tokenizes.",
      style: "flex:1;min-width:220px;background:#0d141c;border:1px solid #33404f;color:#e6edf3;border-radius:8px;padding:8px 10px;font-size:14px"
    });
    ctl.appendChild(input);
    panel.appendChild(ctl);

    const presetRow = P.el("div", { class: "controls" });
    presetRow.appendChild(P.el("span", { class: "name", title: "Quick example sentences." }, ["presets:"]));
    PRESETS.forEach(s => presetRow.appendChild(P.el("button", {
      class: "demo-chip", title: "Use this sentence.",
      onclick: () => { input.value = s; setSentence(s); }
    }, [s])));
    panel.appendChild(presetRow);

    // --- stepper controls ---
    const step = P.el("div", { class: "controls" });
    const prevBtn = P.el("button", { class: "btn", title: "Move the window back one position." }, ["◂ Prev"]);
    const playBtn = P.el("button", { class: "btn primary", title: "Auto-walk the window across the sentence." }, ["Play ▶"]);
    const nextBtn = P.el("button", { class: "btn", title: "Move the window forward one position." }, ["Next ▸"]);
    step.appendChild(prevBtn); step.appendChild(playBtn); step.appendChild(nextBtn);
    panel.appendChild(step);

    const out = P.el("div", { class: "explain" });
    panel.appendChild(out);

    function stopPlay() {
      if (playTimer) { clearInterval(playTimer); playTimer = null; playBtn.textContent = "Play ▶"; }
    }
    function setSentence(s) {
      stopPlay();
      sentence = s;
      pos = 0;
      run();
    }

    input.addEventListener("input", () => setSentence(input.value));
    prevBtn.addEventListener("click", () => { stopPlay(); const n = tokenize(sentence).length; if (pos > 0) pos--; run(); });
    nextBtn.addEventListener("click", () => { stopPlay(); const n = tokenize(sentence).length; if (pos < n - 2) pos++; run(); });
    playBtn.addEventListener("click", () => {
      const toks = tokenize(sentence);
      if (toks.length < 2) return;
      if (playTimer) { stopPlay(); return; }
      playBtn.textContent = "⏸ Pause";
      pos = 0; run();
      playTimer = setInterval(() => {
        if (pos < toks.length - 2) { pos++; run(); }
        else { stopPlay(); }
      }, 1100);
    });

    function run() {
      const toks = tokenize(sentence);
      const n = toks.length;
      // examples: position i (0..n-2): input = toks[0..i], label = toks[i+1]
      const exampleCount = Math.max(0, n - 1);
      pos = Math.max(0, Math.min(pos, Math.max(0, n - 2)));

      let html = "";

      if (n < 2) {
        html += '<div class="phase-tag">need at least 2 words</div>' +
          '<p style="color:#cdd9e5">Type a sentence with two or more words to make a (context → next word) example.</p>';
        out.innerHTML = html;
        prevBtn.disabled = nextBtn.disabled = playBtn.disabled = true;
        return;
      }
      prevBtn.disabled = (pos === 0);
      nextBtn.disabled = (pos >= n - 2);
      playBtn.disabled = false;

      // counter
      html += '<div class="phase-tag">1 sentence → ' + exampleCount + ' training examples</div>';
      html += '<p style="margin:4px 0 10px;color:#cdd9e5">Window at position <b>' + (pos + 1) + '</b> of ' +
        exampleCount + '. <span style="color:#8aa0b4">Input = highlighted context, label = the dashed next word.</span></p>';

      // the token strip with the current window highlighted
      html += '<div style="display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 14px">';
      toks.forEach((t, i) => {
        let bg = "#232d3a", bd = "#33404f", col = "#e6edf3", style = "", tag = "";
        if (i <= pos) {                       // in the context (input)
          bg = "#1f3a4d"; bd = "#4ea1ff"; tag = "context";
        } else if (i === pos + 1) {           // the label (next word to predict)
          bg = "#102a22"; bd = "#46d39a"; col = "#46d39a"; style = "border-style:dashed"; tag = "◀ predict this";
        } else {                              // not yet seen
          col = "#5b6b7b"; tag = "not seen yet";
        }
        html += '<span title="' + esc(t) + ' — ' + tag + '" ' +
          'style="display:inline-flex;flex-direction:column;align-items:center;background:' + bg +
          ';border:1px solid ' + bd + ';' + style + ';border-radius:8px;padding:6px 12px;font-family:monospace;color:' + col + '">' +
          esc(t) +
          (i <= pos + 1 ? '<span style="font-size:10px;color:#8aa0b4;margin-top:2px">' + (i <= pos ? "input" : "label") + '</span>' : '') +
          '</span>';
      });
      html += '</div>';

      // the single current example, written out
      const ctx = toks.slice(0, pos + 1).join(" ");
      const label = toks[pos + 1];
      html += '<div class="phase-tag">this position’s example</div>';
      html += '<div style="font-family:monospace;font-size:14px;margin:6px 0 14px;line-height:1.8">' +
        'input: <span style="color:#4ea1ff">"' + esc(ctx) + '"</span> ' +
        '<span style="color:#8aa0b4">→</span> ' +
        'label: <b style="color:#46d39a">"' + esc(label) + '"</b>' +
        '</div>';

      // full list of all examples generated, current one highlighted
      html += '<div class="phase-tag">all ' + exampleCount + ' examples from this one sentence</div>';
      html += '<div style="margin-top:6px">';
      for (let i = 0; i < n - 1; i++) {
        const c = toks.slice(0, i + 1).join(" ");
        const lbl = toks[i + 1];
        const here = (i === pos);
        html += '<div title="position ' + (i + 1) + ': input = everything before, label = the next word" ' +
          'style="display:flex;align-items:center;gap:8px;margin:3px 0;padding:4px 8px;border-radius:6px;font-family:monospace;font-size:13px;' +
          (here ? 'background:#1f3a4d;border:1px solid #4ea1ff' : 'background:#0d141c;border:1px solid #232d3a') + '">' +
          '<span style="width:22px;color:#8aa0b4">' + (i + 1) + '.</span>' +
          '<span style="flex:1;color:' + (here ? "#cdd9e5" : "#8aa0b4") + '">"' + esc(c) + '"</span>' +
          '<span style="color:#8aa0b4">→</span>' +
          '<span style="width:90px;text-align:right;color:#46d39a">"' + esc(lbl) + '"</span>' +
          '</div>';
      }
      html += '</div>';

      html += '<div class="hint">Every row is a free training example — and nobody labeled a single one. ' +
        'The model practices each: it guesses the next word from the context, then corrects itself. ' +
        'A 1,000-word document is ~1,000 examples; the whole internet is unlimited free data. ' +
        'That is why next-token prediction can train on essentially all text ever written.</div>';

      out.innerHTML = html;
    }

    host.appendChild(panel);
    run();
  }

  // =========================================================================
  // DEMO 2 — Scaling laws + emergence
  // =========================================================================
  // Honest toy math:
  //   scale s runs 0..1 (a single knob standing for data + params + compute grown together).
  //   LOSS is a smooth, monotonically DECREASING power-law-ish curve of scale:
  //       loss(s) = L_inf + A / (1 + K*s)^p
  //     - L_inf  = irreducible loss floor (you can't beat the data's entropy)
  //     - decreasing & smooth for all s  -> "predictable" scaling law
  //   EMERGENCE is a steep logistic (sigmoid) of scale, ~0 below a threshold, ~1 above:
  //       emerge(s) = 1 / (1 + exp(-STEEP*(s - THRESH)))
  //     - flat-near-zero then a sharp switch-on -> "sudden, not gradual"
  function buildScaling(host, P) {
    const panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["Scaling laws + emergence — bigger, predictably better (and surprises)"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["Drag the scale knob (data + parameters + compute, grown together). Watch two very different " +
       "things: the loss slides DOWN a smooth, predictable curve — while a toy ability sits at ~0% " +
       "then switches ON sharply past a threshold. Smooth scaling vs. sudden emergence."]));

    const guide = P.el("details", { class: "guide" });
    guide.innerHTML =
      '<summary>What am I looking at?</summary>' +
      '<div class="guide-body">' +
        'Three dials decide how good an LLM gets, and they must grow <i>together</i>: ' +
        '<b style="color:#4ea1ff">data</b> (how much it reads), ' +
        '<b style="color:#ffd24e">parameters</b> (how much it can store) and ' +
        '<b style="color:#46d39a">compute</b> (study time). The weakest link caps the rest, so here one ' +
        '<b>scale knob</b> grows all three at once.<br><br>' +
        '<b>The blue loss curve (scaling law).</b> As scale rises, the training <b>loss</b> ' +
        '(how wrong the next-word guesses are — lower is better) falls along a <b>smooth, ' +
        'predictable</b> downward curve. Because it’s smooth, labs can <b>predict</b> how much a ' +
        'bigger model will gain <i>before</i> building it — which is why they confidently spend ' +
        'fortunes scaling up. It never quite hits 0: there’s an irreducible floor (no model can ' +
        'out-predict the true randomness of language).<br><br>' +
        '<b>The green emergence curve (emergent ability).</b> A toy skill — “can do 2-step ' +
        'arithmetic” — reads near <b>0%</b> while the model is small, then <b>switches on ' +
        'sharply</b> past a threshold size, like water turning to ice. Nobody trained for it; it just ' +
        '<b>emerges</b> from scaling up next-word prediction. Notice the shapes differ on purpose: ' +
        'scaling is <i>gradual and predictable</i>; emergence is <i>sudden</i>.<br><br>' +
        'Hover the curves and the moving marker for the exact numbers.' +
      '</div>';
    panel.appendChild(guide);

    // --- the scale slider ---
    let scale = 0.18;     // 0..1
    const slr = P.el("div", { class: "slider-row" });
    slr.appendChild(P.el("span", { class: "name", title: "Grows data + parameters + compute together." }, ["scale"]));
    const range = P.el("input", { type: "range", min: "0", max: "1000", value: String(Math.round(scale * 1000)),
      title: "Drag to grow the model (data + parameters + compute together)." });
    const valLbl = P.el("span", { class: "val" }, []);
    slr.appendChild(range); slr.appendChild(valLbl);
    panel.appendChild(slr);

    const canvas = P.el("canvas", { class: "loss", style: "height:240px", title: "Loss (blue, falling) and the emergent ability (green, switching on) vs. scale." });
    panel.appendChild(canvas);

    const out = P.el("div", { class: "explain" });
    panel.appendChild(out);

    // --- honest toy math --------------------------------------------------
    const L_INF = 1.6, A = 4.2, K = 9, P_EXP = 0.7;     // loss(s) params
    const STEEP = 22, THRESH = 0.62;                      // emergence params
    function lossAt(s)   { return L_INF + A / Math.pow(1 + K * s, P_EXP); }
    function emergeAt(s) { return 1 / (1 + Math.exp(-STEEP * (s - THRESH))); }

    // model "size" label just for flavour: exponential in scale (params), honest monotone
    function paramsLabel(s) {
      const p = 1e6 * Math.pow(10, s * 5);   // 1M -> ~100B across the slider
      if (p >= 1e9) return P.fmt(p / 1e9, 1) + "B params";
      return P.fmt(p / 1e6, 0) + "M params";
    }
    function tokensLabel(s) {
      const t = 1e8 * Math.pow(10, s * 5);   // ~100M -> ~10T tokens
      if (t >= 1e12) return P.fmt(t / 1e12, 1) + "T tokens";
      if (t >= 1e9)  return P.fmt(t / 1e9, 0) + "B tokens";
      return P.fmt(t / 1e6, 0) + "M tokens";
    }

    function drawChart() {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth || 600, h = canvas.clientHeight || 240;
      canvas.width = w * dpr; canvas.height = h * dpr;
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const padL = 44, padR = 44, padT = 16, padB = 30;
      const x0 = padL, x1 = w - padR, y0 = h - padB, y1 = padT;
      const lossMax = lossAt(0), lossMin = lossAt(1);
      const sx = (s) => x0 + (x1 - x0) * s;
      const lossY = (L) => y0 - (y0 - y1) * ((lossMax - L) / (lossMax - lossMin)); // loss high at top? -> map so falling line goes down
      const emY   = (e) => y0 - (y0 - y1) * e;   // 0 at bottom, 1 at top

      // axes
      ctx.strokeStyle = "#33404f"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x0, y1); ctx.lineTo(x0, y0); ctx.lineTo(x1, y0); ctx.stroke();
      ctx.fillStyle = "#9aa7b4"; ctx.font = "11px Segoe UI";
      ctx.fillText("loss", 6, y1 + 10);
      ctx.fillText("0%", x1 + 6, y0);
      ctx.fillText("100%", x1 + 6, y1 + 10);
      ctx.fillText("scale →", (x0 + x1) / 2 - 18, h - 6);
      ctx.fillText("small", x0 - 4, h - 6);
      ctx.fillText("huge", x1 - 18, h - 6);

      // loss curve (blue, smooth, falling)
      ctx.strokeStyle = "#4ea1ff"; ctx.lineWidth = 2.5; ctx.beginPath();
      for (let i = 0; i <= 200; i++) { const s = i / 200; const x = sx(s), y = lossY(lossAt(s)); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
      ctx.stroke();

      // loss floor (dashed)
      ctx.strokeStyle = "#2f5a7a"; ctx.lineWidth = 1; ctx.setLineDash([4, 4]); ctx.beginPath();
      ctx.moveTo(x0, lossY(lossMin)); ctx.lineTo(x1, lossY(lossMin)); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = "#2f5a7a"; ctx.fillText("irreducible floor", x0 + 6, lossY(lossMin) - 4);

      // emergence curve (green, steep sigmoid)
      ctx.strokeStyle = "#46d39a"; ctx.lineWidth = 2.5; ctx.beginPath();
      for (let i = 0; i <= 200; i++) { const s = i / 200; const x = sx(s), y = emY(emergeAt(s)); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
      ctx.stroke();

      // threshold marker for emergence
      ctx.strokeStyle = "#2f7a5a"; ctx.lineWidth = 1; ctx.setLineDash([3, 4]); ctx.beginPath();
      ctx.moveTo(sx(THRESH), y0); ctx.lineTo(sx(THRESH), y1); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = "#2f7a5a"; ctx.fillText("threshold", sx(THRESH) + 4, y1 + 22);

      // current scale marker (vertical line + dots on both curves)
      const cx = sx(scale);
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1; ctx.globalAlpha = 0.35; ctx.beginPath();
      ctx.moveTo(cx, y0); ctx.lineTo(cx, y1); ctx.stroke(); ctx.globalAlpha = 1;
      // loss dot
      ctx.fillStyle = "#4ea1ff"; ctx.beginPath(); ctx.arc(cx, lossY(lossAt(scale)), 5, 0, Math.PI * 2); ctx.fill();
      // emergence dot
      ctx.fillStyle = "#46d39a"; ctx.beginPath(); ctx.arc(cx, emY(emergeAt(scale)), 5, 0, Math.PI * 2); ctx.fill();

      // legend
      ctx.font = "11px Segoe UI";
      ctx.fillStyle = "#4ea1ff"; ctx.fillText("— loss (scaling law)", x0 + 6, y1 + 12);
      ctx.fillStyle = "#46d39a"; ctx.fillText("— emergent ability", x0 + 6, y1 + 26);
    }

    function run() {
      const L = lossAt(scale), e = emergeAt(scale);
      const ePct = Math.round(e * 100);
      const on = e >= 0.5;
      valLbl.textContent = Math.round(scale * 100) + "%";
      drawChart();

      let html = "";
      html += '<div class="phase-tag">at this scale</div>';
      html += '<div class="stats" style="margin-top:4px">' +
        '<div class="stat" title="The scale knob: data + parameters + compute, grown together."><div class="k">scale</div><div class="v">' + Math.round(scale * 100) + '%</div></div>' +
        '<div class="stat" title="Rough model size at this scale (toy mapping)."><div class="k">size</div><div class="v" style="font-size:14px">' + paramsLabel(scale) + '</div></div>' +
        '<div class="stat" title="Rough training data at this scale (toy mapping)."><div class="k">data</div><div class="v" style="font-size:14px">' + tokensLabel(scale) + '</div></div>' +
        '<div class="stat" title="Training loss — how wrong the next-word guesses are. Lower is better."><div class="k">loss</div><div class="v" style="color:#4ea1ff">' + P.fmt(L) + '</div></div>' +
        '<div class="stat" title="The toy emergent ability: 2-step arithmetic."><div class="k">2-step math</div><div class="v" style="color:' + (on ? "#46d39a" : "#ff6b6b") + '">' + ePct + '%</div></div>' +
        '</div>';

      html += '<div class="phase-tag" style="margin-top:14px">scaling law (the blue curve)</div>';
      html += '<p style="margin:4px 0;color:#cdd9e5">The loss falls <b>smoothly and predictably</b> as you scale up — ' +
        'a clean downward trend, no jumps. Because it’s this smooth, you can <b>read off</b> how much a ' +
        'bigger model will gain <i>before</i> spending the money. It flattens toward an <b>irreducible floor</b> ' +
        '(' + P.fmt(lossAt(1)) + ') — language has real randomness no model can out-guess.</p>';

      html += '<div class="phase-tag" style="margin-top:10px">emergent ability (the green curve)</div>';
      if (ePct <= 2) {
        html += '<p style="margin:4px 0;color:#cdd9e5">Right now the model is too small: 2-step arithmetic is at ' +
          '<b style="color:#ff6b6b">' + ePct + '%</b> — it simply <b>can’t</b> do it. Keep dragging right…</p>';
      } else if (!on) {
        html += '<p style="margin:4px 0;color:#cdd9e5">You’re near the <b>threshold</b> — the ability is ' +
          'starting to flicker on (<b style="color:#ffd24e">' + ePct + '%</b>). Notice how <b>steep</b> this is ' +
          'compared to the gentle loss curve. A little more scale…</p>';
      } else {
        html += '<p style="margin:4px 0;color:#cdd9e5">It <b>switched on</b> — 2-step arithmetic now works ' +
          '(<b style="color:#46d39a">' + ePct + '%</b>). This didn’t ramp up gradually; it appeared ' +
          '<b>suddenly</b> past a size threshold, like a phase change. Nobody trained for it — it ' +
          '<b>emerged</b> from scaling next-word prediction.</p>';
      }

      html += '<div class="hint">Two different shapes, one knob: scaling laws are <b>gradual and predictable</b> ' +
        '(plan the gain in advance); emergent abilities are <b>sudden</b> (they surprise even the builders). ' +
        'The whole recipe: a simple task + clean data + scale, grown together.</div>';

      out.innerHTML = html;
    }

    range.addEventListener("input", () => { scale = (+range.value) / 1000; run(); });
    window.addEventListener("resize", drawChart);

    host.appendChild(panel);
    run();
  }

  // =========================================================================
  // DEMO 3 — Data cleaning
  // =========================================================================
  // A toy raw crawl. Each snippet is tagged; the cleaning toggles drop matching kinds.
  //   Quality filter  -> drops gibberish, spam, boilerplate
  //   Deduplicate     -> drops exact duplicates, keeping the first copy
  //   Safety filter   -> drops toxic / unwanted
  // A "good" snippet survives everything.
  function buildCleaning(host, P) {
    const panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["Data cleaning — garbage in, garbage out"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["Here’s a tiny “raw web crawl”: good writing mixed with spam, gibberish, exact " +
       "duplicates, a boilerplate menu, and a toxic line. The model becomes what it reads — so toggle the " +
       "cleaning steps and watch the junk drop out before any training happens."]));

    const guide = P.el("details", { class: "guide" });
    guide.innerHTML =
      '<summary>What am I looking at?</summary>' +
      '<div class="guide-body">' +
        'You can’t train straight on a raw web crawl: <b>most of the open web is junk</b> — spam, ' +
        'broken gibberish, endless duplicates, menus, toxic content. And the iron rule is ' +
        '<b>garbage in, garbage out</b>: a model only imitates the patterns in its data, so training on ' +
        'spam teaches it to <i>produce</i> spam. So the crawl is heavily <b>cleaned</b> first, in three jobs:' +
        '<br><br>' +
        '<table><tr><th>step</th><th>removes</th><th>why</th></tr>' +
        '<tr><td><b style="color:#ffd24e">Quality filter</b></td><td>gibberish, spam, boilerplate/menus</td><td>keep real, fluent human writing</td></tr>' +
        '<tr><td><b style="color:#4ea1ff">Deduplicate</b></td><td>exact repeats (keep one copy)</td><td>repeats over-weight text &amp; push the model to memorize/regurgitate</td></tr>' +
        '<tr><td><b style="color:#ff6b6b">Safety filter</b></td><td>toxic / unwanted content</td><td>don’t let the model absorb &amp; reproduce it</td></tr></table>' +
        'Toggle each step and watch the corpus shrink — fewer snippets, fewer tokens — but get ' +
        '<b>much cleaner</b>. The field’s hard-won lesson: <b>quality often beats raw quantity</b>. ' +
        'Less but cleaner text usually beats more but dirtier text. Hover any snippet for its tag.' +
      '</div>';
    panel.appendChild(guide);

    // toy crawl. kind drives which filter removes it. dupOf marks an exact duplicate of an earlier id.
    const DUP_TEXT = "Breaking: scientists discover a new species of deep-sea fish off the coast.";
    const CRAWL = [
      { id: 1,  kind: "good",       text: "The mitochondria is the powerhouse of the cell, producing energy the body uses." },
      { id: 2,  kind: "spam",       text: "BUY CHEAP MEDS NOW!!! click here >>> www.totally-legit-pills.biz CLICK CLICK" },
      { id: 3,  kind: "good",       text: DUP_TEXT },
      { id: 4,  kind: "gibberish",  text: "asdf qwerty zxcv lkjh 0x9f3a ##@@ ndndnd florp florp wug wug zzz" },
      { id: 5,  kind: "good",       text: "Photosynthesis lets plants turn sunlight, water, and CO2 into sugar and oxygen." },
      { id: 6,  kind: "dup",        text: DUP_TEXT, dupOf: 3 },
      { id: 7,  kind: "boilerplate",text: "Home | About | Products | Login | Contact | Privacy Policy | Terms | Sitemap" },
      { id: 8,  kind: "good",       text: "Shakespeare wrote both comedies and tragedies, including Hamlet and Macbeth." },
      { id: 9,  kind: "toxic",      text: "You are all idiots and I hope terrible things happen to people I dislike." },
      { id: 10, kind: "dup",        text: DUP_TEXT, dupOf: 3 },
      { id: 11, kind: "good",       text: "Rivers carry sediment downstream, slowly building deltas where they meet the sea." }
    ];
    // toy token count = word count
    const tokCount = (t) => t.trim().split(/\s+/).length;

    const KIND_META = {
      good:        { label: "good text",   col: "#46d39a", removedBy: null },
      spam:        { label: "spam",        col: "#ff6b6b", removedBy: "quality" },
      gibberish:   { label: "gibberish",   col: "#ff6b6b", removedBy: "quality" },
      boilerplate: { label: "boilerplate", col: "#ffd24e", removedBy: "quality" },
      dup:         { label: "exact duplicate", col: "#4ea1ff", removedBy: "dedup" },
      toxic:       { label: "toxic",       col: "#ff6b6b", removedBy: "safety" }
    };

    // toggles
    let quality = false, dedup = false, safety = false;
    const ctl = P.el("div", { class: "controls" });
    ctl.appendChild(P.el("span", { class: "name", title: "Apply cleaning steps to the raw crawl." }, ["cleaning steps:"]));
    const qBtn = P.el("button", { class: "demo-chip", title: "Remove gibberish, spam, and boilerplate." }, ["Quality filter"]);
    const dBtn = P.el("button", { class: "demo-chip", title: "Remove exact duplicates, keeping one copy." }, ["Deduplicate"]);
    const sBtn = P.el("button", { class: "demo-chip", title: "Remove toxic / unwanted content." }, ["Safety filter"]);
    const allBtn = P.el("button", { class: "btn", title: "Apply all three cleaning steps." }, ["Clean all"]);
    const resetBtn = P.el("button", { class: "btn", title: "Back to the raw crawl." }, ["Reset"]);
    qBtn.addEventListener("click", () => { quality = !quality; sync(); });
    dBtn.addEventListener("click", () => { dedup = !dedup; sync(); });
    sBtn.addEventListener("click", () => { safety = !safety; sync(); });
    allBtn.addEventListener("click", () => { quality = dedup = safety = true; sync(); });
    resetBtn.addEventListener("click", () => { quality = dedup = safety = false; sync(); });
    ctl.appendChild(qBtn); ctl.appendChild(dBtn); ctl.appendChild(sBtn); ctl.appendChild(allBtn); ctl.appendChild(resetBtn);
    panel.appendChild(ctl);

    const out = P.el("div", { class: "explain" });
    panel.appendChild(out);

    // decide, for a snippet, whether it is removed and by which step
    function removalStep(s) {
      const m = KIND_META[s.kind];
      if (m.removedBy === "quality" && quality) return "quality";
      if (m.removedBy === "dedup"   && dedup)   return "dedup";
      if (m.removedBy === "safety"  && safety)  return "safety";
      return null;
    }

    function sync() {
      qBtn.className = "demo-chip" + (quality ? " active" : "");
      dBtn.className = "demo-chip" + (dedup ? " active" : "");
      sBtn.className = "demo-chip" + (safety ? " active" : "");
      run();
    }

    function run() {
      const totalSnip = CRAWL.length;
      const totalTok = CRAWL.reduce((a, s) => a + tokCount(s.text), 0);
      let keptSnip = 0, keptTok = 0, removedSnip = 0, removedTok = 0;

      let rows = "";
      CRAWL.forEach(s => {
        const m = KIND_META[s.kind];
        const step = removalStep(s);
        const removed = !!step;
        const tk = tokCount(s.text);
        if (removed) { removedSnip++; removedTok += tk; } else { keptSnip++; keptTok += tk; }

        const stepLabel = step === "quality" ? "removed by Quality filter"
          : step === "dedup" ? "removed by Deduplicate (kept the first copy)"
          : step === "safety" ? "removed by Safety filter" : "kept";
        const dupNote = s.kind === "dup" ? ' (copy of #' + s.dupOf + ')' : '';

        rows += '<div title="' + esc(m.label + dupNote + " — " + stepLabel) + '" ' +
          'style="display:flex;align-items:flex-start;gap:10px;margin:4px 0;padding:7px 10px;border-radius:8px;' +
          (removed
            ? 'background:#160d0d;border:1px solid #3a2330;opacity:.5'
            : 'background:#0d141c;border:1px solid #232d3a') + '">' +
          '<span style="flex-shrink:0;width:96px;font-size:11px;font-weight:700;color:' + m.col + '">' + m.label + dupNote + '</span>' +
          '<span style="flex:1;font-size:13px;color:' + (removed ? "#6b5560" : "#cdd9e5") + ';' + (removed ? "text-decoration:line-through" : "") + '">' + esc(s.text) + '</span>' +
          '<span style="flex-shrink:0;font-size:11px;color:#8aa0b4;width:54px;text-align:right">' + tk + ' tok</span>' +
          (removed ? '<span style="flex-shrink:0;font-size:14px;color:#ff6b6b">✕</span>' : '<span style="flex-shrink:0;font-size:14px;color:#46d39a">✓</span>') +
          '</div>';
      });

      const allOn = quality && dedup && safety;
      const noneOn = !quality && !dedup && !safety;

      let html = "";
      // before -> after counters
      html += '<div class="stats" style="margin-bottom:10px">' +
        '<div class="stat" title="Snippets before → after cleaning."><div class="k">snippets</div><div class="v">' +
          totalSnip + ' <span style="color:#8aa0b4">→</span> <span style="color:' + (keptSnip < totalSnip ? "#46d39a" : "#e6edf3") + '">' + keptSnip + '</span></div></div>' +
        '<div class="stat" title="Tokens before → after cleaning."><div class="k">tokens</div><div class="v">' +
          totalTok + ' <span style="color:#8aa0b4">→</span> <span style="color:' + (keptTok < totalTok ? "#46d39a" : "#e6edf3") + '">' + keptTok + '</span></div></div>' +
        '<div class="stat" title="How many snippets the active filters dropped."><div class="k">dropped</div><div class="v" style="color:#ff6b6b">' + removedSnip + '</div></div>' +
        '</div>';

      html += '<div class="phase-tag">' + (noneOn ? "raw crawl (nothing cleaned yet)" : "corpus after the active filters") + '</div>';
      html += '<div style="margin-top:6px">' + rows + '</div>';

      // verdict
      if (noneOn) {
        html += '<div style="margin-top:12px;padding:10px 12px;border-radius:8px;background:#3a2330;border:1px solid #ff6b6b" title="Training on this would teach the model to imitate the junk.">' +
          '<b style="color:#ff6b6b">Raw crawl — do not train on this.</b> It’s full of spam, gibberish, ' +
          'duplicates, and a toxic line. Garbage in, garbage out: the model would learn to produce all of it.</div>';
      } else if (allOn) {
        html += '<div style="margin-top:12px;padding:10px 12px;border-radius:8px;background:#1f3a2e;border:1px solid #46d39a" title="A smaller but much cleaner corpus.">' +
          '<b style="color:#46d39a">Clean corpus — better model.</b> Smaller (' + keptSnip + ' of ' + totalSnip +
          ' snippets, ' + keptTok + ' of ' + totalTok + ' tokens) but only fluent, unique, safe writing left. ' +
          'Quality often beats raw quantity.</div>';
      } else {
        html += '<div style="margin-top:12px;padding:10px 12px;border-radius:8px;background:#23303a;border:1px solid #4ea1ff" title="Some junk still remains.">' +
          '<b style="color:#4ea1ff">Partly cleaned.</b> ' + removedSnip + ' junk snippet(s) dropped so far. ' +
          'Turn on the remaining filters (or <b>Clean all</b>) to finish the corpus.</div>';
      }

      html += '<div class="hint">Each step trades <b>quantity</b> for <b>quality</b>: fewer tokens, but real, ' +
        'unique, safe text. Note Deduplicate keeps the <i>first</i> copy and drops the exact repeats — ' +
        'repeats would over-weight that text and push the model toward memorizing it verbatim. The model ' +
        'becomes what it reads, so a clean corpus is worth more than a bigger dirty one.</div>';

      out.innerHTML = html;
    }

    host.appendChild(panel);
    sync();
  }

  // =========================================================================
  // DEMO 4 — Overfitting (train vs. validation loss)
  // =========================================================================
  // Honest toy math. Training time t runs 0..1 (think "epochs over the data").
  //   TRAINING loss  = monotonically DECREASING toward a floor (the model keeps
  //     getting better at the text it sees):  trainFloor + trainAmp * e^(-kTrain * t)
  //   VALIDATION loss = a U-shape: an exponential FALL (it's learning real patterns)
  //     PLUS a rising "overfit" term (it starts memorizing, which hurts unseen text):
  //       valFloor + valAmp * e^(-kVal * t) + over * t
  //     The minimum of this U is where early stopping would halt (best model).
  // Toggles reshape the parameters to show each cause/remedy:
  //   model size  -> capacity to memorize (trainFloor, over)
  //   data amount -> how hard memorizing is (over, valFloor)
  //   regularization -> actively shrinks 'over' (a hurdle; nudges trainFloor up a touch)
  function buildOverfitting(host, P) {
    const panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["Overfitting — watch the two loss curves split"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["Drag the training-time knob (more epochs over the data). The blue TRAINING loss keeps sliding " +
       "down — but the orange VALIDATION loss (a held-out set the model never trains on) falls, bottoms " +
       "out, then turns back UP. Where they split is overfitting; the ★ is the best model."]));

    const guide = P.el("details", { class: "guide" });
    guide.innerHTML =
      '<summary>What am I looking at?</summary>' +
      '<div class="guide-body">' +
        'Training does exactly one thing: push down the loss on the <b>text the model is shown</b>. But ' +
        'that’s only a <b>proxy</b> — what we actually want is good predictions on text it has ' +
        '<b>never seen</b>. So we split our data: a <b style="color:#4ea1ff">training set</b> the model ' +
        'learns from, and a held-out <b style="color:#ff9d5c">validation set</b> we <b>never train on</b>, ' +
        'kept as a fair stand-in for unseen text.<br><br>' +
        'Then we watch <b>two</b> losses as training proceeds (left → right = more epochs):' +
        '<table><tr><th>curve</th><th>what it does</th><th>why</th></tr>' +
        '<tr><td><b style="color:#4ea1ff">training loss</b></td><td>keeps falling toward 0</td><td>it can always get better at text it can see — even by memorizing it</td></tr>' +
        '<tr><td><b style="color:#ff9d5c">validation loss</b></td><td>falls, then turns back <b>UP</b></td><td>real patterns help at first; later it just memorizes, which <i>hurts</i> unseen text</td></tr></table>' +
        '<b>Phase 1:</b> both fall together (learning general patterns). <b>The ★</b> = validation’s ' +
        'lowest point = the <b>best</b> model = where <b>early stopping</b> halts. <b>Phase 2:</b> the curves ' +
        '<b>SPLIT</b> — training keeps dropping while validation rises. That growing <b>gap</b> is ' +
        'overfitting.<br><br>' +
        'Use the toggles to change the three things that drive it: <b>model size</b> (capacity to memorize), ' +
        '<b>data amount</b> (more data makes memorizing harder), and <b>regularization</b> (hurdles like ' +
        'dropout/weight decay that actively fight memorizing). Watch the gap and the ★ move.' +
      '</div>';
    panel.appendChild(guide);

    // ---- state: the three cause/remedy toggles + training-time knob ----
    let size = "right";    // "small" | "right" | "large"
    let data = "normal";   // "little" | "normal" | "lots"
    let reg = false;       // regularization on/off
    let t = 0.5;           // training time 0..1

    function chip(label, active, title, onclick) {
      return P.el("button", { class: "demo-chip" + (active ? " active" : ""), title: title, onclick: onclick }, [label]);
    }

    const row1 = P.el("div", { class: "controls" });
    row1.appendChild(P.el("span", { class: "name", title: "How much capacity the model has to memorize specifics." }, ["model size:"]));
    const sizeWrap = P.el("span", {});
    row1.appendChild(sizeWrap);
    panel.appendChild(row1);

    const row2 = P.el("div", { class: "controls" });
    row2.appendChild(P.el("span", { class: "name", title: "How much (varied) data the model trains on." }, ["data amount:"]));
    const dataWrap = P.el("span", {});
    row2.appendChild(dataWrap);
    panel.appendChild(row2);

    const row3 = P.el("div", { class: "controls" });
    row3.appendChild(P.el("span", { class: "name", title: "Hurdles (dropout / weight decay) that make memorizing harder." }, ["regularization:"]));
    const regWrap = P.el("span", {});
    row3.appendChild(regWrap);
    panel.appendChild(row3);

    function renderChips() {
      sizeWrap.innerHTML = "";
      [["small", "Small (underfits)"], ["right", "Right-sized"], ["large", "Large (overfits)"]].forEach(([k, lbl]) =>
        sizeWrap.appendChild(chip(lbl, size === k, "Set model capacity to " + lbl + ".", () => { size = k; run(); })));
      dataWrap.innerHTML = "";
      [["little", "Little data"], ["normal", "Normal"], ["lots", "Lots of data"]].forEach(([k, lbl]) =>
        dataWrap.appendChild(chip(lbl, data === k, "Set training data to " + lbl + ".", () => { data = k; run(); })));
      regWrap.innerHTML = "";
      regWrap.appendChild(chip(reg ? "ON" : "OFF", reg, "Toggle regularization (dropout / weight decay).", () => { reg = !reg; run(); }));
    }

    // ---- training-time slider ----
    const slr = P.el("div", { class: "slider-row" });
    slr.appendChild(P.el("span", { class: "name", title: "How long we train: more epochs over the same data." }, ["training time"]));
    const range = P.el("input", { type: "range", min: "0", max: "1000", value: String(Math.round(t * 1000)),
      title: "Drag to train longer (more epochs). Watch the two curves." });
    const valLbl = P.el("span", { class: "val" }, []);
    slr.appendChild(range); slr.appendChild(valLbl);
    panel.appendChild(slr);

    const canvas = P.el("canvas", { class: "loss", style: "height:250px",
      title: "Training loss (blue, always falling) and validation loss (orange, U-shaped) vs. training time." });
    panel.appendChild(canvas);

    const out = P.el("div", { class: "explain" });
    panel.appendChild(out);

    // ---- honest toy math: parameters from the toggles -------------------
    function params() {
      // base = right-sized model, normal data, no reg
      let trainFloor = 0.5, trainAmp = 3.4, kTrain = 4.2;
      let valFloor = 1.2, valAmp = 3.0, kVal = 3.8, over = 1.7;

      if (size === "small") {            // underfit: can't get low even on training; little memorizing
        trainFloor = 1.7; trainAmp = 2.4; kTrain = 3.6;
        valFloor = 1.9; valAmp = 2.2; over = 0.35;
      } else if (size === "large") {     // big capacity: memorizes training to ~0, strong overfit
        trainFloor = 0.06; trainAmp = 3.9; kTrain = 4.6;
        valFloor = 1.25; valAmp = 3.1; over = 3.3;
      }

      if (data === "little") { over *= 1.9; valFloor += 0.35; }   // less data -> easier to memorize, worse floor
      else if (data === "lots") { over *= 0.4; valFloor -= 0.22; } // more varied data -> memorizing is harder

      if (reg) { over *= 0.45; trainFloor += 0.18; valFloor -= 0.05; } // hurdle: less overfit, slightly worse train fit

      over = Math.max(0, over);
      return { trainFloor, trainAmp, kTrain, valFloor, valAmp, kVal, over };
    }
    function trainAt(tt, p) { return p.trainFloor + p.trainAmp * Math.exp(-p.kTrain * tt); }
    function valAt(tt, p)   { return p.valFloor + p.valAmp * Math.exp(-p.kVal * tt) + p.over * tt; }
    // find the validation minimum (best model / early-stopping point) by scanning
    function bestT(p) {
      let bt = 0, bv = Infinity;
      for (let i = 0; i <= 400; i++) { const tt = i / 400; const v = valAt(tt, p); if (v < bv) { bv = v; bt = tt; } }
      return { bt, bv };
    }

    function drawChart(p, best) {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth || 600, h = canvas.clientHeight || 250;
      canvas.width = w * dpr; canvas.height = h * dpr;
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const padL = 40, padR = 116, padT = 16, padB = 30;
      const x0 = padL, x1 = w - padR, y0 = h - padB, y1 = padT;
      // y-domain: from 0 to a bit above the highest starting loss
      const top = Math.max(trainAt(0, p), valAt(0, p), valAt(1, p)) * 1.05;
      const sx = (s) => x0 + (x1 - x0) * s;
      const ly = (L) => y0 - (y0 - y1) * (1 - L / top);   // higher loss -> higher up

      // axes
      ctx.strokeStyle = "#33404f"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x0, y1); ctx.lineTo(x0, y0); ctx.lineTo(x1, y0); ctx.stroke();
      ctx.fillStyle = "#9aa7b4"; ctx.font = "11px Segoe UI";
      ctx.fillText("loss", 6, y1 + 10);
      ctx.fillText("training time →", (x0 + x1) / 2 - 34, h - 6);
      ctx.fillText("0", x0 - 4, h - 6);
      ctx.fillText("more epochs", x1 - 56, h - 6);

      // shade the overfitting region (t > best.bt)
      ctx.fillStyle = "rgba(255,107,107,0.08)";
      ctx.fillRect(sx(best.bt), y1, x1 - sx(best.bt), y0 - y1);

      // training curve (blue)
      ctx.strokeStyle = "#4ea1ff"; ctx.lineWidth = 2.5; ctx.beginPath();
      for (let i = 0; i <= 200; i++) { const s = i / 200; const x = sx(s), y = ly(trainAt(s, p)); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
      ctx.stroke();
      // validation curve (orange)
      ctx.strokeStyle = "#ff9d5c"; ctx.lineWidth = 2.5; ctx.beginPath();
      for (let i = 0; i <= 200; i++) { const s = i / 200; const x = sx(s), y = ly(valAt(s, p)); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
      ctx.stroke();

      // best-model marker (star + dashed vertical) at validation minimum
      ctx.strokeStyle = "#46d39a"; ctx.lineWidth = 1; ctx.setLineDash([3, 4]); ctx.beginPath();
      ctx.moveTo(sx(best.bt), y0); ctx.lineTo(sx(best.bt), y1); ctx.stroke(); ctx.setLineDash([]);
      drawStar(ctx, sx(best.bt), ly(best.bv), 7, "#46d39a");
      ctx.fillStyle = "#46d39a"; ctx.fillText("★ best (early stop)", sx(best.bt) + 6, y1 + 12);

      // current training-time marker (vertical + dots on both curves)
      const cx = sx(t);
      ctx.strokeStyle = "#ffffff"; ctx.globalAlpha = 0.35; ctx.lineWidth = 1; ctx.beginPath();
      ctx.moveTo(cx, y0); ctx.lineTo(cx, y1); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillStyle = "#4ea1ff"; ctx.beginPath(); ctx.arc(cx, ly(trainAt(t, p)), 5, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#ff9d5c"; ctx.beginPath(); ctx.arc(cx, ly(valAt(t, p)), 5, 0, Math.PI * 2); ctx.fill();

      // legend (right margin)
      ctx.font = "11px Segoe UI";
      ctx.fillStyle = "#4ea1ff"; ctx.fillText("— training loss", x1 + 8, y1 + 12);
      ctx.fillStyle = "#ff9d5c"; ctx.fillText("— validation loss", x1 + 8, y1 + 28);
      ctx.fillStyle = "#ff6b6b"; ctx.fillText("▒ overfitting", x1 + 8, y1 + 44);
    }
    function drawStar(ctx, cx, cy, r, col) {
      ctx.fillStyle = col; ctx.beginPath();
      for (let i = 0; i < 10; i++) {
        const ang = -Math.PI / 2 + i * Math.PI / 5;
        const rad = i % 2 === 0 ? r : r * 0.45;
        const x = cx + Math.cos(ang) * rad, y = cy + Math.sin(ang) * rad;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.closePath(); ctx.fill();
    }

    function run() {
      renderChips();
      const p = params();
      const best = bestT(p);
      const tr = trainAt(t, p), va = valAt(t, p), gap = va - tr;
      valLbl.textContent = Math.round(t * 100) + "%";
      drawChart(p, best);

      // regime detection for the verdict
      const underfit = (size === "small");
      const pastBest = t > best.bt + 0.02;
      const nearBest = Math.abs(t - best.bt) <= 0.06;

      let html = "";
      html += '<div class="phase-tag">at this point in training</div>';
      html += '<div class="stats" style="margin-top:4px">' +
        '<div class="stat" title="How far through training (epochs over the data)."><div class="k">training time</div><div class="v">' + Math.round(t * 100) + '%</div></div>' +
        '<div class="stat" title="Loss on the text the model trains on. Lower is better."><div class="k">training loss</div><div class="v" style="color:#4ea1ff">' + P.fmt(tr) + '</div></div>' +
        '<div class="stat" title="Loss on held-out text the model never trains on — the real measure."><div class="k">validation loss</div><div class="v" style="color:#ff9d5c">' + P.fmt(va) + '</div></div>' +
        '<div class="stat" title="validation − training. A big, growing gap = overfitting."><div class="k">gap</div><div class="v" style="color:' + (gap > 1.2 ? "#ff6b6b" : "#cdd9e5") + '">' + P.fmt(gap) + '</div></div>' +
        '</div>';

      // narrate the current regime
      if (underfit) {
        html += '<div class="phase-tag" style="margin-top:14px">regime: underfitting</div>';
        html += '<p style="margin:4px 0;color:#cdd9e5">This model is <b>too small</b>. Notice <b>both</b> losses ' +
          'stay <b>high</b> and close together — it can’t even fit the training text, let alone generalize. ' +
          'The fix isn’t early stopping; it’s <b>more capacity</b> (a bigger model). Try “Right-sized”.</p>';
      } else if (!pastBest && !nearBest) {
        html += '<div class="phase-tag" style="margin-top:14px">phase 1: both falling — healthy</div>';
        html += '<p style="margin:4px 0;color:#cdd9e5">Both curves are still dropping together: the model is learning ' +
          '<b>real patterns</b> that help on seen <i>and</i> unseen text. Keep going — you haven’t reached the ' +
          'best point (★) yet.</p>';
      } else if (nearBest) {
        html += '<div class="phase-tag" style="margin-top:14px">the sweet spot (★)</div>';
        html += '<p style="margin:4px 0;color:#cdd9e5">Validation loss is at its <b>lowest</b> — this is the ' +
          '<b>best model</b> on unseen text. <b>Early stopping</b> would halt right here, before things get worse.</p>';
      } else {
        html += '<div class="phase-tag" style="margin-top:14px">phase 2: overfitting</div>';
        html += '<p style="margin:4px 0;color:#cdd9e5">The curves have <b>split</b>: training loss keeps dropping ' +
          '(the model is <b>memorizing</b> the training text) while validation loss has turned <b>back up</b> — ' +
          'it’s getting <b>worse on unseen text</b>. You’ve trained <b>too long</b>; you should have stopped ' +
          'at the ★.</p>';
      }

      // remedy hint tied to current toggles
      let remedy = "";
      if (size === "large" && data !== "lots") remedy = "This big model is memorizing. Try <b>Lots of data</b> or turn <b>regularization ON</b> — watch the gap shrink and the ★ slide right.";
      else if (data === "little") remedy = "With <b>little data</b> the model memorizes fast (steep validation rebound). Switch to <b>Lots of data</b> to flatten the overfit.";
      else if (!reg && size !== "small") remedy = "Turn <b>regularization ON</b> (dropout / weight decay) to actively fight memorizing — the gap shrinks.";
      else if (reg) remedy = "Regularization is ON: the validation rebound is gentler and the ★ comes later. Notice training loss is a touch higher — that’s the hurdle doing its job.";
      if (remedy) html += '<div class="phase-tag" style="margin-top:10px">remedy</div><p style="margin:4px 0;color:#cdd9e5">' + remedy + '</p>';

      html += '<div class="hint">Watch the <b>validation</b> line, not the training line. Training loss falling always ' +
        'looks like progress — but once validation loss starts <b>rising</b>, “progress” has become ' +
        'memorization. Causes → remedies: too much capacity → right-size; too little data → more clean data; ' +
        'too long → early stopping (★); plus regularization to make memorizing harder.</div>';

      out.innerHTML = html;
    }

    range.addEventListener("input", () => { t = (+range.value) / 1000; run(); });
    window.addEventListener("resize", () => run());

    host.appendChild(panel);
    run();
  }
})();
