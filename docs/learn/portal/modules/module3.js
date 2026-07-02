/* ============================================================================
   Module 3 tab — Language as data (interactive demos)

   Two demos, chosen from a sub-menu:
     1. Word map      — words as points in 2-D; similar words cluster; do the
                        famous "king - man + woman ≈ queen" vector arithmetic.
     2. Tokenizer     — type a word, watch BPE merge characters into sub-word
                        tokens by replaying a learned, ranked merge list.

   Registers itself with the portal engine via Portal.register({...}).
   ============================================================================ */
(function () {
  "use strict";

  Portal.register({
    id: "module3",
    tab: "Module 3 — Language as data",
    title: "Module 3 — Language as data",
    intro: "Turn text into numbers the network can read. See words live as vectors in space " +
           "(similar meanings sit close), and watch sub-word tokenization split text into pieces.",
    render: function (mount, P) {
      const demos = [
        { id: "map", name: "1 · Word map (embeddings)", build: buildWordMap },
        { id: "tok", name: "2 · Tokenizer (BPE)", build: buildTokenizer }
      ];
      let current = "map";
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
  // DEMO 1 — Word map (embeddings as points; vector arithmetic)
  // =========================================================================
  // Each word has a hand-placed 2-D "embedding" so clusters are visible and the
  // analogy parallelograms work (king-man+woman = queen; dog-puppy+kitten = cat).
  const WORDS = [
    { w: "man",    x: 0,    y: 0,    c: "people" },
    { w: "woman",  x: 2,    y: 0,    c: "people" },
    { w: "king",   x: 0,    y: 2,    c: "people" },
    { w: "queen",  x: 2,    y: 2,    c: "people" },
    { w: "kitten", x: 10,   y: 9,    c: "animal" },
    { w: "cat",    x: 10,   y: 11,   c: "animal" },
    { w: "puppy",  x: 12,   y: 9,    c: "animal" },
    { w: "dog",    x: 12,   y: 11,   c: "animal" },
    { w: "lion",   x: 9,    y: 13,   c: "animal" },
    { w: "tiger",  x: 11,   y: 13.5, c: "animal" },
    { w: "car",    x: 1,    y: 11,   c: "vehicle" },
    { w: "truck",  x: 2.5,  y: 11.5, c: "vehicle" },
    { w: "bus",    x: 1.5,  y: 12.8, c: "vehicle" },
    { w: "bike",   x: 0.5,  y: 9.8,  c: "vehicle" },
    { w: "apple",  x: 11,   y: 1,    c: "food" },
    { w: "bread",  x: 12.5, y: 1.6,  c: "food" },
    { w: "cheese", x: 11.5, y: 2.6,  c: "food" }
  ];
  const CAT_COLOR = { people: "#4ea1ff", animal: "#46d39a", vehicle: "#ffd24e", food: "#ff8db4" };

  function buildWordMap(host, P) {
    const panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["Words as vectors — the meaning map"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["Every word is a point (a 2-D embedding). Similar meanings sit close. Real models use " +
       "hundreds of dimensions, but the idea is exactly this — distance = relatedness."]));

    const guide = P.el("details", { class: "guide" });
    guide.innerHTML =
      '<summary>What am I looking at?</summary>' +
      '<div class="guide-body">' +
        'Each dot is a <b>word</b>, placed at its <b>embedding</b> — the learned vector that ' +
        'represents its meaning. Here the vector is just 2 numbers (x, y) so we can see it; real ' +
        'embeddings have hundreds, but the principle is identical: <b>words with similar meaning ' +
        'sit close together</b>, unrelated words sit far apart. Colors mark rough categories.<br><br>' +
        '<b>Vector arithmetic:</b> because directions carry meaning, you can do math on words. ' +
        '<code>king − man + woman</code> lands almost exactly on <b>queen</b> — the "make it royal" ' +
        'and "make it female" directions are consistent across the space. Try it below (also: ' +
        '<code>dog − puppy + kitten ≈ cat</code>, the "grown-up" direction). Hover any dot to see ' +
        'its vector and nearest neighbours.' +
      '</div>';
    panel.appendChild(guide);

    const layout = P.el("div", { class: "layout" });
    const stage = P.el("div", {});
    const side = P.el("div", {});
    layout.appendChild(stage); layout.appendChild(side);
    panel.appendChild(layout);

    const mapBox = P.el("div", {});
    mapBox.title = "The meaning map. Each dot is a word at its embedding vector. Hover a dot for its vector and nearest word.";
    stage.appendChild(mapBox);
    const legend = P.el("div", { class: "legend" });
    legend.innerHTML = Object.keys(CAT_COLOR).map(c =>
      `<span class="item"><span class="swatch" style="background:${CAT_COLOR[c]}"></span> ${c}</span>`).join("") +
      '<span class="item">distance = relatedness</span>' +
      '<span class="item"><span class="swatch" style="background:none;border:2px solid #fff"></span> arithmetic result / match</span>';
    stage.appendChild(legend);
    stage.appendChild(P.el("p", { class: "caption" },
      ["Hover any dot to see its vector (x, y) and its nearest neighbour. Words in the same colour are roughly the same kind of thing — and they cluster, because similar meanings get similar vectors."]));

    // --- arithmetic controls ---
    side.appendChild(P.el("div", { class: "explain" }, [P.el("div", { class: "phase-tag" }, ["vector arithmetic"]),
      P.el("div", { html: "Directions in this space carry meaning, so you can do math on words: <code>A − B + C</code>. Pick three words and hit compute — e.g. <b>king − man + woman</b>. The result is a new point; we then find the nearest real word to it." })]));
    function wordSelect(def, tip) {
      const s = P.el("select", { title: tip || "Pick a word for the arithmetic." });
      s.style.cssText = "background:#0d141c;color:#e6edf3;border:1px solid #33404f;border-radius:8px;padding:7px 10px;font-size:13px;";
      WORDS.forEach(o => { const opt = P.el("option", { value: o.w }, [o.w]); if (o.w === def) opt.selected = true; s.appendChild(opt); });
      return s;
    }
    const selA = wordSelect("king", "Start word (A). We begin at this word's vector."),
          selB = wordSelect("man", "Subtract this word's vector (B) — removes its 'direction of meaning'."),
          selC = wordSelect("woman", "Add this word's vector (C) — adds its 'direction of meaning'.");
    const row = P.el("div", { class: "controls" });
    row.appendChild(selA); row.appendChild(P.el("span", { title: "minus" }, ["−"]));
    row.appendChild(selB); row.appendChild(P.el("span", { title: "plus" }, ["+"]));
    row.appendChild(selC);
    const btn = P.el("button", { class: "btn primary", title: "Compute A − B + C and find the nearest real word to the result.", onclick: compute }, ["= compute"]);
    row.appendChild(btn);
    side.appendChild(row);
    const result = P.el("div", { class: "explain" });
    side.appendChild(result);
    [selA, selB, selC].forEach(s => s.addEventListener("change", () => { arith = null; draw(); result.innerHTML = ""; }));

    let arith = null;   // {ax,ay, result:{x,y}, nearest:word}

    function find(w) { return WORDS.find(o => o.w === w); }
    function nearest(x, y, exclude) {
      let best = null, bd = Infinity;
      WORDS.forEach(o => {
        if (exclude.indexOf(o.w) >= 0) return;
        const d = Math.hypot(o.x - x, o.y - y);
        if (d < bd) { bd = d; best = o; }
      });
      return { word: best, dist: bd };
    }
    function compute() {
      const a = find(selA.value), b = find(selB.value), c = find(selC.value);
      const sx = a.x - b.x, sy = a.y - b.y;          // after subtracting B
      const rx = sx + c.x, ry = sy + c.y;            // after adding C
      const nn = nearest(rx, ry, [a.w, b.w, c.w]);
      arith = { a, b, c, rx, ry, nn };
      const hit = nn.dist < 0.6;
      result.innerHTML =
        '<div class="phase-tag">step by step</div>' +
        `<div><b>1.</b> start at <b>${a.w}</b> = <code>(${P.fmt(a.x)}, ${P.fmt(a.y)})</code></div>` +
        `<div><b>2.</b> − <b>${b.w}</b> <code>(${P.fmt(b.x)}, ${P.fmt(b.y)})</code> → <code>(${P.fmt(sx)}, ${P.fmt(sy)})</code></div>` +
        `<div><b>3.</b> + <b>${c.w}</b> <code>(${P.fmt(c.x)}, ${P.fmt(c.y)})</code> → result <code>(${P.fmt(rx)}, ${P.fmt(ry)})</code></div>` +
        `<div style="margin-top:8px"><b>4.</b> nearest real word to the result: <b style="color:${hit ? '#46d39a' : '#ffd24e'}">${nn.word.w}</b> (distance ${P.fmt(nn.dist)})</div>` +
        (hit
          ? `<span class="hint">It landed essentially ON “${nn.word.w}”. That’s the point: the step “− ${b.w} + ${c.w}” is a consistent <i>direction</i> of meaning in the space, so applying it to ${a.w} moves you to the matching word.</span>`
          : '<span class="hint">Not an exact landing — these toy vectors only encode a few clean analogies. Try <b>king − man + woman</b> or <b>dog − puppy + kitten</b>.</span>');
      draw();
    }

    // --- draw the scatter ---
    function draw() {
      const W = 540, H = 440, pad = 36;
      const xs = WORDS.map(o => o.x), ys = WORDS.map(o => o.y);
      const minX = Math.min(...xs) - 1, maxX = Math.max(...xs) + 1;
      const minY = Math.min(...ys) - 1, maxY = Math.max(...ys) + 1;
      const px = x => pad + (W - 2 * pad) * (x - minX) / (maxX - minX);
      const py = y => H - (pad + (H - 2 * pad) * (y - minY) / (maxY - minY)); // flip so +y is up
      const svg = P.svgEl("svg", { class: "net", viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "xMidYMid meet" });

      // arithmetic overlay (arrows + target star)
      if (arith) {
        const line = (x1, y1, x2, y2, col) => { const l = P.svgEl("line", { x1: px(x1), y1: py(y1), x2: px(x2), y2: py(y2), stroke: col, "stroke-width": 2, "stroke-dasharray": "5 4" }); svg.appendChild(l); };
        line(arith.b.x, arith.b.y, arith.a.x, arith.a.y, "#ff6b6b");        // B→A
        line(arith.a.x, arith.a.y, arith.rx, arith.ry, "#46d39a");          // A→result (+C direction)
        const star = P.svgEl("circle", { cx: px(arith.rx), cy: py(arith.ry), r: 8, fill: "none", stroke: "#fff", "stroke-width": 2 });
        svg.appendChild(star);
        const lbl = P.svgEl("text", { x: px(arith.rx) + 10, y: py(arith.ry) - 8, class: "net-sub-label" }); lbl.textContent = "result"; svg.appendChild(lbl);
      }

      WORDS.forEach(o => {
        const isArith = arith && [arith.a.w, arith.b.w, arith.c.w].indexOf(o.w) >= 0;
        const isHit = arith && arith.nn.word.w === o.w;
        const dot = P.svgEl("circle", { cx: px(o.x), cy: py(o.y), r: isHit ? 9 : 6, fill: CAT_COLOR[o.c], stroke: isHit ? "#fff" : (isArith ? "#fff" : "#0d141c"), "stroke-width": isHit ? 3 : 1 });
        const nn = nearest(o.x, o.y, [o.w]);
        const tt = P.svgEl("title"); tt.textContent = `${o.w}  —  vector (${P.fmt(o.x)}, ${P.fmt(o.y)}), category: ${o.c}. Nearest word: ${nn.word.w}.`;
        dot.appendChild(tt);
        svg.appendChild(dot);
        const t = P.svgEl("text", { x: px(o.x) + 9, y: py(o.y) + 4, class: "net-sub-label" }); t.textContent = o.w; svg.appendChild(t);
      });

      mapBox.innerHTML = ""; mapBox.appendChild(svg);
    }

    host.appendChild(panel);
    draw();
  }

  // =========================================================================
  // DEMO 2 — Tokenizer (BPE merge-replay)
  // =========================================================================
  // A small, ordered merge list "learned" from a tiny corpus. Tokenizing replays
  // these merges in priority order (rank 0 = highest), exactly like real BPE.
  const MERGES = [
    ["e", "r"], ["l", "o"], ["lo", "w"], ["low", "er"], ["e", "s"], ["es", "t"],
    ["low", "est"], ["n", "e"], ["ne", "w"], ["new", "er"], ["h", "u"], ["hu", "g"], ["hug", "s"]
  ];
  const RANK = {}; MERGES.forEach((m, i) => { RANK[m[0] + "" + m[1]] = i; });

  function buildTokenizer(host, P) {
    const panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["Sub-word tokenizer — BPE in action"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["Type a word and watch it split into sub-word tokens by replaying a learned, ranked list of " +
       "merges. Try: lower, lowest, newer, hugs — or a made-up word."]));

    const guide = P.el("details", { class: "guide" });
    guide.innerHTML =
      '<summary>What am I looking at?</summary>' +
      '<div class="guide-body">' +
        'A word starts as single <b>characters</b>. The tokenizer has a fixed, <b>ranked list of ' +
        'merge rules</b> (learned once from data). At each step it applies the <b>highest-priority ' +
        'merge that fits</b>, and repeats until none apply. The leftover pieces are the <b>tokens</b>.<br><br>' +
        'This is why common words collapse to one token while rare/unknown words stay in pieces ' +
        '(down to characters) — so nothing is ever out-of-vocabulary. The merge order is fixed, so ' +
        'the same word always tokenizes the same way.<br><br>' +
        '<b>Learned merge rules (in priority order):</b><br><code>' +
        MERGES.map((m, i) => `${i}: ${m[0]}+${m[1]}→${m[0] + m[1]}`).join("&nbsp; ") + '</code>' +
      '</div>';
    panel.appendChild(guide);

    const row = P.el("div", { class: "controls" });
    const input = P.el("input", { type: "text", value: "lower", class: "tokin", title: "Type a word to tokenize." });
    input.style.cssText = "background:#0d141c;color:#e6edf3;border:1px solid #33404f;border-radius:8px;padding:8px 12px;font-size:14px;";
    const go = P.el("button", { class: "btn primary", title: "Split this word into sub-word tokens by replaying the merge rules.", onclick: run }, ["Tokenize ▶"]);
    row.appendChild(P.el("span", { class: "name" }, ["word:"])); row.appendChild(input); row.appendChild(go);
    panel.appendChild(row);
    const exRow = P.el("div", { class: "controls" });
    exRow.appendChild(P.el("span", { class: "name" }, ["try:"]));
    [["lower", "known word → one token"], ["lowest", "known → one token"], ["newer", "known → one token"],
     ["hugs", "known → one token"], ["flower", "splits into [f][lower]"], ["blorp", "unknown → small pieces (never OOV)"]].forEach(pair =>
      exRow.appendChild(P.el("button", { class: "demo-chip", title: pair[1], onclick: () => { input.value = pair[0]; run(); } }, [pair[0]])));
    panel.appendChild(exRow);
    panel.appendChild(P.el("p", { class: "caption" },
      ["Each row below is one merge step; the two green-highlighted pieces are the pair being merged (the highest-priority rule that fits). Common words collapse to one token; unknown words stay in small pieces — never out-of-vocabulary."]));

    const steps = P.el("div", { class: "explain" });
    panel.appendChild(steps);

    function tokenize(word) {
      let toks = word.split("");
      const trace = [{ toks: toks.slice(), note: "start: split into characters", mi: -1 }];
      while (true) {
        // find the highest-priority applicable adjacent pair
        let bestRank = Infinity, bestIdx = -1;
        for (let i = 0; i < toks.length - 1; i++) {
          const r = RANK[toks[i] + "" + toks[i + 1]];
          if (r !== undefined && r < bestRank) { bestRank = r; bestIdx = i; }
        }
        if (bestIdx < 0) break;
        const a = toks[bestIdx], b = toks[bestIdx + 1];
        toks = toks.slice(0, bestIdx).concat([a + b], toks.slice(bestIdx + 2));
        trace.push({ toks: toks.slice(), note: `apply rule ${bestRank}:  ${a}+${b} → ${a + b}`, mi: bestIdx });
      }
      return trace;
    }

    function run() {
      const word = (input.value || "").toLowerCase().replace(/[^a-z]/g, "");
      if (!word) { steps.innerHTML = '<span class="hint">Type a word first.</span>'; return; }
      const trace = tokenize(word);
      const chip = (t, hot) => `<span style="display:inline-block;background:${hot ? '#2c5a44' : '#232d3a'};border:1px solid ${hot ? '#46d39a' : '#33404f'};border-radius:6px;padding:2px 8px;margin:2px;font-family:monospace">${t}</span>`;
      let html = '<div class="phase-tag">tokenizing, step by step</div>';
      trace.forEach((s) => {
        html += `<div style="margin:5px 0">${s.toks.map((t, i) => chip(t, i === s.mi)).join("")} <span class="hint">${s.note}</span></div>`;
      });
      const final = trace[trace.length - 1].toks;
      html += `<div style="margin-top:10px"><b>Final tokens (${final.length}):</b> ${final.map(t => chip(t, false)).join("")}</div>`;
      html += `<div class="hint">No more rules apply. Each token now becomes an ID number → an embedding vector → into the network.</div>`;
      steps.innerHTML = html;
    }

    host.appendChild(panel);
    run();
  }
})();
