/* ============================================================================
   Module 4 tab — The Transformer (interactive demos)

   Demo 1 · Attention blend — the attention idea made visible:
     A 4-word sentence "the [subject] will eat ___". The focus word "eat" looks
     at every word, scores each by a DOT PRODUCT (similarity), softens the scores
     into WEIGHTS that sum to 1 (softmax), then rebuilds itself as a WEIGHTED
     BLEND of the word vectors. Toggle the subject (lion / rabbit / …) and watch
     "eat" become predator-flavoured or plant-flavoured — so the next-word guess
     flips between meat and grass. This is the exact computation worked through by hand in the notes.

   Demo 2 · Self-attention (Q/K/V) — the same blend, made learnable:
     Each word emits THREE vectors via three learned matrices — a query (what I
     look for), a key (how I'm found), a value (what I contribute). Score =
     query·key, softmax → weights, output = weighted sum of the VALUES. The three
     roles are shown side by side so "match on the key, consume the value" is visible.

   Demo 3 · Multi-head attention — several attentions in parallel:
     Two heads, each with its OWN Q/K/V, learn DIFFERENT relationships (verb→subject
     vs. verb→object). Each head's separate attention weights are shown, then the
     head outputs are concatenated and combined by W_O into one rich vector. The
     point made visible: different heads specialize.

   Demo 4 · Positional information — attention is order-blind:
     Take "dog bites man", reorder the SAME words, and with NO position vectors the
     blend is byte-for-byte identical (a bag of words). Toggle position vectors ON
     and watch the same word at a different slot become a different vector, so the
     two orders finally differ. Shows position vectors being ADDED onto embeddings.

   Demo 5 · A transformer block — one vector through the whole block:
     LayerNorm → attention → +residual → LayerNorm → feed-forward (matrix→ReLU→
     matrix) → +residual → output, narrated step by step. The "add, don't replace"
     residual and the "recenter + rescale" normalize step are made visible.

   Registers itself with the portal engine via Portal.register({...}).
   ============================================================================ */
(function () {
  "use strict";

  Portal.register({
    id: "module4",
    tab: "Module 4 — The Transformer",
    title: "Module 4 — The Transformer",
    intro: "Attention lets every word look at the others and rewrite itself as a blend of the " +
           "relevant ones. These five demos build the Transformer piece by piece: the attention " +
           "blend, the learnable Q/K/V version, many heads in parallel, the position stamp that " +
           "gives word order, and finally the whole transformer block assembled end to end.",
    render: function (mount, P) {
      const demos = [
        { id: "attn",     name: "1 · Attention blend",        build: buildAttention },
        { id: "qkv",      name: "2 · Self-attention (Q/K/V)", build: buildSelfAttention },
        { id: "multi",    name: "3 · Multi-head attention",   build: buildMultiHead },
        { id: "pos",      name: "4 · Positional information",  build: buildPositional },
        { id: "block",    name: "5 · A transformer block",    build: buildBlock }
      ];
      let current = "attn";
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
  // shared little helpers (toy 2-D vector math), reused by every demo below
  // =========================================================================
  function softmax(scores) {
    const m = Math.max.apply(null, scores);
    const ex = scores.map(s => Math.exp(s - m));
    const sum = ex.reduce((a, b) => a + b, 0);
    return ex.map(e => e / sum);
  }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1]; }
  // 2x2 matrix (rows) times a 2-vector
  function matVec(M, v) { return [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]]; }

  // =========================================================================
  // DEMO 1 — Attention blend
  // =========================================================================
  // Toy 2-D embeddings. Slot 1 = "meat / predator-ness", slot 2 = "grass / plant-ness".
  // Every subject sums to 10 on its two slots, so the *scores* are identical across
  // subjects (fair comparison) and only the *direction* of the blend differs.
  const SUBJECTS = [
    { w: "lion",   v: [10, 0],  note: "predator → expects meat" },
    { w: "tiger",  v: [9, 1],   note: "predator → expects meat" },
    { w: "rabbit", v: [0, 10],  note: "herbivore → expects grass" },
    { w: "cow",    v: [1, 9],   note: "herbivore → expects grass" }
  ];
  const TEMP = 5;                  // divides raw scores before softmax, so the blend mixes
                                   // (real attention divides by √d for the same reason)
  const FOCUS = { w: "eat", v: [1, 1] };   // the word doing the looking (predicting the next word)

  function buildAttention(host, P) {
    const panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["Attention — a word rewrites itself from its neighbours"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["The sentence is “the [subject] will eat ___”. The word “eat” looks at every word, " +
       "scores each by similarity, and blends in the relevant ones. Switch the subject and watch “eat” change."]));

    const guide = P.el("details", { class: "guide" });
    guide.innerHTML =
      '<summary>What am I looking at?</summary>' +
      '<div class="guide-body">' +
        'Every word is a tiny <b>2-number vector (embedding)</b>. Here slot 1 = ' +
        '<b style="color:#ff8db4">meat-ness</b> and slot 2 = <b style="color:#46d39a">grass-ness</b> ' +
        '(real embeddings have hundreds of unnamed slots — same idea, bigger).<br><br>' +
        '<b>1 · Score (similarity).</b> The focus word <code>eat</code> takes a <b>dot product</b> with ' +
        'every word. A dot product is a similarity score — the subject scores high, filler words ' +
        '(<code>the</code>, <code>will</code>) score ~0.<br>' +
        '<b>2 · Weights (sum to 1).</b> The raw scores are softened into fractions that add up to ' +
        '100% (via <b>softmax</b>). These are how much of each word to mix in.<br>' +
        '<b>3 · Blend.</b> Multiply each word’s vector by its weight and add them up — a ' +
        '<b>weighted average</b>. The new <code>eat</code> vector now leans toward the subject’s flavour, ' +
        'so the next-word guess flips between <b>meat</b> and <b>grass</b>.<br><br>' +
        'Scores are divided by a small number before softmax so the blend <i>mixes</i> rather than ' +
        'snapping 100% onto one word (real attention divides by √d for exactly this reason). ' +
        'Hover any chip, bar, or number for what it means.' +
      '</div>';
    panel.appendChild(guide);

    // --- subject picker ---
    let subjectIdx = 0;
    const pick = P.el("div", { class: "controls" });
    pick.appendChild(P.el("span", { class: "name", title: "The subject of the sentence — who is doing the eating." }, ["subject:"]));
    SUBJECTS.forEach((s, i) => pick.appendChild(P.el("button", {
      class: "demo-chip" + (i === 0 ? " active" : ""),
      title: s.note,
      onclick: () => { subjectIdx = i; Array.from(pick.querySelectorAll(".demo-chip")).forEach((b, j) => b.className = "demo-chip" + (j === i ? " active" : "")); run(); }
    }, [s.w])));
    panel.appendChild(pick);

    panel.appendChild(P.el("p", { class: "caption" },
      ["The focus word is “eat” (it’s about to predict the next word). Each bar shows how much " +
       "attention “eat” pays to that word; the blend below rebuilds “eat” from those weights."]));

    const out = P.el("div", { class: "explain" });
    panel.appendChild(out);

    function run() {
      const subj = SUBJECTS[subjectIdx];
      // The sentence words (the focus word "eat" attends over all of them, itself included).
      const words = [
        { w: "the",   v: [0, 0] },
        { w: subj.w,  v: subj.v },
        { w: "will",  v: [0, 0] },
        { w: "eat",   v: FOCUS.v }
      ];
      const rawScores = words.map(o => dot(FOCUS.v, o.v));
      const scaled = rawScores.map(s => s / TEMP);
      const weights = softmax(scaled);
      // weighted blend
      const blend = [0, 0];
      words.forEach((o, i) => { blend[0] += weights[i] * o.v[0]; blend[1] += weights[i] * o.v[1]; });
      const predMeat = blend[0] > blend[1];
      const predWord = predMeat ? "meat" : "grass";
      const predColor = predMeat ? "#ff8db4" : "#46d39a";

      const maxW = Math.max.apply(null, weights);

      let html = '';

      // sentence row
      html += '<div class="phase-tag">the sentence (each word = a 2-D vector)</div><div style="margin:4px 0 12px">';
      words.forEach((o) => {
        const isFocus = o.w === "eat";
        html += `<span title="${o.w} = vector [${P.fmt(o.v[0])}, ${P.fmt(o.v[1])}]  (meat-ness, grass-ness)" ` +
          `style="display:inline-block;background:${isFocus ? '#1f3a4d' : '#232d3a'};border:1px solid ${isFocus ? '#4ea1ff' : '#33404f'};border-radius:8px;padding:6px 12px;margin:3px;font-family:monospace">` +
          `${o.w}${isFocus ? ' ◀ focus' : ''}<br><span style="color:#8aa0b4;font-size:12px">[${P.fmt(o.v[0])}, ${P.fmt(o.v[1])}]</span></span>`;
      });
      html += '</div>';

      // step 1+2: scores -> weights as bars
      html += '<div class="phase-tag">1· score (eat · word)  →  2· weight (softmax, sums to 1)</div>';
      words.forEach((o, i) => {
        const pct = Math.round(weights[i] * 100);
        const barW = Math.round((weights[i] / maxW) * 100);
        const hot = weights[i] === maxW;
        html += `<div style="display:flex;align-items:center;gap:8px;margin:3px 0" title="eat · ${o.w} = ${P.fmt(rawScores[i])}  → softened → weight ${P.fmt(weights[i])}">` +
          `<span style="width:48px;font-family:monospace;color:#e6edf3">${o.w}</span>` +
          `<span style="width:74px;font-family:monospace;color:#8aa0b4">score ${P.fmt(rawScores[i])}</span>` +
          `<span style="flex:1;background:#0d141c;border-radius:6px;overflow:hidden;height:18px;border:1px solid #33404f">` +
            `<span style="display:block;height:100%;width:${barW}%;background:${hot ? '#4ea1ff' : '#33506b'}"></span></span>` +
          `<span style="width:42px;text-align:right;font-family:monospace;color:${hot ? '#4ea1ff' : '#8aa0b4'}">${pct}%</span>` +
        `</div>`;
      });

      // step 3: the weighted blend, slot by slot
      const term = (i) => `${P.fmt(weights[i])}·[${P.fmt(words[i].v[0])},${P.fmt(words[i].v[1])}]`;
      html += '<div class="phase-tag" style="margin-top:12px">3· blend = weighted sum of the vectors</div>';
      html += `<div style="font-family:monospace;color:#cdd9e5;margin:4px 0;line-height:1.7">new eat = ` +
        words.map((o, i) => term(i)).join(" + ") + `</div>`;
      html += `<div style="font-family:monospace;margin:6px 0">= <b style="color:#fff">[${P.fmt(blend[0])}, ${P.fmt(blend[1])}]</b> ` +
        `<span style="color:#8aa0b4">(meat-ness ${P.fmt(blend[0])}, grass-ness ${P.fmt(blend[1])})</span></div>`;

      // result: meat vs grass meter + prediction
      const total = blend[0] + blend[1] || 1;
      const meatPct = Math.round((blend[0] / total) * 100);
      html += `<div style="display:flex;align-items:center;gap:8px;margin:10px 0" title="The new eat vector leans this way.">` +
        `<span style="width:48px;font-family:monospace;color:#ff8db4">meat</span>` +
        `<span style="flex:1;background:#46d39a;border-radius:6px;overflow:hidden;height:20px;display:flex">` +
          `<span style="display:block;height:100%;width:${meatPct}%;background:#ff8db4"></span></span>` +
        `<span style="width:48px;font-family:monospace;color:#46d39a;text-align:right">grass</span></div>`;
      html += `<div style="margin-top:10px;font-size:15px">predicted next word after “eat”: ` +
        `<b style="color:${predColor};font-size:17px">${predWord}</b></div>`;
      html += `<div class="hint">Same context-free <code>eat = [1, 1]</code> — but after attention blended in ` +
        `<b>${subj.w}</b>, its vector leans <b style="color:${predColor}">${predMeat ? 'meat' : 'grass'}</b>. ` +
        `Switch the subject above and the very same word predicts the opposite. That is attention.</div>`;

      out.innerHTML = html;
    }

    host.appendChild(panel);
    run();
  }

  // =========================================================================
  // DEMO 2 — Self-attention (Q/K/V)
  // =========================================================================
  // Same toy world, but now each word emits three vectors via three LEARNED
  // matrices. Score = query·key (not raw embedding·embedding), and the blend
  // sums the VALUES (not raw embeddings). The three roles are shown distinctly.
  //
  // The matrices are toy "lenses". W_Q is a near-identity (the searcher asks
  // with roughly its meaning). W_K leans the key toward the predator/prey axis
  // so animals advertise strongly. W_V passes meaning through (what gets poured
  // into the blend). Small numbers chosen so q·k for eat·subject ends up large.
  const QKV = {
    // eat's query is pulled toward "looking for an animal subject"
    W_Q: [[1, 0], [0, 1]],          // identity-ish: ask with your own meaning
    W_K: [[1.2, 0], [0, 1.2]],      // keys: scale up so animals advertise loudly
    W_V: [[1, 0], [0, 1]]           // values: pass the meaning through
  };

  function buildSelfAttention(host, P) {
    const panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["Self-attention — query, key, value (the learnable engine)"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["Same blend as demo 1, but now each word emits THREE vectors through three learned matrices: " +
       "a query (what I look for), a key (how I'm found), a value (what I contribute). " +
       "Score = query·key, then blend the values."]));

    const guide = P.el("details", { class: "guide" });
    guide.innerHTML =
      '<summary>What am I looking at?</summary>' +
      '<div class="guide-body">' +
        'In demo 1 each word used its <b>one</b> embedding for everything. But that one vector was ' +
        'forced to do three different jobs at once — and they genuinely differ. Self-attention gives ' +
        'each word <b>three hats</b>, made by multiplying its embedding by three learned matrices:' +
        '<br><br>' +
        '<table><tr><th>vector</th><th>made by</th><th>role — answers…</th></tr>' +
        '<tr><td><b style="color:#4ea1ff">query q</b></td><td><code>W_Q · x</code></td><td>“who is relevant <i>to me</i>?” (the text you type in a search box)</td></tr>' +
        '<tr><td><b style="color:#ffd24e">key k</b></td><td><code>W_K · x</code></td><td>“what do I match <i>against</i>?” (the keywords a page is listed under)</td></tr>' +
        '<tr><td><b style="color:#46d39a">value v</b></td><td><code>W_V · x</code></td><td>“what do I pour into the blend?” (the page content you read after clicking)</td></tr></table>' +
        '<b>The change from demo 1:</b> score is now <b>query · key</b> (your question vs. my label), ' +
        'and the blend is a weighted sum of the <b>values</b> (everyone’s content). ' +
        'The three matrices are <b>learned</b> by training — that is what makes attention learnable, ' +
        'rather than stuck using one raw embedding for all three jobs.<br><br>' +
        'The focus word <code>eat</code> is the searcher here. Watch its <b>query</b> match each ' +
        'word’s <b>key</b>, then its winning neighbours pour in their <b>value</b>. ' +
        'Hover any vector or bar for the exact numbers.' +
      '</div>';
    panel.appendChild(guide);

    // --- subject picker (reuse the same flavour) ---
    let subjectIdx = 0;
    const pick = P.el("div", { class: "controls" });
    pick.appendChild(P.el("span", { class: "name", title: "The subject of the sentence — who is doing the eating." }, ["subject:"]));
    SUBJECTS.forEach((s, i) => pick.appendChild(P.el("button", {
      class: "demo-chip" + (i === 0 ? " active" : ""),
      title: s.note,
      onclick: () => { subjectIdx = i; Array.from(pick.querySelectorAll(".demo-chip")).forEach((b, j) => b.className = "demo-chip" + (j === i ? " active" : "")); run(); }
    }, [s.w])));
    panel.appendChild(pick);

    panel.appendChild(P.el("p", { class: "caption" },
      ["“eat” is the searcher. It scores its query against every word’s key, softmaxes, " +
       "then blends in the winning words’ values to become its new context-aware vector."]));

    const out = P.el("div", { class: "explain" });
    panel.appendChild(out);

    function run() {
      const subj = SUBJECTS[subjectIdx];
      const words = [
        { w: "the",   x: [0, 0] },
        { w: subj.w,  x: subj.v },
        { w: "will",  x: [0, 0] },
        { w: "eat",   x: FOCUS.v }
      ];
      // build q/k/v for each word
      words.forEach(o => {
        o.q = matVec(QKV.W_Q, o.x);
        o.k = matVec(QKV.W_K, o.x);
        o.v = matVec(QKV.W_V, o.x);
      });
      const focus = words.find(o => o.w === "eat");
      const rawScores = words.map(o => dot(focus.q, o.k));     // query · key
      const scaled = rawScores.map(s => s / TEMP);
      const weights = softmax(scaled);
      const blend = [0, 0];
      words.forEach((o, i) => { blend[0] += weights[i] * o.v[0]; blend[1] += weights[i] * o.v[1]; });
      const predMeat = blend[0] > blend[1];
      const predWord = predMeat ? "meat" : "grass";
      const predColor = predMeat ? "#ff8db4" : "#46d39a";
      const maxW = Math.max.apply(null, weights);

      const vchip = (label, vec, col) =>
        `<span title="${label} = [${P.fmt(vec[0])}, ${P.fmt(vec[1])}]" ` +
        `style="display:inline-block;background:#232d3a;border:1px solid ${col};border-radius:6px;padding:3px 8px;margin:2px;font-family:monospace;font-size:12px">` +
        `<span style="color:${col}">${label}</span> [${P.fmt(vec[0])}, ${P.fmt(vec[1])}]</span>`;

      let html = '';

      // the three hats per word
      html += '<div class="phase-tag">each word emits 3 vectors:  q = W_Q·x,  k = W_K·x,  v = W_V·x</div>';
      html += '<div style="margin:4px 0 12px">';
      words.forEach(o => {
        const isFocus = o.w === "eat";
        html += `<div style="background:${isFocus ? '#1f3a4d' : '#232d3a'};border:1px solid ${isFocus ? '#4ea1ff' : '#33404f'};border-radius:8px;padding:6px 10px;margin:4px 0">` +
          `<span style="font-family:monospace;color:#e6edf3">${o.w}${isFocus ? ' ◀ searcher' : ''}</span> ` +
          `<span title="embedding (what word it is)" style="color:#8aa0b4;font-family:monospace;font-size:12px">x [${P.fmt(o.x[0])}, ${P.fmt(o.x[1])}]</span> &nbsp;→&nbsp; ` +
          vchip("q", o.q, "#4ea1ff") + vchip("k", o.k, "#ffd24e") + vchip("v", o.v, "#46d39a") +
        `</div>`;
      });
      html += '</div>';

      // score = query · key, then softmax
      html += '<div class="phase-tag">1· score = eat’s <span style="color:#4ea1ff">query</span> · each word’s <span style="color:#ffd24e">key</span>  →  2· softmax weights</div>';
      words.forEach((o, i) => {
        const pct = Math.round(weights[i] * 100);
        const barW = Math.round((weights[i] / maxW) * 100);
        const hot = weights[i] === maxW;
        html += `<div style="display:flex;align-items:center;gap:8px;margin:3px 0" title="q_eat · k_${o.w} = [${P.fmt(focus.q[0])},${P.fmt(focus.q[1])}] · [${P.fmt(o.k[0])},${P.fmt(o.k[1])}] = ${P.fmt(rawScores[i])} → weight ${P.fmt(weights[i])}">` +
          `<span style="width:48px;font-family:monospace;color:#e6edf3">${o.w}</span>` +
          `<span style="width:120px;font-family:monospace;color:#8aa0b4">q·k = ${P.fmt(rawScores[i])}</span>` +
          `<span style="flex:1;background:#0d141c;border-radius:6px;overflow:hidden;height:18px;border:1px solid #33404f">` +
            `<span style="display:block;height:100%;width:${barW}%;background:${hot ? '#4ea1ff' : '#33506b'}"></span></span>` +
          `<span style="width:42px;text-align:right;font-family:monospace;color:${hot ? '#4ea1ff' : '#8aa0b4'}">${pct}%</span>` +
        `</div>`;
      });

      // blend the VALUES
      const term = (i) => `${P.fmt(weights[i])}·v_${words[i].w}`;
      html += '<div class="phase-tag" style="margin-top:12px">3· blend = weighted sum of the <span style="color:#46d39a">values</span> (not the raw embeddings)</div>';
      html += `<div style="font-family:monospace;color:#cdd9e5;margin:4px 0;line-height:1.7">new eat = ` +
        words.map((o, i) => term(i)).join(" + ") + `</div>`;
      html += `<div style="font-family:monospace;margin:6px 0">= <b style="color:#fff">[${P.fmt(blend[0])}, ${P.fmt(blend[1])}]</b> ` +
        `<span style="color:#8aa0b4">(meat-ness ${P.fmt(blend[0])}, grass-ness ${P.fmt(blend[1])})</span></div>`;

      // prediction
      html += `<div style="margin-top:10px;font-size:15px">predicted next word after “eat”: ` +
        `<b style="color:${predColor};font-size:17px">${predWord}</b></div>`;
      html += `<div class="hint">The searcher matched on each word’s <b style="color:#ffd24e">key</b>, ` +
        `then consumed the winners’ <b style="color:#46d39a">value</b> — “match on the key, consume the value.” ` +
        `Because <code>W_Q</code>, <code>W_K</code>, <code>W_V</code> are <b>learned</b>, the model can ` +
        `discover which words should attend to which — that is the upgrade over demo 1.</div>`;

      out.innerHTML = html;
    }

    host.appendChild(panel);
    run();
  }

  // =========================================================================
  // DEMO 3 — Multi-head attention
  // =========================================================================
  // Two heads, each its OWN Q/K/V, learning a DIFFERENT relationship. The
  // sentence carries a subject AND an object so the two heads have different
  // things to find. Head 1 = verb→subject (eat looks at lion). Head 2 =
  // verb→object (eat looks at meat). Each head's weights are shown separately,
  // then the two head-outputs are concatenated and combined by W_O.
  function buildMultiHead(host, P) {
    const panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["Multi-head attention — different heads, different jobs"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["Sentence: “the big lion will eat meat”. The focus word “eat” wants to look at BOTH its " +
       "subject (lion: who eats) and its object (meat: what’s eaten) — two relationships. " +
       "One head can only do one. So we run two heads in parallel, then combine."]));

    const guide = P.el("details", { class: "guide" });
    guide.innerHTML =
      '<summary>What am I looking at?</summary>' +
      '<div class="guide-body">' +
        'A <b>head</b> is one complete self-attention computation — one set of Q/K/V doing ' +
        'score q·k → softmax → blend values. A single head can only learn <b>one</b> notion of ' +
        '“what’s relevant”. But a sentence has many relationships at once:' +
        '<br><br>' +
        '<b style="color:#4ea1ff">Head 1</b> has learned <b>verb → subject</b>: when <code>eat</code> ' +
        'searches, this head’s query/key make it light up on <b>lion</b> (who is doing the eating).<br>' +
        '<b style="color:#ff8db4">Head 2</b> has learned <b>verb → object</b>: the <i>same</i> word ' +
        '<code>eat</code>, with this head’s different query/key, lights up on <b>meat</b> ' +
        '(what is being eaten).<br><br>' +
        'Each head produces its own little output vector for <code>eat</code>. We then ' +
        '<b>concatenate</b> them side by side into one long vector, and multiply by a learned ' +
        '<b>output matrix W_O</b> that mixes both findings into one final vector — so the new ' +
        '<code>eat</code> now knows its subject AND its object at once.<br><br>' +
        'Nobody assigned the heads their jobs — those specialties <b>emerge from training</b>. ' +
        'Hover any bar or number for the detail.' +
      '</div>';
    panel.appendChild(guide);

    panel.appendChild(P.el("p", { class: "caption" },
      ["Both heads run on the same sentence at the same time. Compare the two attention rows: " +
       "the same word “eat” attends to different neighbours in each head."]));

    const out = P.el("div", { class: "explain" });
    panel.appendChild(out);

    // toy 2-D embeddings; slot1 = subject/animacy axis, slot2 = food/object axis.
    const words = [
      { w: "the",  x: [0, 0] },
      { w: "big",  x: [1, 0] },
      { w: "lion", x: [9, 1] },     // strong on the subject axis
      { w: "will", x: [0, 0] },
      { w: "eat",  x: [2, 2] },     // the searcher
      { w: "meat", x: [1, 9] }      // strong on the object/food axis
    ];
    // Head 1 — verb→subject: query & keys emphasise slot 1 (the subject axis).
    const HEAD1 = {
      W_Q: [[3, 0], [0, 0]],        // eat's query points along the subject axis
      W_K: [[1, 0], [0, 0]],        // keys advertise their subject-ness
      W_V: [[1, 0], [0, 0]]         // value = the subject-ness it contributes
    };
    // Head 2 — verb→object: query & keys emphasise slot 2 (the food/object axis).
    const HEAD2 = {
      W_Q: [[0, 0], [0, 3]],        // eat's query points along the object axis
      W_K: [[0, 0], [0, 1]],        // keys advertise their object-ness
      W_V: [[0, 0], [0, 1]]         // value = the object-ness it contributes
    };
    // Output matrix W_O: takes the 2-number concat [head1out, head2out] and mixes it
    // into a final 2-D vector (here it simply spreads each head onto one slot).
    const W_O = [[1, 0], [0, 1]];

    function runHead(H) {
      const ws = words.map(o => ({ w: o.w, q: matVec(H.W_Q, o.x), k: matVec(H.W_K, o.x), v: matVec(H.W_V, o.x) }));
      const focus = ws.find(o => o.w === "eat");
      const raw = ws.map(o => dot(focus.q, o.k));
      const weights = softmax(raw.map(s => s / 3));
      // each head here contributes a single number (its value is 1-D-ish on its axis)
      let outVal = 0;
      ws.forEach((o, i) => { outVal += weights[i] * (o.v[0] + o.v[1]); });
      return { ws, raw, weights, outVal };
    }

    function headRows(label, color, res) {
      const maxW = Math.max.apply(null, res.weights);
      let h = `<div class="phase-tag" style="background:${color};color:#08121f">${label}</div>`;
      res.ws.forEach((o, i) => {
        const pct = Math.round(res.weights[i] * 100);
        const barW = Math.round((res.weights[i] / maxW) * 100);
        const hot = res.weights[i] === maxW;
        h += `<div style="display:flex;align-items:center;gap:8px;margin:3px 0" title="${label}: q_eat · k_${o.w} = ${P.fmt(res.raw[i])} → weight ${P.fmt(res.weights[i])}">` +
          `<span style="width:48px;font-family:monospace;color:#e6edf3">${o.w}</span>` +
          `<span style="width:96px;font-family:monospace;color:#8aa0b4">q·k = ${P.fmt(res.raw[i])}</span>` +
          `<span style="flex:1;background:#0d141c;border-radius:6px;overflow:hidden;height:16px;border:1px solid #33404f">` +
            `<span style="display:block;height:100%;width:${barW}%;background:${hot ? color : '#33506b'}"></span></span>` +
          `<span style="width:42px;text-align:right;font-family:monospace;color:${hot ? color : '#8aa0b4'}">${pct}%</span>` +
        `</div>`;
      });
      return h;
    }

    function run() {
      const r1 = runHead(HEAD1);
      const r2 = runHead(HEAD2);
      // concatenate the two head outputs into one 2-vector, then × W_O
      const concat = [r1.outVal, r2.outVal];
      const finalVec = matVec(W_O, concat);

      const top1 = r1.ws[r1.weights.indexOf(Math.max.apply(null, r1.weights))].w;
      const top2 = r2.ws[r2.weights.indexOf(Math.max.apply(null, r2.weights))].w;

      let html = '';

      // sentence
      html += '<div class="phase-tag">the sentence (focus word = eat)</div><div style="margin:4px 0 12px">';
      words.forEach(o => {
        const isFocus = o.w === "eat";
        html += `<span title="${o.w} = [${P.fmt(o.x[0])}, ${P.fmt(o.x[1])}]  (subject-axis, object-axis)" ` +
          `style="display:inline-block;background:${isFocus ? '#1f3a4d' : '#232d3a'};border:1px solid ${isFocus ? '#4ea1ff' : '#33404f'};border-radius:8px;padding:5px 10px;margin:3px;font-family:monospace;font-size:13px">` +
          `${o.w}${isFocus ? ' ◀' : ''}</span>`;
      });
      html += '</div>';

      // two heads side by side
      html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">';
      html += '<div>' + headRows("head 1 · verb→subject", "#4ea1ff", r1) +
        `<div class="hint">Head 1’s query/key point along the <b>subject</b> axis, so “eat” attends to <b style="color:#4ea1ff">${top1}</b>.</div></div>`;
      html += '<div>' + headRows("head 2 · verb→object", "#ff8db4", r2) +
        `<div class="hint">Head 2’s query/key point along the <b>object</b> axis, so the same “eat” attends to <b style="color:#ff8db4">${top2}</b>.</div></div>`;
      html += '</div>';

      // concat + W_O
      html += '<div class="phase-tag" style="margin-top:14px">concatenate the two head outputs, then × W_O → one rich vector</div>';
      html += `<div style="font-family:monospace;color:#cdd9e5;margin:4px 0;line-height:1.8">` +
        `concat = [ <span title="head 1's output number" style="color:#4ea1ff">${P.fmt(r1.outVal)}</span> | ` +
        `<span title="head 2's output number" style="color:#ff8db4">${P.fmt(r2.outVal)}</span> ]` +
        ` &nbsp;×&nbsp; W_O &nbsp;=&nbsp; <b style="color:#fff" title="the final vector for eat, carrying BOTH relationships">[${P.fmt(finalVec[0])}, ${P.fmt(finalVec[1])}]</b></div>`;
      html += `<div class="hint">One head could only find the subject OR the object. Running two in ` +
        `parallel and fusing them with <code>W_O</code> lets the new “eat” carry <b style="color:#4ea1ff">${top1}</b> ` +
        `<i>and</i> <b style="color:#ff8db4">${top2}</b> at once. Real models use 8–12 heads, each a learned specialty.</div>`;

      out.innerHTML = html;
    }

    host.appendChild(panel);
    run();
  }

  // =========================================================================
  // DEMO 4 — Positional information
  // =========================================================================
  // "dog bites man" vs. "man bites dog": same three words, reordered. With NO
  // position vectors the blend is identical (a bag of words). Toggle positions
  // ON and the same word at a different slot becomes a different vector, so the
  // two orders finally differ. The position vectors are shown being ADDED on.
  function buildPositional(host, P) {
    const panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["Positional information — giving attention a sense of order"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["Attention is a weighted SUM, and a sum doesn’t care about order — so on its own it can’t tell " +
       "“dog bites man” from “man bites dog”. The fix: add a small position vector to each word " +
       "before attention. Toggle it below and watch order start to matter."]));

    const guide = P.el("details", { class: "guide" });
    guide.innerHTML =
      '<summary>What am I looking at?</summary>' +
      '<div class="guide-body">' +
        'Two sentences made of the <b>same three words</b> in different orders. Each word is a tiny ' +
        '2-number embedding (its meaning). For each sentence we run the focus word <code>bites</code> ' +
        'through attention and print the blended result.<br><br>' +
        '<b>Positions OFF.</b> The two sentences have the exact same bag of words, so the same ' +
        'embeddings, the same scores, the same blend — the two results come out <b>identical</b>. ' +
        'Attention sees a <i>bag of words, not a sequence</i>.<br><br>' +
        '<b>Positions ON.</b> Before attention, each slot’s <b>position vector</b> is <b>added</b> onto ' +
        'the word’s embedding: <code>input = embedding + position</code>. Now <code>dog</code> at ' +
        'slot 1 is a different vector from <code>dog</code> at slot 3, so the ingredients going into ' +
        'the sum differ — and the two orders finally produce <b>different</b> results. Note we never ' +
        'changed attention itself; we only position-stamped its inputs.<br><br>' +
        'Hover any vector to see the embedding, the position added, and the sum.' +
      '</div>';
    panel.appendChild(guide);

    // toggle
    let usePos = false;
    const ctl = P.el("div", { class: "controls" });
    ctl.appendChild(P.el("span", { class: "name", title: "Add a position vector onto each word before attention?" }, ["position vectors:"]));
    const offBtn = P.el("button", { class: "demo-chip active", title: "No position — pure bag of words." }, ["OFF (bag of words)"]);
    const onBtn  = P.el("button", { class: "demo-chip", title: "Add embedding + position before attention." }, ["ON (add position)"]);
    offBtn.addEventListener("click", () => { usePos = false; offBtn.className = "demo-chip active"; onBtn.className = "demo-chip"; run(); });
    onBtn.addEventListener("click",  () => { usePos = true;  onBtn.className  = "demo-chip active"; offBtn.className = "demo-chip"; run(); });
    ctl.appendChild(offBtn); ctl.appendChild(onBtn);
    panel.appendChild(ctl);

    const out = P.el("div", { class: "explain" });
    panel.appendChild(out);

    // toy embeddings + position table (same tiny numbers as the note)
    const EMB = { dog: [2, 0], bites: [0, 2], man: [1, 1] };
    const POS = [[0.1, 0], [0, 0.1], [0.2, 0.2]];   // slot 1, 2, 3
    const SENT_A = ["dog", "bites", "man"];
    const SENT_B = ["man", "bites", "dog"];

    // run attention on one sentence; bites is the focus word. returns blend + per-word input vectors.
    function attend(order) {
      const inputs = order.map((w, slot) => {
        const e = EMB[w];
        const p = POS[slot];
        const inp = usePos ? [e[0] + p[0], e[1] + p[1]] : [e[0], e[1]];
        return { w, slot, e, p, inp };
      });
      const focus = inputs.find(o => o.w === "bites");
      const raw = inputs.map(o => dot(focus.inp, o.inp));
      const weights = softmax(raw.map(s => s / 3));
      const blend = [0, 0];
      inputs.forEach((o, i) => { blend[0] += weights[i] * o.inp[0]; blend[1] += weights[i] * o.inp[1]; });
      return { inputs, weights, blend };
    }

    function sentenceBlock(title, order, res) {
      let h = `<div class="phase-tag">${title}</div>`;
      h += '<div style="margin:4px 0 8px">';
      res.inputs.forEach(o => {
        const isFocus = o.w === "bites";
        const tip = usePos
          ? `${o.w}@slot${o.slot + 1}: embedding [${P.fmt(o.e[0])},${P.fmt(o.e[1])}] + position [${P.fmt(o.p[0])},${P.fmt(o.p[1])}] = [${P.fmt(o.inp[0])},${P.fmt(o.inp[1])}]`
          : `${o.w}: embedding [${P.fmt(o.e[0])},${P.fmt(o.e[1])}]  (no position added)`;
        h += `<span title="${tip}" style="display:inline-block;background:${isFocus ? '#1f3a4d' : '#232d3a'};border:1px solid ${isFocus ? '#4ea1ff' : '#33404f'};border-radius:8px;padding:5px 9px;margin:3px;font-family:monospace;font-size:12px">` +
          `${o.w}${isFocus ? ' ◀' : ''}<br>` +
          (usePos
            ? `<span style="color:#8aa0b4">[${P.fmt(o.e[0])},${P.fmt(o.e[1])}]</span>+<span style="color:#ffd24e">[${P.fmt(o.p[0])},${P.fmt(o.p[1])}]</span>=<b style="color:#cdd9e5">[${P.fmt(o.inp[0])},${P.fmt(o.inp[1])}]</b>`
            : `<span style="color:#8aa0b4">[${P.fmt(o.inp[0])},${P.fmt(o.inp[1])}]</span>`) +
          `</span>`;
      });
      h += '</div>';
      h += `<div style="font-family:monospace;margin:4px 0" title="bites' new vector after attention">` +
        `blended “bites” = <b style="color:#fff">[${P.fmt(res.blend[0])}, ${P.fmt(res.blend[1])}]</b></div>`;
      return h;
    }

    function run() {
      const a = attend(SENT_A);
      const b = attend(SENT_B);
      const same = Math.abs(a.blend[0] - b.blend[0]) < 1e-9 && Math.abs(a.blend[1] - b.blend[1]) < 1e-9;

      let html = '';
      if (usePos) {
        html += '<div class="phase-tag" style="background:#46d39a;color:#08121f">positions ON — input = embedding + position</div>';
      } else {
        html += '<div class="phase-tag">positions OFF — input = embedding only (bag of words)</div>';
      }
      html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:8px">';
      html += '<div>' + sentenceBlock("“dog bites man”", SENT_A, a) + '</div>';
      html += '<div>' + sentenceBlock("“man bites dog”", SENT_B, b) + '</div>';
      html += '</div>';

      // verdict banner
      if (same) {
        html += `<div style="margin-top:12px;padding:10px 12px;border-radius:8px;background:#3a2330;border:1px solid #ff8db4" title="The two results are byte-for-byte equal.">` +
          `<b style="color:#ff8db4">Identical results.</b> Same bag of words → same blend. Attention cannot tell ` +
          `“dog bites man” from “man bites dog”. Something is missing: <b>word order</b>.</div>`;
        html += `<div class="hint">Now flip <b>position vectors: ON</b> above and watch the two results split apart.</div>`;
      } else {
        html += `<div style="margin-top:12px;padding:10px 12px;border-radius:8px;background:#1f3a2e;border:1px solid #46d39a" title="The two results now differ.">` +
          `<b style="color:#46d39a">Different results.</b> By adding a position vector, <code>dog</code> at slot 1 ` +
          `became a different vector from <code>dog</code> at slot 3 — so the two orders no longer match. ` +
          `Order now matters.</div>`;
        html += `<div class="hint">We didn’t touch attention (still a sum). We only <b>added position</b> onto the ` +
          `inputs, so the same word at a different slot is a different ingredient. That’s positional encoding.</div>`;
      }

      out.innerHTML = html;
    }

    host.appendChild(panel);
    run();
  }

  // =========================================================================
  // DEMO 5 — A transformer block
  // =========================================================================
  // One word vector flows through the whole block, narrated step by step:
  //   LayerNorm → Attention → +residual → LayerNorm → FeedForward → +residual.
  // The "add, don't replace" residual and the "recenter + rescale" norm are made
  // visible. A "next step" button walks through the six stages one at a time.
  function buildBlock(host, P) {
    const panel = P.el("div", { class: "panel" });
    panel.appendChild(P.el("h3", { class: "demo-title" }, ["A transformer block — mix, then think (with residuals + norm)"]));
    panel.appendChild(P.el("p", { class: "demo-sub" },
      ["Follow one word’s vector through the whole block, one step at a time: normalize → attention " +
       "(mix) → add residual → normalize → feed-forward (think) → add residual → block output. " +
       "Watch the “add, don’t replace” and the “reset the volume” moves happen."]));

    const guide = P.el("details", { class: "guide" });
    guide.innerHTML =
      '<summary>What am I looking at?</summary>' +
      '<div class="guide-body">' +
        'A transformer block is the repeatable unit a GPT stacks dozens of times. One word’s vector ' +
        '<code>x</code> flows through six steps. Two sub-layers do the work — <b>attention</b> (mix ' +
        'information <i>between</i> words) then a <b>feed-forward</b> network (each word <i>thinks</i> ' +
        'about what it gathered). Each is wrapped with two helpers:' +
        '<br><br>' +
        '<b>LayerNorm (the reset).</b> Before each sub-layer, the vector is <b>recentered</b> ' +
        '(subtract its mean → centered on 0) and <b>rescaled</b> (divide by its spread → standard ' +
        'volume ≈ 1). This stops the numbers drifting huge or tiny as they pass up a deep stack. The ' +
        'relative <i>pattern</i> is kept — only the scale changes.<br>' +
        '<b>Residual (add, don’t replace).</b> A sub-layer doesn’t overwrite the vector — it ' +
        '<b>adds its change on top</b>: <code>output = input + sublayer(input)</code>. The original ' +
        'survives on a clean straight path, and each sub-layer only has to learn a small correction.<br><br>' +
        'The feed-forward network is nothing new — it’s the Module 2 layers: <b>matrix → ReLU → matrix</b>, ' +
        'applied to each word on its own. Press <b>Next step ▸</b> to walk the block; hover any number ' +
        'to see what it is.' +
      '</div>';
    panel.appendChild(guide);

    // --- controls: step through ---
    let step = 0;                       // 0..6 (0 = just the input)
    const TOTAL = 6;
    const ctl = P.el("div", { class: "controls" });
    const nextBtn = P.el("button", { class: "btn primary", title: "Advance to the next stage of the block." }, ["Next step ▸"]);
    const backBtn = P.el("button", { class: "btn", title: "Go back one stage." }, ["◂ Back"]);
    const resetBtn = P.el("button", { class: "btn", title: "Return to the input." }, ["Reset"]);
    ctl.appendChild(backBtn); ctl.appendChild(nextBtn); ctl.appendChild(resetBtn);
    panel.appendChild(ctl);

    const out = P.el("div", { class: "explain" });
    panel.appendChild(out);

    // --- the toy computation (fixed, hand-checkable numbers) -----------------
    // One word's vector with 3 slots. Big drifted numbers so LayerNorm clearly does something.
    const X0 = [100, 300, 200];

    function layerNorm(v) {
      const d = v.length;
      const mu = v.reduce((a, b) => a + b, 0) / d;
      const variance = v.reduce((a, b) => a + (b - mu) * (b - mu), 0) / d;
      const sigma = Math.sqrt(variance) || 1;
      const out = v.map(x => (x - mu) / sigma);
      return { out, mu, sigma };
    }
    // toy attention sub-layer: returns a small "context" vector to ADD (the mixing it found).
    // (Hand-waved as a fixed contribution so the residual add is the visible point.)
    function attentionSublayer(v) { return [0.6, -0.4, 0.2]; }
    // toy feed-forward: matrix → ReLU → matrix, per slot. Returns the change to ADD.
    function feedForward(v) {
      // W1 (3x3-ish) → ReLU → W2; kept tiny & fixed so the result is small and addable.
      const h = v.map(x => Math.max(0, x));              // ReLU on a simple pass-through "matrix"
      // a small second matrix that scales down and shifts — produces a modest correction
      return h.map((x, i) => (x * 0.2) - [0.1, 0.3, 0.2][i]);
    }

    // precompute the whole pipeline so we can show any step
    function pipeline() {
      const x = X0.slice();
      const ln1 = layerNorm(x);                                   // step 1
      const attn = attentionSublayer(ln1.out);                    // step 2 (the change)
      const afterAttn = x.map((xi, i) => xi + attn[i]);           // step 3 residual: x + attn
      const ln2 = layerNorm(afterAttn);                           // step 4
      const ff = feedForward(ln2.out);                            // step 5 (the change)
      const afterFF = afterAttn.map((xi, i) => xi + ff[i]);       // step 6 residual: a + ff
      return { x, ln1, attn, afterAttn, ln2, ff, afterFF };
    }
    const PIPE = pipeline();

    const STEPS = [
      { tag: "input", title: "the word’s vector arrives" },
      { tag: "1 · LayerNorm", title: "reset #1 — recenter + rescale before attention" },
      { tag: "2 · Attention (mix)", title: "look at the other words, gather context (the change to add)" },
      { tag: "3 · + Residual", title: "ADD the attention result on top of the original (don’t replace)" },
      { tag: "4 · LayerNorm", title: "reset #2 — recenter + rescale before feed-forward" },
      { tag: "5 · Feed-forward (think)", title: "matrix → ReLU → matrix, per word (the change to add)" },
      { tag: "6 · + Residual → output", title: "ADD the feed-forward result on top — block output" }
    ];

    function vec(v, col) {
      return `[${v.map(x => `<span style="color:${col || '#cdd9e5'}">${P.fmt(x)}</span>`).join(", ")}]`;
    }
    function vecChip(label, v, tip, border) {
      return `<span title="${tip}" style="display:inline-block;background:#232d3a;border:1px solid ${border || '#33404f'};border-radius:8px;padding:6px 12px;margin:3px;font-family:monospace">` +
        `<span style="color:#8aa0b4;font-size:12px">${label}</span><br>${vec(v)}</span>`;
    }

    function run() {
      let html = '';
      // pipeline rail — show which step we're on
      html += '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:12px">';
      STEPS.forEach((s, i) => {
        const done = i < step, here = i === step;
        html += `<span title="${s.title}" style="font-size:11px;padding:3px 8px;border-radius:999px;` +
          `background:${here ? '#4ea1ff' : (done ? '#1f3a4d' : '#232d3a')};` +
          `color:${here ? '#08121f' : (done ? '#cdd9e5' : '#8aa0b4')};` +
          `border:1px solid ${here ? '#4ea1ff' : '#33404f'};font-weight:${here ? 700 : 400}">${s.tag}</span>`;
      });
      html += '</div>';

      const s = STEPS[step];
      html += `<div class="phase-tag">${s.tag}</div>`;
      html += `<p style="margin:4px 0 10px;color:#cdd9e5">${s.title}.</p>`;

      if (step === 0) {
        html += vecChip("x (the word’s vector, 3 slots)", PIPE.x, "The raw vector arriving into the block. Slots have drifted to large values.", "#4ea1ff");
        html += `<div class="hint">This single vector will flow through the whole block. Press <b>Next step ▸</b>.</div>`;
      }
      else if (step === 1) {
        html += vecChip("x (before)", PIPE.x, "the vector before normalizing", "#33404f");
        html += '<span style="font-size:20px;color:#8aa0b4">→</span>';
        html += vecChip("LayerNorm(x)", PIPE.ln1.out, `subtract mean ${P.fmt(PIPE.ln1.mu)}, divide by spread ${P.fmt(PIPE.ln1.sigma)}`, "#ffd24e");
        html += `<div class="hint">Recenter: subtract the mean <b>μ = ${P.fmt(PIPE.ln1.mu)}</b> (now centered on 0). ` +
          `Rescale: divide by the spread <b>σ = ${P.fmt(PIPE.ln1.sigma)}</b> (now a standard volume ≈ 1). ` +
          `The big drifted numbers <code>[${PIPE.x.join(", ")}]</code> become gentle ones — same pattern, tidy scale.</div>`;
      }
      else if (step === 2) {
        html += vecChip("LayerNorm(x)", PIPE.ln1.out, "the normalized vector going into attention", "#33404f");
        html += '<span style="font-size:20px;color:#8aa0b4">→ attention →</span>';
        html += vecChip("Attention(·) — the change", PIPE.attn, "the context this word gathered from the other words; this is what will be ADDED, not the whole new vector", "#46d39a");
        html += `<div class="hint">Attention looks at the other words and returns <b>just the change</b> ` +
          `<code>[${PIPE.attn.map(x => P.fmt(x)).join(", ")}]</code> — the context it gathered. It does <i>not</i> ` +
          `replace the vector; the next step adds it on.</div>`;
      }
      else if (step === 3) {
        html += vecChip("x (original)", PIPE.x, "the original vector — it skips around untouched", "#4ea1ff");
        html += '<span style="font-size:20px;color:#46d39a">+</span>';
        html += vecChip("Attention change", PIPE.attn, "the change attention computed", "#46d39a");
        html += '<span style="font-size:20px;color:#8aa0b4">=</span>';
        html += vecChip("x + Attention(x)", PIPE.afterAttn, "the original PLUS the change — residual add", "#fff");
        html += `<div class="hint"><b>Residual:</b> <code>output = input + sublayer(input)</code>. The original ` +
          `<code>x</code> survives on a straight path and attention’s change is added on top — not a rewrite. ` +
          `This is the “highway” that keeps deep stacks trainable.</div>`;
      }
      else if (step === 4) {
        html += vecChip("x + Attention(x)", PIPE.afterAttn, "the vector after the first residual", "#33404f");
        html += '<span style="font-size:20px;color:#8aa0b4">→</span>';
        html += vecChip("LayerNorm again", PIPE.ln2.out, `recenter (mean ${P.fmt(PIPE.ln2.mu)}) + rescale (spread ${P.fmt(PIPE.ln2.sigma)})`, "#ffd24e");
        html += `<div class="hint">The residual add nudged the numbers, so we normalize again before the next ` +
          `sub-layer (mean <b>${P.fmt(PIPE.ln2.mu)}</b>, spread <b>${P.fmt(PIPE.ln2.sigma)}</b>). ` +
          `A LayerNorm guards <i>each</i> sub-layer — two per block.</div>`;
      }
      else if (step === 5) {
        html += vecChip("LayerNorm", PIPE.ln2.out, "the normalized vector going into the feed-forward net", "#33404f");
        html += '<span style="font-size:20px;color:#8aa0b4">→ matrix → ReLU → matrix →</span>';
        html += vecChip("FeedForward(·) — the change", PIPE.ff, "the feed-forward network's output; the 'thinking' done per word, to be ADDED", "#46d39a");
        html += `<div class="hint">The feed-forward net is the <b>Module 2 layers</b>: matrix → ReLU → matrix, ` +
          `applied to this word alone. It returns another <b>change to add</b> ` +
          `<code>[${PIPE.ff.map(x => P.fmt(x)).join(", ")}]</code> — the word “thinking” about what it gathered.</div>`;
      }
      else if (step === 6) {
        html += vecChip("after attention", PIPE.afterAttn, "the vector before the feed-forward residual", "#4ea1ff");
        html += '<span style="font-size:20px;color:#46d39a">+</span>';
        html += vecChip("FeedForward change", PIPE.ff, "the change the feed-forward net computed", "#46d39a");
        html += '<span style="font-size:20px;color:#8aa0b4">=</span>';
        html += vecChip("block output", PIPE.afterFF, "the final vector leaving this block — same shape as it entered", "#fff");
        html += `<div class="hint"><b>Second residual:</b> add the feed-forward change on top. The vector leaves ` +
          `the block <b>the same shape</b> it entered (3 slots) but richer. Stack N of these blocks and you have a GPT body — ` +
          `each block refines the understanding a little more.</div>`;
      }

      out.innerHTML = html;
      backBtn.disabled = (step === 0);
      nextBtn.disabled = (step === TOTAL);
    }

    nextBtn.addEventListener("click", () => { if (step < TOTAL) step++; run(); });
    backBtn.addEventListener("click", () => { if (step > 0) step--; run(); });
    resetBtn.addEventListener("click", () => { step = 0; run(); });

    host.appendChild(panel);
    run();
  }
})();
