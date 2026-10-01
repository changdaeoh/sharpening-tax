/* All graphics are original SVGs rendered from aggregate counts or stated equations. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const DATA = window.RESEARCH_DATA;
  const M = window.ResearchMath;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const names = { webshop: "WebShop", bfcl: "BFCL v4 MT", acebench: "ACEBench" };
  const modelNames = { "gemma-4-31B": "Gemma 4 · 31B", "Qwen2.5-32B": "Qwen 2.5 · 32B", "Qwen3.5-35B-A3B": "Qwen 3.5 · 35B-A3B", "Ministral-3-14B": "Ministral 3 · 14B" };
  const state = { model: "gemma-4-31B", benchmark: "webshop", K: 128, taxK: 128, taxMetric: "S" };
  const blue = "#0081fb", ink = "#17191c";
  const pct = p => (100 * p).toFixed(1) + "%";
  const cell = () => DATA.cells[state.model + "/" + state.benchmark];
  const number = value => value === null ? "N/A" : value.toFixed(3);
  const svgText = (x, y, value, extra = "") => `<text x="${x}" y="${y}" ${extra}>${value}</text>`;
  const line = (x1, y1, x2, y2, extra = "") => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${extra}/>`;
  const path = (values, x, y) => values.map((p, i) => `${i ? "L" : "M"}${x(i + 1).toFixed(2)},${y(p).toFixed(2)}`).join(" ");

  const config = window.SITE_CONFIG || {};
  if (config.paperUrl) {
    $("paper-link").href = config.paperUrl;
    $("paper-link").hidden = false;
  }
  if (config.codeUrl) document.querySelectorAll("[data-code-link]").forEach(a => { a.href = config.codeUrl; });

  function renderCoverage() {
    const current = cell();
    const x = k => 39 + Math.log2(k) / 7 * 441;
    const y = p => 191 - 161 * p;
    const position = x(state.K);
    let html = `<title>Coverage with repeated attempts</title><desc>${modelNames[state.model]} on ${names[state.benchmark]}. At ${state.K} attempts, base ${pct(current.base.curve[state.K - 1])}, post-trained ${pct(current.post.curve[state.K - 1])}. Unbiased pass-at-k estimates from 128 rollouts per task. Attempt axis uses powers of two.</desc>`;
    html += `<defs><clipPath id="coverage-clip"><rect x="36" y="17" width="${position - 33}" height="185"/></clipPath></defs>`;
    [0, .25, .5, .75, 1].forEach(p => {
      html += line(39, y(p), 480, y(p), 'class="grid"');
      html += svgText(30, y(p) + 3, Math.round(p * 100), 'class="axis" text-anchor="end"');
    });
    html += svgText(39, 12, "Tasks solved (%)", 'class="axis"');
    [1, 2, 4, 8, 16, 32, 64, 128].forEach(k => {
      html += svgText(x(k), 210, k, `class="axis" text-anchor="middle"${k === state.K ? ' style="fill:#0668d7;font-weight:700"' : ""}`);
    });
    html += svgText(260, 231, "Attempts per task · k", 'class="axis" text-anchor="middle"');
    html += line(position, 25, position, 192, 'stroke="#c4cbd4" stroke-dasharray="3 4"');
    ["base", "post"].forEach(arm => {
      const d = path(current[arm].curve, x, y);
      html += `<path d="${d}" class="${arm}-line" opacity=".14"/>`;
      html += `<path d="${d}" class="${arm}-line" clip-path="url(#coverage-clip)"/>`;
    });
    const pb = current.base.curve[state.K - 1], pp = current.post.curve[state.K - 1];
    let by = y(pb), py = y(pp);
    if (Math.abs(by - py) < 18) { if (pb >= pp) { by -= 9; py += 9; } else { py -= 9; by += 9; } }
    [["base", pb, by, blue], ["post", pp, py, ink]].forEach(([arm, p, labelY, color]) => {
      html += `<circle cx="${position}" cy="${y(p)}" r="4" fill="${color}" stroke="white" stroke-width="2"/>`;
      const right = state.K < 64;
      html += svgText(position + (right ? 10 : -10), Math.max(24, Math.min(190, labelY + 4)), pct(p), `class="point-label" fill="${arm === "base" ? "#0668d7" : ink}" text-anchor="${right ? "start" : "end"}" style="paint-order:stroke;stroke:white;stroke-width:4px;stroke-linejoin:round"`);
    });
    $("coverage-chart").innerHTML = html;
    $("coverage-chart").setAttribute("aria-label", `Measured coverage at ${state.K} attempts: base ${pct(pb)}, post-trained ${pct(pp)}`);
    $("coverage-k").value = state.K;
    $("coverage-budget").setAttribute("aria-valuetext", `${state.K} attempts per task`);
    $("coverage-summary").textContent = `At ${state.K} ${state.K === 1 ? "attempt" : "attempts"}: base ${pct(pb)} · post-trained ${pct(pp)}`;
    $("sample-note").textContent = `Measured · ${current.base.tasks} tasks · 128 rollouts each`;
  }

  function initCategoryBars() {
    $("category-bars").innerHTML = ["base", "post"].map(arm => `<div><p class="bar-label">${arm === "base" ? "Base" : "Post-trained"}</p><div class="stacked-bar" id="bar-${arm}" role="img">${["never", "sometimes", "always"].map(kind => `<span class="${kind}" style="width:0"></span>`).join("")}</div></div>`).join("");
  }
  function renderPolarization() {
    const current = cell();
    ["base", "post"].forEach(arm => {
      const categories = current[arm].categories;
      const bar = $("bar-" + arm);
      bar.setAttribute("aria-label", `${arm === "base" ? "Base" : "Post-trained"}: ${pct(categories[0])} never solved, ${pct(categories[1])} sometimes solved, ${pct(categories[2])} every time`);
      [...bar.children].forEach((span, i) => {
        span.style.width = `${categories[i] * 100}%`;
        span.textContent = categories[i] >= .07 ? pct(categories[i]) : "";
        span.title = ["Never solved", "Sometimes solved", "Solved every time"][i] + ": " + pct(categories[i]);
      });
    });
    $("middle-change").textContent = `${pct(current.base.categories[1])} → ${pct(current.post.categories[1])}`;
  }

  function taxPlot(arm, curve, K, metrics) {
    const calibrated = state.taxMetric === "S";
    const value = metrics[state.taxMetric];
    const headroom = 1 - curve[0];
    const values = calibrated ? curve.map(p => headroom > 0 ? (p - curve[0]) / headroom : 0) : curve;
    const x = k => 30 + (k - 1) / (K - 1) * 221;
    const y = p => 145 - p * 115;
    const ceiling = values[K - 1];
    const color = arm === "base" ? blue : ink;
    const description = calibrated
      ? `Calibrated scalability S(${K}) is ${number(value)}. The vertical axis is recovered first-try failure headroom. The horizontal axis is normalized budget, (k-1)/(K-1). The shaded area is S.`
      : `Raw scalability A(${K}) is ${number(value)}. Shading sums pass-at-${K} minus pass-at-k for k from 1 to ${K - 1}, on a linear attempt axis.`;
    let html = `<title>${arm === "base" ? "Base" : "Post-trained"} ${calibrated ? "calibrated" : "raw"} scalability</title><desc>${description}</desc>`;
    [0, .5, 1].forEach(p => {
      html += line(30, y(p), 251, y(p), 'class="grid"');
      html += svgText(24, y(p) + 3, calibrated ? p.toFixed(1) : Math.round(p * 100), 'class="axis" text-anchor="end"');
    });
    // Raw: unit k-width and coverage-gap height. Calibrated: width 1/(K-1)
    // and height divided by first-try failure headroom, giving exactly S(K).
    for (let k = 1; k < K; k++) html += `<rect class="tax-area" x="${x(k)}" y="${y(ceiling)}" width="${x(k + 1) - x(k)}" height="${Math.max(0, y(values[k - 1]) - y(ceiling))}" fill="${color}" fill-opacity=".15"/>`;
    html += line(30, y(ceiling), 251, y(ceiling), `stroke="${color}" stroke-width="1" stroke-dasharray="3 3" opacity=".6"`);
    let steps = `M${x(1)},${y(values[0])}`;
    for (let k = 2; k <= K; k++) steps += `H${x(k)}V${y(values[k - 1])}`;
    html += `<path d="${steps}" stroke="${color}" stroke-width="1.8" fill="none"/>`;
    if (calibrated) [0, .5, 1].forEach(u => { html += svgText(30 + u * 221, 162, u.toFixed(1), 'class="axis" text-anchor="middle"'); });
    else [...new Set([1, Math.round(K / 2), K])].forEach(k => { html += svgText(x(k), 162, k, 'class="axis" text-anchor="middle"'); });
    html += svgText(251, 17, `${state.taxMetric} = ${value === null ? "N/A" : value.toFixed(calibrated ? 3 : 2)}`, `text-anchor="end" font-size="12" fill="${arm === "base" ? "#0668d7" : ink}"`);
    html += svgText(30, 17, calibrated ? "Headroom recovered" : "% solved", 'class="axis"');
    html += svgText(140, 182, calibrated ? "Normalized budget · (k − 1)/(K − 1)" : "Attempts · k (linear)", 'class="axis" text-anchor="middle"');
    const chart = $("tax-" + arm + "-chart");
    chart.innerHTML = html;
    chart.setAttribute("aria-label", `${arm === "base" ? "Base" : "Post-trained"} ${calibrated ? "calibrated" : "raw"} scalability: ${number(value)}`);
  }
  function renderTax() {
    const current = cell(), b = M.metrics(current.base.curve, state.taxK), p = M.metrics(current.post.curve, state.taxK);
    const metric = state.taxMetric, calibrated = metric === "S";
    const format = value => value === null ? "N/A" : value.toFixed(calibrated ? 3 : 2);
    taxPlot("base", current.base.curve, state.taxK, b);
    taxPlot("post", current.post.curve, state.taxK, p);
    $("tax-context").textContent = `${modelNames[state.model]} / ${names[state.benchmark]}`;
    $("tax-equation").innerHTML = `Tax<sub>${metric}</sub> = ${metric}<sub>Base</sub> − ${metric}<sub>Post</sub>`;
    $("scalability-base").textContent = format(b[metric]);
    $("scalability-post").textContent = format(p[metric]);
    $("tax-value").textContent = b[metric] === null || p[metric] === null ? "N/A" : format(b[metric] - p[metric]);
    $("tax-normalization-note").textContent = calibrated ? "S normalizes the area by budget and first-try failure rate." : "A sums the coverage still recoverable by spending the remaining attempts.";
    $("tax-area-caption").textContent = calibrated ? "Shaded area = S(K). Both axes are normalized by the available headroom and budget." : "Shaded area = A(K). Each curve’s ceiling is its own pass@K.";
    $("tax-k").value = state.taxK;
    $("tax-budget").setAttribute("aria-valuetext", `Budget of ${state.taxK} attempts`);
  }

  let playTimer = null, manuallyInteracted = false;
  function stopPlayback() {
    clearInterval(playTimer);
    playTimer = null;
    $("play-budget").textContent = "▶";
    $("play-budget").setAttribute("aria-label", "Play retry budget animation");
    $("play-budget").title = "Play retry budget animation";
  }
  function playBudget() {
    if (playTimer) { stopPlayback(); return; }
    // Reduced-motion users get the final state immediately, including on click.
    if (reducedMotion.matches) {
      state.K = 128; $("coverage-budget").value = 7; renderCoverage(); return;
    }
    let exponent = 0;
    state.K = 1; $("coverage-budget").value = 0; renderCoverage();
    $("play-budget").textContent = "Ⅱ";
    $("play-budget").setAttribute("aria-label", "Pause retry budget animation");
    $("play-budget").title = "Pause retry budget animation";
    playTimer = setInterval(() => {
      exponent++;
      state.K = 2 ** exponent;
      $("coverage-budget").value = exponent;
      renderCoverage();
      if (exponent === 7) stopPlayback();
    }, 520);
  }
  $("play-budget").addEventListener("click", () => { manuallyInteracted = true; playBudget(); });
  $("coverage-budget").addEventListener("input", e => {
    manuallyInteracted = true; stopPlayback(); state.K = 2 ** Number(e.target.value); renderCoverage();
  });
  $("tax-budget").addEventListener("input", e => { state.taxK = 2 ** Number(e.target.value); renderTax(); });
  document.querySelectorAll("[data-tax-metric]").forEach(button => button.addEventListener("click", () => {
    state.taxMetric = button.dataset.taxMetric;
    document.querySelectorAll("[data-tax-metric]").forEach(b => b.setAttribute("aria-pressed", String(b === button)));
    renderTax();
  }));
  document.querySelectorAll("[data-benchmark]").forEach(button => button.addEventListener("click", () => {
    manuallyInteracted = true; stopPlayback(); state.benchmark = button.dataset.benchmark;
    document.querySelectorAll("[data-benchmark]").forEach(b => b.setAttribute("aria-pressed", String(b === button)));
    renderCoverage(); renderPolarization(); renderTax();
  }));
  $("model-select").addEventListener("change", e => {
    manuallyInteracted = true; stopPlayback(); state.model = e.target.value;
    renderCoverage(); renderPolarization(); renderTax();
  });
  document.addEventListener("visibilitychange", () => { if (document.hidden) stopPlayback(); });
  reducedMotion.addEventListener("change", () => { if (reducedMotion.matches) stopPlayback(); });

  const reference = [.64, .2, .09, .045, .025];
  function renderPolicy() {
    const p = Number($("difficulty").value) / 100, T = M.temperature(p);
    const probabilities = M.tempered(reference, T);
    $("temperature-value").value = T.toFixed(2);
    $("difficulty-value").value = pct(p);
    $("difficulty").setAttribute("aria-valuetext", `${pct(p)} sampled success probability; temperature ${T.toFixed(2)}`);
    const action = p < .48 ? "Explore more paths." : p > .52 ? "Commit to likely paths." : "Keep the balance.";
    $("temperature-action").textContent = action;
    const y = probability => 155 - probability * 145;
    let html = `<title>Illustrative next-action distribution</title><desc>At sampled success probability ${pct(p)}, PTGS sets temperature ${T.toFixed(2)}. Gray bars show fixed temperature one; blue bars show the same example policy tempered by PTGS. These are illustrative action probabilities, not measured success rates.</desc>`;
    html += svgText(15, 13, "Action probability", 'class="axis"');
    html += line(15, 155, 425, 155, 'class="grid"');
    probabilities.forEach((probability, i) => {
      const x = 33 + i * 82;
      html += `<rect x="${x}" y="${y(reference[i])}" width="25" height="${155 - y(reference[i])}" fill="#e0e4e9" rx="2"/>`;
      html += `<rect x="${x + 28}" y="${y(probability)}" width="25" height="${155 - y(probability)}" fill="${blue}" rx="2"/>`;
      html += svgText(x + 40.5, y(probability) - 7, `${Math.round(probability * 100)}%`, 'font-size="11" text-anchor="middle" fill="#0668d7"');
      html += svgText(x + 26.5, 176, String.fromCharCode(65 + i), 'class="axis" text-anchor="middle"');
    });
    $("policy-chart").innerHTML = html;
    renderTemperatureIllustration(p, T);
  }
  $("difficulty").addEventListener("input", renderPolicy);

  function renderTemperatureIllustration(p, T) {
    const x = q => 38 + q * 302, bottom = 139;
    const density = q => 110 * q * (1 - q) ** 9; // Beta(2,10).
    const py = d => bottom - d / density(.1) * 96;
    const samples = Array.from({ length: 201 }, (_, i) => i / 200);
    const posteriorPath = samples.map((q, i) => `${i ? "L" : "M"}${x(q)},${py(density(q))}`).join(" ");
    let posterior = `<title>Example prompt posterior: Beta(2,10)</title><desc>One observed success and nine failures plus a Beta(1,1) prior. The vertical marker shows the current sampled probability ${pct(p)}. This is an illustrative posterior.</desc>`;
    posterior += `<path d="M${x(0)},${bottom} ${posteriorPath.replace(/^M/, "L")} L${x(1)},${bottom} Z" fill="${blue}" fill-opacity=".09"/>`;
    posterior += `<path d="${posteriorPath}" class="base-line"/>`;
    posterior += line(x(p), 28, x(p), bottom, 'stroke="#0668d7" stroke-dasharray="3 3"');
    posterior += `<circle cx="${x(p)}" cy="${py(density(p))}" r="4" fill="${blue}" stroke="white" stroke-width="2"/>`;
    posterior += svgText(38, 15, "Posterior density", 'class="axis"');
    posterior += svgText(340, 15, "Beta(2,10)", 'class="axis" text-anchor="end"');
    [0, .5, 1].forEach(q => { posterior += svgText(x(q), 157, q, 'class="axis" text-anchor="middle"'); });
    posterior += svgText(190, 177, `Sampled probability · p̂ = ${p.toFixed(3)}`, 'class="axis" text-anchor="middle"');
    $("posterior-chart").innerHTML = posterior;

    const ty = t => 139 - (t - 2 / 3) / (1.5 - 2 / 3) * 104;
    const rule = samples.map((q, i) => `${i ? "L" : "M"}${x(q)},${ty(M.temperature(q))}`).join(" ");
    let diagram = `<title>PTGS temperature map</title><desc>With pivot one half and tau 1.5, temperature decreases from 1.5 for the hardest prompts to two thirds for the easiest. At the current draw ${p.toFixed(3)}, temperature is ${T.toFixed(2)}.</desc>`;
    diagram += `<rect x="38" y="30" width="151" height="109" fill="${blue}" fill-opacity=".04"/>`;
    [2 / 3, 1, 1.5].forEach(t => {
      diagram += line(38, ty(t), 340, ty(t), 'class="grid"');
      diagram += svgText(30, ty(t) + 3, t === 2 / 3 ? "⅔" : t, 'class="axis" text-anchor="end"');
    });
    diagram += line(x(.5), 30, x(.5), bottom, 'stroke="#c4cbd4" stroke-dasharray="3 3"');
    diagram += `<path d="${rule}" class="base-line"/>`;
    diagram += line(x(p), ty(T), x(p), bottom, 'stroke="#0668d7" stroke-dasharray="3 3"');
    diagram += `<circle cx="${x(p)}" cy="${ty(T)}" r="5" fill="${blue}" stroke="white" stroke-width="2"/>`;
    diagram += svgText(38, 15, "Heat · T > 1", 'class="axis"');
    diagram += svgText(340, 15, "Cool · T < 1", 'class="axis" text-anchor="end"');
    [0, .5, 1].forEach(q => { diagram += svgText(x(q), 157, q === .5 ? "0.5 · pivot" : q, 'class="axis" text-anchor="middle"'); });
    diagram += svgText(190, 177, "Sampled success probability · p̂", 'class="axis" text-anchor="middle"');
    $("temperature-rule-chart").innerHTML = diagram;
    $("temperature-rule-caption").textContent = `p̂ = ${p.toFixed(3)} → T = ${T.toFixed(2)}. ${T > 1 ? "Heat up to explore." : T < 1 ? "Cool down to exploit." : "Neutral at the pivot."}`;
  }
  $("draw-posterior").addEventListener("click", () => {
    // Integer-shape Gamma sums give an exact Beta(2,10) draw before display rounding.
    const gamma = shape => Array.from({ length: shape }, () => -Math.log(1 - Math.random())).reduce((a, b) => a + b, 0);
    const a = gamma(2), b = gamma(10);
    $("difficulty").value = (100 * a / (a + b)).toFixed(1);
    renderPolicy();
  });

  function renderCollapse() {
    const fraction = Number($("collapse-fraction").value);
    $("collapse-value").value = `${fraction}%`;
    $("retained-area").style.width = `${100 - fraction}%`;
    $("lost-area").style.width = `${fraction}%`;
    $("collapse-caption").textContent = `Collapse ${fraction}% of tasks → lose ${fraction}% of raw scalability.`;
    $("collapse-fraction").setAttribute("aria-valuetext", `${fraction} percent collapsed; ${100 - fraction} percent of base scalability retained`);
  }
  $("collapse-fraction").addEventListener("input", renderCollapse);

  function renderSignal() {
    const p = Number($("signal-probability").value) / 100, n = 4;
    $("signal-probability-value").value = `${Math.round(100 * p)}%`;
    const events = [["Positive · ≥1 success", M.positive(.1, n), M.positive(p, n)], ["Mixed · success + failure", M.mixed(.1, n), M.mixed(p, n)]];
    $("signal-comparison").innerHTML = events.map(([label, before, after]) => `<div class="signal-event"><div><span>${label}</span><span>${pct(before)} <b>→ ${pct(after)}</b></span></div><div class="signal-track" role="img" aria-label="${label}: ${pct(before)} before, ${pct(after)} after"><i style="width:${after * 100}%"></i><i class="signal-before" style="width:${before * 100}%"></i></div></div>`).join("");
    $("signal-probability").setAttribute("aria-valuetext", `Improved success probability ${pct(p)}; positive group ${pct(events[0][2])}; mixed group ${pct(events[1][2])}`);
  }
  $("signal-probability").addEventListener("input", renderSignal);

  // paper.tex, tab:ptgs-combined. The base here is Qwen2.5-7B-Instruct.
  const results = {
    "sokoban-ppo": { before: [46.5, 55.0, .094], after: [61.1, 69.7, .081] },
    "sokoban-grpo": { before: [36.5, 55.3, .081], after: [39.1, 72.5, .025] },
    "frozenlake-ppo": { before: [63.7, 74.1, .039], after: [65.0, 80.0, .020] },
    "frozenlake-grpo": { before: [63.4, 77.8, .029], after: [67.3, 81.2, .028] },
  };
  function renderResults() {
    const key = $("result-select").value, r = results[key];
    ["p1", "pk", "tax"].forEach((metric, i) => {
      $("result-" + metric + "-before").textContent = r.before[i].toFixed(i === 2 ? 3 : 1);
      $("result-" + metric + "-after").textContent = r.after[i].toFixed(i === 2 ? 3 : 1) + (i < 2 ? "%" : "");
    });
    const checkpoint = key.endsWith("ppo") ? "final PPO checkpoints" : "best validation GRPO checkpoints per run";
    $("result-note").textContent = `Fixed-temperature RL → RL with PTGS. Qwen2.5-7B-Instruct · five-run means · ${checkpoint}. Same fixed evaluation temperature.`;
  }
  $("result-select").addEventListener("change", renderResults);

  const tabs = [...document.querySelectorAll('[role="tab"]')];
  function activateTab(tab) {
    tabs.forEach(t => {
      const selected = t === tab;
      t.setAttribute("aria-selected", String(selected));
      t.tabIndex = selected ? 0 : -1;
      $(t.getAttribute("aria-controls")).hidden = !selected;
    });
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activateTab(tab));
    tab.addEventListener("keydown", event => {
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      else if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = tabs.length - 1;
      else return;
      event.preventDefault(); activateTab(tabs[next]); tabs[next].focus();
    });
  });
  $("copy-citation").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText($("bibtex").textContent);
      $("copy-status").textContent = "Citation copied.";
    } catch (_) {
      // Supports file:// previews and browsers that block clipboard access.
      const selection = window.getSelection(), range = document.createRange();
      range.selectNodeContents($("bibtex")); selection.removeAllRanges(); selection.addRange(range);
      $("copy-status").textContent = "Citation selected. Press Ctrl+C (or ⌘C) to copy.";
    }
  });

  initCategoryBars(); renderCoverage(); renderPolarization(); renderTax(); renderPolicy(); renderResults(); renderCollapse(); renderSignal();
  if ("IntersectionObserver" in window) {
    const playObserver = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) {
        if (!manuallyInteracted && !reducedMotion.matches) playBudget();
        playObserver.disconnect();
      }
    }, { threshold: .7 });
    playObserver.observe($("coverage-chart"));
    const observedSections = ["observation", "diagnostic", "solution", "theory"];
    const activeObserver = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        document.querySelectorAll("nav a").forEach(a => {
          if (a.hash === "#" + entry.target.id) a.setAttribute("aria-current", "location");
          else a.removeAttribute("aria-current");
        });
      });
    }, { rootMargin: "-15% 0px -55% 0px" });
    observedSections.forEach(id => activeObserver.observe($(id)));
  }
})();
