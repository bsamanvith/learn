// Shared helpers for widgets. Plain browser JS, no build step.
// Include with <script src="../../../tools/widgets/lib.js"></script> from a subject's widgets/.

const Learn = (() => {
  const SVGNS = "http://www.w3.org/2000/svg";

  // Seeded RNG (mulberry32) so a "simulate" run can be reproduced.
  function rng(seed = Date.now()) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Uniform integer in {0, ..., n-1}.
  const randInt = (rand, n) => Math.floor(rand() * n);

  // Fisher–Yates shuffle, in place; returns the array.
  function shuffle(arr, rand) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = randInt(rand, i + 1);
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }

  // Count values into `bins` equal-width buckets over [lo, hi).
  function histogram(values, lo, hi, bins) {
    const counts = new Array(bins).fill(0);
    const w = (hi - lo) / bins;
    for (const v of values) {
      const i = Math.floor((v - lo) / w);
      if (i >= 0 && i < bins) counts[i]++;
    }
    return counts;
  }

  // Read colours from CSS custom properties so widgets follow light/dark mode.
  function palette() {
    const s = getComputedStyle(document.documentElement);
    const get = (n) => s.getPropertyValue(n).trim();
    return {
      a: get("--event-a"), b: get("--event-b"), both: get("--event-both"),
      muted: get("--muted"), fg: get("--fg"), bg: get("--bg"), panel: get("--panel"),
      good: get("--good"), bad: get("--bad"),
    };
  }

  // Bind a range input to state[key]; calls onChange with the new state.
  function slider(id, state, key, onChange, fmt = (v) => v) {
    const el = document.getElementById(id);
    const out = document.getElementById(id + "-value");
    const update = () => {
      state[key] = Number(el.value);
      if (out) out.textContent = fmt(state[key]);
      onChange(state);
    };
    el.addEventListener("input", update);
    update();
  }

  // Linear map from [d0, d1] to [r0, r1].
  const scale = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);

  // Create an SVG element with attributes, optionally appended to a parent.
  function el(tag, attrs = {}, parent) {
    const e = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "text") e.textContent = v;
      else e.setAttribute(k, v);
    }
    if (parent) parent.appendChild(e);
    return e;
  }

  // Draw x/y axes with ticks into `g`. Returns {sx, sy} scales.
  // opts: {x:[lo,hi], y:[lo,hi], W, H, M:{l,r,t,b}, xTicks:[], yTicks:[], xLabel, yLabel, fmtX, fmtY}
  function axes(g, opts) {
    const c = palette();
    const M = Object.assign({ l: 44, r: 16, t: 16, b: 36 }, opts.M || {});
    const sx = scale(opts.x[0], opts.x[1], M.l, opts.W - M.r);
    const sy = scale(opts.y[0], opts.y[1], opts.H - M.b, M.t);
    const fx = opts.fmtX || ((v) => v), fy = opts.fmtY || ((v) => v);
    el("line", { x1: M.l, y1: opts.H - M.b, x2: opts.W - M.r, y2: opts.H - M.b, stroke: c.muted }, g);
    el("line", { x1: M.l, y1: M.t, x2: M.l, y2: opts.H - M.b, stroke: c.muted }, g);
    for (const t of opts.xTicks || []) {
      el("line", { x1: sx(t), x2: sx(t), y1: opts.H - M.b, y2: opts.H - M.b + 4, stroke: c.muted }, g);
      el("text", { x: sx(t), y: opts.H - M.b + 16, "text-anchor": "middle", fill: c.muted, "font-size": 12, text: fx(t) }, g);
    }
    for (const t of opts.yTicks || []) {
      el("line", { x1: M.l - 4, x2: M.l, y1: sy(t), y2: sy(t), stroke: c.muted }, g);
      el("text", { x: M.l - 7, y: sy(t) + 4, "text-anchor": "end", fill: c.muted, "font-size": 12, text: fy(t) }, g);
    }
    if (opts.xLabel) el("text", { x: (M.l + opts.W - M.r) / 2, y: opts.H - 4, "text-anchor": "middle", fill: c.muted, "font-size": 12, text: opts.xLabel }, g);
    if (opts.yLabel) el("text", { x: 12, y: (M.t + opts.H - M.b) / 2, "text-anchor": "middle", fill: c.muted, "font-size": 12,
      transform: `rotate(-90 12 ${(M.t + opts.H - M.b) / 2})`, text: opts.yLabel }, g);
    return { sx, sy };
  }

  // Fixed-decimal formatting that trims "-0.000".
  const fmt = (x, d = 3) => (Math.abs(x) < 0.5 * 10 ** -d ? 0 : x).toFixed(d);

  // Re-render on light/dark switch so palette-derived colours stay right.
  function onThemeChange(fn) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", fn);
  }

  return { rng, randInt, shuffle, histogram, palette, slider, scale, el, axes, fmt, onThemeChange };
})();
