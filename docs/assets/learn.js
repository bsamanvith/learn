// Shared animation engine for the learn site. No dependencies: plain Canvas 2D.
// Each scene mounts on an element with data-scene="...", owns its own clock,
// and only animates while it is on screen.
(() => {
  // ---------- small helpers ----------
  const rng = seed => () => { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const gauss = r => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  const ease = t => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  const lerp = (a, b, t) => a + (b - a) * t;
  const sigmoid = z => 1 / (1 + Math.exp(-z));
  const reduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const fmt = (x, d = 3) => !isFinite(x) ? "—" : Math.abs(x) >= 1e4 ? x.toExponential(1) : x.toFixed(d);

  let C = {};
  function readColors() {
    const cs = getComputedStyle(document.documentElement), g = k => cs.getPropertyValue(k).trim();
    C = { panel: g("--panel"), ink: g("--ink"), muted: g("--muted"), line: g("--line"), accent: g("--accent"), data: g("--data"), class1: g("--class1"), soft: g("--soft"), bad: g("--bad"), good: g("--good"), mono: g("--f-mono"), display: g("--f-display"), body: g("--f-body") };
  }
  const probe = document.createElement("canvas").getContext("2d");
  function alpha(color, a) {
    probe.fillStyle = "#000"; probe.fillStyle = color; const s = probe.fillStyle;
    if (s[0] === "#") return `rgba(${parseInt(s.slice(1, 3), 16)},${parseInt(s.slice(3, 5), 16)},${parseInt(s.slice(5, 7), 16)},${a})`;
    return s.replace(/rgba?\(([^)]+)\)/, (_, p) => `rgba(${p.split(",").slice(0, 3).join(",")},${a})`);
  }

  // Size a canvas to its CSS width at device resolution; returns a 2D context in CSS pixels.
  function fit(canvas, aspect) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = canvas.clientWidth || 300, H = Math.round(W * aspect);
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
      canvas.style.height = H + "px";
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    }
    const k = canvas.getContext("2d");
    k.setTransform(dpr, 0, 0, dpr, 0, 0);
    k.clearRect(0, 0, W, H);
    return { k, W, H };
  }
  // A linear map from data range to a pixel box.
  function box(x0, x1, y0, y1, left, top, width, height) {
    return { X: x => left + (x - x0) / (x1 - x0) * width, Y: y => top + (1 - (y - y0) / (y1 - y0)) * height, left, top, width, height };
  }
  function label(k, text, x, y, { align = "left", base = "alphabetic", color = C.muted, size = 12, font = C.mono } = {}) {
    k.fillStyle = color; k.font = `${size}px ${font}`; k.textAlign = align; k.textBaseline = base; k.fillText(text, x, y);
  }
  function dot(k, x, y, r, fill, stroke, lw = 2) {
    k.beginPath(); k.arc(x, y, r, 0, Math.PI * 2);
    if (fill) { k.fillStyle = fill; k.fill(); }
    if (stroke) { k.strokeStyle = stroke; k.lineWidth = lw; k.stroke(); }
  }

  // ---------- shared data: points around a line ----------
  function lineData() {
    const r = rng(11), xs = [], ys = [], n = 28;
    for (let i = 0; i < n; i++) { const x = -1 + 2 * i / (n - 1) + 0.04 * gauss(r); xs.push(x); ys.push(1.6 * x + 0.4 + 0.35 * gauss(r)); }
    const m = a => a.reduce((s, v) => s + v, 0) / n;
    const S = { n, xs, ys, Sx: m(xs), Sy: m(ys), Sxx: m(xs.map(x => x * x)), Sxy: m(xs.map((x, i) => x * ys[i])), Syy: m(ys.map(y => y * y)) };
    S.wStar = (S.Sxy - S.Sx * S.Sy) / (S.Sxx - S.Sx * S.Sx); S.bStar = S.Sy - S.wStar * S.Sx;
    S.loss = (w, b) => w * w * S.Sxx + b * b + 2 * w * b * S.Sx - 2 * w * S.Sxy - 2 * b * S.Sy + S.Syy;
    S.grad = (w, b) => [2 * (w * S.Sxx + b * S.Sx - S.Sxy), 2 * (w * S.Sx + b - S.Sy)];
    S.yMin = Math.min(...ys) - 0.9; S.yMax = Math.max(...ys) + 0.9;
    return S;
  }
  const D = lineData();

  function drawScatter(k, B, w, b, { residuals = true, ghosts = [], pointR = 4 } = {}) {
    const xa = -1.25, xb = 1.25;
    k.strokeStyle = C.line; k.lineWidth = 1;
    k.beginPath(); k.moveTo(B.left, B.Y(0)); k.lineTo(B.left + B.width, B.Y(0)); k.stroke();
    k.save(); k.beginPath(); k.rect(B.left, B.top, B.width, B.height); k.clip();
    ghosts.forEach(([gw, gb], i) => {
      k.strokeStyle = alpha(C.accent, 0.08 + 0.1 * i / Math.max(1, ghosts.length)); k.lineWidth = 2;
      k.beginPath(); k.moveTo(B.X(xa), B.Y(gw * xa + gb)); k.lineTo(B.X(xb), B.Y(gw * xb + gb)); k.stroke();
    });
    if (residuals) {
      k.strokeStyle = alpha(C.accent, 0.45); k.lineWidth = 1.2;
      D.xs.forEach((x, i) => { k.beginPath(); k.moveTo(B.X(x), B.Y(D.ys[i])); k.lineTo(B.X(x), B.Y(w * x + b)); k.stroke(); });
    }
    k.strokeStyle = C.accent; k.lineWidth = 3; k.lineCap = "round";
    k.beginPath(); k.moveTo(B.X(xa), B.Y(w * xa + b)); k.lineTo(B.X(xb), B.Y(w * xb + b)); k.stroke();
    D.xs.forEach((x, i) => dot(k, B.X(x), B.Y(D.ys[i]), pointR, C.data));
    k.restore();
  }

  // ---------- a clock that walks through a list of keyframes ----------
  // durations[i] is how long the move from frame i to i+1 takes.
  function timeline(count, durFn, hold) {
    const starts = [0];
    for (let i = 0; i < count - 1; i++) starts.push(starts[i] + durFn(i));
    const end = starts[count - 1];
    return {
      end, total: end + hold,
      at(t) { // -> [index, eased fraction]
        if (t >= end) return [count - 1, 0];
        let i = 0; while (i < count - 2 && starts[i + 1] <= t) i++;
        const f = (t - starts[i]) / (starts[i + 1] - starts[i]);
        return [i, reduced() ? (f > 0.5 ? 1 : 0) : ease(Math.min(1, f))];
      },
    };
  }

  // ---------- scene: a line learning to fit the points ----------
  function LineFit(root) {
    const ambient = root.hasAttribute("data-ambient");
    const fitCv = root.querySelector("canvas[data-role=fit]"), lossCv = root.querySelector("canvas[data-role=loss]");
    const out = root.querySelector("[data-role=readout]");
    const S = { lr: 0.3, t: 0, playing: !reduced(), path: [], tl: null };

    function build() {
      let w = -1.2, b = 2.2; S.path = [[w, b]];
      for (let i = 0; i < 90; i++) {
        const [gw, gb] = D.grad(w, b); w -= S.lr * gw; b -= S.lr * gb; S.path.push([w, b]);
        if (Math.hypot(gw, gb) < 2e-3) break;
      }
      S.tl = timeline(S.path.length, i => (i < 10 ? 480 : 150), 2200);
      S.losses = S.path.map(([w, b]) => D.loss(w, b));
      S.t = 0;
    }
    build();

    function current() {
      const [i, f] = S.tl.at(S.t);
      const a = S.path[i], z = S.path[Math.min(i + 1, S.path.length - 1)];
      return { i, f, w: lerp(a[0], z[0], f), b: lerp(a[1], z[1], f) };
    }

    function draw() {
      const c = current();
      { // data view
        const { k, W, H } = fit(fitCv, ambient ? 0.62 : 0.66);
        const B = box(-1.25, 1.25, D.yMin, D.yMax, 8, 8, W - 16, H - 16);
        const ghosts = S.path.slice(Math.max(0, c.i - 4), c.i + 1);
        drawScatter(k, B, c.w, c.b, { ghosts, pointR: ambient ? 3.5 : 4.2 });
        const step = c.i + (c.f > 0.5 ? 1 : 0);
        label(k, `step ${step}`, 16, 18, { color: C.ink, size: 13, base: "middle" });
        label(k, `loss ${fmt(D.loss(c.w, c.b), 3)}`, 16, 36, { size: 12, base: "middle" });
        if (!ambient) label(k, `ŷ = ${fmt(c.w, 2)}·x ${c.b < 0 ? "−" : "+"} ${fmt(Math.abs(c.b), 2)}`, W - 16, H - 16, { align: "right", color: C.accent, size: 15, font: C.display });
      }
      if (lossCv) { // loss curve over steps
        // Side by side: match the data view's height. Stacked (phone): a shorter strip.
        const beside = lossCv.clientWidth < fitCv.clientWidth * 0.95;
        const { k, W, H } = fit(lossCv, beside ? fitCv.clientHeight / lossCv.clientWidth : 0.5);
        const n = S.path.length, lo = Math.log10(Math.min(...S.losses)) - 0.05, hi = Math.log10(Math.max(...S.losses)) + 0.1;
        const B = box(0, Math.max(n - 1, 1), lo, hi, 44, 14, W - 58, H - 44);
        k.strokeStyle = C.line; k.lineWidth = 1;
        k.beginPath(); k.moveTo(B.left, B.top); k.lineTo(B.left, B.top + B.height); k.lineTo(B.left + B.width, B.top + B.height); k.stroke();
        for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) {
          const y = B.Y(e); label(k, e >= 0 ? String(10 ** e) : (10 ** e).toFixed(-e), B.left - 6, y, { align: "right", base: "middle", size: 11 });
          k.strokeStyle = alpha(C.line, 0.6); k.beginPath(); k.moveTo(B.left, y); k.lineTo(B.left + B.width, y); k.stroke();
        }
        label(k, "loss", B.left + 6, B.top + 2, { base: "top", size: 11 });
        label(k, "0", B.left, B.top + B.height + 18, { align: "center", size: 11 });
        label(k, `step ${n - 1} →`, B.left + B.width, B.top + B.height + 18, { align: "right", size: 11 });
        k.strokeStyle = alpha(C.muted, 0.25); k.lineWidth = 1.5; k.beginPath();
        S.losses.forEach((L, i) => { const x = B.X(i), y = B.Y(Math.log10(L)); i ? k.lineTo(x, y) : k.moveTo(x, y); }); k.stroke();
        const upto = c.i + c.f, L = D.loss(c.w, c.b);
        k.strokeStyle = C.accent; k.lineWidth = 2.5; k.beginPath();
        for (let i = 0; i <= c.i; i++) { const x = B.X(i), y = B.Y(Math.log10(S.losses[i])); i ? k.lineTo(x, y) : k.moveTo(x, y); }
        k.lineTo(B.X(upto), B.Y(Math.log10(L))); k.stroke();
        dot(k, B.X(upto), B.Y(Math.log10(L)), 5, C.panel, C.accent, 2.5);
      }
      if (out) out.innerHTML = `η = <b>${S.lr}</b> · step <b>${c.i}</b> of ${S.path.length - 1} · w <b>${fmt(c.w, 2)}</b> · b <b>${fmt(c.b, 2)}</b>`;
    }

    const playBtn = root.querySelector("[data-act=play]");
    const sync = () => { if (playBtn) playBtn.textContent = S.playing ? "Pause" : "Play"; };
    playBtn?.addEventListener("click", () => { S.playing = !S.playing; if (S.t >= S.tl.end) S.t = 0; sync(); });
    root.querySelector("[data-act=restart]")?.addEventListener("click", () => { S.t = 0; S.playing = true; sync(); });
    root.querySelectorAll("[data-lr]").forEach(btn => btn.addEventListener("click", () => {
      root.querySelectorAll("[data-lr]").forEach(b => b.setAttribute("aria-pressed", String(b === btn)));
      S.lr = +btn.dataset.lr; build(); S.playing = true; sync();
    }));
    sync();
    return {
      el: root,
      tick(dt) { if (S.playing) { S.t += dt; if (S.t > S.tl.total) S.t = 0; } draw(); },
    };
  }

  // ---------- scene: four learning rates racing ----------
  function Race(root) {
    const lanes = [...root.querySelectorAll("[data-lane]")];
    // Only w learns here; b stays at its best value. Then L(w) is a parabola with curvature 2·Sxx.
    const curv = 2 * D.Sxx, wS = D.wStar, bS = D.bStar, span = 3.2, HOPS = 16;
    const Lw = w => D.loss(w, bS);
    const S = { t: 0, playing: !reduced() };
    const runs = lanes.map(el => {
      const factor = +el.dataset.lane, lr = factor / curv, path = [wS - 2.4];
      let diverged = false;
      for (let i = 0; i < HOPS; i++) {
        const w = path[path.length - 1], next = w - lr * 2 * (D.Sxx * w + bS * D.Sx - D.Sxy);
        path.push(next);
        if (Math.abs(next - wS) > span + 0.6) { diverged = true; break; }
      }
      el.querySelector("[data-role=eta]").textContent = `η = ${lr.toFixed(2)}`;
      return { el, cv: el.querySelector("canvas"), out: el.querySelector("[data-role=readout]"), lr, path, diverged };
    });
    const tl = timeline(HOPS + 1, () => 560, 2000);

    function drawLane(run) {
      let [i, f] = tl.at(S.t);
      const last = run.path.length - 1;
      if (i >= last) { i = last; f = 0; }
      const w0 = run.path[i], w1 = run.path[Math.min(i + 1, last)];
      const { k, W, H } = fit(run.cv, 1.25);
      const topH = H * 0.54, Lmax = Lw(wS + span) * 1.08;
      const B = box(wS - span, wS + span, 0, Lmax, 10, 18, W - 20, topH - 26);
      // the bowl
      k.strokeStyle = alpha(C.data, 0.7); k.lineWidth = 2; k.beginPath();
      for (let s = 0; s <= 120; s++) { const w = wS - span + 2 * span * s / 120, x = B.X(w), y = B.Y(Lw(w)); s ? k.lineTo(x, y) : k.moveTo(x, y); }
      k.stroke();
      k.strokeStyle = C.line; k.lineWidth = 1; k.beginPath(); k.moveTo(B.left, B.Y(0)); k.lineTo(B.left + B.width, B.Y(0)); k.stroke();
      label(k, "loss", B.left, 10, { base: "middle", size: 11 });
      label(k, "w", B.left + B.width, B.Y(0) + 12, { align: "right", base: "middle", size: 11, font: C.display, color: C.ink });
      // tick at the bottom of the bowl
      k.strokeStyle = C.muted; k.beginPath(); k.moveTo(B.X(wS), B.Y(0) - 4); k.lineTo(B.X(wS), B.Y(0) + 4); k.stroke();
      // footprints
      k.save(); k.beginPath(); k.rect(0, 0, W, topH); k.clip();
      for (let j = 0; j <= i; j++) dot(k, B.X(run.path[j]), B.Y(Lw(run.path[j])), 2.6, alpha(C.accent, 0.25 + 0.5 * j / Math.max(1, i)));
      // the hop: a bezier arc from w0 to w1
      const p0 = [B.X(w0), B.Y(Lw(w0))], p1 = [B.X(w1), B.Y(Lw(w1))];
      const h = Math.max(14, 0.35 * Math.abs(p1[0] - p0[0])), cx = (p0[0] + p1[0]) / 2, cy = Math.min(p0[1], p1[1]) - h;
      const bx = (1 - f) ** 2 * p0[0] + 2 * f * (1 - f) * cx + f * f * p1[0], by = (1 - f) ** 2 * p0[1] + 2 * f * (1 - f) * cy + f * f * p1[1];
      if (f > 0 && f < 1) {
        k.setLineDash([3, 4]); k.strokeStyle = alpha(C.accent, 0.5); k.lineWidth = 1.2;
        k.beginPath(); k.moveTo(p0[0], p0[1]); k.quadraticCurveTo(cx, cy, p1[0], p1[1]); k.stroke(); k.setLineDash([]);
      }
      dot(k, bx, by, 6.5, C.accent, C.panel, 2);
      k.restore();
      // the line this w gives, on the data
      const wNow = lerp(w0, w1, f);
      const B2 = box(-1.25, 1.25, D.yMin, D.yMax, 6, topH + 8, W - 12, H - topH - 14);
      k.strokeStyle = C.line; k.strokeRect(B2.left, B2.top, B2.width, B2.height);
      drawScatter(k, B2, wNow, bS, { residuals: true, pointR: 2.6 });
      // status
      const hop = i + (f > 0.5 ? 1 : 0), off = run.diverged && i >= last;
      const tag = run.el.querySelector("[data-role=state]");
      const near = Math.abs(wNow - wS) < 0.05;
      tag.textContent = off ? "diverged" : near ? "landed" : `hop ${hop}`;
      tag.className = "tag" + (off ? " bad" : near ? " good" : "");
      run.out.innerHTML = `loss <b>${off ? "∞" : fmt(Lw(wNow), 3)}</b>`;
    }

    const playBtn = root.querySelector("[data-act=play]");
    const sync = () => { if (playBtn) playBtn.textContent = S.playing ? "Pause" : "Play"; };
    playBtn?.addEventListener("click", () => { S.playing = !S.playing; sync(); });
    root.querySelector("[data-act=restart]")?.addEventListener("click", () => { S.t = 0; S.playing = true; sync(); });
    sync();
    return { el: root, tick(dt) { if (S.playing) { S.t += dt; if (S.t > tl.total) S.t = 0; } runs.forEach(drawLane); } };
  }

  // ---------- scene: logistic regression learning an S-curve ----------
  function Logistic(root) {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const r = rng(5), n = 46, xs = [], ys = [], jit = [];
    for (let i = 0; i < n; i++) { const x = -3 + 6 * i / (n - 1) + 0.05 * gauss(r); xs.push(x); ys.push(r() < sigmoid(2.2 * x - 0.8) ? 1 : 0); jit.push((r() - 0.5) * 0.06); }
    const logloss = (w, b) => -xs.reduce((s, x, i) => { const p = Math.min(1 - 1e-9, Math.max(1e-9, sigmoid(w * x + b))); return s + (ys[i] ? Math.log(p) : Math.log(1 - p)); }, 0) / n;
    const iters = [[0, 0]];
    for (let t = 0; t < 320; t++) {
      const [w, b] = iters[t]; let gw = 0, gb = 0;
      xs.forEach((x, i) => { const e = sigmoid(w * x + b) - ys[i]; gw += e * x; gb += e; });
      iters.push([w - 1.2 * gw / n, b - 1.2 * gb / n]);
    }
    const snapIdx = [...new Set(Array.from({ length: 41 }, (_, i) => Math.round(320 * (i / 40) ** 2)))];
    const snaps = snapIdx.map(i => iters[i]);
    const tl = timeline(snaps.length, i => (i < 6 ? 520 : 240), 2400);
    const S = { t: 0, playing: !reduced() };

    function draw() {
      const [i, f] = tl.at(S.t), a = snaps[i], z = snaps[Math.min(i + 1, snaps.length - 1)];
      const w = lerp(a[0], z[0], f), b = lerp(a[1], z[1], f);
      const { k, W, H } = fit(cv, cv.clientWidth < 600 ? 0.75 : 0.42);
      const B = box(-3.3, 3.3, -0.12, 1.12, 40, 26, W - 54, H - 52);
      const xb = Math.abs(w) > 1e-3 ? -b / w : null;
      // the two decision regions
      if (xb !== null) {
        const px = Math.max(B.left, Math.min(B.left + B.width, B.X(xb)));
        k.fillStyle = alpha(C.data, 0.07); k.fillRect(B.left, B.top, px - B.left, B.height);
        k.fillStyle = alpha(C.class1, 0.08); k.fillRect(px, B.top, B.left + B.width - px, B.height);
      }
      for (const p of [0, 0.5, 1]) {
        k.strokeStyle = alpha(C.line, p === 0.5 ? 1 : 0.8); k.lineWidth = 1; k.setLineDash(p === 0.5 ? [4, 4] : []);
        k.beginPath(); k.moveTo(B.left, B.Y(p)); k.lineTo(B.left + B.width, B.Y(p)); k.stroke(); k.setLineDash([]);
        label(k, String(p), B.left - 8, B.Y(p), { align: "right", base: "middle", size: 11 });
      }
      label(k, "P(pass)", B.left, 10, { base: "middle", size: 11 });
      label(k, "hours studied (centred) →", B.left + B.width, B.top + B.height + 22, { align: "right", size: 11 });
      // errors: from each point to the curve
      xs.forEach((x, j) => {
        const p = sigmoid(w * x + b);
        k.strokeStyle = alpha(ys[j] ? C.class1 : C.data, 0.35); k.lineWidth = 1.2;
        k.beginPath(); k.moveTo(B.X(x), B.Y(ys[j] + jit[j])); k.lineTo(B.X(x), B.Y(p)); k.stroke();
      });
      // the S-curve
      k.strokeStyle = C.accent; k.lineWidth = 3; k.lineCap = "round"; k.beginPath();
      for (let s = 0; s <= 160; s++) { const x = -3.3 + 6.6 * s / 160; const px = B.X(x), py = B.Y(sigmoid(w * x + b)); s ? k.lineTo(px, py) : k.moveTo(px, py); }
      k.stroke();
      xs.forEach((x, j) => dot(k, B.X(x), B.Y(ys[j] + jit[j]), 4, ys[j] ? C.class1 : C.data));
      if (xb !== null && xb > -3.3 && xb < 3.3) {
        k.strokeStyle = C.ink; k.lineWidth = 1.2; k.setLineDash([5, 4]);
        k.beginPath(); k.moveTo(B.X(xb), B.top); k.lineTo(B.X(xb), B.top + B.height); k.stroke(); k.setLineDash([]);
        label(k, "boundary", B.X(xb) + 6, B.Y(0.25), { color: C.ink, size: 11, base: "middle" });
      }
      const step = snapIdx[i] + Math.round((snapIdx[Math.min(i + 1, snapIdx.length - 1)] - snapIdx[i]) * f);
      out.innerHTML = `step <b>${step}</b> · w <b>${fmt(w, 2)}</b> · b <b>${fmt(b, 2)}</b> · log-loss <b>${fmt(logloss(w, b), 3)}</b>${xb !== null ? ` · boundary at x = <b>${fmt(xb, 2)}</b>` : ""}`;
    }
    const playBtn = root.querySelector("[data-act=play]");
    const sync = () => { if (playBtn) playBtn.textContent = S.playing ? "Pause" : "Play"; };
    playBtn?.addEventListener("click", () => { S.playing = !S.playing; sync(); });
    root.querySelector("[data-act=restart]")?.addEventListener("click", () => { S.t = 0; S.playing = true; sync(); });
    sync();
    return { el: root, tick(dt) { if (S.playing) { S.t += dt; if (S.t > tl.total) S.t = 0; } draw(); } };
  }

  // ---------- mount and run ----------
  const SCENES = { linefit: LineFit, race: Race, logistic: Logistic };
  readColors();
  const scenes = [...document.querySelectorAll("[data-scene]")].map(el => {
    const s = SCENES[el.dataset.scene](el); s.visible = true; return s;
  });
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver(es => es.forEach(e => { const s = scenes.find(s => s.el === e.target); if (s) s.visible = e.isIntersecting; }), { rootMargin: "80px" });
    scenes.forEach(s => io.observe(s.el));
  }
  const refresh = () => { readColors(); scenes.forEach(s => s.tick(0)); };
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", refresh);
  new MutationObserver(refresh).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  if (document.fonts) document.fonts.ready.then(refresh);
  let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(refresh, 80); });

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(100, now - last); last = now;
    scenes.forEach(s => { if (s.visible) s.tick(dt); });
    requestAnimationFrame(frame);
  }
  scenes.forEach(s => s.tick(0));
  requestAnimationFrame(frame);
})();
