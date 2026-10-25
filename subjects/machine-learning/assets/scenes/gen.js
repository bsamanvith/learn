// Interactive figures for the generative-model and LLM lessons (24–30). Each registers with learn.js.
(() => {
  const L = window.Learn; if (!L) return;
  const { rng, gauss, fmt, C, alpha, fit, shape, panes, boxIn, box, label, dot, knobs, wire, axes, curve, path, stepTween, reduced } = L;
  const sum = a => a.reduce((s, v) => s + v, 0);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

  // ---------- 24 · Noising a spiral, and sampling back with the exact denoiser ----------
  L.register("diffuse", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const T = 200, beta = Array.from({ length: T }, (_, i) => 1e-4 + (0.08 - 1e-4) * i / (T - 1)), abar = [];
    beta.reduce((p, b, i) => (abar[i] = p * (1 - b)), 1);
    const r = rng(2), n = 300, data = [], eps = [];
    for (let i = 0; i < n; i++) { const s = Math.sqrt(r()) * 3 * Math.PI + Math.PI / 2; data.push([s * Math.cos(s) / (3 * Math.PI) + 0.02 * gauss(r), s * Math.sin(s) / (3 * Math.PI) + 0.02 * gauss(r)]); eps.push([gauss(r), gauss(r)]); }
    const S = { t: 0, tween: null };
    const show = knobs(root, S, 0);
    let gen = null; // a running reverse process: { x, t, seed }
    root.querySelectorAll("input[data-k=t]").forEach(el => el.addEventListener("input", () => { gen = null; }));
    function x0hat(x, t) { // E[x0 | x_t] for the 300 training points: a softmax-weighted average
      const a = Math.sqrt(abar[t]), v = 1 - abar[t], lw = data.map(d => -((x[0] - a * d[0]) ** 2 + (x[1] - a * d[1]) ** 2) / (2 * v)), m = Math.max(...lw);
      let z = 0, s0 = 0, s1 = 0; lw.forEach((l, i) => { const w = Math.exp(l - m); z += w; s0 += w * data[i][0]; s1 += w * data[i][1]; });
      return [s0 / z, s1 / z];
    }
    let seed = 1;
    wire(root, S, { generate: () => { const g = rng(seed++ * 31); gen = { x: Array.from({ length: 400 }, () => [gauss(g), gauss(g)]), t: T - 1, g }; } });
    function reverseStep() {
      const t = gen.t, g = gen.g;
      gen.x = gen.x.map(x => {
        const x0 = x0hat(x, t); if (t === 0) return x0;
        const ab = abar[t], ap = abar[t - 1], m0 = Math.sqrt(1 - beta[t]) * (1 - ap) / (1 - ab), m1 = Math.sqrt(ap) * beta[t] / (1 - ab), sd = Math.sqrt(beta[t] * (1 - ap) / (1 - ab));
        return [m0 * x[0] + m1 * x0[0] + sd * gauss(g), m0 * x[1] + m1 * x0[1] + sd * gauss(g)];
      });
      gen.t--;
    }
    function draw() {
      const { k, W, H } = shape(cv, 0.5, 0.85), side = Math.min(W, H) - 8, x0 = (W - side) / 2;
      const B = box(-2.3, 2.3, -2.3, 2.3, x0, 4, side, side);
      k.strokeStyle = C.line; k.lineWidth = 1; k.strokeRect(B.left, B.top, B.width, B.height);
      let pts, t, note;
      if (gen) {
        t = Math.max(gen.t, 0); pts = gen.x;
        data.forEach(d => dot(k, B.X(d[0]), B.Y(d[1]), 1.6, alpha(C.data, 0.35), null));
        note = gen.t >= 0 ? `sampling backwards: step <b>${gen.t + 1}</b> of ${T}` : (() => {
          const near = pts.map(p => Math.min(...data.map(d => Math.hypot(p[0] - d[0], p[1] - d[1])))).sort((a, b) => a - b);
          return `done · median distance from a sample to its nearest training point <b>${fmt(near[200], 4)}</b>: the exact denoiser can only redraw its training set`;
        })();
      } else {
        t = Math.round(S.t); const a = t ? Math.sqrt(abar[t - 1]) : 1, s = t ? Math.sqrt(1 - abar[t - 1]) : 0;
        pts = data.map((d, i) => [a * d[0] + s * eps[i][0], a * d[1] + s * eps[i][1]]);
        note = `t = <b>${t}</b> · x<sub>t</sub> = <b>${fmt(a, 3)}</b>·x<sub>0</sub> + <b>${fmt(s, 3)}</b>·ε`;
      }
      const col = gen ? C.accent : C.data;
      pts.forEach(p => dot(k, B.X(clamp(p[0], -2.25, 2.25)), B.Y(clamp(p[1], -2.25, 2.25)), 2.4, col, null));
      label(k, gen ? "400 new samples (orange) and the 300 training points (faint)" : "300 training points, noised to step t", B.left + 8, B.top + 14, { size: 11, base: "middle" });
      out.innerHTML = note;
    }
    return { el: root, tick(dt) { if (gen && gen.t >= 0 && dt > 0) { const per = reduced() ? T : 2; for (let i = 0; i < per && gen.t >= 0; i++) reverseStep(); } stepTween(S, dt, show); draw(); } };
  });

  // ---------- 27 · Compute-optimal models: the isoFLOP curves ----------
  L.register("scaling", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const FITS = { hoffmann: [1.69, 406.4, 410.7, 0.34, 0.28], besiroglu: [1.82, 482, 2085, 0.348, 0.366] };
    const S = { c: 21, fit: "hoffmann", tween: null }; // c is log10 of training compute in FLOPs
    const loss = (N, D) => { const [E, A, Bc, al, be] = FITS[S.fit]; return E + A / N ** al + Bc / D ** be; };
    const show = knobs(root, S, 1);
    wire(root, S, {}, () => {});
    const best = c => { let b = [Infinity, 0]; for (let s = 0; s <= 600; s++) { const ln = 6 + 8 * s / 600, N = 10 ** ln, l = loss(N, 10 ** c / (6 * N)); if (l < b[0]) b = [l, ln]; } return b; };
    const big = x => x >= 1e12 ? `${fmt(x / 1e12, 1)} trillion` : x >= 1e9 ? `${fmt(x / 1e9, 1)} billion` : `${fmt(x / 1e6, 0)} million`;
    function draw() {
      const { k, W, H } = shape(cv, 0.46, 0.8), B = box(6, 13, 1.7, 4.4, 48, 14, W - 64, H - 50);
      axes(k, B, { xs: [6, 7, 8, 9, 10, 11, 12, 13], ys: [2, 2.5, 3, 3.5, 4], fx: x => `1e${x}`, fy: v => v.toFixed(1), xl: "parameters N (log) →", yl: "predicted loss", grid: true });
      const opt = [];
      for (let c = 18; c <= 26; c++) { curve(k, B, ln => loss(10 ** ln, 10 ** c / (6 * 10 ** ln)), 6, 13, alpha(C.muted, 0.4), 1.2); opt.push(best(c)); }
      path(k, B, opt.map(([l, ln]) => [ln, l]), alpha(C.ink, 0.6), 1.4, [4, 4]);
      curve(k, B, ln => loss(10 ** ln, 10 ** S.c / (6 * 10 ** ln)), 6, 13, C.accent, 2.8);
      const [lb, lnb] = best(S.c), N = 10 ** lnb, D = 10 ** S.c / (6 * N);
      dot(k, B.X(lnb), B.Y(lb), 6, C.panel, C.accent, 3);
      label(k, "optimum for each budget", B.X(opt[0][1]) + 6, B.Y(opt[0][0]) + 16, { size: 11 });
      out.innerHTML = `C = 10<sup>${fmt(S.c, 1)}</sup> FLOPs · best model <b>${big(N)}</b> parameters trained on <b>${big(D)}</b> tokens (<b>${fmt(D / N, 0)}</b> tokens per parameter) · loss <b>${fmt(lb, 3)}</b> · left of the dot the model is too small, right of it it sees too little data`;
    }
    return { el: root, tick(dt) { stepTween(S, dt, show); draw(); } };
  });

  // ---------- 29 · Quantising weights with an outlier ----------
  L.register("quant", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const r = rng(29), w = Array.from({ length: 4000 }, () => 0.02 * gauss(r)); w[17] = 0.4; w[901] = -0.33; w[2500] = 0.37;
    const S = { bits: 4, keep: 0, tween: null };
    const show = knobs(root, S, 0);
    wire(root, S, {}, () => {});
    function quant() {
      const b = Math.round(S.bits), out = [], big = w.map(v => Math.abs(v) > 0.15);
      const amax = Math.max(...w.filter((v, i) => !(S.keep && big[i])).map(Math.abs)), levels = 2 ** (b - 1) - 1, sc = amax / levels;
      w.forEach((v, i) => out.push(S.keep && big[i] ? v : Math.round(v / sc) * sc));
      return { q: out, sc, levels, amax };
    }
    function draw() {
      const { q, sc, levels, amax } = quant(), { k, W, H } = shape(cv, 0.42, 0.8);
      const lim = S.keep ? 0.09 : 0.42, B = box(-lim, lim, 0, 1, 40, 22, W - 56, H - 64);
      axes(k, B, { xs: [-lim, 0, lim].map(v => +v.toFixed(2)), xl: "weight value →" });
      const bins = 90, h = new Array(bins).fill(0); w.forEach(v => { const i = Math.floor((v + lim) / (2 * lim) * bins); if (i >= 0 && i < bins) h[i]++; });
      const mh = Math.max(...h), bw = B.width / bins;
      h.forEach((c, i) => { k.fillStyle = alpha(C.data, 0.55); const y = B.Y(0.85 * c / mh); k.fillRect(B.left + i * bw, y, bw - 0.5, B.Y(0) - y); });
      for (let j = -levels; j <= levels; j++) { const x = j * sc; if (Math.abs(x) > lim) continue; k.strokeStyle = alpha(C.accent, 0.8); k.lineWidth = 1.5; k.beginPath(); k.moveTo(B.X(x), B.Y(0.98)); k.lineTo(B.X(x), B.Y(0)); k.stroke(); }
      label(k, `${2 * levels + 1} levels (orange), spaced ${fmt(sc, 4)} apart`, B.left, 10, { size: 12, color: C.ink, base: "middle" });
      const err = sum(w.map((v, i) => (v - q[i]) ** 2)) / w.length, sig = sum(w.map(v => v * v)) / w.length, used = new Set(q.map(v => Math.round(v / sc))).size;
      out.innerHTML = `${Math.round(S.bits)}-bit, scaled by the largest |w| = ${fmt(amax, 3)} · the 4,000 weights use <b>${used}</b> distinct levels · signal-to-noise <b>${fmt(10 * Math.log10(sig / Math.max(err, 1e-30)), 1)} dB</b>${S.keep ? " · the three outliers stay in 16 bits" : " · three outliers stretch the grid, so nearly every weight rounds to one of a few levels near zero"}`;
    }
    return { el: root, tick(dt) { stepTween(S, dt, show); draw(); } };
  });

  // ---------- 30 · Temperature, top-k and top-p on one next-token distribution ----------
  L.register("sampling", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const tok = ["mat", "floor", "sofa", "bed", "chair", "windowsill", "roof", "table", "rug", "keyboard", "grass", "moon"];
    const p0 = [0.32, 0.14, 0.1, 0.08, 0.06, 0.05, 0.04, 0.035, 0.03, 0.025, 0.02, 0.01], z = p0.map(Math.log);
    const S = { T: 0, p: 1, k: 12, tween: null }; // T is log10 of the temperature
    const show = knobs(root, S, 2);
    let samples = [], sr = rng(30);
    function dist() {
      const T = 10 ** S.T, m = Math.max(...z), e = z.map(v => Math.exp((v - m) / T)), Z = sum(e), q = e.map(v => v / Z);
      const order = q.map((v, i) => i).sort((a, b) => q[b] - q[a]), keep = new Set(); let c = 0;
      for (const i of order) { if (keep.size >= Math.round(S.k) || (keep.size && c >= S.p - 1e-12)) break; keep.add(i); c += q[i]; }
      const kz = sum([...keep].map(i => q[i]));
      return { q, keep, f: q.map((v, i) => keep.has(i) ? v / kz : 0) };
    }
    wire(root, S, { draw: () => { const { f } = dist(); samples = Array.from({ length: 12 }, () => { let u = sr(), i = 0; while (i < f.length - 1 && (u -= f[i]) > 0) i++; return tok[i]; }); } });
    function draw() {
      root.querySelectorAll("[data-out=T]").forEach(o => { o.textContent = fmt(10 ** S.T, 2); });
      root.querySelectorAll("[data-out=k]").forEach(o => { o.textContent = Math.round(S.k); });
      const { q, keep, f } = dist(), { k, W, H } = shape(cv, 0.42, 0.85), narrow = W < 600;
      const left = narrow ? 82 : 96, B = box(0, 0.75, 0, 12, left, 24, W - left - 48, H - 30);
      label(k, "The cat sat on the …", 4, 10, { color: C.ink, size: 13, base: "middle", font: C.display });
      const rh = B.height / 12;
      tok.forEach((t, i) => {
        const y = B.top + i * rh, kept = keep.has(i);
        label(k, t, left - 8, y + rh / 2, { align: "right", base: "middle", size: 12, color: kept ? C.ink : C.muted });
        k.strokeStyle = alpha(C.muted, 0.6); k.lineWidth = 1; k.strokeRect(B.X(0), y + rh * 0.2, B.X(Math.min(q[i], 0.75)) - B.X(0), rh * 0.6);
        if (kept) { k.fillStyle = alpha(C.accent, 0.85); k.fillRect(B.X(0), y + rh * 0.2, B.X(Math.min(f[i], 0.75)) - B.X(0), rh * 0.6); label(k, fmt(f[i], 2), B.X(Math.min(f[i], 0.75)) + 6, y + rh / 2, { base: "middle", size: 11, color: C.ink }); }
      });
      const Hb = -sum(f.map(v => v > 0 ? v * Math.log2(v) : 0));
      out.innerHTML = `outline: after temperature · filled: what is actually sampled from · <b>${keep.size}</b> of 12 tokens survive · entropy <b>${fmt(Hb, 2)}</b> bits${samples.length ? ` · samples: <b>${samples.join(", ")}</b>` : ""}`;
    }
    return { el: root, tick(dt) { stepTween(S, dt, show); draw(); } };
  });
})();
