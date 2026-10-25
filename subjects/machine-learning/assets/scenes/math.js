// Interactive figures for the math lessons (M1–M4). Each registers with learn.js.
(() => {
  const L = window.Learn; if (!L) return;
  const { rng, gauss, fmt, C, alpha, shape, box, boxIn, label, dot, knobs, wire, axes, curve, tweenTo, stepTween } = L;
  const sum = a => a.reduce((s, v) => s + v, 0);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

  // ---------- M1 · What a 2×2 matrix does to the plane: eigenvectors and singular vectors ----------
  L.register("matrix", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const S = { a: 1.2, b: 0.6, c: 0.3, d: 0.9, tween: null };
    const show = knobs(root, S, 2);
    const presets = { rotation: [0.71, -0.71, 0.71, 0.71], shear: [1, 1, 0, 1], symmetric: [1.5, 0.5, 0.5, 0.7], singular: [1, 0.5, 2, 1] };
    const acts = {}; for (const [n, [a, b, c, d]] of Object.entries(presets)) acts[n] = () => tweenTo(S, { a, b, c, d });
    wire(root, S, acts);
    function svd(a, b, c, d) { // eigen-decomposition of AᵀA gives V and σ²; U = AV/σ
      const p = a * a + c * c, q = a * b + c * d, r = b * b + d * d, m = (p + r) / 2, h = Math.sqrt(((p - r) / 2) ** 2 + q * q);
      const s1 = Math.sqrt(m + h), s2 = Math.sqrt(Math.max(0, m - h)), th = 0.5 * Math.atan2(2 * q, p - r), v1 = [Math.cos(th), Math.sin(th)];
      const u1 = [(a * v1[0] + b * v1[1]) / (s1 || 1), (c * v1[0] + d * v1[1]) / (s1 || 1)];
      return { s1, s2, u1, u2: [-u1[1], u1[0]] };
    }
    function draw() {
      const { a, b, c, d } = S, A = (x, y) => [a * x + b * y, c * x + d * y], det = a * d - b * c, tr = a + d;
      const { k, W, H } = shape(cv, 0.62, 0.9), sc = Math.min(W, H) / 7.2, X = x => W / 2 + x * sc, Y = y => H / 2 - y * sc;
      k.save(); k.beginPath(); k.rect(0, 0, W, H); k.clip();
      for (let i = -6; i <= 6; i++) { // the original grid, faint, and its image
        k.strokeStyle = alpha(C.line, 0.5); k.lineWidth = 1; k.beginPath(); k.moveTo(X(i), 0); k.lineTo(X(i), H); k.moveTo(0, Y(i)); k.lineTo(W, Y(i)); k.stroke();
        k.strokeStyle = alpha(C.data, i === 0 ? 0.55 : 0.2); k.lineWidth = i === 0 ? 1.6 : 1; k.beginPath();
        let p = A(i, -8), q = A(i, 8); k.moveTo(X(p[0]), Y(p[1])); k.lineTo(X(q[0]), Y(q[1])); p = A(-8, i); q = A(8, i); k.moveTo(X(p[0]), Y(p[1])); k.lineTo(X(q[0]), Y(q[1])); k.stroke();
      }
      const sq = [A(0, 0), A(1, 0), A(1, 1), A(0, 1)];
      k.fillStyle = alpha(det >= 0 ? C.data : C.bad || C.accent, 0.18); k.beginPath(); sq.forEach(([x, y], i) => i ? k.lineTo(X(x), Y(y)) : k.moveTo(X(x), Y(y))); k.closePath(); k.fill();
      k.strokeStyle = C.accent; k.lineWidth = 2.5; k.beginPath(); for (let s = 0; s <= 120; s++) { const t = 2 * Math.PI * s / 120, [x, y] = A(Math.cos(t), Math.sin(t)); s ? k.lineTo(X(x), Y(y)) : k.moveTo(X(x), Y(y)); } k.stroke();
      k.strokeStyle = alpha(C.muted, 0.6); k.lineWidth = 1; k.setLineDash([3, 4]); k.beginPath(); k.arc(X(0), Y(0), sc, 0, 2 * Math.PI); k.stroke(); k.setLineDash([]);
      const { s1, s2, u1, u2 } = svd(a, b, c, d);
      [[u1, s1], [u2, s2]].forEach(([u, s]) => { k.strokeStyle = C.accent; k.lineWidth = 1.5; k.setLineDash([2, 3]); k.beginPath(); k.moveTo(X(0), Y(0)); k.lineTo(X(u[0] * s), Y(u[1] * s)); k.stroke(); k.setLineDash([]); });
      const disc = tr * tr - 4 * det, eig = [];
      if (disc >= 0) for (const l of [(tr + Math.sqrt(disc)) / 2, (tr - Math.sqrt(disc)) / 2]) { // (A − λI)v = 0
        let v = Math.abs(b) > 1e-9 ? [b, l - a] : Math.abs(c) > 1e-9 ? [l - d, c] : (Math.abs(l - a) < 1e-9 ? [1, 0] : [0, 1]); const m = Math.hypot(...v); v = v.map(x => x / m); eig.push([l, v]);
        k.strokeStyle = C.ink; k.lineWidth = 1.4; k.setLineDash([8, 5]); k.beginPath(); k.moveTo(X(-5 * v[0]), Y(-5 * v[1])); k.lineTo(X(5 * v[0]), Y(5 * v[1])); k.stroke(); k.setLineDash([]);
        label(k, `λ = ${fmt(l, 2)}`, X(2.6 * v[0]) + 6, Y(2.6 * v[1]) - 6, { size: 12, color: C.ink });
      }
      const e1 = A(1, 0), e2 = A(0, 1);
      dot(k, X(e1[0]), Y(e1[1]), 5, C.data, C.panel, 1.5); dot(k, X(e2[0]), Y(e2[1]), 5, C.data, C.panel, 1.5);
      label(k, "Ae₁", X(e1[0]) + 8, Y(e1[1]) + 14, { size: 12, color: C.data }); label(k, "Ae₂", X(e2[0]) + 8, Y(e2[1]) - 8, { size: 12, color: C.data });
      k.restore();
      out.innerHTML = `det A = <b>${fmt(det, 2)}</b> (area of the shaded square's image${det < 0 ? ", flipped" : ""}) · singular values <b>${fmt(s1, 2)}</b>, <b>${fmt(s2, 2)}</b> (the ellipse's semi-axes) · ${eig.length ? `real eigenvalues <b>${fmt(eig[0][0], 2)}</b>, <b>${fmt(eig[1][0], 2)}</b>: dashed lines map to themselves` : `complex eigenvalues ${fmt(tr / 2, 2)} ± ${fmt(Math.sqrt(-disc) / 2, 2)}i: no direction stays put, the plane turns`}`;
    }
    return { el: root, tick(dt) { stepTween(S, dt, show); draw(); } };
  });

  // ---------- M3 · The central limit theorem, by histogram ----------
  L.register("clt", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const S = { n: 1, dist: "exponential", tween: null };
    const show = knobs(root, S, 0);
    const D = { // sampler, mean, sd, skewness
      uniform: [r => r(), 0.5, Math.sqrt(1 / 12), 0], exponential: [r => -Math.log(1 - r()), 1, 1, 2], coin: [r => r() < 0.2 ? 1 : 0, 0.2, 0.4, (1 - 0.4) / 0.4],
    };
    let seed = 1, key = "", z = [];
    wire(root, S, { resample: () => { seed++; key = ""; } }, () => { key = ""; });
    function draw() {
      const n = Math.round(S.n), kk = `${n}|${S.dist}|${seed}`;
      if (kk !== key) { key = kk; const r = rng(seed * 977), [f, m, sd] = D[S.dist]; z = Array.from({ length: 6000 }, () => { let s = 0; for (let i = 0; i < n; i++) s += f(r); return (s - n * m) / (sd * Math.sqrt(n)); }); }
      const { k, W, H } = shape(cv, 0.42, 0.75), B = box(-4, 4, 0, 0.75, 36, 14, W - 52, H - 48);
      axes(k, B, { xs: [-3, -2, -1, 0, 1, 2, 3], ys: [0, 0.25, 0.5], xl: "(sum − nμ) / (σ√n) →" });
      const bins = 64, w = 8 / bins, h = new Array(bins).fill(0); z.forEach(v => { const i = Math.floor((v + 4) / w); if (i >= 0 && i < bins) h[i]++; });
      h.forEach((c, i) => { const v = Math.min(0.75, c / (z.length * w)); k.fillStyle = alpha(C.data, 0.6); k.fillRect(B.X(-4 + i * w) + 0.5, B.Y(v), B.X(w) - B.X(0) - 1, B.Y(0) - B.Y(v)); });
      curve(k, B, x => Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI), -4, 4, C.accent, 2.5);
      const sk = D[S.dist][3] / Math.sqrt(n);
      out.innerHTML = `${n} draw${n > 1 ? "s" : ""} per sum, 6,000 sums · orange: the standard normal · skewness of the sum <b>${fmt(sk, 2)}</b> = skewness of one draw ÷ √n${S.dist === "coin" && n < 20 ? " · a sum of coins can only take whole-number values, hence the gaps" : ""}`;
    }
    return { el: root, tick(dt) { stepTween(S, dt, show); draw(); } };
  });

  // ---------- M4 · Fitting one Gaussian to two bumps: forward against reverse KL ----------
  L.register("kldir", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const S = { sep: 2, tween: null };
    const show = knobs(root, S, 2);
    const xs = Array.from({ length: 401 }, (_, i) => -8 + 16 * i / 400), dx = 16 / 400, s0 = 0.6;
    const N = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
    let key = "", fit;
    function solve() {
      const p = xs.map(x => 0.5 * N(x, -S.sep, s0) + 0.5 * N(x, S.sep, s0));
      const kl = (a, b) => { let s = 0; for (let i = 0; i < xs.length; i++) if (a[i] > 1e-300) s += a[i] * Math.log(a[i] / Math.max(b[i], 1e-300)) * dx; return s; };
      const fm = [0, Math.sqrt(s0 * s0 + S.sep * S.sep)]; // forward KL(p‖q): match the mean and variance
      let best = [Infinity, 0, 1]; // reverse KL(q‖p): grid search, then refine
      for (let m = 0; m <= 4; m += 0.1) for (let s = 0.2; s <= 4; s += 0.05) { const q = xs.map(x => N(x, m, s)), v = kl(q, p); if (v < best[0]) best = [v, m, s]; }
      for (let it = 0; it < 3; it++) { const [, m0, s1] = best, h = 0.1 / 3 ** it; for (let m = m0 - h; m <= m0 + h; m += h / 4) for (let s = s1 - h / 2; s <= s1 + h / 2; s += h / 8) { if (s <= 0.05) continue; const q = xs.map(x => N(x, m, s)), v = kl(q, p); if (v < best[0]) best = [v, m, s]; } }
      const qf = xs.map(x => N(x, fm[0], fm[1])), qr = xs.map(x => N(x, best[1], best[2]));
      fit = { p, fm, rm: [best[1], best[2]], fwd: kl(p, qf), rev: best[0], fwdOfRev: kl(p, qr), revOfFwd: kl(qf, p) };
    }
    function draw() {
      const kk = S.sep.toFixed(3); if (kk !== key) { key = kk; solve(); }
      const { k, W, H } = shape(cv, 0.42, 0.75), B = box(-6, 6, 0, 0.75, 36, 14, W - 52, H - 48);
      axes(k, B, { xs: [-6, -4, -2, 0, 2, 4, 6], ys: [0, 0.25, 0.5] });
      k.fillStyle = alpha(C.muted, 0.25); k.beginPath(); k.moveTo(B.X(-6), B.Y(0)); xs.forEach((x, i) => { if (x >= -6 && x <= 6) k.lineTo(B.X(x), B.Y(Math.min(fit.p[i], 0.75))); }); k.lineTo(B.X(6), B.Y(0)); k.fill();
      curve(k, B, x => N(x, fit.fm[0], fit.fm[1]), -6, 6, C.data, 2.5, { dash: [6, 4] });
      curve(k, B, x => N(x, fit.rm[0], fit.rm[1]), -6, 6, C.accent, 2.5);
      label(k, "p: two bumps (grey)", B.left + 6, B.top + 10, { size: 12, color: C.ink });
      label(k, "argmin KL(p‖q): covers both (dashed)", B.left + 6, B.top + 26, { size: 12, color: C.data });
      label(k, "argmin KL(q‖p): picks one (solid)", B.left + 6, B.top + 42, { size: 12, color: C.accent });
      out.innerHTML = `forward fit N(${fmt(fit.fm[0], 2)}, ${fmt(fit.fm[1], 2)}²): KL(p‖q) = <b>${fmt(fit.fwd, 3)}</b>, but KL(q‖p) = <b>${fmt(fit.revOfFwd, 2)}</b> · reverse fit N(${fmt(fit.rm[0], 2)}, ${fmt(fit.rm[1], 2)}²): KL(q‖p) = <b>${fmt(fit.rev, 3)}</b>, but KL(p‖q) = <b>${fmt(fit.fwdOfRev, 2)}</b>`;
    }
    return { el: root, tick(dt) { stepTween(S, dt, show); draw(); } };
  });
})();
