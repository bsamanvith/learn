// Interactive figures for the classical-ML lessons (04–11). Each registers with learn.js.
(() => {
  const L = window.Learn; if (!L) return;
  const { rng, gauss, fmt, C, alpha, fit, shape, panes, boxIn, box, label, dot, knobs, wire, axes, curve, path, tweenTo, stepTween, reduced } = L;
  const sum = a => a.reduce((s, v) => s + v, 0);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const Phi = x => 0.5 * (1 + erf(x / Math.SQRT2));
  function erf(x) { // Abramowitz–Stegun 7.1.26, |error| < 1.5e-7
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
  }
  // Least squares (A w ≈ y, plus lam·|w|² if lam > 0) by Householder QR on the augmented system.
  function lstsq(A, y, lam = 0) {
    const p = A[0].length, R = A.map(r => r.slice()), b = y.slice();
    if (lam > 0) for (let j = 0; j < p; j++) { const r = new Array(p).fill(0); r[j] = Math.sqrt(lam); R.push(r); b.push(0); }
    const m = R.length;
    for (let j = 0; j < p; j++) {
      let nrm = 0; for (let i = j; i < m; i++) nrm += R[i][j] ** 2; nrm = Math.sqrt(nrm);
      if (nrm < 1e-300) continue;
      const a = R[j][j] > 0 ? -nrm : nrm, v = [];
      for (let i = j; i < m; i++) v.push(R[i][j]); v[0] -= a;
      const vv = sum(v.map(x => x * x)); if (vv < 1e-300) continue;
      for (let c = j; c < p; c++) { let d = 0; for (let i = j; i < m; i++) d += v[i - j] * R[i][c]; d = 2 * d / vv; for (let i = j; i < m; i++) R[i][c] -= d * v[i - j]; }
      let d = 0; for (let i = j; i < m; i++) d += v[i - j] * b[i]; d = 2 * d / vv; for (let i = j; i < m; i++) b[i] -= d * v[i - j];
    }
    const w = new Array(p).fill(0);
    for (let j = p - 1; j >= 0; j--) { let s = b[j]; for (let c = j + 1; c < p; c++) s -= R[j][c] * w[c]; w[j] = Math.abs(R[j][j]) > 1e-12 ? s / R[j][j] : 0; }
    return w;
  }
  // Legendre polynomials P_0..P_d at t in [-1, 1]: a well-conditioned polynomial basis.
  function legendre(t, d) { const P = [1, t]; for (let n = 1; n < d; n++) P.push(((2 * n + 1) * t * P[n] - n * P[n - 1]) / (n + 1)); return P.slice(0, d + 1); }

  // ---------- 04 · Fit polynomials of every degree to 15 noisy points ----------
  L.register("polyfit", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const f = x => Math.sin(2 * Math.PI * x), sigma = 0.3, n = 15, D = 12;
    const S = { d: 3, seed: 4, tween: null };
    const show = knobs(root, S, 0);
    let data, fits, errs;
    function build() {
      const r = rng(S.seed);
      const xs = Array.from({ length: n }, (_, i) => (i + 0.15 + 0.7 * r()) / n);
      const ys = xs.map(x => f(x) + sigma * gauss(r));
      data = { xs, ys };
      fits = []; errs = [];
      for (let d = 0; d <= D; d++) {
        const w = lstsq(xs.map(x => legendre(2 * x - 1, d)), ys);
        const g = x => sum(legendre(2 * x - 1, d).map((v, i) => v * w[i]));
        const train = sum(xs.map((x, i) => (g(x) - ys[i]) ** 2)) / n;
        let test = 0; for (let s = 0; s < 400; s++) { const x = (s + 0.5) / 400; test += (g(x) - f(x)) ** 2; }
        fits.push(g); errs.push({ train, test: test / 400 + sigma * sigma });
      }
    }
    build();
    wire(root, S, { resample: () => { S.seed++; build(); } });
    function draw() {
      const d = Math.round(S.d), P = panes(cv, { frac: 0.58 }), k = P.k;
      const A = boxIn(P.a, 0, 1, -1.9, 1.9, 8, 10, 8, 10);
      curve(k, A, f, 0, 1, alpha(C.muted, 0.8), 1.5, { dash: [5, 5] });
      curve(k, A, fits[d], 0, 1, C.accent, 3, { n: 400 });
      data.xs.forEach((x, i) => dot(k, A.X(x), A.Y(data.ys[i]), 4.2, C.data, C.panel, 1.5));
      label(k, `degree ${d}`, A.left + 6, A.top + 12, { color: C.ink, size: 13, base: "middle" });
      // right: train and test error against degree, on a log scale
      const B = boxIn(P.b, -0.5, D + 0.5, -3, 1, 44, 16, 14, 36), ly = v => Math.log10(clamp(v, 1e-3, 10));
      axes(k, B, { xs: [0, 3, 6, 9, 12], ys: [-3, -2, -1, 0, 1], fy: v => v === 0 ? "1" : `1e${v}`, xl: "degree →", yl: "mean squared error", grid: true });
      path(k, B, errs.map((e, i) => [i, ly(e.train)]), C.data, 2);
      path(k, B, errs.map((e, i) => [i, ly(e.test)]), C.accent, 2.5);
      errs.forEach((e, i) => { dot(k, B.X(i), B.Y(ly(e.train)), i === d ? 5.5 : 3, i === d ? C.data : C.panel, C.data, 2); dot(k, B.X(i), B.Y(ly(e.test)), i === d ? 5.5 : 3, i === d ? C.accent : C.panel, C.accent, 2); });
      label(k, "test", B.X(D) - 4, B.Y(ly(errs[D].test)) + (errs[D].test > 9 ? 14 : -10), { align: "right", color: C.accent, size: 12 });
      label(k, "train", B.X(D) - 4, B.Y(ly(errs[D].train)) - 10, { align: "right", color: C.data, size: 12 });
      k.strokeStyle = alpha(C.muted, 0.5); k.setLineDash([3, 4]); k.lineWidth = 1;
      k.beginPath(); k.moveTo(B.left, B.Y(ly(sigma * sigma))); k.lineTo(B.left + B.width, B.Y(ly(sigma * sigma))); k.stroke(); k.setLineDash([]);
      label(k, "noise floor σ²", B.left + 4, B.Y(ly(sigma * sigma)) + 12, { size: 11 });
      const e = errs[d];
      out.innerHTML = `${d + 1} parameters for ${n} points · train error <b>${fmt(e.train, 3)}</b> · test error <b>${fmt(e.test, 3)}</b>${d >= 10 ? " · <b>memorising the noise</b>" : d <= 1 ? " · <b>too stiff to bend</b>" : ""}`;
    }
    return { el: root, tick(dt) { stepTween(S, dt, show); draw(); } };
  });

  // ---------- 05 · Ridge and lasso as a ball growing until it touches the loss contours ----------
  L.register("ballpath", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const w0 = [2.1, 0.75], th = 0.5, l1 = 0.6, l2 = 2.2; // loss (w − w0)ᵀ H (w − w0), H = U diag(l1, l2) Uᵀ
    const u1 = [Math.cos(th), Math.sin(th)], u2 = [-Math.sin(th), Math.cos(th)];
    const H = [[l1 * u1[0] ** 2 + l2 * u2[0] ** 2, l1 * u1[0] * u1[1] + l2 * u2[0] * u2[1]], [0, l1 * u1[1] ** 2 + l2 * u2[1] ** 2]]; H[1][0] = H[0][1];
    const loss = w => { const a = w[0] - w0[0], b = w[1] - w0[1]; return H[0][0] * a * a + 2 * H[0][1] * a * b + H[1][1] * b * b; };
    const S = { lam: -0.3, mode: "l1", tween: null };
    const show = knobs(root, S, 2);
    function solve(lam, mode) {
      if (mode === "l2") { // (H + λI) w = H w0
        const a = H[0][0] + lam, b = H[0][1], d = H[1][1] + lam, r0 = H[0][0] * w0[0] + H[0][1] * w0[1], r1 = H[1][0] * w0[0] + H[1][1] * w0[1], det = a * d - b * b;
        return [(d * r0 - b * r1) / det, (a * r1 - b * r0) / det];
      }
      const w = [0, 0], soft = (z, t) => Math.sign(z) * Math.max(0, Math.abs(z) - t); // coordinate descent
      for (let it = 0; it < 300; it++) for (let j = 0; j < 2; j++) { const o = 1 - j, c = H[j][o] * (w[o] - w0[o]); w[j] = soft(w0[j] - c / H[j][j], lam / (2 * H[j][j])); }
      return w;
    }
    wire(root, S, {});
    function draw() {
      const lam = 10 ** S.lam, w = solve(lam, S.mode);
      const { k, W, H: Hh } = shape(cv, 0.5, 0.75);
      // equal scale on both axes, centred on the interesting region
      const sc = Math.min(W / 5.2, Hh / 4.4), cx = 0.7, cy = 0.45;
      const B = box(cx - W / (2 * sc), cx + W / (2 * sc), cy - Hh / (2 * sc), cy + Hh / (2 * sc), 0, 0, W, Hh);
      const X = B.X, Y = B.Y, span = cy + Hh / (2 * sc) - 0.3;
      // axes
      k.strokeStyle = C.line; k.lineWidth = 1;
      k.beginPath(); k.moveTo(0, Y(0)); k.lineTo(W, Y(0)); k.moveTo(X(0), 0); k.lineTo(X(0), Hh); k.stroke();
      label(k, "w₁", W - 14, Y(0) - 8, { size: 12, align: "right" }); label(k, "w₂", X(0) + 8, 16, { size: 12 });
      // loss contours, through the solution and a few more
      const lv = loss(w), levels = [lv / 4, lv, lv * 2.2, lv * 4].filter(v => v > 1e-4);
      levels.forEach((c, i) => {
        k.strokeStyle = i === 1 ? C.data : alpha(C.data, 0.35); k.lineWidth = i === 1 ? 2 : 1.2; k.beginPath();
        for (let s = 0; s <= 120; s++) { const a = 2 * Math.PI * s / 120, p = Math.sqrt(c) * Math.cos(a) / Math.sqrt(l1), q = Math.sqrt(c) * Math.sin(a) / Math.sqrt(l2);
          const x = w0[0] + p * u1[0] + q * u2[0], y = w0[1] + p * u1[1] + q * u2[1]; s ? k.lineTo(X(x), Y(y)) : k.moveTo(X(x), Y(y)); }
        k.stroke();
      });
      // the ball through the solution
      const r = S.mode === "l1" ? Math.abs(w[0]) + Math.abs(w[1]) : Math.hypot(w[0], w[1]);
      k.fillStyle = alpha(C.accent, 0.12); k.strokeStyle = C.accent; k.lineWidth = 2; k.beginPath();
      if (S.mode === "l1") { k.moveTo(X(r), Y(0)); k.lineTo(X(0), Y(r)); k.lineTo(X(-r), Y(0)); k.lineTo(X(0), Y(-r)); k.closePath(); }
      else k.ellipse(X(0), Y(0), Math.abs(X(r) - X(0)), Math.abs(X(r) - X(0)), 0, 0, 2 * Math.PI);
      k.fill(); k.stroke();
      // the whole regularisation path
      const pts = []; for (let s = 0; s <= 120; s++) pts.push(solve(10 ** (-3 + 5 * s / 120), S.mode));
      path(k, B, pts, alpha(C.ink, 0.55), 1.5, [4, 4]);
      dot(k, X(w0[0]), Y(w0[1]), 5, C.data, C.panel, 1.5);
      label(k, "least squares", X(w0[0]), Y(w0[1]) - 12, { color: C.data, size: 12, align: "center" });
      dot(k, X(w[0]), Y(w[1]), 6.5, C.panel, C.accent, 3);
      const zero = S.mode === "l1" && Math.abs(w[1]) < 1e-9 && Math.abs(w[0]) > 1e-9;
      out.innerHTML = `λ = <b>${fmt(lam, 3)}</b> · w = (<b>${fmt(w[0], 3)}</b>, <b>${fmt(w[1], 3)}</b>) · ${S.mode === "l1" ? "‖w‖₁" : "‖w‖₂"} = <b>${fmt(r, 3)}</b>${zero ? " · <b>w₂ is exactly zero</b>: the diamond's corner touched first" : Math.abs(w[0]) + Math.abs(w[1]) < 1e-9 ? " · <b>everything is zero</b>" : ""}`;
    }
    return { el: root, tick(dt) { stepTween(S, dt, show); draw(); } };
  });

  // ---------- 06 · Softmax with a temperature ----------
  L.register("softmax", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const names = ["cat", "dog", "fox", "owl"];
    const S = { z0: 2, z1: 1, z2: 0.2, z3: -1, T: 0, tween: null }; // T is log10 of the temperature
    const show = knobs(root, S, 1);
    root.querySelectorAll("[data-out=T]").forEach(o => { o.dataset.fmt = "pow"; });
    function draw() {
      const T = 10 ** S.T, z = [S.z0, S.z1, S.z2, S.z3], m = Math.max(...z), e = z.map(v => Math.exp((v - m) / T)), Z = sum(e), p = e.map(v => v / Z);
      root.querySelectorAll("[data-out=T]").forEach(o => { o.textContent = fmt(T, 2); });
      const P = panes(cv, { frac: 0.5, wide: 0.36, narrow: 1.0 }), k = P.k;
      const A = boxIn(P.a, -0.5, 3.5, -5, 5, 36, 22, 8, 26), B = boxIn(P.b, -0.5, 3.5, 0, 1, 36, 22, 8, 26);
      label(k, "logits z", A.left, P.a.y + 8, { color: C.ink, size: 12, base: "middle" });
      label(k, `probabilities softmax(z / T), T = ${fmt(T, 2)}`, B.left, P.b.y + 8, { color: C.ink, size: 12, base: "middle" });
      axes(k, A, { ys: [-4, 0, 4] }); axes(k, B, { ys: [0, 0.5, 1] });
      k.strokeStyle = C.line; k.beginPath(); k.moveTo(A.left, A.Y(0)); k.lineTo(A.left + A.width, A.Y(0)); k.stroke();
      const bw = A.width / 4 * 0.56;
      z.forEach((v, i) => {
        k.fillStyle = alpha(C.data, 0.85); k.fillRect(A.X(i) - bw / 2, Math.min(A.Y(0), A.Y(v)), bw, Math.abs(A.Y(v) - A.Y(0)));
        label(k, names[i], A.X(i), A.top + A.height + 16, { align: "center", size: 12 });
        k.fillStyle = i === z.indexOf(m) ? C.accent : alpha(C.accent, 0.55); k.fillRect(B.X(i) - bw / 2, B.Y(p[i]), bw, B.Y(0) - B.Y(p[i]));
        label(k, fmt(p[i], 2), B.X(i), B.Y(p[i]) - 6, { align: "center", color: C.ink, size: 12 });
        label(k, names[i], B.X(i), B.top + B.height + 16, { align: "center", size: 12 });
      });
      const H = -sum(p.map(v => v > 0 ? v * Math.log2(v) : 0));
      out.innerHTML = `entropy <b>${fmt(H, 2)}</b> bits (max 2) · ${T < 0.2 ? "<b>T → 0: argmax</b>, all mass on the largest logit" : T > 4 ? "<b>T large: nearly uniform</b>" : "the order never changes; T only sharpens or flattens"} · adding the same number to every logit changes nothing`;
    }
    return { el: root, tick(dt) { stepTween(S, dt, show); draw(); } };
  });

  // ---------- 07 · Minibatch SGD with momentum on a stretched bowl ----------
  L.register("sgd", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const r0 = rng(7), N = 256, X = [], Y = [];
    for (let i = 0; i < N; i++) { const a = gauss(r0), b = 0.85 * a + 0.53 * gauss(r0); X.push([a, b]); Y.push(1.2 * a - 0.8 * b + 0.5 * gauss(r0)); }
    const G = [[0, 0], [0, 0]], g0 = [0, 0];
    X.forEach((x, i) => { for (let a = 0; a < 2; a++) { g0[a] += x[a] * Y[i] / N; for (let b = 0; b < 2; b++) G[a][b] += x[a] * x[b] / N; } });
    const yy = sum(Y.map(y => y * y)) / N, det = G[0][0] * G[1][1] - G[0][1] ** 2;
    const wS = [(G[1][1] * g0[0] - G[0][1] * g0[1]) / det, (G[0][0] * g0[1] - G[0][1] * g0[0]) / det];
    const loss = w => 0.5 * (w[0] * w[0] * G[0][0] + 2 * w[0] * w[1] * G[0][1] + w[1] * w[1] * G[1][1]) - (w[0] * g0[0] + w[1] * g0[1]) + 0.5 * yy;
    const Lmin = loss(wS), steps = 150, start = [-1.6, 2.4];
    const S = { lr: 0.25, bs: 3, mom: 0, seed: 1, t: 0, tween: null }; // bs is log2 of the batch size
    const show = knobs(root, S, 2);
    let run;
    function build() {
      const r = rng(S.seed * 97 + 3), B = Math.round(2 ** S.bs);
      let w = start.slice(), v = [0, 0]; const pts = [w.slice()], ls = [loss(w) - Lmin];
      for (let t = 0; t < steps; t++) {
        const g = [0, 0];
        for (let j = 0; j < B; j++) { const i = Math.floor(r() * N), e = X[i][0] * w[0] + X[i][1] * w[1] - Y[i]; g[0] += e * X[i][0] / B; g[1] += e * X[i][1] / B; }
        v = [S.mom * v[0] + g[0], S.mom * v[1] + g[1]]; w = [w[0] - S.lr * v[0], w[1] - S.lr * v[1]];
        if (!isFinite(w[0]) || Math.abs(w[0]) + Math.abs(w[1]) > 1e3) { pts.push(w); ls.push(Infinity); break; }
        pts.push(w.slice()); ls.push(loss(w) - Lmin);
      }
      run = { pts, ls, B }; S.t = reduced() ? 1e9 : 0;
      root.querySelectorAll("[data-out=bs]").forEach(o => { o.textContent = B; });
    }
    build();
    root.querySelectorAll("input[data-k]").forEach(el => el.addEventListener("input", build));
    wire(root, S, { rerun: () => { S.seed++; build(); } });
    function draw() {
      const P = panes(cv, { frac: 0.55, wide: 0.44 }), k = P.k, shown = Math.min(run.pts.length, 1 + Math.floor(S.t / 18));
      const A = boxIn(P.a, -2.2, 3.2, -1.8, 2.8, 8, 8, 8, 8);
      k.save(); k.beginPath(); k.rect(A.left, A.top, A.width, A.height); k.clip();
      for (const c of [0.02, 0.1, 0.3, 0.7, 1.4, 2.5, 4]) { // contours: solve the quadratic along rays from w*
        k.strokeStyle = alpha(C.data, 0.3); k.lineWidth = 1; k.beginPath();
        for (let s = 0; s <= 100; s++) { const a = 2 * Math.PI * s / 100, d = [Math.cos(a), Math.sin(a)], q = 0.5 * (d[0] * d[0] * G[0][0] + 2 * d[0] * d[1] * G[0][1] + d[1] * d[1] * G[1][1]), rr = Math.sqrt(c / q);
          s ? k.lineTo(A.X(wS[0] + rr * d[0]), A.Y(wS[1] + rr * d[1])) : k.moveTo(A.X(wS[0] + rr * d[0]), A.Y(wS[1] + rr * d[1])); }
        k.stroke();
      }
      k.restore();
      path(k, A, run.pts.slice(0, shown).filter(p => isFinite(p[0])), C.accent, 1.6);
      dot(k, A.X(wS[0]), A.Y(wS[1]), 4, C.ink, null);
      const cur = run.pts[shown - 1]; if (isFinite(cur[0])) dot(k, A.X(clamp(cur[0], -2.2, 3.2)), A.Y(clamp(cur[1], -1.8, 2.8)), 5.5, C.panel, C.accent, 2.5);
      const B = boxIn(P.b, 0, steps, -4, 1, 44, 16, 14, 36), lg = v => Math.log10(clamp(v, 1e-4, 10));
      axes(k, B, { xs: [0, 50, 100, 150], ys: [-4, -3, -2, -1, 0, 1], fy: v => v === 0 ? "1" : v === 1 ? "10" : `1e${v}`, xl: "step →", yl: "loss − minimum", grid: true });
      path(k, B, run.ls.slice(0, shown).map((v, i) => [i, lg(v)]), C.accent, 2);
      const tail = run.ls.slice(-40).filter(isFinite), floor = tail.length ? sum(tail) / tail.length : Infinity;
      const div = !isFinite(run.ls[run.ls.length - 1]);
      out.innerHTML = div ? `batch ${run.B} · <b>diverged</b>: the step is too long for the steepest direction (η·λ<sub>max</sub> > 2${S.mom ? "(1 + β)" : ""})`
        : `batch <b>${run.B}</b> · η = ${fmt(S.lr, 2)} · β = ${fmt(S.mom, 2)} · loss above the minimum over the last 40 steps: <b>${floor.toExponential(1)}</b>${run.B <= 2 ? " · noisy: it rattles around the bottom" : ""}`;
    }
    return { el: root, tick(dt) { S.t += dt; stepTween(S, dt, show); draw(); } };
  });

  // ---------- 08 · Threshold, ROC curve and precision under a base rate ----------
  L.register("roc", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const S = { dp: 1.5, tau: 0.75, prev: -1, tween: null }; // prev is log10 of the share of positives
    const show = knobs(root, S, 2);
    const pdf = (x, m) => Math.exp(-0.5 * (x - m) ** 2) / Math.sqrt(2 * Math.PI);
    function draw() {
      const pi = 10 ** S.prev, tpr = 1 - Phi(S.tau - S.dp), fpr = 1 - Phi(S.tau), prec = pi * tpr / (pi * tpr + (1 - pi) * fpr || 1);
      root.querySelectorAll("[data-out=prev]").forEach(o => { o.textContent = `${fmt(100 * pi, pi < 0.1 ? 1 : 0)}%`; });
      const P = panes(cv, { frac: 0.6, wide: 0.42 }), k = P.k;
      const A = boxIn(P.a, -3.5, 6, 0, 0.45, 12, 12, 8, 36);
      axes(k, A, { xs: [-2, 0, 2, 4], xl: "classifier score →" });
      const fill = (m, color, from) => { k.fillStyle = color; k.beginPath(); k.moveTo(A.X(from), A.Y(0)); for (let s = 0; s <= 80; s++) { const x = from + (6 - from) * s / 80; k.lineTo(A.X(x), A.Y(pdf(x, m))); } k.lineTo(A.X(6), A.Y(0)); k.closePath(); k.fill(); };
      fill(0, alpha(C.data, 0.25), Math.max(-3.5, S.tau)); fill(S.dp, alpha(C.accent, 0.3), Math.max(-3.5, S.tau));
      curve(k, A, x => pdf(x, 0), -3.5, 6, C.data, 2.2); curve(k, A, x => pdf(x, S.dp), -3.5, 6, C.accent, 2.2);
      k.strokeStyle = C.ink; k.lineWidth = 1.5; k.setLineDash([5, 4]); k.beginPath(); k.moveTo(A.X(S.tau), A.top); k.lineTo(A.X(S.tau), A.top + A.height); k.stroke(); k.setLineDash([]);
      label(k, "negatives", A.X(0) - 8, A.Y(pdf(0, 0)) - 4, { align: "right", color: C.data, size: 12 });
      label(k, "positives", A.X(S.dp) + 8, A.Y(pdf(0, 0)) - 4, { color: C.accent, size: 12 });
      label(k, "flagged →", A.X(S.tau) + 6, A.Y(0) - 8, { size: 11, color: C.ink });
      const B = boxIn(P.b, 0, 1, 0, 1, 40, 16, 14, 36);
      axes(k, B, { xs: [0, 0.5, 1], ys: [0, 0.5, 1], xl: "false positive rate →", yl: "true positive rate" });
      k.strokeStyle = alpha(C.muted, 0.5); k.setLineDash([3, 4]); k.lineWidth = 1; k.beginPath(); k.moveTo(B.X(0), B.Y(0)); k.lineTo(B.X(1), B.Y(1)); k.stroke(); k.setLineDash([]);
      const roc = []; for (let s = 0; s <= 120; s++) { const t = 7 - 11 * s / 120; roc.push([1 - Phi(t), 1 - Phi(t - S.dp)]); }
      path(k, B, roc, C.accent, 2.5);
      dot(k, B.X(fpr), B.Y(tpr), 6, C.panel, C.ink, 2.5);
      label(k, `AUC ${fmt(Phi(S.dp / Math.SQRT2), 3)}`, B.X(0.96), B.Y(0.06), { align: "right", color: C.ink, size: 12 });
      out.innerHTML = `true positive rate <b>${fmt(tpr, 2)}</b> · false positive rate <b>${fmt(fpr, 3)}</b> · with ${fmt(100 * pi, pi < 0.1 ? 1 : 0)}% positives, precision is <b>${fmt(prec, 2)}</b>${prec < 0.5 && tpr > 0.5 ? ": most alarms are false, though the ROC point looks fine" : ""}`;
    }
    return { el: root, tick(dt) { stepTween(S, dt, show); draw(); } };
  });

  // ---------- 09 · Choosing a split: impurity as a function of the threshold ----------
  L.register("split", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const r = rng(23), pts = [];
    for (let i = 0; i < 26; i++) pts.push([clamp(0.3 + 0.16 * gauss(r), 0.02, 0.98), clamp(0.36 + 0.17 * gauss(r), 0.02, 0.98), 0]);
    for (let i = 0; i < 22; i++) pts.push([clamp(0.68 + 0.14 * gauss(r), 0.02, 0.98), clamp(0.62 + 0.15 * gauss(r), 0.02, 0.98), 1]);
    const S = { t: 0.42, axis: 0, tween: null };
    const show = knobs(root, S, 2);
    const gini = q => 2 * q * (1 - q), ent = q => q <= 0 || q >= 1 ? 0 : -(q * Math.log2(q) + (1 - q) * Math.log2(1 - q)) / 2, mis = q => Math.min(q, 1 - q);
    function score(t, a, fn) {
      const l = pts.filter(p => p[a] <= t), rr = pts.filter(p => p[a] > t), q = s => s.length ? s.filter(p => p[2]).length / s.length : 0;
      return { l, r: rr, imp: (l.length * fn(q(l)) + rr.length * fn(q(rr))) / pts.length, ql: q(l), qr: q(rr) };
    }
    const best = a => { let b = [1, 0.5]; for (let s = 1; s < 200; s++) { const t = s / 200, v = score(t, a, gini).imp; if (v < b[0] - 1e-12) b = [v, t]; } return b; };
    wire(root, S, { best: () => tweenTo(S, { t: best(S.axis)[1] }) }, () => {});
    function draw() {
      const P = panes(cv, { frac: 0.48, wide: 0.46, narrow: 1.5 }), k = P.k, a = S.axis;
      const side = Math.min(P.a.w, P.a.h) - 20, A = box(0, 1, 0, 1, P.a.x + (P.a.w - side) / 2, P.a.y + 6, side, side);
      const sc = score(S.t, a, gini);
      k.fillStyle = alpha(sc.ql > 0.5 ? C.accent : C.data, 0.08);
      a === 0 ? k.fillRect(A.X(0), A.Y(1), A.X(S.t) - A.X(0), A.height) : k.fillRect(A.X(0), A.Y(S.t), A.width, A.Y(0) - A.Y(S.t));
      k.fillStyle = alpha(sc.qr > 0.5 ? C.accent : C.data, 0.08);
      a === 0 ? k.fillRect(A.X(S.t), A.Y(1), A.X(1) - A.X(S.t), A.height) : k.fillRect(A.X(0), A.Y(1), A.width, A.Y(S.t) - A.Y(1));
      k.strokeStyle = C.line; k.strokeRect(A.left, A.top, A.width, A.height);
      pts.forEach(p => p[2] ? dot(k, A.X(p[0]), A.Y(p[1]), 4.5, C.accent, C.panel, 1.2) : dot(k, A.X(p[0]), A.Y(p[1]), 4.5, C.data, C.panel, 1.2));
      k.strokeStyle = C.ink; k.lineWidth = 2; k.beginPath();
      a === 0 ? (k.moveTo(A.X(S.t), A.top), k.lineTo(A.X(S.t), A.top + A.height)) : (k.moveTo(A.left, A.Y(S.t)), k.lineTo(A.left + A.width, A.Y(S.t)));
      k.stroke();
      label(k, a === 0 ? "x₁ →" : "x₂ ↑", A.left + A.width - 4, A.top + A.height - 6, { align: "right", size: 11 });
      const B = boxIn(P.b, 0, 1, 0, 0.55, 40, 16, 14, 36);
      axes(k, B, { xs: [0, 0.5, 1], ys: [0, 0.25, 0.5], xl: `threshold on ${a === 0 ? "x₁" : "x₂"} →`, yl: "weighted impurity after the split" });
      curve(k, B, t => score(t, a, gini).imp, 0, 1, C.accent, 2.5, { n: 200 });
      curve(k, B, t => score(t, a, ent).imp, 0, 1, C.data, 1.6, { n: 200, dash: [5, 4] });
      curve(k, B, t => score(t, a, mis).imp, 0, 1, alpha(C.ink, 0.6), 1.4, { n: 200 });
      label(k, "Gini", B.X(0.03), B.Y(0) - 38, { color: C.accent, size: 12 });
      label(k, "entropy / 2 (dashed)", B.X(0.03), B.Y(0) - 23, { color: C.data, size: 12 });
      label(k, "misclassification (grey)", B.X(0.03), B.Y(0) - 8, { color: C.muted, size: 12 });
      dot(k, B.X(S.t), B.Y(sc.imp), 6, C.panel, C.ink, 2.5);
      const [bi, bt] = best(a);
      const cnt = s => `${s.filter(p => !p[2]).length} blue · ${s.filter(p => p[2]).length} orange`;
      out.innerHTML = `left: <b>${cnt(sc.l)}</b> · right: <b>${cnt(sc.r)}</b> · Gini after split <b>${fmt(sc.imp, 3)}</b> (before ${fmt(gini(22 / 48), 3)}) · best on this axis <b>${fmt(bi, 3)}</b> at ${fmt(bt, 2)}`;
    }
    return { el: root, tick(dt) { stepTween(S, dt, show); draw(); } };
  });

  // ---------- 10 · Averaging correlated trees ----------
  L.register("forestvar", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const S = { rho: 0.3, B: 1.5, seed: 1, tween: null }; // B is log10 of the number of trees
    const show = knobs(root, S, 2);
    wire(root, S, { resample: () => S.seed++ });
    function draw() {
      const nb = Math.round(10 ** S.B), rho = S.rho, v = rho + (1 - rho) / nb;
      root.querySelectorAll("[data-out=B]").forEach(o => { o.textContent = nb; });
      const P = panes(cv, { frac: 0.5, wide: 0.42 }), k = P.k;
      // left: 24 forests; each row shows its trees (ticks) and their average (dot)
      const A = boxIn(P.a, -3.2, 3.2, 0, 24, 10, 22, 10, 30), r = rng(S.seed * 131 + Math.round(rho * 100));
      label(k, "24 forests grown on 24 fresh datasets", A.left, P.a.y + 8, { color: C.ink, size: 12, base: "middle" });
      for (let f = 0; f < 24; f++) {
        const shared = gauss(r) * Math.sqrt(rho); let s = 0; const y = A.Y(f + 0.5);
        const show = Math.min(nb, 60);
        for (let b = 0; b < nb; b++) { const t = shared + Math.sqrt(1 - rho) * gauss(r); s += t; if (b < show) { k.strokeStyle = alpha(C.data, 0.35); k.lineWidth = 1; k.beginPath(); k.moveTo(A.X(clamp(t, -3.2, 3.2)), y - 4); k.lineTo(A.X(clamp(t, -3.2, 3.2)), y + 4); k.stroke(); } }
        dot(k, A.X(clamp(s / nb, -3.2, 3.2)), y, 4, C.accent, null);
      }
      k.strokeStyle = C.ink; k.setLineDash([3, 3]); k.beginPath(); k.moveTo(A.X(0), A.top); k.lineTo(A.X(0), A.top + A.height); k.stroke(); k.setLineDash([]);
      label(k, "blue ticks: single trees · orange: the forest's average · dashed: the truth", A.left, A.top + A.height + 18, { size: 11 });
      const B = boxIn(P.b, 0, 3, 0, 1.05, 40, 16, 14, 36);
      axes(k, B, { xs: [0, 1, 2, 3], fx: x => String(10 ** x), ys: [0, 0.5, 1], xl: "number of trees (log) →", yl: "variance of the average / σ²" });
      for (const g of [0, 0.1, 0.3, 0.6]) if (Math.abs(g - rho) > 0.02) curve(k, B, x => g + (1 - g) / 10 ** x, 0, 3, alpha(C.muted, 0.45), 1.2);
      curve(k, B, x => rho + (1 - rho) / 10 ** x, 0, 3, C.accent, 2.6);
      k.strokeStyle = C.ink; k.setLineDash([4, 4]); k.lineWidth = 1; k.beginPath(); k.moveTo(B.left, B.Y(rho)); k.lineTo(B.left + B.width, B.Y(rho)); k.stroke(); k.setLineDash([]);
      label(k, "floor ρσ²", B.left + B.width - 4, B.Y(rho) - 7, { align: "right", size: 11 });
      dot(k, B.X(Math.log10(nb)), B.Y(v), 6, C.panel, C.accent, 2.5);
      out.innerHTML = `${nb} trees with correlation ρ = ${fmt(rho, 2)} · variance of the average = ρσ² + (1 − ρ)σ²/B = <b>${fmt(v, 3)}</b>σ² · more trees can never beat <b>${fmt(rho, 2)}</b>σ²; only decorrelating them can`;
    }
    return { el: root, tick(dt) { stepTween(S, dt, show); draw(); } };
  });

  // ---------- 11 · Gradient boosting with stumps, one residual at a time ----------
  L.register("boost", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const f = x => Math.sin(5 * x) + 0.6 * Math.cos(13 * x) * x, r = rng(5);
    const tr = Array.from({ length: 70 }, () => { const x = r(); return [x, f(x) + 0.28 * gauss(r)]; }).sort((a, b) => a[0] - b[0]);
    const te = Array.from({ length: 400 }, () => { const x = r(); return [x, f(x) + 0.28 * gauss(r)]; });
    const M = 300, S = { m: 12, nu: 0.3, tween: null };
    const show = knobs(root, S, 2);
    let stumps, curves, lastNu;
    function build() {
      lastNu = S.nu; stumps = []; curves = [];
      const F = tr.map(() => 0), Ft = te.map(() => 0), n = tr.length;
      const mse = (P, D) => sum(D.map((d, i) => (d[1] - P[i]) ** 2)) / D.length;
      curves.push([mse(F, tr), mse(Ft, te)]);
      for (let m = 0; m < M; m++) {
        const res = tr.map((d, i) => d[1] - F[i]), tot = sum(res); let bestS = [-1, 0, 0, 0], left = 0;
        for (let i = 0; i < n - 1; i++) { left += res[i]; const nl = i + 1, nr = n - nl, gain = left * left / nl + (tot - left) ** 2 / nr; if (gain > bestS[0]) bestS = [gain, (tr[i][0] + tr[i + 1][0]) / 2, left / nl, (tot - left) / nr]; }
        const [, t, a, b] = bestS; stumps.push([t, S.nu * a, S.nu * b]);
        tr.forEach((d, i) => { F[i] += d[0] <= t ? S.nu * a : S.nu * b; }); te.forEach((d, i) => { Ft[i] += d[0] <= t ? S.nu * a : S.nu * b; });
        curves.push([mse(F, tr), mse(Ft, te)]);
      }
    }
    build();
    const model = (x, m) => { let s = 0; for (let j = 0; j < m; j++) s += x <= stumps[j][0] ? stumps[j][1] : stumps[j][2]; return s; };
    function draw() {
      if (S.nu !== lastNu) build();
      root.querySelectorAll("[data-out=m]").forEach(o => { o.textContent = Math.round(S.m); });
      const m = Math.round(S.m), P = panes(cv, { frac: 0.58, wide: 0.44 }), k = P.k;
      const A = boxIn(P.a, 0, 1, -2.2, 2.2, 8, 10, 8, 10);
      tr.forEach(d => { const p = model(d[0], m); k.strokeStyle = alpha(C.bad || C.accent, 0.35); k.lineWidth = 1; k.beginPath(); k.moveTo(A.X(d[0]), A.Y(d[1])); k.lineTo(A.X(d[0]), A.Y(p)); k.stroke(); });
      tr.forEach(d => dot(k, A.X(d[0]), A.Y(d[1]), 3.6, C.data, C.panel, 1.2));
      curve(k, A, x => model(x, m), 0, 1, C.accent, 2.6, { n: 500 });
      if (m > 0) { const [t, a, b] = stumps[m - 1]; curve(k, A, x => -1.75 + (x <= t ? a : b) * 3, 0, 1, alpha(C.ink, 0.6), 1.5, { n: 500, dash: [4, 3] }); label(k, "last stump added (×3, offset)", A.left + 6, A.Y(-1.75) + 18, { size: 11 }); }
      label(k, `${m} stumps, shrinkage ν = ${fmt(S.nu, 2)}`, A.left + 6, A.top + 12, { color: C.ink, size: 13, base: "middle" });
      const B = boxIn(P.b, 0, M, 0, 0.3, 44, 16, 14, 36);
      axes(k, B, { xs: [0, 100, 200, 300], ys: [0, 0.1, 0.2, 0.3], xl: "boosting rounds →", yl: "mean squared error", grid: true });
      path(k, B, curves.map((c, i) => [i, Math.min(c[0], 0.3)]), C.data, 2);
      path(k, B, curves.map((c, i) => [i, Math.min(c[1], 0.3)]), C.accent, 2.5);
      dot(k, B.X(m), B.Y(Math.min(curves[m][1], 0.3)), 5.5, C.panel, C.accent, 2.5);
      dot(k, B.X(m), B.Y(Math.min(curves[m][0], 0.3)), 4.5, C.panel, C.data, 2);
      let bi = 0; curves.forEach((c, i) => { if (c[1] < curves[bi][1]) bi = i; });
      out.innerHTML = `train <b>${fmt(curves[m][0], 3)}</b> · test <b>${fmt(curves[m][1], 3)}</b> (noise floor 0.078) · best test error at <b>${bi}</b> rounds for this ν${m > bi * 1.6 + 10 ? " · past it, new stumps chase noise" : ""}`;
    }
    return { el: root, tick(dt) { stepTween(S, dt, show); draw(); } };
  });
})();
