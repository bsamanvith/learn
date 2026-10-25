// Interactive figures for the deep-learning lessons (12–21). Each registers with learn.js.
(() => {
  const L = window.Learn; if (!L) return;
  const { rng, gauss, fmt, C, alpha, fit, shape, panes, boxIn, box, label, dot, knobs, wire, axes, curve, path, tweenTo, stepTween, reduced, sigmoid } = L;
  const sum = a => a.reduce((s, v) => s + v, 0);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const initial = (root, S) => { try { Object.assign(S, JSON.parse(root.dataset.init || "{}")); } catch {} };
  // [r, g, b] of any CSS colour, for writing pixels.
  const probe = document.createElement("canvas").getContext("2d", { willReadFrequently: true });
  function rgb(color) {
    probe.clearRect(0, 0, 1, 1); probe.fillStyle = "#000"; probe.fillStyle = color; probe.fillRect(0, 0, 1, 1);
    const d = probe.getImageData(0, 0, 1, 1).data; return [d[0], d[1], d[2]];
  }

  // ---------- 12 · A tiny MLP learning a 2D classification, live ----------
  L.register("mlp", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const S = { h: 6, act: "tanh", set: "circles", seed: 1, tween: null };
    initial(root, S);
    const show = knobs(root, S, 0);
    const N = 200;
    function makeData() {
      const r = rng(17 + ["xor", "circles", "spiral"].indexOf(S.set)), X = [], Y = [];
      for (let i = 0; i < N; i++) {
        if (S.set === "xor") { const a = 2 * r() - 1, b = 2 * r() - 1; X.push([a + 0.08 * Math.sign(a), b + 0.08 * Math.sign(b)].map(v => clamp(v, -1, 1))); Y.push(a * b > 0 ? 1 : 0); }
        else if (S.set === "circles") { const y = i % 2, rad = y ? 0.25 + 0.2 * r() : 0.62 + 0.3 * r(), t = 2 * Math.PI * r(); X.push([rad * Math.cos(t), rad * Math.sin(t)]); Y.push(y); }
        else { const y = i % 2, s = r(), t = 3.6 * s * Math.PI / 1.1 + y * Math.PI, rad = 0.12 + 0.85 * s; X.push([rad * Math.cos(t) + 0.04 * gauss(r), rad * Math.sin(t) + 0.04 * gauss(r)]); Y.push(y); }
      }
      return { X, Y };
    }
    let D = makeData(), net, steps;
    const act = { tanh: [Math.tanh, a => 1 - a * a], relu: [z => Math.max(0, z), a => a > 0 ? 1 : 0], linear: [z => z, () => 1] };
    function init() {
      const r = rng(S.seed * 7919), H = Math.round(S.h);
      net = { H, W1: Array.from({ length: H }, () => [gauss(r) * 1.4, gauss(r) * 1.4]), b1: Array.from({ length: H }, () => 0.3 * gauss(r)), w2: Array.from({ length: H }, () => gauss(r) / Math.sqrt(H)), b2: 0 };
      net.m = new Float64Array(4 * H + 1); net.v = new Float64Array(4 * H + 1); net.t = 0; steps = 0;
    }
    init();
    function forward(x) {
      const [f] = act[S.act], h = net.W1.map((w, j) => f(w[0] * x[0] + w[1] * x[1] + net.b1[j]));
      return { h, p: sigmoid(sum(h.map((v, j) => v * net.w2[j])) + net.b2) };
    }
    function trainStep(lr = 0.03) {
      const H = net.H, [, df] = act[S.act], g = new Float64Array(4 * H + 1); let loss = 0;
      for (let i = 0; i < N; i++) {
        const x = D.X[i], y = D.Y[i], { h, p } = forward(x), e = (p - y) / N;
        loss -= (y ? Math.log(p + 1e-12) : Math.log(1 - p + 1e-12)) / N;
        for (let j = 0; j < H; j++) {
          g[2 * H + j] += e * h[j]; const dz = e * net.w2[j] * df(h[j]);
          g[j * 2] += dz * x[0]; g[j * 2 + 1] += dz * x[1]; g[3 * H + j] += dz;
        }
        g[4 * H] += e;
      }
      net.t++; const th = [];
      for (let q = 0; q < g.length; q++) {
        net.m[q] = 0.9 * net.m[q] + 0.1 * g[q]; net.v[q] = 0.999 * net.v[q] + 0.001 * g[q] * g[q];
        th.push(lr * (net.m[q] / (1 - 0.9 ** net.t)) / (Math.sqrt(net.v[q] / (1 - 0.999 ** net.t)) + 1e-8));
      }
      for (let j = 0; j < H; j++) { net.W1[j][0] -= th[2 * j]; net.W1[j][1] -= th[2 * j + 1]; net.w2[j] -= th[2 * H + j]; net.b1[j] -= th[3 * H + j]; }
      net.b2 -= th[4 * H]; steps++;
      return loss;
    }
    let loss = NaN, playing = true;
    root.querySelectorAll("input[data-k=h]").forEach(el => el.addEventListener("input", init));
    wire(root, S, { reset: () => { S.seed++; init(); playing = true; }, pause: e => { playing = !playing; e.currentTarget.textContent = playing ? "Pause" : "Train"; } },
      key => { if (key === "set") D = makeData(); init(); playing = true; });
    const off = document.createElement("canvas"), G = 64; off.width = off.height = G;
    function draw() {
      const { k, W, H } = shape(cv, 0.62, 1.0), side = Math.min(W, H) - 8, x0 = (W - side) / 2, y0 = 4;
      const B = box(-1.1, 1.1, -1.1, 1.1, x0, y0, side, side);
      const oc = off.getContext("2d"), img = oc.createImageData(G, G), cA = rgb(C.data), cB = rgb(C.accent), cP = rgb(C.panel);
      for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
        const p = forward([-1.1 + 2.2 * (j + 0.5) / G, 1.1 - 2.2 * (i + 0.5) / G]).p, c = p > 0.5 ? cB : cA, s = Math.abs(p - 0.5) * 2 * 0.55, o = 4 * (i * G + j);
        for (let q = 0; q < 3; q++) img.data[o + q] = cP[q] + (c[q] - cP[q]) * s; img.data[o + 3] = 255;
      }
      oc.putImageData(img, 0, 0); k.imageSmoothingEnabled = true; k.drawImage(off, x0, y0, side, side);
      // each hidden unit's line w·x + b = 0
      if (S.act !== "linear") net.W1.forEach((w, j) => {
        const b = net.b1[j]; if (Math.hypot(w[0], w[1]) < 1e-6) return;
        k.save(); k.beginPath(); k.rect(x0, y0, side, side); k.clip(); k.strokeStyle = alpha(C.ink, 0.28); k.lineWidth = 1; k.setLineDash([4, 4]); k.beginPath();
        if (Math.abs(w[1]) > Math.abs(w[0])) { k.moveTo(B.X(-1.1), B.Y((-b + 1.1 * w[0]) / w[1])); k.lineTo(B.X(1.1), B.Y((-b - 1.1 * w[0]) / w[1])); }
        else { k.moveTo(B.X((-b + 1.1 * w[1]) / w[0]), B.Y(-1.1)); k.lineTo(B.X((-b - 1.1 * w[1]) / w[0]), B.Y(1.1)); }
        k.stroke(); k.restore();
      });
      D.X.forEach((x, i) => dot(k, B.X(x[0]), B.Y(x[1]), 3.4, D.Y[i] ? C.accent : C.data, C.panel, 1));
      k.strokeStyle = C.line; k.lineWidth = 1; k.strokeRect(x0, y0, side, side);
      const acc = sum(D.X.map((x, i) => (forward(x).p > 0.5) === !!D.Y[i] ? 1 : 0)) / N;
      out.innerHTML = `${net.H} hidden ${S.act} units · step <b>${steps}</b> · loss <b>${fmt(loss, 3)}</b> · training accuracy <b>${fmt(100 * acc, 0)}%</b>${S.act === "linear" ? " · with no nonlinearity the network is one straight line, however many units" : ""}`;
    }
    return { el: root, tick(dt) { if (playing && dt > 0 && steps < 6000) for (let i = 0; i < 6; i++) loss = trainStep(); stepTween(S, dt, show); draw(); } };
  });

  // ---------- 13 · Forward values and backward gradients on a small graph ----------
  L.register("backprop", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const S = { x: 1.5, w: -0.5, b: 0.2, tween: null }, y = 1, eta = 1;
    const show = knobs(root, S, 2);
    function vals() {
      const u = S.w * S.x, z = u + S.b, a = sigmoid(z), Lv = (a - y) ** 2;
      const dL = 1, da = 2 * (a - y), dz = da * a * (1 - a), du = dz, db = dz, dw = du * S.x, dx = du * S.w;
      return { u, z, a, Lv, dL, da, dz, du, db, dw, dx };
    }
    wire(root, S, { step: () => { const v = vals(); tweenTo(S, { w: clamp(S.w - eta * v.dw, -3, 3), b: clamp(S.b - eta * v.db, -3, 3) }, 600); }, reset: () => tweenTo(S, { x: 1.5, w: -0.5, b: 0.2 }) });
    function draw() {
      const v = vals(), { k, W, H } = fit(cv, cv.clientWidth < 450 ? 1.25 : 0.8), narrow = W < 600, fs = W < 450 ? 11 : 12;
      // node positions on a 0..1 grid
      const nodes = narrow ? { x: [0.17, 0.06], w: [0.17, 0.4], b: [0.17, 0.74], u: [0.52, 0.23], z: [0.52, 0.57], a: [0.83, 0.57], L: [0.83, 0.91] }
        : { x: [0.07, 0.2], w: [0.07, 0.55], b: [0.07, 0.88], u: [0.33, 0.37], z: [0.55, 0.6], a: [0.75, 0.6], L: [0.93, 0.6] };
      const P = n => [nodes[n][0] * W, 18 + nodes[n][1] * (H - 50)];
      const edges = [["x", "u"], ["w", "u"], ["u", "z"], ["b", "z"], ["z", "a"], ["a", "L"]];
      const grad = { x: v.dx, w: v.dw, b: v.db, u: v.du, z: v.dz, a: v.da, L: v.dL }, val = { x: S.x, w: S.w, b: S.b, u: v.u, z: v.z, a: v.a, L: v.Lv };
      const op = { u: "×", z: "+", a: "σ", L: "(·−1)²" };
      edges.forEach(([s, t]) => {
        const [x1, y1] = P(s), [x2, y2] = P(t);
        k.strokeStyle = C.line; k.lineWidth = 2; k.beginPath(); k.moveTo(x1, y1); k.lineTo(x2, y2); k.stroke();
        // gradient flowing backwards: an arrow, thicker for larger |gradient|
        const gw = clamp(Math.abs(grad[s]) * 3, 0.6, 7), mx = (x1 + x2) / 2, my = (y1 + y2) / 2, ang = Math.atan2(y1 - y2, x1 - x2);
        k.strokeStyle = alpha(C.accent, 0.75); k.lineWidth = gw; k.beginPath(); k.moveTo(mx - 14 * Math.cos(ang), my - 14 * Math.sin(ang)); k.lineTo(mx + 10 * Math.cos(ang), my + 10 * Math.sin(ang)); k.stroke();
        k.fillStyle = C.accent; k.beginPath(); k.moveTo(mx + 16 * Math.cos(ang), my + 16 * Math.sin(ang)); k.lineTo(mx + 8 * Math.cos(ang + 0.5) , my + 8 * Math.sin(ang + 0.5)); k.lineTo(mx + 8 * Math.cos(ang - 0.5), my + 8 * Math.sin(ang - 0.5)); k.fill();
      });
      for (const n in nodes) {
        const [x, yy] = P(n), r = narrow ? 18 : 22;
        k.fillStyle = C.panel; k.strokeStyle = n === "L" ? C.accent : C.ink; k.lineWidth = 1.8; k.beginPath(); k.arc(x, yy, r, 0, 2 * Math.PI); k.fill(); k.stroke();
        label(k, op[n] || n, x, yy, { align: "center", base: "middle", color: C.ink, size: op[n] && op[n].length > 2 ? 11 : 15, font: op[n] ? C.mono : C.display });
        label(k, `${n} = ${fmt(val[n], 2)}`, x, yy - r - 8, { align: "center", color: C.ink, size: fs });
        label(k, `∂L/∂${n} = ${fmt(grad[n], 3)}`, x, yy + r + 15, { align: "center", color: C.accent, size: fs });
      }
      out.innerHTML = `∂L/∂w = ∂L/∂a · σ′(z) · x = <b>${fmt(v.da, 3)}</b> × <b>${fmt(v.a * (1 - v.a), 3)}</b> × <b>${fmt(S.x, 2)}</b> = <b>${fmt(v.dw, 3)}</b> · one backward pass gives every gradient at once${Math.abs(v.z) > 4 ? " · σ is saturated, so every gradient upstream of it is tiny" : ""}`;
    }
    return { el: root, tick(dt) { stepTween(S, dt, show); draw(); } };
  });

  // ---------- 14 / 16 · Signals and gradients through 30 layers ----------
  L.register("depth", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const S = { g: 0, act: "tanh", res: 0, tween: null }; // g is log2 of the gain
    initial(root, S);
    const show = knobs(root, S, 2);
    const n = 96, depth = 30, batch = 32, r = rng(3);
    const Z = Array.from({ length: depth }, () => Float64Array.from({ length: n * n }, () => gauss(r)));
    const X0 = Array.from({ length: batch }, () => Float64Array.from({ length: n }, () => gauss(r)));
    const G0 = Array.from({ length: batch }, () => Float64Array.from({ length: n }, () => gauss(r)));
    const std = arr => { let s = 0, c = 0; for (const a of arr) for (const v of a) { s += v * v; c++; } return Math.sqrt(s / c); };
    let res, key = "";
    function run() {
      const gain = 2 ** S.g, sc = gain / Math.sqrt(n), f = S.act === "tanh" ? Math.tanh : S.act === "relu" ? z => Math.max(0, z) : z => z;
      const df = S.act === "tanh" ? (z => { const t = Math.tanh(z); return 1 - t * t; }) : S.act === "relu" ? (z => z > 0 ? 1 : 0) : (() => 1);
      let h = X0.map(a => a.slice()); const pre = [], hs = [h], fw = [std(h)];
      for (let l = 0; l < depth; l++) {
        const W = Z[l], zs = h.map(x => { const z = new Float64Array(n); for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += W[i * n + j] * x[j]; z[i] = s * sc; } return z; });
        pre.push(zs);
        h = zs.map((z, b) => { const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = (S.res ? h[b][i] : 0) + f(z[i]); return o; });
        hs.push(h); fw.push(std(h));
        if (!isFinite(fw[l + 1]) || fw[l + 1] > 1e12) { while (fw.length <= depth) fw.push(Infinity); break; }
      }
      // backward: a random gradient at the top, pushed down through every layer
      let g = G0.map(a => a.slice()); const bw = new Array(depth + 1); bw[depth] = std(g);
      for (let l = depth - 1; l >= 0; l--) {
        if (!pre[l]) { bw[l] = Infinity; continue; }
        const W = Z[l];
        g = g.map((gb, b) => { const dz = new Float64Array(n); for (let i = 0; i < n; i++) dz[i] = gb[i] * df(pre[l][b][i]) * sc; const o = new Float64Array(n); for (let j = 0; j < n; j++) { let s = 0; for (let i = 0; i < n; i++) s += W[i * n + j] * dz[i]; o[j] = (S.res ? gb[j] : 0) + s; } return o; });
        bw[l] = std(g);
      }
      res = { fw, bw };
    }
    wire(root, S, {}, () => {});
    function draw() {
      const kk = `${S.g.toFixed(3)}|${S.act}|${S.res}`; if (kk !== key) { key = kk; run(); }
      root.querySelectorAll("[data-out=g]").forEach(o => { o.textContent = fmt(2 ** S.g, 2); });
      const { k, W, H } = shape(cv, 0.42, 0.75), B = box(0, depth, -8, 8, 48, 14, W - 64, H - 50), lg = v => clamp(Math.log10(v), -8, 8);
      axes(k, B, { xs: [0, 10, 20, 30], ys: [-8, -4, 0, 4, 8], fy: v => v === 0 ? "1" : `1e${v}`, xl: "layer →", yl: "standard deviation (log)", grid: true });
      k.fillStyle = alpha(C.good || C.data, 0.08); k.fillRect(B.left, B.Y(1), B.width, B.Y(-1) - B.Y(1));
      path(k, B, res.fw.map((v, l) => [l, lg(v)]), C.data, 2.5);
      path(k, B, res.bw.map((v, l) => [l, lg(v)]), C.accent, 2.5);
      label(k, "activations ↑", B.X(depth) - 4, B.Y(lg(res.fw[depth])) - 9, { align: "right", color: C.data, size: 12 });
      label(k, "gradients ↓", B.X(0) + 6, B.Y(lg(res.bw[0])) + (lg(res.bw[0]) > -6 ? 16 : -9), { color: C.accent, size: 12 });
      const ratio = res.bw[0] / res.bw[depth];
      const verdict = !isFinite(ratio) || ratio > 1e3 ? "<b>exploding</b>" : ratio < 1e-3 ? "<b>vanishing</b>" : "<b>healthy</b>";
      out.innerHTML = `${S.res ? "residual" : "plain"} ${S.act} network, 30 layers of width ${n}, weights N(0, g²/n) with g = ${fmt(2 ** S.g, 2)} · gradient at layer 0 ÷ gradient at the top = <b>${isFinite(ratio) ? ratio.toExponential(1) : "∞"}</b>: ${verdict}`;
    }
    return { el: root, tick(dt) { stepTween(S, dt, show); draw(); } };
  });

  // ---------- 15 · A 3×3 kernel sliding over an image ----------
  L.register("conv", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const K = {
      vertical: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], horizontal: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
      blur: [[1, 2, 1], [2, 4, 2], [1, 2, 1]].map(r => r.map(v => v / 16)), sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    };
    const art = ["............", "............", "..########..", "..#......#..", "..#......#..", "..#..##..#..", "..#..##..#..", "..#......#..", "..########..", "......#.....", ".....###....", "............"];
    const S = { kern: "vertical", shift: 0, pos: 0, playing: !reduced(), tween: null };
    const img = () => art.map((row, i) => [...row].map((c, j) => { const jj = j - S.shift; return jj >= 0 && jj < 12 && art[i][jj] === "#" ? 1 : 0; }));
    wire(root, S, { shift: () => { S.shift = (S.shift + 1) % 3; }, pause: e => { S.playing = !S.playing; e.currentTarget.textContent = S.playing ? "Pause" : "Play"; } }, () => {});
    let acc = 0;
    function draw() {
      const I = img(), k3 = K[S.kern], O = [];
      for (let i = 0; i < 10; i++) { O.push([]); for (let j = 0; j < 10; j++) { let s = 0; for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) s += k3[a][b] * I[i + a][j + b]; O[i].push(s); } }
      const W0 = cv.clientWidth || 300, gap = W0 < 600 ? 14 : 30, cell = Math.min(28, (W0 - 2 * gap - 8) / (12 + 3 * 1.6 + 10)), kc = cell * 1.6;
      const { k, W } = fit(cv, (24 + 12 * cell + 26) / W0);
      const xi = 4, xk = xi + 12 * cell + gap, xo = xk + 3 * kc + gap, y0 = 24, pi = Math.floor(S.pos) % 100, oi = Math.floor(pi / 10), oj = pi % 10;
      const mx = Math.max(1e-9, ...O.flat().map(Math.abs));
      label(k, "input", xi, 12, { color: C.ink, size: 12, base: "middle" }); label(k, "kernel", xk, 12, { color: C.ink, size: 12, base: "middle" }); label(k, "output", xo, 12, { color: C.ink, size: 12, base: "middle" });
      for (let i = 0; i < 12; i++) for (let j = 0; j < 12; j++) { k.fillStyle = I[i][j] ? C.ink : alpha(C.line, 0.35); k.fillRect(xi + j * cell + 0.5, y0 + i * cell + 0.5, cell - 1, cell - 1); }
      k.strokeStyle = C.accent; k.lineWidth = 2.5; k.strokeRect(xi + oj * cell, y0 + oi * cell, 3 * cell, 3 * cell);
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
        const v = k3[a][b], m = Math.max(...k3.flat().map(Math.abs));
        k.fillStyle = v >= 0 ? alpha(C.accent, 0.15 + 0.6 * v / m) : alpha(C.data, 0.15 + 0.6 * -v / m);
        k.fillRect(xk + b * kc + 1, y0 + a * kc + 1, kc - 2, kc - 2);
        label(k, Number.isInteger(v) ? String(v) : fmt(v, 2), xk + (b + 0.5) * kc, y0 + (a + 0.5) * kc, { align: "center", base: "middle", color: C.ink, size: Math.min(12, kc / 3) });
      }
      for (let i = 0; i < 10; i++) for (let j = 0; j < 10; j++) {
        const done = i * 10 + j <= pi, v = O[i][j] / mx;
        k.fillStyle = !done ? alpha(C.line, 0.2) : v >= 0 ? alpha(C.accent, 0.08 + 0.9 * v) : alpha(C.data, 0.08 + 0.9 * -v);
        k.fillRect(xo + j * cell + 0.5, y0 + i * cell + 0.5, cell - 1, cell - 1);
      }
      k.strokeStyle = C.ink; k.lineWidth = 2; k.strokeRect(xo + oj * cell, y0 + oi * cell, cell, cell);
      const terms = []; for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) if (I[oi + a][oj + b] && k3[a][b]) terms.push(Number.isInteger(k3[a][b]) ? k3[a][b] : fmt(k3[a][b], 2));
      out.innerHTML = `output[${oi}, ${oj}] = Σ kernel × patch = ${terms.length ? terms.join(" + ").replace(/\+ -/g, "− ") : "0"} = <b>${fmt(O[oi][oj], 2)}</b> (orange positive, blue negative)${S.shift ? ` · image shifted ${S.shift} pixel${S.shift > 1 ? "s" : ""} right: the output shifts with it (equivariance)` : ""}`;
    }
    return { el: root, tick(dt) { if (S.playing) { acc += dt; if (acc > 140) { S.pos = (S.pos + 1) % 100; acc = 0; } } draw(); } };
  });

  // ---------- 17 · Gradients through time: a tanh RNN against an LSTM's cell path ----------
  L.register("bptt", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const S = { rho: 0.9, f: 0.97, tween: null };
    const show = knobs(root, S, 2);
    const n = 32, T = 100, r = rng(9);
    // a random orthogonal matrix (Gram–Schmidt), so its spectral radius is exactly what we scale it to
    const Q = []; for (let i = 0; i < n; i++) { let v = Array.from({ length: n }, () => gauss(r)); for (const q of Q) { const d = sum(v.map((x, j) => x * q[j])); v = v.map((x, j) => x - d * q[j]); } const m = Math.hypot(...v); Q.push(v.map(x => x / m)); }
    const U = Array.from({ length: T }, () => Array.from({ length: n }, () => 0.5 * gauss(r)));
    let key = "", curveR = [];
    function run() {
      // forward: h_t = tanh(ρ Q h_{t-1} + u_t); backward: push a unit vector from step T down to every earlier step
      const hs = [new Array(n).fill(0)];
      for (let t = 0; t < T; t++) { const h = hs[t], z = Q.map((q, i) => S.rho * sum(q.map((x, j) => x * h[j])) + U[t][i]); hs.push(z.map(Math.tanh)); }
      let g = new Array(n).fill(0); g[0] = 1; curveR = [[T, 0]];
      for (let t = T; t >= 1; t--) { const d = g.map((x, i) => x * (1 - hs[t][i] ** 2)); g = Array.from({ length: n }, (_, j) => S.rho * sum(Q.map((q, i) => q[j] * d[i]))); curveR.push([t - 1, Math.log10(Math.hypot(...g) + 1e-300)]); }
    }
    function draw() {
      const kk = `${S.rho}`; if (kk !== key) { key = kk; run(); }
      const { k, W, H } = shape(cv, 0.42, 0.75), B = box(0, T, -12, 1, 48, 14, W - 64, H - 50);
      axes(k, B, { xs: [0, 25, 50, 75, 100], ys: [-12, -9, -6, -3, 0], fy: v => v === 0 ? "1" : `1e${v}`, xl: "how many steps back from step 100 →", yl: "size of ∂h₁₀₀/∂h_t", grid: true });
      path(k, B, curveR.map(([t, v]) => [T - t, clamp(v, -12, 1)]), C.data, 2.5);
      curve(k, B, s => s * Math.log10(S.f), 0, T, C.accent, 2.5);
      const back = T - curveR[curveR.length - 1][0], rr = curveR[curveR.length - 1][1];
      label(k, "tanh RNN", B.X(40), B.Y(clamp(curveR[40][1], -12, 1)) + 16, { color: C.data, size: 12 });
      label(k, "LSTM cell path, forget gate f", B.X(T) - 4, B.Y(T * Math.log10(S.f)) + 16, { align: "right", color: C.accent, size: 12 });
      out.innerHTML = `after 100 steps the RNN's gradient is <b>1e${fmt(rr, 1)}</b>; the LSTM's cell path keeps f¹⁰⁰ = <b>${fmt(S.f ** 100, 3)}</b> · the RNN multiplies by ρ times tanh′ ≤ 1 at every step, so it vanishes even at ρ = 1`;
    }
    return { el: root, tick(dt) { stepTween(S, dt, show); draw(); } };
  });

  // ---------- 19 · One query attending over six keys ----------
  L.register("attention", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const keys = [[1, 0.15], [0.7, 0.75], [-0.2, 1], [-0.9, 0.35], [-0.6, -0.8], [0.5, -0.9]].map(([a, b]) => { const m = Math.hypot(a, b); return [a / m, b / m]; });
    const S = { ang: 30, mag: 1.5, tween: null };
    const show = knobs(root, S, 1);
    function draw() {
      const q = [S.mag * Math.cos(S.ang * Math.PI / 180), S.mag * Math.sin(S.ang * Math.PI / 180)];
      const s = keys.map(kv => kv[0] * q[0] + kv[1] * q[1]), m = Math.max(...s), e = s.map(v => Math.exp(v - m)), Z = sum(e), w = e.map(v => v / Z);
      const P = panes(cv, { frac: 0.5, wide: 0.48, narrow: 1.15 }), k = P.k;
      const side = Math.min(P.a.w, P.a.h) - 16, A = box(-1.6, 1.6, -1.6, 1.6, P.a.x + (P.a.w - side) / 2, P.a.y + 8, side, side);
      label(k, "query and keys", A.left, P.a.y + 6, { color: C.ink, size: 12, base: "middle" });
      k.strokeStyle = C.line; k.lineWidth = 1; k.beginPath(); k.arc(A.X(0), A.Y(0), A.X(1) - A.X(0), 0, 2 * Math.PI); k.stroke();
      const arrow = (v, color, lw) => { const x = A.X(v[0]), y = A.Y(v[1]), a = Math.atan2(y - A.Y(0), x - A.X(0)); k.strokeStyle = color; k.fillStyle = color; k.lineWidth = lw; k.beginPath(); k.moveTo(A.X(0), A.Y(0)); k.lineTo(x, y); k.stroke(); k.beginPath(); k.moveTo(x, y); k.lineTo(x - 9 * Math.cos(a - 0.4), y - 9 * Math.sin(a - 0.4)); k.lineTo(x - 9 * Math.cos(a + 0.4), y - 9 * Math.sin(a + 0.4)); k.fill(); };
      keys.forEach((kv, i) => { arrow(kv, alpha(C.data, 0.25 + 0.75 * w[i] / Math.max(...w)), 1.5 + 5 * w[i]); label(k, `k${i + 1}`, A.X(kv[0] * 1.18), A.Y(kv[1] * 1.18), { align: "center", base: "middle", size: 12, color: C.data }); });
      arrow([clamp(q[0], -1.55, 1.55), clamp(q[1], -1.55, 1.55)], C.accent, 3);
      label(k, "q", A.X(clamp(q[0], -1.5, 1.5)) + 10, A.Y(clamp(q[1], -1.5, 1.5)), { color: C.accent, size: 14, base: "middle", font: C.display });
      const B = boxIn(P.b, -0.5, 5.5, 0, 1, 36, 22, 14, 30);
      label(k, "attention weights softmax(q·kᵢ)", B.left, P.b.y + 8, { color: C.ink, size: 12, base: "middle" });
      axes(k, B, { ys: [0, 0.5, 1] });
      const bw = B.width / 6 * 0.6;
      w.forEach((a, i) => { k.fillStyle = alpha(C.accent, 0.35 + 0.65 * a); k.fillRect(B.X(i) - bw / 2, B.Y(a), bw, B.Y(0) - B.Y(a)); label(k, `k${i + 1}`, B.X(i), B.top + B.height + 16, { align: "center", size: 12 }); label(k, fmt(a, 2), B.X(i), B.Y(a) - 6, { align: "center", size: 11, color: C.ink }); });
      const H = -sum(w.map(a => a > 0 ? a * Math.log2(a) : 0));
      out.innerHTML = `scores q·kᵢ range over <b>${fmt(Math.min(...s), 2)}</b> to <b>${fmt(m, 2)}</b> · largest weight <b>${fmt(Math.max(...w), 2)}</b> · entropy <b>${fmt(H, 2)}</b> bits of a possible ${fmt(Math.log2(6), 2)}${S.mag > 6 ? " · nearly one-hot: the softmax is saturated and passes almost no gradient to the other keys" : ""}`;
    }
    return { el: root, tick(dt) { stepTween(S, dt, show); draw(); } };
  });
})();
