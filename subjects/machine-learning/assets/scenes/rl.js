// Interactive figures for the reinforcement-learning lessons (31–39). Each registers with learn.js.
(() => {
  const L = window.Learn; if (!L) return;
  const { rng, gauss, fmt, C, alpha, fit, shape, panes, boxIn, box, label, dot, knobs, wire, axes, curve, path, stepTween, reduced } = L;
  const sum = a => a.reduce((s, v) => s + v, 0);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const argmax = a => a.reduce((b, v, i) => v > a[b] ? i : b, 0);
  const arrowHead = (k, x, y, ang, s, color) => { k.fillStyle = color; k.beginPath(); k.moveTo(x, y); k.lineTo(x - s * Math.cos(ang - 0.45), y - s * Math.sin(ang - 0.45)); k.lineTo(x - s * Math.cos(ang + 0.45), y - s * Math.sin(ang + 0.45)); k.fill(); };

  // ---------- 31 · 400 walkers on a three-state chain, settling into π ----------
  L.register("markov", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const S = { stay: 0.5, tween: null };
    const show = knobs(root, S, 2);
    const names = ["A", "B", "C"], M = 400;
    const P = () => { const m = 1 - S.stay; return [[S.stay, 0.75 * m, 0.25 * m], [0.2 * m, S.stay, 0.8 * m], [0.6 * m, 0.4 * m, S.stay]]; };
    function stationary(Pm) { let p = [1, 0, 0]; for (let i = 0; i < 2000; i++) p = [0, 1, 2].map(j => sum(p.map((v, a) => v * Pm[a][j]))); return p; }
    function lambda2(Pm) { // the other two eigenvalues from the trace and determinant
      const tr = Pm[0][0] + Pm[1][1] + Pm[2][2] - 1, det = Pm[0][0] * (Pm[1][1] * Pm[2][2] - Pm[1][2] * Pm[2][1]) - Pm[0][1] * (Pm[1][0] * Pm[2][2] - Pm[1][2] * Pm[2][0]) + Pm[0][2] * (Pm[1][0] * Pm[2][1] - Pm[1][1] * Pm[2][0]);
      const disc = tr * tr - 4 * det; return disc < 0 ? Math.sqrt(det) : Math.max(Math.abs((tr + Math.sqrt(disc)) / 2), Math.abs((tr - Math.sqrt(disc)) / 2));
    }
    const r = rng(31); let walk, steps, acc = 0, hist;
    function reset() { walk = Array.from({ length: M }, () => ({ s: 0, from: 0, jit: [gauss(r), gauss(r)] })); steps = 0; hist = []; }
    reset();
    wire(root, S, { reset });
    function step() { const Pm = P(); walk.forEach(w => { let u = r(), j = 0; while (j < 2 && (u -= Pm[w.s][j]) > 0) j++; w.from = w.s; w.s = j; }); steps++; }
    function draw(f) {
      const Pm = P(), pi = stationary(Pm), occ = [0, 1, 2].map(i => walk.filter(w => w.s === i).length / M), tv = 0.5 * sum(occ.map((v, i) => Math.abs(v - pi[i])));
      const Pn = panes(cv, { frac: 0.58, wide: 0.46, narrow: 1.55 }), k = Pn.k, a = Pn.a;
      const cx = a.x + a.w / 2, cy = a.y + a.h / 2 + 8, R = Math.min(a.w, a.h) * 0.34;
      const pos = [0, 1, 2].map(i => [cx + R * Math.cos(-Math.PI / 2 + 2 * Math.PI * i / 3), cy + R * Math.sin(-Math.PI / 2 + 2 * Math.PI * i / 3)]);
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) if (i !== j && Pm[i][j] > 0.001) { // arrows bent to one side so i→j and j→i don't overlap
        const [x1, y1] = pos[i], [x2, y2] = pos[j], mx = (x1 + x2) / 2, my = (y1 + y2) / 2, nx = -(y2 - y1), ny = x2 - x1, nl = Math.hypot(nx, ny), qx = mx + 0.16 * nx, qy = my + 0.16 * ny;
        k.strokeStyle = alpha(C.ink, 0.25 + 0.6 * Pm[i][j]); k.lineWidth = 1 + 4 * Pm[i][j]; k.beginPath(); k.moveTo(x1, y1); k.quadraticCurveTo(qx, qy, x2, y2); k.stroke();
        const tx = 0.25 * x1 + 0.5 * qx + 0.25 * x2, ty = 0.25 * y1 + 0.5 * qy + 0.25 * y2;
        arrowHead(k, tx + 4 * (x2 - x1) / nl, ty + 4 * (y2 - y1) / nl, Math.atan2(y2 - y1, x2 - x1), 9, alpha(C.ink, 0.6));
        if (a.w >= 450) label(k, fmt(Pm[i][j], 2), tx + 0.1 * nx, ty + 0.1 * ny, { align: "center", base: "middle", size: 11 });
      }
      walk.forEach(w => { const e = reduced() ? 1 : Math.min(1, f * 1.6), [x1, y1] = pos[w.from], [x2, y2] = pos[w.s]; dot(k, x1 + (x2 - x1) * e + 9 * w.jit[0], y1 + (y2 - y1) * e + 9 * w.jit[1], 1.8, alpha(C.accent, 0.7), null); });
      pos.forEach(([x, y], i) => { k.strokeStyle = C.ink; k.lineWidth = 1.5; k.beginPath(); k.arc(x, y, 30, 0, 2 * Math.PI); k.stroke(); label(k, names[i], x, y - 38, { align: "center", color: C.ink, size: 14, font: C.display }); });
      const B = boxIn(Pn.b, -0.5, 2.5, 0, 1, 36, 22, 14, 30); axes(k, B, { ys: [0, 0.5, 1] });
      label(k, `share of walkers in each state, step ${steps}`, B.left, Pn.b.y + 8, { color: C.ink, size: 12, base: "middle" });
      const bw = B.width / 3 * 0.5;
      occ.forEach((v, i) => { k.fillStyle = alpha(C.accent, 0.8); k.fillRect(B.X(i) - bw / 2, B.Y(v), bw, B.Y(0) - B.Y(v)); k.strokeStyle = C.ink; k.lineWidth = 2; k.setLineDash([4, 3]); k.beginPath(); k.moveTo(B.X(i) - bw * 0.7, B.Y(pi[i])); k.lineTo(B.X(i) + bw * 0.7, B.Y(pi[i])); k.stroke(); k.setLineDash([]); label(k, names[i], B.X(i), B.top + B.height + 16, { align: "center", size: 12 }); });
      label(k, "dashed: π", B.left + B.width - 4, B.top + 4, { align: "right", base: "top", size: 11 });
      const l2 = lambda2(Pm);
      out.innerHTML = `π = (${pi.map(v => fmt(v, 3)).join(", ")}) · distance to π <b>${fmt(tv, 3)}</b> after ${steps} steps · second eigenvalue |λ₂| = <b>${fmt(l2, 3)}</b>: the gap to π shrinks by that factor each step`;
    }
    return { el: root, tick(dt) { if (dt > 0 && steps < 60) { acc += dt; if (acc > 650) { acc = 0; step(); } } draw(acc / 650); } };
  });

  // ---------- 32 · ε-greedy, UCB and Thompson sampling on five slot machines ----------
  L.register("bandit", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const mu = [0.2, 0.45, 0.3, 0.7, 0.55], K = 5, best = Math.max(...mu), Tmax = 5000;
    const S = { algo: "ucb", tween: null };
    const runs = {}; let cur, seed = 1;
    function betaSample(r, a, b) { const g = s => { // Marsaglia–Tsang for shape ≥ 1
      const d = s - 1 / 3, c = 1 / Math.sqrt(9 * d); for (;;) { let x, v; do { x = gauss(r); v = 1 + c * x; } while (v <= 0); v = v * v * v; const u = r(); if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v; } };
      const x = g(a), y = g(b); return x / (x + y); }
    function start() { cur = { algo: S.algo, n: new Array(K).fill(0), s: new Array(K).fill(0), t: 0, regret: [0], r: rng(seed * 101), env: rng(seed * 7) }; runs[S.algo] = cur; }
    start();
    wire(root, S, { restart: () => { seed++; for (const k in runs) delete runs[k]; start(); } }, () => { if (!runs[S.algo] || runs[S.algo].algo !== S.algo) start(); else cur = runs[S.algo]; });
    function pull() {
      const c = cur, r = c.r; let a;
      if (c.algo === "greedy") a = r() < 0.1 ? Math.floor(r() * K) : argmax(c.n.map((n, i) => n ? c.s[i] / n : 1));
      else if (c.algo === "ucb") a = c.t < K ? c.t : argmax(c.n.map((n, i) => c.s[i] / n + Math.sqrt(2 * Math.log(c.t) / n)));
      else a = argmax(c.n.map((n, i) => betaSample(r, 1 + c.s[i], 1 + n - c.s[i])));
      const rew = c.env() < mu[a] ? 1 : 0; c.n[a]++; c.s[a] += rew; c.t++; c.regret.push(c.regret[c.regret.length - 1] + best - mu[a]);
    }
    const names = { greedy: "ε-greedy (ε = 0.1)", ucb: "UCB", thompson: "Thompson" }, cols = () => ({ greedy: C.muted, ucb: C.data, thompson: C.accent });
    function draw() {
      const c = cur, P = panes(cv, { frac: 0.5, wide: 0.46, narrow: 1.15 }), k = P.k;
      const A = boxIn(P.a, -0.5, 4.5, 0, 1, 36, 22, 14, 34); axes(k, A, { ys: [0, 0.5, 1] });
      label(k, `arm estimates after ${c.t} pulls`, A.left, P.a.y + 8, { color: C.ink, size: 12, base: "middle" });
      for (let i = 0; i < K; i++) {
        const n = c.n[i], m = n ? c.s[i] / n : 0, w = n ? (c.algo === "thompson" ? 2 * Math.sqrt(m * (1 - m) / (n + 3)) : Math.sqrt(2 * Math.log(Math.max(c.t, 2)) / n)) : 1;
        k.fillStyle = alpha(cols()[c.algo], 0.18); k.fillRect(A.X(i) - 14, A.Y(clamp(m + w, 0, 1)), 28, A.Y(clamp(m - w, 0, 1)) - A.Y(clamp(m + w, 0, 1)));
        k.strokeStyle = C.ink; k.lineWidth = 2; k.setLineDash([4, 3]); k.beginPath(); k.moveTo(A.X(i) - 18, A.Y(mu[i])); k.lineTo(A.X(i) + 18, A.Y(mu[i])); k.stroke(); k.setLineDash([]);
        if (n) dot(k, A.X(i), A.Y(m), 5, cols()[c.algo], C.panel, 1.5);
        label(k, `${n}`, A.X(i), A.top + A.height + 15, { align: "center", size: 11, color: C.ink });
        label(k, `arm ${i + 1}`, A.X(i), A.top + A.height + 29, { align: "center", size: 10 });
      }
      const B = boxIn(P.b, 0, Tmax, 0, 250, 40, 22, 14, 34); axes(k, B, { xs: [0, 2500, 5000], ys: [0, 100, 200], xl: "pulls →", grid: true });
      label(k, "regret: reward lost to exploring", B.left, P.b.y + 8, { color: C.ink, size: 12, base: "middle" });
      for (const a in runs) { const rr = runs[a].regret, st = Math.max(1, Math.floor(rr.length / 300)); path(k, B, rr.filter((_, i) => i % st === 0 || i === rr.length - 1).map((v, i, arr) => [i === arr.length - 1 ? rr.length - 1 : i * st, Math.min(v, 250)]), cols()[a], a === c.algo ? 2.6 : 1.4); label(k, names[a], B.X(rr.length - 1) - 4, B.Y(Math.min(rr[rr.length - 1], 250)) - 8, { align: "right", size: 11, color: cols()[a] }); }
      out.innerHTML = `${names[c.algo]} · pulled the best arm <b>${fmt(100 * c.n[3] / Math.max(c.t, 1), 0)}%</b> of the time · regret <b>${fmt(c.regret[c.t], 1)}</b> · the band is ${c.algo === "thompson" ? "about two posterior standard deviations" : "the UCB exploration bonus"}; dashed lines are the true win rates`;
    }
    return { el: root, tick(dt) { if (dt > 0 && cur.t < Tmax) for (let i = 0; i < 25 && cur.t < Tmax; i++) pull(); draw(); } };
  });

  // ---------- 34 · Value iteration on a small gridworld ----------
  L.register("valueiter", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const grid = ["....+", ".#.#-", ".#...", "....#", "S...."], R = grid.length, Cn = grid[0].length;
    const S = { gamma: 0.9, slip: 0.2, tween: null };
    const show = knobs(root, S, 2);
    const moves = [[-1, 0], [0, 1], [1, 0], [0, -1]];
    let V, sweeps, delta, auto = false, acc = 0;
    const cell = (i, j) => grid[i][j], term = (i, j) => cell(i, j) === "+" || cell(i, j) === "-";
    function reset() { V = Array.from({ length: R }, () => new Array(Cn).fill(0)); sweeps = 0; delta = NaN; }
    reset();
    function next(i, j, a) { const ni = i + moves[a][0], nj = j + moves[a][1]; return ni < 0 || nj < 0 || ni >= R || nj >= Cn || cell(ni, nj) === "#" ? [i, j] : [ni, nj]; }
    function q(i, j, a, V0) { // intended move with probability 1 − slip, each perpendicular move with slip / 2; −0.04 per step
      let s = 0; for (const [b, p] of [[a, 1 - S.slip], [(a + 1) % 4, S.slip / 2], [(a + 3) % 4, S.slip / 2]]) { const [ni, nj] = next(i, j, b); s += p * (cell(ni, nj) === "+" ? 1 : cell(ni, nj) === "-" ? -1 : -0.04 + S.gamma * V0[ni][nj]); }
      return s;
    }
    function sweep() { const V0 = V.map(r => r.slice()); delta = 0; for (let i = 0; i < R; i++) for (let j = 0; j < Cn; j++) { if (cell(i, j) === "#" || term(i, j)) continue; V[i][j] = Math.max(...[0, 1, 2, 3].map(a => q(i, j, a, V0))); delta = Math.max(delta, Math.abs(V[i][j] - V0[i][j])); } sweeps++; }
    root.querySelectorAll("input[data-k]").forEach(el => el.addEventListener("input", () => { reset(); auto = false; }));
    wire(root, S, { sweep: () => { auto = false; sweep(); }, run: () => { auto = true; }, reset: () => { reset(); auto = false; } });
    function draw() {
      const { k, W, H } = shape(cv, 0.56, 0.9), cs = Math.min((W - 8) / Cn, (H - 8) / R), x0 = (W - cs * Cn) / 2, y0 = 4;
      const pos = alpha(C.good || C.data, 1), neg = C.bad || C.accent;
      for (let i = 0; i < R; i++) for (let j = 0; j < Cn; j++) {
        const x = x0 + j * cs, y = y0 + i * cs, c = cell(i, j);
        if (c === "#") { k.fillStyle = alpha(C.muted, 0.55); k.fillRect(x + 1, y + 1, cs - 2, cs - 2); continue; }
        const v = c === "+" ? 1 : c === "-" ? -1 : V[i][j];
        k.fillStyle = v >= 0 ? alpha(pos, 0.08 + 0.55 * Math.min(1, v)) : alpha(neg, 0.08 + 0.55 * Math.min(1, -v)); k.fillRect(x + 1, y + 1, cs - 2, cs - 2);
        k.strokeStyle = C.line; k.lineWidth = 1; k.strokeRect(x + 1, y + 1, cs - 2, cs - 2);
        if (term(i, j)) { label(k, c === "+" ? "+1" : "−1", x + cs / 2, y + cs / 2, { align: "center", base: "middle", size: 18, color: C.ink, font: C.display }); continue; }
        label(k, fmt(v, 2), x + 6, y + 14, { size: Math.max(10, cs / 7), color: C.ink, base: "middle" });
        if (sweeps) { const a = argmax([0, 1, 2, 3].map(b => q(i, j, b, V))), cx = x + cs / 2, cy = y + cs / 2 + 6, L2 = cs * 0.22, ang = Math.atan2(moves[a][0], moves[a][1]);
          k.strokeStyle = C.ink; k.lineWidth = 2; k.beginPath(); k.moveTo(cx - L2 * Math.cos(ang), cy - L2 * Math.sin(ang)); k.lineTo(cx + L2 * Math.cos(ang), cy + L2 * Math.sin(ang)); k.stroke(); arrowHead(k, cx + L2 * Math.cos(ang) + 4 * Math.cos(ang), cy + L2 * Math.sin(ang) + 4 * Math.sin(ang), ang, 9, C.ink); }
        if (c === "S") label(k, "start", x + cs - 6, y + cs - 8, { align: "right", size: 10 });
      }
      out.innerHTML = `sweeps <b>${sweeps}</b> · largest change in the last sweep <b>${isNaN(delta) ? "—" : delta.toExponential(1)}</b> · the error after k sweeps is at most γᵏ = <b>${fmt(S.gamma ** sweeps, 3)}</b> times the first · each step costs 0.04, and a move slips sideways with probability ${fmt(S.slip, 2)}`;
    }
    return { el: root, tick(dt) { if (auto && dt > 0) { acc += dt; if (acc > 220) { acc = 0; sweep(); if (delta < 1e-4) auto = false; } } stepTween(S, dt, show); draw(); } };
  });

  // ---------- 36 · SARSA against Q-learning on the cliff ----------
  L.register("cliff", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const R = 4, Cn = 12, moves = [[-1, 0], [0, 1], [1, 0], [0, -1]], alphaLR = 0.1;
    const S = { eps: 0.1, tween: null };
    const show = knobs(root, S, 2);
    let agents, seed = 1;
    const mk = name => ({ name, Q: Array.from({ length: R * Cn }, () => [0, 0, 0, 0]), returns: [], last: [] });
    function reset() { agents = [mk("sarsa"), mk("q")]; agents.r = rng(seed++ * 13); }
    reset();
    root.querySelectorAll("input[data-k]").forEach(el => el.addEventListener("input", reset));
    wire(root, S, { reset });
    const stepEnv = (s, a) => { const i = Math.floor(s / Cn), j = s % Cn, ni = clamp(i + moves[a][0], 0, R - 1), nj = clamp(j + moves[a][1], 0, Cn - 1); if (ni === R - 1 && nj > 0 && nj < Cn - 1) return [(R - 1) * Cn, -100, false]; return [ni * Cn + nj, -1, ni === R - 1 && nj === Cn - 1]; };
    const pick = (Q, s, r) => r() < S.eps ? Math.floor(r() * 4) : argmax(Q[s].map(v => v + 1e-9 * r()));
    function episode(ag) {
      const r = agents.r, start = (R - 1) * Cn; let s = start, a = pick(ag.Q, s, r), G = 0;
      for (let t = 0; t < 500; t++) {
        const [s2, rew, done] = stepEnv(s, a); G += rew; const a2 = pick(ag.Q, s2, r);
        const target = done ? rew : rew + (ag.name === "q" ? Math.max(...ag.Q[s2]) : ag.Q[s2][a2]);
        ag.Q[s][a] += alphaLR * (target - ag.Q[s][a]); s = s2; a = a2; if (done) break;
      }
      ag.returns.push(G);
    }
    function greedyPath(ag) { let s = (R - 1) * Cn; const p = [s]; for (let t = 0; t < 40; t++) { const [s2, , done] = stepEnv(s, argmax(ag.Q[s])); p.push(s2); s = s2; if (done || s2 === (R - 1) * Cn) break; } return p; }
    function draw() {
      const n = agents[0].returns.length, P = panes(cv, { frac: 0.6, wide: 0.42, narrow: 0.95 }), k = P.k;
      const cs = Math.min((P.a.w - 4) / Cn, (P.a.h - 30) / R), x0 = P.a.x + (P.a.w - cs * Cn) / 2, y0 = P.a.y + 22;
      for (let i = 0; i < R; i++) for (let j = 0; j < Cn; j++) {
        const cliff = i === R - 1 && j > 0 && j < Cn - 1;
        k.fillStyle = cliff ? alpha(C.bad || C.accent, 0.3) : alpha(C.line, 0.25); k.fillRect(x0 + j * cs + 1, y0 + i * cs + 1, cs - 2, cs - 2);
      }
      label(k, "S", x0 + cs / 2, y0 + (R - 0.5) * cs, { align: "center", base: "middle", color: C.ink, size: 13 });
      label(k, "G", x0 + (Cn - 0.5) * cs, y0 + (R - 0.5) * cs, { align: "center", base: "middle", color: C.ink, size: 13 });
      label(k, "the cliff: −100 and back to S", x0 + Cn * cs / 2, y0 + (R - 0.5) * cs, { align: "center", base: "middle", size: 11, color: C.ink });
      [[agents[0], C.data, -0.12], [agents[1], C.accent, 0.12]].forEach(([ag, col, off]) => {
        const p = greedyPath(ag).map(s => [x0 + (s % Cn + 0.5 + off) * cs, y0 + (Math.floor(s / Cn) + 0.5 + off) * cs]);
        k.strokeStyle = col; k.lineWidth = 3; k.beginPath(); p.forEach(([x, y], i) => i ? k.lineTo(x, y) : k.moveTo(x, y)); k.stroke();
      });
      label(k, "greedy paths:", x0, P.a.y + 8, { size: 12, color: C.ink, base: "middle" });
      label(k, "SARSA", x0 + 100, P.a.y + 8, { size: 12, color: C.data, base: "middle" }); label(k, "Q-learning", x0 + 150, P.a.y + 8, { size: 12, color: C.accent, base: "middle" });
      const B = boxIn(P.b, 0, 500, -100, 0, 44, 22, 14, 34); axes(k, B, { xs: [0, 250, 500], ys: [-100, -50, 0], xl: "episode →", grid: true });
      label(k, "return per episode, averaged over 20", B.left, P.b.y + 8, { size: 12, color: C.ink, base: "middle" });
      const smooth = a => a.map((_, i) => { const w = a.slice(Math.max(0, i - 19), i + 1); return [i, Math.max(-100, sum(w) / w.length)]; });
      path(k, B, smooth(agents[0].returns), C.data, 2); path(k, B, smooth(agents[1].returns), C.accent, 2);
      const avg = ag => { const w = ag.returns.slice(-50); return w.length ? sum(w) / w.length : NaN; };
      out.innerHTML = `episode ${n} · average return over the last 50: SARSA <b>${fmt(avg(agents[0]), 1)}</b>, Q-learning <b>${fmt(avg(agents[1]), 1)}</b> · Q-learning learns the shortest path along the edge, but while exploring with ε = ${fmt(S.eps, 2)} it keeps falling off`;
    }
    return { el: root, tick(dt) { if (dt > 0 && agents[0].returns.length < 500) for (let i = 0; i < 2; i++) { episode(agents[0]); episode(agents[1]); } stepTween(S, dt, show); draw(); } };
  });

  // ---------- 39 · PPO's clipped objective as a function of the probability ratio ----------
  L.register("ppoclip", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const S = { eps: 0.2, r: 1.3, tween: null };
    const show = knobs(root, S, 2);
    const clip = (r, e) => clamp(r, 1 - e, 1 + e);
    function draw() {
      const P = panes(cv, { frac: 0.5, wide: 0.42, narrow: 1.05 }), k = P.k, e = S.eps;
      [[P.a, 1, "A > 0: better than average"], [P.b, -1, "A < 0: worse than average"]].forEach(([p, A, title]) => {
        const B = boxIn(p, 0, 2, A > 0 ? -0.2 : -2.1, A > 0 ? 2.1 : 0.2, 40, 24, 14, 36);
        axes(k, B, { xs: [0, 0.5, 1, 1.5, 2], ys: A > 0 ? [0, 1, 2] : [-2, -1, 0], xl: "ratio r = π_new / π_old →" });
        label(k, title, B.left, p.y + 8, { color: C.ink, size: 12, base: "middle" });
        k.fillStyle = alpha(C.accent, 0.08); k.fillRect(B.X(1 - e), B.top, B.X(1 + e) - B.X(1 - e), B.height);
        curve(k, B, r => r * A, 0, 2, alpha(C.muted, 0.7), 1.5, { dash: [5, 4] });
        curve(k, B, r => Math.min(r * A, clip(r, e) * A), 0, 2, C.accent, 3);
        const v = Math.min(S.r * A, clip(S.r, e) * A), flat = (A > 0 && S.r > 1 + e) || (A < 0 && S.r < 1 - e);
        dot(k, B.X(S.r), B.Y(v), 6, C.panel, flat ? C.ink : C.accent, 2.5);
        label(k, flat ? "flat: zero gradient" : "slope A", B.X(S.r) + (S.r > 1.5 ? -10 : 10), B.Y(v) + (A > 0 ? -12 : 18), { align: S.r > 1.5 ? "right" : "left", size: 11, color: C.ink });
      });
      out.innerHTML = `ε = ${fmt(e, 2)}, r = ${fmt(S.r, 2)} · objective min(r·A, clip(r, ${fmt(1 - e, 2)}, ${fmt(1 + e, 2)})·A) · dashed: the unclipped surrogate r·A · the clip only ever removes incentive: once the policy has moved more than ε in the helpful direction, the gradient is zero, but moving the wrong way is never protected`;
    }
    return { el: root, tick(dt) { stepTween(S, dt, show); draw(); } };
  });
})();
