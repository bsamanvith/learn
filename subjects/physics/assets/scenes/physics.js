// Interactive figures for the physics problem sets. Each registers with learn.js.
(() => {
  const L = window.Learn; if (!L) return;
  const { fmt, C, alpha, panes, boxIn, label, dot, knobs, wire, axes, curve, reduced } = L;

  // T / T0 = 1 / AGM(1, cos(θ0/2)), exact for any amplitude below 180°.
  const agm = (a, b) => { for (let i = 0; i < 30 && Math.abs(a - b) > 1e-15; i++) [a, b] = [(a + b) / 2, Math.sqrt(a * b)]; return a; };
  const ratio = th => 1 / agm(1, Math.cos(th / 2));

  // ---------- P1 · The pendulum's real period ----------
  // Left: the exact pendulum (integrated) beside one that obeys the small-angle law, released together.
  // Right: T/T0 against amplitude, exact and the θ0²/16 approximation.
  L.register("pendulum", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const S = { amp: 90, tween: null };
    knobs(root, S, 0);
    let th = 0, w = 0, t = 0, lastAmp = null; // exact state in units where g/L = 1, so T0 = 2π
    const release = () => { th = S.amp * Math.PI / 180; w = 0; t = 0; lastAmp = S.amp; };
    wire(root, S, { restart: release });
    const acc = x => -Math.sin(x);
    function step(h) { // RK4 for θ'' = −sin θ
      const k1 = w, l1 = acc(th), k2 = w + h / 2 * l1, l2 = acc(th + h / 2 * k1), k3 = w + h / 2 * l2, l3 = acc(th + h / 2 * k2), k4 = w + h * l3, l4 = acc(th + h * k3);
      th += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); w += h / 6 * (l1 + 2 * l2 + 2 * l3 + l4); t += h;
    }
    function draw() {
      const { k, a, b } = panes(cv, { frac: 0.48, wide: 0.44, narrow: 1.25 });
      const th0 = S.amp * Math.PI / 180, small = th0 * Math.cos(t); // the small-angle pendulum: θ0 cos t
      // the two pendulums share a pivot
      const px = a.x + a.w / 2, py = a.y + a.h * 0.42, len = Math.min(a.w * 0.42, a.h * 0.5);
      k.strokeStyle = alpha(C.line, 0.9); k.lineWidth = 1; k.setLineDash([2, 4]);
      k.beginPath(); k.arc(px, py, len, Math.PI / 2 - th0, Math.PI / 2 + th0); k.stroke(); k.setLineDash([]);
      const bob = (ang, color, r) => {
        const x = px + len * Math.sin(ang), y = py + len * Math.cos(ang);
        k.strokeStyle = color; k.lineWidth = 2; k.beginPath(); k.moveTo(px, py); k.lineTo(x, y); k.stroke();
        dot(k, x, y, r, color, C.panel, 2);
      };
      bob(small, alpha(C.muted, 0.7), 9);
      bob(th, C.accent, 11);
      dot(k, px, py, 3.5, C.ink, C.ink, 1);
      label(k, "exact", a.x + 6, a.y + 16, { color: C.accent, size: 12 });
      label(k, "small-angle law", a.x + 6, a.y + 34, { color: C.muted, size: 12 });
      // the period curve
      const B = boxIn(b, 0, 180, 1, 3, 44, 14, 12, 40);
      axes(k, B, { xs: [0, 45, 90, 135, 180], ys: [1, 1.5, 2, 2.5, 3], fx: x => x + "°", fy: y => y.toFixed(1), xl: "amplitude θ₀ →", yl: "T / T₀", grid: true });
      curve(k, B, x => 1 + (x * Math.PI / 180) ** 2 / 16, 0, 180, alpha(C.data, 0.9), 2, { dash: [6, 5] });
      curve(k, B, x => ratio(x * Math.PI / 180), 0, 179.4, C.accent, 2.5, { n: 400 });
      const r = ratio(th0);
      dot(k, B.X(S.amp), B.Y(Math.min(3, r)), 6, C.panel, C.accent, 3);
      label(k, "1 + θ₀²/16", B.X(150), B.Y(1 + (150 * Math.PI / 180) ** 2 / 16) + 18, { color: C.data, size: 11, align: "center" });
      const series = 1 + th0 * th0 / 16;
      out.innerHTML = `θ₀ = <b>${S.amp}°</b> · exact T/T₀ = <b>${fmt(r, 4)}</b> · approximation 1 + θ₀²/16 = <b>${fmt(series, 4)}</b> (off by ${fmt(100 * Math.abs(series - r) / r, 1)}%) · each swing takes ${fmt(100 * (r - 1), 1)}% longer than Galileo's rule says`;
    }
    release();
    return {
      el: root,
      tick(dt) {
        if (S.amp !== lastAmp) release();
        if (!reduced()) { const h = 0.004, n = Math.round((dt / 1000) * 2.2 / h); for (let i = 0; i < n; i++) step(h); }
        draw();
      },
    };
  });
})();
