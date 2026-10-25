// Interactive figures for the mathematics problem sets. Each registers with learn.js.
(() => {
  const L = window.Learn; if (!L) return;
  const { fmt, C, alpha, shape, box, axes, curve, dot, label, knobs, wire, stepTween } = L;

  // ---------- P2 · The Gibbs phenomenon: Fourier partial sums of a square wave ----------
  L.register("gibbs", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const S = { n: 5, zoom: 0, tween: null };
    const show = knobs(root, S, 0);
    wire(root, S);
    const LIMIT = 1.1789797; // (2/π) Si(π)
    const sum = (x, N) => { let s = 0; for (let k = 1; k <= N; k++) s += Math.sin((2 * k - 1) * x) / (2 * k - 1); return 4 / Math.PI * s; };
    function draw() {
      const N = Math.round(S.n), zoom = S.zoom === 1;
      const x0 = zoom ? -0.6 : -Math.PI, x1 = zoom ? 0.6 : Math.PI, y0 = zoom ? -1.35 : -1.45, y1 = zoom ? 1.35 : 1.45;
      const { k, W, H } = shape(cv, 0.42, 0.78), B = box(x0, x1, y0, y1, 44, 14, W - 60, H - 50);
      const xs = zoom ? [-0.5, -0.25, 0, 0.25, 0.5] : [-3, -2, -1, 0, 1, 2, 3];
      axes(k, B, { xs, ys: [-1, 0, 1], fx: x => zoom ? x.toFixed(2) : String(x), fy: y => y.toFixed(0), xl: "x →", grid: true });
      // the overshoot level and the square wave itself
      k.strokeStyle = alpha(C.accent, 0.7); k.lineWidth = 1; k.setLineDash([4, 4]);
      for (const y of [LIMIT, -LIMIT]) { k.beginPath(); k.moveTo(B.left, B.Y(y)); k.lineTo(B.left + B.width, B.Y(y)); k.stroke(); }
      k.setLineDash([]);
      label(k, "1.179: the limit of the overshoot", B.left + B.width - 4, B.Y(LIMIT) - 6, { align: "right", size: 11, color: C.accent });
      k.strokeStyle = alpha(C.muted, 0.8); k.lineWidth = 2;
      k.beginPath(); k.moveTo(B.X(x0), B.Y(-1)); k.lineTo(B.X(0), B.Y(-1)); k.lineTo(B.X(0), B.Y(1)); k.lineTo(B.X(x1), B.Y(1)); k.stroke();
      curve(k, B, x => sum(x, N), x0, x1, C.data, 2.5, { n: zoom ? 900 : 1400 });
      const xm = Math.PI / (2 * N), ym = sum(xm, N);
      dot(k, B.X(xm), B.Y(ym), 6, C.panel, C.accent, 3);
      out.innerHTML = `N = <b>${N}</b> terms · first peak at x = π/2N = <b>${fmt(xm, 4)}</b>, height <b>${fmt(ym, 4)}</b> · overshoot <b>${fmt(100 * (ym - 1) / 2, 2)}%</b> of the jump · limit as N → ∞: 1.1790, an overshoot of 8.95%`;
    }
    return { el: root, tick(dt) { stepTween(S, dt, show); draw(); } };
  });
})();
