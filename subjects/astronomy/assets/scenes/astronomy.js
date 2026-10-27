// Interactive figures for the astronomy problem sets. Each registers with learn.js.
(() => {
  const L = window.Learn; if (!L) return;
  const { fmt, C, alpha, shape, label, dot, knobs, wire, tweenTo, stepTween, reduced } = L;

  // ---------- P1 · Launch sideways from 1 AU: circle, ellipse, parabola, hyperbola ----------
  // Units: GM = 1 and r = 1 AU at launch, so the circular speed is 1 and a year is 2π time units.
  L.register("orbit", root => {
    const cv = root.querySelector("canvas"), out = root.querySelector("[data-role=readout]");
    const S = { v: 1.1, tween: null };
    const show = knobs(root, S, 3);
    const MARS = 1.524, HOHMANN = Math.sqrt(2 * MARS / (1 + MARS));
    wire(root, S, {
      circle: () => tweenTo(S, { v: 1 }),
      mars: () => tweenTo(S, { v: HOHMANN }),
      escape: () => tweenTo(S, { v: Math.SQRT2 }),
    });
    let p = [1, 0], q = [0, 1], last = null, trail = [];
    const launch = () => { p = [1, 0]; q = [0, S.v]; last = S.v; trail = []; };
    function step(h) { // leapfrog (kick–drift–kick) keeps the orbit closed over many turns
      const r3 = Math.hypot(...p) ** 3;
      q = [q[0] - h / 2 * p[0] / r3, q[1] - h / 2 * p[1] / r3];
      p = [p[0] + h * q[0], p[1] + h * q[1]];
      const s3 = Math.hypot(...p) ** 3;
      q = [q[0] - h / 2 * p[0] / s3, q[1] - h / 2 * p[1] / s3];
    }
    function draw() {
      const { k, W, H } = shape(cv, 0.5, 0.95), sc = Math.min(W, H) / 2 / 2.35, cx = W / 2, cy = H / 2;
      const X = x => cx + x * sc, Y = y => cy - y * sc;
      k.save(); k.beginPath(); k.rect(0, 0, W, H); k.clip();
      // the planets' orbits for scale
      for (const [r, name, ang] of [[1, "Earth's orbit", -0.8], [MARS, "Mars's orbit", 0.75]]) {
        k.strokeStyle = alpha(C.muted, 0.55); k.lineWidth = 1; k.setLineDash([3, 5]);
        k.beginPath(); k.arc(cx, cy, r * sc, 0, 2 * Math.PI); k.stroke(); k.setLineDash([]);
        label(k, name, X(r * Math.cos(ang)) + 6, Y(r * Math.sin(ang)) + (ang < 0 ? 14 : -6), { size: 11 });
      }
      // the conic: r = h² / (1 + e cos(θ − θp)), launched at θ = 0 from r = 1 with speed v
      const v = S.v, h2 = v * v, e = Math.abs(v * v - 1), thp = v >= 1 ? 0 : Math.PI;
      k.strokeStyle = C.accent; k.lineWidth = 2.5; k.beginPath();
      let on = false;
      for (let s = 0; s <= 720; s++) {
        const th = -Math.PI + 2 * Math.PI * s / 720, den = 1 + e * Math.cos(th - thp);
        if (den <= 0.02) { on = false; continue; }
        const r = h2 / den; if (r > 8) { on = false; continue; }
        const x = r * Math.cos(th), y = r * Math.sin(th);
        on ? k.lineTo(X(x), Y(y)) : k.moveTo(X(x), Y(y)); on = true;
      }
      k.stroke();
      // trail and body
      if (trail.length > 1) { k.strokeStyle = alpha(C.data, 0.5); k.lineWidth = 2; k.beginPath(); trail.forEach(([x, y], i) => i ? k.lineTo(X(x), Y(y)) : k.moveTo(X(x), Y(y))); k.stroke(); }
      dot(k, cx, cy, 9, C.accent, alpha(C.accent, 0.3), 6);
      dot(k, X(p[0]), Y(p[1]), 6, C.data, C.panel, 2);
      // the launch arrow at (1, 0)
      k.strokeStyle = C.ink; k.lineWidth = 1.6; k.beginPath(); k.moveTo(X(1), Y(0)); k.lineTo(X(1), Y(0.5 * v)); k.stroke();
      k.beginPath(); k.moveTo(X(1) - 5, Y(0.5 * v) + 7); k.lineTo(X(1), Y(0.5 * v)); k.lineTo(X(1) + 5, Y(0.5 * v) + 7); k.stroke();
      label(k, "launch", X(1) + 8, Y(0) + 14, { size: 11, color: C.ink });
      k.restore();
      const E = v * v / 2 - 1, kind = e < 0.004 ? "a circle" : v < Math.SQRT2 - 0.002 ? "an ellipse" : v <= Math.SQRT2 + 0.002 ? "a parabola: just escapes" : "a hyperbola: escapes";
      let extra = "";
      if (v < Math.SQRT2 - 0.002) {
        const a = 1 / (2 - v * v), other = 2 * a - 1;
        extra = ` · a = <b>${fmt(a, 3)}</b> AU · ${v >= 1 ? "farthest" : "closest"} point <b>${fmt(other, 3)}</b> AU · period <b>${fmt(a ** 1.5, 3)}</b> yr`;
      } else extra = ` · speed left far away <b>${fmt(Math.sqrt(Math.max(0, v * v - 2)), 3)}</b> × 29.8 km/s`;
      out.innerHTML = `v = <b>${fmt(v, 3)}</b> × circular speed · energy per mass <b>${fmt(E, 3)}</b> (GM/r units) · orbit is ${kind}, e = <b>${fmt(e, 3)}</b>${extra}`;
    }
    launch();
    return {
      el: root,
      tick(dt) {
        stepTween(S, dt, show);
        if (Math.abs(S.v - last) > 1e-9) launch();
        if (!reduced()) {
          // advance 2.4 time units per second, in steps small enough for a close pass by the Sun
          let left = Math.min(dt, 50) / 1000 * 2.4;
          while (left > 1e-9) { const h = Math.min(left, 0.01, 0.02 * Math.hypot(...p) ** 1.5); step(h); left -= h; }
          trail.push([...p]); if (trail.length > 260) trail.shift();
          if (Math.hypot(...p) > 9) launch();
        }
        draw();
      },
    };
  });
})();
