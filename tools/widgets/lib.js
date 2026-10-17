// Shared helpers for widgets. Plain browser JS, no build step.
// Include with <script src="../../tools/widgets/lib.js"></script> (adjust path).

const Learn = (() => {
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
      muted: get("--muted"), fg: get("--fg"), bg: get("--bg"),
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

  return { rng, histogram, palette, slider };
})();
