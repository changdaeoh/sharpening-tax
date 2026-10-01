/* Small, dependency-free numerical helpers. Equations (1), (2), and (4). */
(function (root) {
  "use strict";
  const metrics = (curve, K) => {
    if (!Number.isInteger(K) || K < 2 || K > curve.length) throw new RangeError("Invalid budget");
    const A = curve.slice(0, K - 1).reduce((sum, value) => sum + curve[K - 1] - value, 0);
    const headroom = 1 - curve[0];
    return { A, S: headroom > 0 ? A / ((K - 1) * headroom) : null };
  };
  const temperature = (p, pivot = 0.5, tau = 1.5) => {
    const h = (pivot - p) / (p <= pivot ? pivot : 1 - pivot);
    return tau ** h;
  };
  const tempered = (probabilities, T) => {
    const weights = probabilities.map(p => p ** (1 / T));
    const sum = weights.reduce((a, b) => a + b, 0);
    return weights.map(w => w / sum);
  };
  const mixed = (p, n) => 1 - p ** n - (1 - p) ** n;
  const positive = (p, n) => 1 - (1 - p) ** n;
  const api = { metrics, temperature, tempered, mixed, positive };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ResearchMath = api;
})(typeof window !== "undefined" ? window : globalThis);
