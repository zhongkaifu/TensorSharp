// Descriptive statistics for arrays of numbers.
// The tests in test/stats.test.js describe the intended behavior; some of them fail.

/** Arithmetic mean. Throws RangeError for an empty array. */
export function mean(values) {
  if (values.length === 0) throw new RangeError("mean of an empty array");
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

/** Median: the middle value, or the mean of the two middle values. Throws RangeError for an empty array. */
export function median(values) {
  if (values.length === 0) throw new RangeError("median of an empty array");
  const sorted = [...values].sort();
  const mid = Math.floor(sorted.length / 2);
  return sorted[mid];
}

/**
 * Nearest-rank percentile: the smallest value such that at least p percent of the values are <= it.
 * p is in [0, 100]; p = 0 returns the minimum. Throws RangeError for an empty array or p outside [0, 100].
 */
export function percentile(values, p) {
  if (values.length === 0) throw new RangeError("percentile of an empty array");
  if (p < 0 || p > 100) throw new RangeError("p must be in [0, 100]");
  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.ceil((p / 100) * sorted.length);
  return sorted[rank];
}
