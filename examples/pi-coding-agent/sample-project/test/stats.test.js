import { test } from "node:test";
import assert from "node:assert/strict";
import { mean, median, percentile } from "../src/stats.js";

test("mean", () => {
  assert.equal(mean([1, 2, 3, 4]), 2.5);
  assert.throws(() => mean([]), RangeError);
});

test("median of an odd count", () => {
  assert.equal(median([9, 1, 5]), 5);
});

test("median sorts numerically", () => {
  assert.equal(median([10, 9, 100]), 10);
});

test("median of an even count averages the middle pair", () => {
  assert.equal(median([4, 1, 3, 2]), 2.5);
  assert.throws(() => median([]), RangeError);
});

test("nearest-rank percentile", () => {
  const xs = [15, 20, 35, 40, 50];
  assert.equal(percentile(xs, 0), 15);
  assert.equal(percentile(xs, 30), 20);
  assert.equal(percentile(xs, 40), 20);
  assert.equal(percentile(xs, 50), 35);
  assert.equal(percentile(xs, 100), 50);
  assert.throws(() => percentile(xs, 101), RangeError);
  assert.throws(() => percentile([], 50), RangeError);
});
