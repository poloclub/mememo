/**
 * Compute n choose k
 * @param n n to choose from
 * @param k to choose k
 * @returns Result
 */
export const comb = (n: number, k: number): number => {
  const minK = Math.min(k, n - k);
  return Array.from(new Array(minK), (_, i) => i + 1).reduce(
    (a, b) => (a * (n + 1 - b)) / b,
    1
  );
};

/**
 * Return all n-length combinations from an array
 * @param array Input array
 * @param n Length of combinations
 * @returns An array of n-length combinations
 */
export const getCombinations = <T>(array: T[], n: number): T[][] => {
  const result: T[][] = [];

  function backtrack(first = 0, current: T[] = []) {
    if (current.length === n) {
      result.push(current);
      return;
    }
    for (let i = first; i < array.length; i++) {
      backtrack(i + 1, [...current, array[i]]);
    }
  }

  backtrack();
  return result;
};
