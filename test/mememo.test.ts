import { describe, test, expect, beforeEach } from 'vitest';
import { add, HNSW } from '../src/mememo';
import embeddingDataJSON from './data/accident-report-embeddings-100.json';
import graph10Layer0JSON from './data/insert-10-layer-0.json';

interface EmbeddingData {
  embeddings: number[][];
  reportNumbers: number[];
}
const embeddingData = embeddingDataJSON as EmbeddingData;

const graph10Layer0 = graph10Layer0JSON as Record<
  string,
  Record<string, number>
>;

test('add()', () => {
  expect(add(10, 1)).toBe(11);
});

test('constructor', () => {
  const hnsw = new HNSW({
    distanceFunction: 'cosine',
    efConstruction: 100,
    m: 16
  });
});

//==========================================================================||
//                                 Insert                                   ||
//==========================================================================||

test('insert() 10 items, 1 layer', () => {
  const hnsw = new HNSW({
    distanceFunction: 'cosine',
    seed: 20240101
  });

  // Insert 10 embeddings
  const size = 10;

  // The random levels with this seed is 0,0,0,0,0,0,0,0,0,0
  const reportIDs: string[] = [];
  for (let i = 0; i < size; i++) {
    const curReportID = String(embeddingData.reportNumbers[i]);
    reportIDs.push(curReportID);
    hnsw.insert(curReportID, embeddingData.embeddings[i]);
  }

  // There should be only one layer, and all nodes are fully connected
  expect(hnsw.graphLayers.length).toBe(1);

  for (const reportID of reportIDs) {
    const curNode = hnsw.graphLayers[0].graph.get(reportID);
    expect(curNode).to.not.toBeUndefined();
    expect(curNode!.size).toBe(9);

    // Test the distances
    const expectedNeighbors = graph10Layer0[reportID];
    for (const [neighborKey, neighborDistance] of curNode!.entries()) {
      expect(neighborDistance).toBeCloseTo(
        expectedNeighbors[neighborKey],
        1e-6
      );
    }
  }
});

//==========================================================================||
//                          Helper Functions                                ||
//==========================================================================||

test('_getRandomLevel()', () => {
  const hnsw = new HNSW({
    distanceFunction: 'cosine',
    seed: 20240101
  });

  const levels: number[] = [];
  for (let i = 0; i < 10000; i++) {
    levels.push(hnsw._getRandomLevel());
  }

  // Count the different levels
  const levelCounter = new Map<number, number>();
  for (const level of levels) {
    if (levelCounter.has(level)) {
      levelCounter.set(level, levelCounter.get(level)! + 1);
    } else {
      levelCounter.set(level, 1);
    }
  }

  expect(levelCounter.get(0)! > 9000);
  expect(levelCounter.get(1)! > 400);
  expect(levelCounter.get(1)! < 700);
  expect(levelCounter.get(2)! < 50);
  expect(levelCounter.get(3)! < 10);
});

test('distance function (cosine)', () => {
  const hnsw = new HNSW({
    distanceFunction: 'cosine'
  });

  const a = [0.44819598, 0.26875241, 0.02164449, 0.33802939, 0.2482019];
  const b = [0.99448402, 0.29269615, 0.98586198, 0.57482737, 0.12994758];

  expect(hnsw.distanceFunction(a, b)).closeTo(0.2554613725418178, 1e-6);
});

test('distance function (cosine-normalized)', () => {
  const hnsw = new HNSW({
    distanceFunction: 'cosine-normalized'
  });

  const a = [0.3448653, 0.4612705, 0.79191367, 0.057099, 0.19470466];
  const b = [0.39233533, 0.37618326, 0.12894695, 0.50411272, 0.65863662];

  expect(hnsw.distanceFunction(a, b)).closeTo(0.43203611706139833, 1e-6);
});

test('distance function (custom)', () => {
  const l1Distance = (a: number[], b: number[]) => {
    return a.reduce((sum, value, index) => sum + Math.abs(value - b[index]), 0);
  };

  const hnsw = new HNSW({
    distanceFunction: l1Distance
  });

  const a = [0.44819598, 0.26875241, 0.02164449, 0.33802939, 0.2482019];
  const b = [0.99448402, 0.29269615, 0.98586198, 0.57482737, 0.12994758];

  expect(hnsw.distanceFunction(a, b)).closeTo(1.8895015711439895, 1e-6);
});
