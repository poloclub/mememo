import { describe, it, expect, beforeEach } from 'vitest';
import { HNSW } from '../src/mememo';
import { randomLcg, randomUniform } from 'd3-random';
import embeddingDataJSON from './data/accident-report-embeddings-100.json';
import graph10Layer1JSON from './data/insert-10-1-layer.json';
import graph10Layer2JSON from './data/insert-10-2-layer.json';
import graph30Layer3JSON from './data/insert-30-3-layer.json';
import graph100Layer6JSON from './data/insert-100-6-layer.json';
import graph100Layer3M3JSON from './data/insert-100-3-layer-m=3.json';
import graph50Update10JSON from './data/update-50-3-layer-10.json';
import graph50Delete1JSON from './data/delete-50-insert-30-delete-20-insert-20.json';
import graph50Delete2JSON from './data/delete-50-insert-30-delete-20-undelete-10-insert-20.json';
import query1JSON from './data/query-50.json';

interface EmbeddingData {
  embeddings: number[][];
  reportNumbers: number[];
}

type GraphLayer = Record<string, Record<string, number | undefined>>;

type QueryResult = [string, number];

interface QueryData {
  i: number;
  k: number;
  result: QueryResult[];
}

const embeddingData = embeddingDataJSON as EmbeddingData;

const graph10Layer1 = graph10Layer1JSON as GraphLayer[];
const graph10Layer2 = graph10Layer2JSON as GraphLayer[];
const graph30Layer3 = graph30Layer3JSON as GraphLayer[];
const graph100Layer6 = graph100Layer6JSON as GraphLayer[];
const graph100Layer3M3 = graph100Layer3M3JSON as GraphLayer[];
const graph50Update10 = graph50Update10JSON as GraphLayer[];
const graph50Delete1 = graph50Delete1JSON as GraphLayer[];
const graph50Delete2 = graph50Delete2JSON as GraphLayer[];

const query1 = query1JSON as QueryData[];

/**
 * Check if the graphs in HNSW match the expected graph layers from json
 * @param reportIDs Report IDs in the hnsw
 * @param hnsw HNSW index
 * @param expectedGraphs Expected graph layers loaded from json
 */
const _checkGraphLayers = (
  reportIDs: string[],
  hnsw: HNSW,
  expectedGraphs: GraphLayer[]
) => {
  for (const reportID of reportIDs) {
    for (const [l, graphLayer] of hnsw.graphLayers.entries()) {
      const curNode = graphLayer.graph.get(reportID);

      if (curNode === undefined) {
        expect(expectedGraphs[l][reportID]).toBeUndefined();
      } else {
        expect(expectedGraphs[l][reportID]).not.to.toBeUndefined();
        // Check the distances
        const expectedNeighbors = expectedGraphs[l][reportID];
        for (const [neighborKey, neighborDistance] of curNode.entries()) {
          expect(expectedNeighbors[neighborKey]).to.not.toBeUndefined();
          expect(neighborDistance).toBeCloseTo(
            expectedNeighbors[neighborKey]!,
            1e-6
          );
        }
      }
    }
  }
};

describe('constructor', () => {
  it('constructor', () => {
    const hnsw = new HNSW({
      distanceFunction: 'cosine-normalized',
      efConstruction: 100,
      m: 16
    });
  });
});

//==========================================================================||
//                                 Insert                                   ||
//==========================================================================||

describe('insert()', () => {
  it('insert() 10 items, 1 layer', async () => {
    const hnsw = new HNSW({
      distanceFunction: 'cosine-normalized',
      seed: 20240101
    });

    // Insert 10 embeddings
    const size = 10;

    // The random levels with this seed is 0,0,0,0,0,0,0,0,0,0
    const reportIDs: string[] = [];
    for (let i = 0; i < size; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      await hnsw.insert(curReportID, embeddingData.embeddings[i]);
    }

    // There should be only one layer, and all nodes are fully connected
    expect(hnsw.graphLayers.length).toBe(1);
    _checkGraphLayers(reportIDs, hnsw, graph10Layer1);
  });

  it('insert() 10 items, 2 layer', async () => {
    const hnsw = new HNSW({
      distanceFunction: 'cosine-normalized',
      seed: 10
    });

    // Insert 10 embeddings
    const size = 10;

    // The random levels with this seed is 0,0,0,1,1,0,0,0,0,0
    const reportIDs: string[] = [];
    for (let i = 0; i < size; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      await hnsw.insert(curReportID, embeddingData.embeddings[i]);
    }

    expect(hnsw.graphLayers.length).toBe(2);
    _checkGraphLayers(reportIDs, hnsw, graph10Layer2);
  });

  it('insert() 30 items, 3 layer', async () => {
    const hnsw = new HNSW({
      distanceFunction: 'cosine-normalized',
      seed: 262
    });

    // Insert 30 embeddings
    const size = 30;

    // The random levels with seed 262 is: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 0, 1, 1, 0, 0, 0]
    const reportIDs: string[] = [];
    for (let i = 0; i < size; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      await hnsw.insert(curReportID, embeddingData.embeddings[i]);
    }

    expect(hnsw.graphLayers.length).toBe(3);
    _checkGraphLayers(reportIDs, hnsw, graph30Layer3);
  });

  it('insert() 100 items, 6 layer', async () => {
    const hnsw = new HNSW({
      distanceFunction: 'cosine-normalized',
      seed: 11906
    });

    // Insert 100 embeddings
    const size = 100;

    // The random levels with seed 11906 is: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 0, 0, 0, 3, 0, 5, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    // 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    const reportIDs: string[] = [];
    for (let i = 0; i < size; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      await hnsw.insert(curReportID, embeddingData.embeddings[i]);
    }

    expect(hnsw.graphLayers.length).toBe(6);
    _checkGraphLayers(reportIDs, hnsw, graph100Layer6);
  });

  it('insert() 100 items, 3 layer, m=3', async () => {
    const hnsw = new HNSW({
      distanceFunction: 'cosine-normalized',
      seed: 21574,
      m: 3
    });

    // Insert 100 embeddings
    const size = 100;

    // The random levels with seed 21574 (need to manually set it, because it
    // would change since m and ml are different from default)
    const levels = [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 2, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 1, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0
    ];

    const reportIDs: string[] = [];
    for (let i = 0; i < size; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      await hnsw.insert(curReportID, embeddingData.embeddings[i], levels[i]);
    }

    expect(hnsw.graphLayers.length).toBe(3);
    _checkGraphLayers(reportIDs, hnsw, graph100Layer3M3);
  });

  it.skip('Find random seeds', () => {
    // Find random seed that give a nice level sequence to test
    const size = 50;
    const start = 100000;
    for (let i = start; i < start + 100000; i++) {
      const rng = randomLcg(i);
      const curLevels: number[] = [];
      const ml = 1 / Math.log(16);

      for (let j = 0; j < size; j++) {
        const level = Math.floor(-Math.log(rng()) * ml);
        curLevels.push(level);
      }

      if (Math.max(...curLevels) < 4) {
        const levelSum = curLevels.reduce((sum, value) => sum + value, 0);
        if (levelSum > 12) {
          console.log('Good seed: ', i);
          console.log(curLevels);
          break;
        }
      }
    }
  });
});

//==========================================================================||
//                                 Update                                   ||
//==========================================================================||

describe('update()', () => {
  it('update() 10 / 50 items', async () => {
    const hnsw = new HNSW({
      distanceFunction: 'cosine-normalized',
      seed: 65975
    });

    // Insert 50 embeddings
    const size = 50;

    // The random levels with this seed is [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    // 1, 1, 0, 0, 1, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0 ]
    const reportIDs: string[] = [];
    for (let i = 0; i < size; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      await hnsw.insert(curReportID, embeddingData.embeddings[i]);
    }

    // Update 10 nodes
    const updateIndexes = [
      [3, 71],
      [6, 63],
      [36, 82],
      [9, 67],
      [31, 91],
      [1, 55],
      [43, 65],
      [4, 85],
      [37, 61],
      [45, 86]
    ];

    for (const pair of updateIndexes) {
      const oldKey = String(embeddingData.reportNumbers[pair[0]]);
      const newValue = embeddingData.embeddings[pair[1]];
      await hnsw.update(oldKey, newValue);
    }

    expect(hnsw.graphLayers.length).toBe(3);
    _checkGraphLayers(reportIDs, hnsw, graph50Update10);
  });
});

//==========================================================================||
//                                 Delete                                   ||
//==========================================================================||

describe('markDelete()', () => {
  it('markDelete(): insert 30 => delete 20 => insert 20', async () => {
    const hnsw = new HNSW({
      distanceFunction: 'cosine-normalized',
      seed: 113082
    });

    // Insert 50 embeddings
    const size = 50;

    // The random levels with this seed is [1, 0, 2, 0, 0, 0, 0, 1, 1, 0, 2, 0,
    // 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
    // 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    const reportIDs: string[] = [];

    // Insert 30 nodes
    for (let i = 0; i < 30; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      await hnsw.insert(curReportID, embeddingData.embeddings[i]);
    }

    // Delete 30 random nodes
    const deleteIndexes = [
      7, 12, 4, 14, 20, 27, 5, 21, 2, 19, 10, 15, 24, 6, 3, 0, 22, 8, 11, 1
    ];

    for (const i of deleteIndexes) {
      const key = String(embeddingData.reportNumbers[i]);
      await hnsw.markDeleted(key);
    }

    // Insert the rest 20 nodes
    for (let i = 30; i < size; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      await hnsw.insert(curReportID, embeddingData.embeddings[i]);
    }

    expect(hnsw.graphLayers.length).toBe(2);
    _checkGraphLayers(reportIDs, hnsw, graph50Delete1);
  });

  it('markDelete(): insert 30 => delete 20 => un-delete 10 => insert 20', async () => {
    const hnsw = new HNSW({
      distanceFunction: 'cosine-normalized',
      seed: 113082
    });

    // Insert 50 embeddings
    const size = 50;

    // The random levels with this seed is [1, 0, 2, 0, 0, 0, 0, 1, 1, 0, 2, 0,
    // 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
    // 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    const reportIDs: string[] = [];

    // Insert 30 nodes
    for (let i = 0; i < 30; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      await hnsw.insert(curReportID, embeddingData.embeddings[i]);
    }

    // Delete 20 random nodes
    const deleteIndexes = [
      7, 12, 4, 14, 20, 27, 5, 21, 2, 19, 10, 15, 24, 6, 3, 0, 22, 8, 11, 1
    ];

    for (const i of deleteIndexes) {
      const key = String(embeddingData.reportNumbers[i]);
      await hnsw.markDeleted(key);
    }

    // Un-delete 10 random nodes
    const unDeleteIndexes = [12, 22, 4, 14, 19, 5, 2, 15, 21, 0];

    for (const i of unDeleteIndexes) {
      const key = String(embeddingData.reportNumbers[i]);
      await hnsw.unMarkDeleted(key);
    }

    // Insert the rest 20 nodes
    for (let i = 30; i < size; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      await hnsw.insert(curReportID, embeddingData.embeddings[i]);
    }

    expect(hnsw.graphLayers.length).toBe(2);
    // graph50Delete2 is actually the same as graph50Delete1
    _checkGraphLayers(reportIDs, hnsw, graph50Delete2);
  });
});

//==========================================================================||
//                                 query                                    ||
//==========================================================================||

const createHNSW30201020 = async () => {
  const hnsw = new HNSW({
    distanceFunction: 'cosine-normalized',
    seed: 113082
  });

  // Insert 50 embeddings
  const size = 50;

  // The random levels with this seed is [1, 0, 2, 0, 0, 0, 0, 1, 1, 0, 2, 0,
  // 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
  // 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  const reportIDs: string[] = [];

  // Insert 30 nodes
  for (let i = 0; i < 30; i++) {
    const curReportID = String(embeddingData.reportNumbers[i]);
    reportIDs.push(curReportID);
    await hnsw.insert(curReportID, embeddingData.embeddings[i]);
  }

  // Delete 20 random nodes
  const deleteIndexes = [
    7, 12, 4, 14, 20, 27, 5, 21, 2, 19, 10, 15, 24, 6, 3, 0, 22, 8, 11, 1
  ];

  for (const i of deleteIndexes) {
    const key = String(embeddingData.reportNumbers[i]);
    await hnsw.markDeleted(key);
  }

  // Un-delete 10 random nodes
  const unDeleteIndexes = [12, 22, 4, 14, 19, 5, 2, 15, 21, 0];

  for (const i of unDeleteIndexes) {
    const key = String(embeddingData.reportNumbers[i]);
    await hnsw.unMarkDeleted(key);
  }

  // Insert the rest 20 nodes
  for (let i = 30; i < size; i++) {
    const curReportID = String(embeddingData.reportNumbers[i]);
    reportIDs.push(curReportID);
    await hnsw.insert(curReportID, embeddingData.embeddings[i]);
  }
  return hnsw;
};

describe('query()', () => {
  it('query(): 90/50 items, insert 30 => delete 20 => un-delete 10 => insert 20', async () => {
    const hnsw = await createHNSW30201020();

    // Check query results
    for (const q of query1) {
      const { keys, distances } = await hnsw.query(
        embeddingData.embeddings[q.i],
        q.k
      );
      expect(keys.length).toBe(q.result.length);

      for (const [i, key] of keys.entries()) {
        expect(key).toBe(q.result[i][0]);
        expect(distances[i]).toBeCloseTo(q.result[i][1], 4);
      }
    }
  });
});

//==========================================================================||
//                                 Export                                   ||
//==========================================================================||

describe('loadIndex()', () => {
  it('Export and load index', async () => {
    // Export the index
    const hnsw1 = await createHNSW30201020();
    const index1 = hnsw1.exportIndex();

    // Create a new hnsw using the index json
    const hnsw2 = new HNSW({
      distanceFunction: 'cosine-normalized',
      seed: 113082
    });
    hnsw2.loadIndex(index1);

    // The export of the new index should be the same as the old index
    const index2 = hnsw2.exportIndex();

    expect(JSON.stringify(index1)).toBe(JSON.stringify(index2));
  });

  it('Export and load index, re-create embeddings', async () => {
    // Export the index
    const hnsw1 = await createHNSW30201020();
    const index1 = hnsw1.exportIndex();

    // Create a new hnsw using the index json
    const hnsw2 = new HNSW({
      distanceFunction: 'cosine-normalized',
      seed: 113082
    });
    hnsw2.loadIndex(index1);

    // Insert 50 embeddings
    const size = 50;

    // The random levels with this seed is [1, 0, 2, 0, 0, 0, 0, 1, 1, 0, 2, 0,
    // 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
    // 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    const reportIDs: string[] = [];

    // Insert 30 nodes
    for (let i = 0; i < 30; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      await hnsw2.insertSkipIndex(curReportID, embeddingData.embeddings[i]);
    }

    // Delete 20 random nodes
    const deleteIndexes = [
      7, 12, 4, 14, 20, 27, 5, 21, 2, 19, 10, 15, 24, 6, 3, 0, 22, 8, 11, 1
    ];

    for (const i of deleteIndexes) {
      const key = String(embeddingData.reportNumbers[i]);
      await hnsw2.markDeleted(key);
    }

    // Un-delete 10 random nodes
    const unDeleteIndexes = [12, 22, 4, 14, 19, 5, 2, 15, 21, 0];

    for (const i of unDeleteIndexes) {
      const key = String(embeddingData.reportNumbers[i]);
      await hnsw2.unMarkDeleted(key);
    }

    // Insert the rest 20 nodes
    for (let i = 30; i < size; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      await hnsw2.insertSkipIndex(curReportID, embeddingData.embeddings[i]);
    }

    // The export of the new index should be the same as the old index
    const index2 = hnsw2.exportIndex();
    expect(JSON.stringify(index1)).toBe(JSON.stringify(index2));

    // Check query results
    for (const q of query1) {
      const { keys, distances } = await hnsw2.query(
        embeddingData.embeddings[q.i],
        q.k
      );
      expect(keys.length).toBe(q.result.length);

      for (const [i, key] of keys.entries()) {
        expect(key).toBe(q.result[i][0]);
        expect(distances[i]).toBeCloseTo(q.result[i][1], 4);
      }
    }
  });
});

//==========================================================================||
//                          Helper Functions                                ||
//==========================================================================||

describe('_getRandomLevel()', () => {
  it('_getRandomLevel()', () => {
    const hnsw = new HNSW({
      distanceFunction: 'cosine-normalized',
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
});

describe('Distance functions', () => {
  it('Distance function (cosine)', () => {
    const hnsw = new HNSW({
      distanceFunction: 'cosine'
    });

    const a = [0.44819598, 0.26875241, 0.02164449, 0.33802939, 0.2482019];
    const b = [0.99448402, 0.29269615, 0.98586198, 0.57482737, 0.12994758];

    expect(hnsw.distanceFunction(a, b, null, null)).closeTo(
      0.2554613725418178,
      1e-6
    );
  });

  it('Distance function (cosine-normalized)', () => {
    const hnsw = new HNSW({
      distanceFunction: 'cosine-normalized'
    });

    const a = [0.3448653, 0.4612705, 0.79191367, 0.057099, 0.19470466];
    const b = [0.39233533, 0.37618326, 0.12894695, 0.50411272, 0.65863662];

    expect(hnsw.distanceFunction(a, b, null, null)).closeTo(
      0.43203611706139833,
      1e-6
    );
  });

  it('Distance function (custom)', () => {
    const l1Distance = (a: number[], b: number[]) => {
      return a.reduce(
        (sum, value, index) => sum + Math.abs(value - b[index]),
        0
      );
    };

    const hnsw = new HNSW({
      distanceFunction: l1Distance
    });

    const a = [0.44819598, 0.26875241, 0.02164449, 0.33802939, 0.2482019];
    const b = [0.99448402, 0.29269615, 0.98586198, 0.57482737, 0.12994758];

    expect(hnsw.distanceFunction(a, b, null, null)).closeTo(
      1.8895015711439895,
      1e-6
    );
  });
});
