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

const useIndexedDB = true;

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

//==========================================================================||
//                                 Insert                                   ||
//==========================================================================||

describe('insert()', () => {
  it('insert() 10 items, 1 layer', async () => {
    const hnsw = new HNSW({
      distanceFunction: 'cosine',
      seed: 20240101,
      useIndexedDB
    });

    // Insert 10 embeddings
    const size = 10;

    // The random levels with this seed is 0,0,0,0,0,0,0,0,0,0
    const reportIDs: string[] = [];
    const embeddings: number[][] = [];
    for (let i = 0; i < size; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      embeddings.push(embeddingData.embeddings[i]);
    }

    await hnsw.bulkInsert(reportIDs, embeddings);

    // There should be only one layer, and all nodes are fully connected
    expect(hnsw.graphLayers.length).toBe(1);
    _checkGraphLayers(reportIDs, hnsw, graph10Layer1);
  });

  it('insert() 10 items, 2 layer', async () => {
    const hnsw = new HNSW({
      distanceFunction: 'cosine',
      seed: 10,
      useIndexedDB
    });

    // Insert 10 embeddings
    const size = 10;

    // The random levels with this seed is 0,0,0,1,1,0,0,0,0,0
    const reportIDs: string[] = [];
    const embeddings: number[][] = [];
    for (let i = 0; i < size; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      embeddings.push(embeddingData.embeddings[i]);
    }

    await hnsw.bulkInsert(reportIDs, embeddings);

    expect(hnsw.graphLayers.length).toBe(2);
    _checkGraphLayers(reportIDs, hnsw, graph10Layer2);
  });

  it('insert() 30 items, 3 layer', async () => {
    const hnsw = new HNSW({
      distanceFunction: 'cosine',
      seed: 262,
      useIndexedDB
    });

    // Insert 30 embeddings
    const size = 30;

    // The random levels with seed 262 is: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 0, 1, 1, 0, 0, 0]
    const reportIDs: string[] = [];
    const embeddings: number[][] = [];
    for (let i = 0; i < size; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      embeddings.push(embeddingData.embeddings[i]);
    }

    await hnsw.bulkInsert(reportIDs, embeddings);

    expect(hnsw.graphLayers.length).toBe(3);
    _checkGraphLayers(reportIDs, hnsw, graph30Layer3);
  });

  it('insert() 100 items, 6 layer', async () => {
    const hnsw = new HNSW({
      distanceFunction: 'cosine',
      seed: 11906,
      useIndexedDB
    });

    // Insert 100 embeddings
    const size = 100;

    // The random levels with seed 11906 is: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 0, 0, 0, 3, 0, 5, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    // 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    const reportIDs: string[] = [];
    const embeddings: number[][] = [];
    for (let i = 0; i < size; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      embeddings.push(embeddingData.embeddings[i]);
    }

    await hnsw.bulkInsert(reportIDs, embeddings);

    expect(hnsw.graphLayers.length).toBe(6);
    _checkGraphLayers(reportIDs, hnsw, graph100Layer6);
  });

  it('insert() 100 items, 3 layer, m=3', async () => {
    const hnsw = new HNSW({
      distanceFunction: 'cosine',
      seed: 21574,
      m: 3,
      useIndexedDB
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
    const embeddings: number[][] = [];
    for (let i = 0; i < size; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      embeddings.push(embeddingData.embeddings[i]);
    }

    await hnsw.bulkInsert(reportIDs, embeddings, levels);

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
      distanceFunction: 'cosine',
      seed: 65975,
      useIndexedDB
    });

    // Insert 50 embeddings
    const size = 50;

    // The random levels with this seed is [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    // 1, 1, 0, 0, 1, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0 ]
    const reportIDs: string[] = [];
    const embeddings: number[][] = [];
    for (let i = 0; i < size; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      embeddings.push(embeddingData.embeddings[i]);
    }

    await hnsw.bulkInsert(reportIDs, embeddings);

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
      distanceFunction: 'cosine',
      seed: 113082,
      useIndexedDB
    });

    // Insert 50 embeddings
    const size = 50;

    // The random levels with this seed is [1, 0, 2, 0, 0, 0, 0, 1, 1, 0, 2, 0,
    // 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
    // 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    let reportIDs: string[] = [];
    let embeddings: number[][] = [];
    for (let i = 0; i < 30; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      embeddings.push(embeddingData.embeddings[i]);
    }

    await hnsw.bulkInsert(reportIDs, embeddings);

    // Delete 30 random nodes
    const deleteIndexes = [
      7, 12, 4, 14, 20, 27, 5, 21, 2, 19, 10, 15, 24, 6, 3, 0, 22, 8, 11, 1
    ];

    for (const i of deleteIndexes) {
      const key = String(embeddingData.reportNumbers[i]);
      await hnsw.markDeleted(key);
    }

    // Insert the rest 20 nodes
    reportIDs = [];
    embeddings = [];
    for (let i = 30; i < size; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      embeddings.push(embeddingData.embeddings[i]);
    }

    await hnsw.bulkInsert(reportIDs, embeddings);

    expect(hnsw.graphLayers.length).toBe(2);
    _checkGraphLayers(reportIDs, hnsw, graph50Delete1);
  });

  it('markDelete(): insert 30 => delete 20 => un-delete 10 => insert 20', async () => {
    const hnsw = new HNSW({
      distanceFunction: 'cosine',
      seed: 113082,
      useIndexedDB
    });

    // Insert 50 embeddings
    const size = 50;

    // The random levels with this seed is [1, 0, 2, 0, 0, 0, 0, 1, 1, 0, 2, 0,
    // 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
    // 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    // Insert 30 nodes
    let reportIDs: string[] = [];
    let embeddings: number[][] = [];
    for (let i = 0; i < 30; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      embeddings.push(embeddingData.embeddings[i]);
    }
    await hnsw.bulkInsert(reportIDs, embeddings);

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
    reportIDs = [];
    embeddings = [];
    for (let i = 30; i < size; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      embeddings.push(embeddingData.embeddings[i]);
    }
    await hnsw.bulkInsert(reportIDs, embeddings);

    expect(hnsw.graphLayers.length).toBe(2);
    // graph50Delete2 is actually the same as graph50Delete1
    _checkGraphLayers(reportIDs, hnsw, graph50Delete2);
  });
});

//==========================================================================||
//                                 query                                    ||
//==========================================================================||

describe('query()', () => {
  it('query(): 90/50 items, insert 30 => delete 20 => un-delete 10 => insert 20', async () => {
    const hnsw = new HNSW({
      distanceFunction: 'cosine',
      seed: 113082,
      useIndexedDB
    });

    // Insert 50 embeddings
    const size = 50;

    // The random levels with this seed is [1, 0, 2, 0, 0, 0, 0, 1, 1, 0, 2, 0,
    // 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
    // 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    // Insert 30 nodes
    let reportIDs: string[] = [];
    let embeddings: number[][] = [];
    for (let i = 0; i < 30; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      embeddings.push(embeddingData.embeddings[i]);
    }
    await hnsw.bulkInsert(reportIDs, embeddings);

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
    reportIDs = [];
    embeddings = [];
    for (let i = 30; i < size; i++) {
      const curReportID = String(embeddingData.reportNumbers[i]);
      reportIDs.push(curReportID);
      embeddings.push(embeddingData.embeddings[i]);
    }
    await hnsw.bulkInsert(reportIDs, embeddings);

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
