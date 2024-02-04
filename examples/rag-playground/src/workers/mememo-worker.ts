import { HNSW } from '../../../../src/index';
import type {
  DocumentRecord,
  DocumentRecordStreamData
} from '../types/common-types';
import {
  timeit,
  splitStreamTransform,
  parseJSONTransform
} from '@xiaohk/utils';
import Flexsearch from 'flexsearch';
import { openDB, IDBPDatabase } from 'idb';

//==========================================================================||
//                            Types & Constants                             ||
//==========================================================================||

export type MememoWorkerMessage =
  | {
      command: 'startLoadData';
      payload: {
        /** NDJSON data url */
        url: string;
        datasetName: string;
      };
    }
  | {
      command: 'transferLoadData';
      payload: {
        isFirstBatch: boolean;
        isLastBatch: boolean;
        documents: string[];
        loadedPointCount: number;
      };
    }
  | {
      command: 'startLexicalSearch';
      payload: {
        query: string;
        requestID: number;
        limit: number;
      };
    }
  | {
      command: 'finishLexicalSearch';
      payload: {
        requestID: number;
        results: string[];
      };
    };

const DEV_MODE = import.meta.env.DEV;
const POINT_THRESHOLD = 100;

// Data loading
let pendingDataPoints: DocumentRecord[] = [];
let loadedPointCount = 0;
let sentPointCount = 0;
let lastDrawnPoints: DocumentRecord[] | null = null;

// Indexes
const flexIndex: Flexsearch.Index = new Flexsearch.Index() as Flexsearch.Index;
let workerDatasetName = 'my-dataset';
let documentDBPromise: Promise<IDBPDatabase<string>> | null = null;

//==========================================================================||
//                                Functions                                 ||
//==========================================================================||

/**
 * Handle message events from the main thread
 * @param e Message event
 */
self.onmessage = (e: MessageEvent<MememoWorkerMessage>) => {
  // Stream point data
  switch (e.data.command) {
    case 'startLoadData': {
      console.log('Worker: start streaming data');
      timeit('Stream data', true);
      const { url, datasetName } = e.data.payload;
      startLoadCompressedData(url, datasetName);
      break;
    }

    case 'startLexicalSearch': {
      const { query, limit, requestID } = e.data.payload;
      searchPoint(query, limit, requestID);
      break;
    }

    default: {
      console.error('Worker: unknown message', e.data.command);
      break;
    }
  }
};

/**
 * Start loading the text data
 * @param url URL to the zipped NDJSON file
 * @param datasetName Name of the dataset
 */
const startLoadCompressedData = (url: string, datasetName: string) => {
  // Update the indexed db store reference
  workerDatasetName = datasetName;

  documentDBPromise = openDB<string>(`${workerDatasetName}-store`, 1, {
    upgrade(db) {
      db.createObjectStore(workerDatasetName);
    }
  });

  fetch(url).then(async response => {
    if (!response.ok) {
      console.error('Failed to load data', response);
      return;
    }

    const reader = response.body
      ?.pipeThrough(new DecompressionStream('gzip'))
      ?.pipeThrough(new TextDecoderStream())
      ?.pipeThrough(splitStreamTransform('\n'))
      ?.pipeThrough(parseJSONTransform())
      ?.getReader();

    while (true && reader !== undefined) {
      const result = await reader.read();
      const point = result.value as DocumentRecordStreamData;
      const done = result.done;

      if (done) {
        timeit('Stream data', DEV_MODE);
        pointStreamFinished();
        break;
      } else {
        await processPointStream(point);
      }
    }
  });
};

/**
 * Process one data point
 * @param point Loaded data point
 */
const processPointStream = async (point: DocumentRecordStreamData) => {
  if (documentDBPromise === null) {
    throw Error('documentDB is null');
  }
  const documentDB = await documentDBPromise;

  const documentPoint: DocumentRecord = {
    text: point[0],
    embedding: point[1],
    id: loadedPointCount
  };

  // Index the point
  pendingDataPoints.push(documentPoint);
  flexIndex.add(documentPoint.id, documentPoint.text);
  await documentDB.put(workerDatasetName, documentPoint.text, documentPoint.id);
  loadedPointCount += 1;

  // Notify the main thread if we have load enough data
  if (pendingDataPoints.length >= POINT_THRESHOLD) {
    const result: MememoWorkerMessage = {
      command: 'transferLoadData',
      payload: {
        isFirstBatch: lastDrawnPoints === null,
        isLastBatch: false,
        documents: pendingDataPoints.map(d => d.text),
        loadedPointCount
      }
    };

    await new Promise<void>(resolve => {
      setTimeout(resolve, 500);
    });

    postMessage(result);

    sentPointCount += pendingDataPoints.length;
    lastDrawnPoints = pendingDataPoints.slice();
    pendingDataPoints = [];
  }
};

/**
 * Construct tree and notify the main thread when finish reading all data
 */
const pointStreamFinished = () => {
  // Send any left over points

  const result: MememoWorkerMessage = {
    command: 'transferLoadData',
    payload: {
      isFirstBatch: lastDrawnPoints === null,
      isLastBatch: true,
      documents: pendingDataPoints.map(d => d.text),
      loadedPointCount
    }
  };
  postMessage(result);

  sentPointCount += pendingDataPoints.length;
  lastDrawnPoints = pendingDataPoints.slice();
  pendingDataPoints = [];
};

/**
 * Start a lexical query
 * @param query Query string
 * @param limit Number of query items
 */
const searchPoint = async (query: string, limit: number, requestID: number) => {
  if (documentDBPromise === null) {
    throw Error('documentDB is null');
  }
  const documentDB = await documentDBPromise;
  const resultIndexes = flexIndex.search(query, {
    limit
  }) as unknown as number[];

  // Look up the indexes in indexedDB
  const results = [];
  for (const i of resultIndexes) {
    const result = (await documentDB.get(workerDatasetName, i)) as string;
    results.push(result);
  }

  const message: MememoWorkerMessage = {
    command: 'finishLexicalSearch',
    payload: {
      results,
      requestID
    }
  };
  postMessage(message);
};
