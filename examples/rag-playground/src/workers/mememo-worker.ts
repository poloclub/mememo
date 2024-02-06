import { HNSW } from '../../../../src/index';
import type { MememoIndexJSON } from '../../../../src/index';
import type {
  DocumentRecord,
  DocumentDBEntry,
  DocumentRecordStreamData
} from '../types/common-types';
import {
  timeit,
  splitStreamTransform,
  parseJSONTransform
} from '@xiaohk/utils';
import Flexsearch from 'flexsearch';
import Dexie from 'dexie';
import type { Table, PromiseExtended } from 'dexie';

//==========================================================================||
//                            Types & Constants                             ||
//==========================================================================||

export type MememoWorkerMessage =
  | {
      command: 'startLoadData';
      payload: {
        /** NDJSON data url */
        url: string;
        /** Index json url */
        indexURL?: string;
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
        query: string;
        requestID: number;
        results: string[];
      };
    }
  | {
      command: 'startExportIndex';
      payload: {
        requestID: number;
      };
    }
  | {
      command: 'finishExportIndex';
      payload: {
        indexJSON: MememoIndexJSON;
      };
    };

const DEV_MODE = import.meta.env.DEV;
const POINT_THRESHOLD = 100;

// Data loading
let pendingDataPoints: DocumentRecord[] = [];
let loadedPointCount = 0;
let lastDrawnPoints: DocumentRecord[] | null = null;

// Indexes
const flexIndex: Flexsearch.Index = new Flexsearch.Index({
  tokenize: 'forward'
}) as Flexsearch.Index;

let documentDBPromise: PromiseExtended<Table<DocumentDBEntry, number>> | null =
  null;
const hnswIndex = new HNSW({
  distanceFunction: 'cosine-normalized',
  seed: 123,
  useIndexedDB: true
});

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
      const { url, indexURL } = e.data.payload;
      startLoadCompressedData(url, indexURL).then(
        () => {},
        () => {}
      );
      break;
    }

    case 'startLexicalSearch': {
      const { query, limit, requestID } = e.data.payload;
      searchPoint(query, limit, requestID).then(
        () => {},
        () => {}
      );
      break;
    }

    case 'startExportIndex': {
      const indexJSON = hnswIndex.exportIndex();
      const message: MememoWorkerMessage = {
        command: 'finishExportIndex',
        payload: {
          indexJSON: indexJSON
        }
      };
      postMessage(message);
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
const startLoadCompressedData = async (url: string, indexURL?: string) => {
  // Create a new store, clear content from previous sessions
  const myDexie = new Dexie('mememo-document-store');
  myDexie.version(1).stores({
    mememo: 'id'
  });
  const db = myDexie.table<DocumentDBEntry, number>('mememo');
  documentDBPromise = db.clear().then(() => db);

  // Load the index if the url is given
  let skipIndex = false;
  if (indexURL !== undefined) {
    try {
      const indexJSON = (await (
        await fetch(indexURL)
      ).json()) as MememoIndexJSON;
      hnswIndex.loadIndex(indexJSON);
      skipIndex = true;
    } catch (error) {
      console.error(error);
    }
  }

  fetch(url).then(
    async response => {
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
          await processPointStream(point, skipIndex);
        }
      }
    },
    () => {}
  );
};

/**
 * Process one data point
 * @param point Loaded data point
 */
const processPointStream = async (
  point: DocumentRecordStreamData,
  skipIndex: boolean
) => {
  if (documentDBPromise === null) {
    throw Error('documentDB is null');
  }
  const documentDB = await documentDBPromise;

  const documentPoint: DocumentRecord = {
    text: point[0],
    embedding: point[1],
    id: loadedPointCount
  };

  // Index the point in flex
  pendingDataPoints.push(documentPoint);
  flexIndex.add(documentPoint.id, documentPoint.text);

  loadedPointCount += 1;

  if (pendingDataPoints.length >= POINT_THRESHOLD) {
    // Batched index the documents to IndexedDB and MeMemo
    const keys = pendingDataPoints.map(d => String(d.id));
    const embeddings = pendingDataPoints.map(d => d.embedding);
    const documentEntries: DocumentDBEntry[] = pendingDataPoints.map(d => ({
      id: d.id,
      text: d.text
    }));

    await documentDB.bulkPut(documentEntries);

    if (skipIndex) {
      await hnswIndex.bulkInsertSkipIndex(keys, embeddings);
    } else {
      await hnswIndex.bulkInsert(keys, embeddings);
    }

    // Notify the main thread if we have load enough data
    const result: MememoWorkerMessage = {
      command: 'transferLoadData',
      payload: {
        isFirstBatch: lastDrawnPoints === null,
        isLastBatch: false,
        documents: pendingDataPoints.map(d => d.text),
        loadedPointCount
      }
    };

    // await new Promise<void>(resolve => {
    //   setTimeout(resolve, 100);
    // });

    postMessage(result);

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
  const results = await documentDB.bulkGet(resultIndexes);
  const documents: string[] = [];
  for (const r of results) {
    if (r !== undefined) {
      documents.push(r.text);
    }
  }

  const message: MememoWorkerMessage = {
    command: 'finishLexicalSearch',
    payload: {
      query,
      results: documents,
      requestID
    }
  };
  postMessage(message);
};
