import { HNSW } from 'mememo';
import type { MememoIndexJSON } from 'mememo';
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
import type * as Flexsearch from 'flexsearch';
import Index from 'flexsearch';
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
    }
  | {
      command: 'startSemanticSearch';
      payload: {
        embedding: number[];
        requestID: number;
        topK: number;
        maxDistance: number;
      };
    }
  | {
      command: 'finishSemanticSearch';
      payload: {
        embedding: number[];
        requestID: number;
        topK: number;
        maxDistance: number;
        documents: string[];
        documentDistances: number[];
      };
    };

const DEV_MODE = import.meta.env.DEV;
const POINT_THRESHOLD_INDEX = 500;
const POINT_THRESHOLD_SKIP_INDEX = 500;

// Data loading
let pendingDataPoints: DocumentRecord[] = [];
let loadedPointCount = 0;
let lastDrawnPoints: DocumentRecord[] | null = null;

// Indexes
// const flexIndex: Flexsearch.Index = new Flexsearch.Index({
//   tokenize: 'forward'
// }) as Flexsearch.Index;

// @ts-ignore
const flexIndex: Flexsearch.Index<string> = new Index({
  tokenize: 'forward'
}) as Flexsearch.Index<string>;

let documentDBPromise: PromiseExtended<Table<DocumentDBEntry, string>> | null =
  null;
const hnswIndex = new HNSW({
  distanceFunction: 'cosine-normalized',
  seed: 123,
  useIndexedDB: true,
  efConstruction: 40
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

    case 'startSemanticSearch': {
      const { requestID, embedding, topK, maxDistance } = e.data.payload;
      semanticSearch(embedding, topK, maxDistance, requestID).then(
        () => {},
        () => {}
      );
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
  const db = myDexie.table<DocumentDBEntry, string>('mememo');
  documentDBPromise = db.clear().then(() => db);

  // Load the index if the url is given
  let skipIndex = false;
  if (indexURL !== undefined) {
    try {
      console.time('Load index');
      const ds = new DecompressionStream('gzip');
      const response = await fetch(indexURL);
      const blobIn = await response.blob();
      const streamIn = blobIn.stream().pipeThrough(ds);
      const blobOut = await new Response(streamIn).blob();
      const indexJSON = JSON.parse(await blobOut.text()) as MememoIndexJSON;
      hnswIndex.loadIndex(indexJSON);
      console.timeEnd('Load index');
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
    id: String(point[0]),
    text: point[1],
    embedding: point[2]
  };

  const pointThreshold = skipIndex
    ? POINT_THRESHOLD_SKIP_INDEX
    : POINT_THRESHOLD_INDEX;

  // Index the point in flex
  pendingDataPoints.push(documentPoint);
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore
  flexIndex.add(documentPoint.id, documentPoint.text);

  loadedPointCount += 1;

  if (pendingDataPoints.length >= pointThreshold) {
    // Batched index the documents to IndexedDB and MeMemo
    const keys = pendingDataPoints.map(d => d.id);
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
  const resultIDs = await flexIndex.search(query, {
    limit
  });

  // Look up the indexes in indexedDB
  const results = await documentDB.bulkGet(resultIDs);
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

/**
 * Semantic search relevant documents
 * @param embedding Query embedding
 * @param topK Top-k items
 * @param maxDistance Max distance threshold
 */
const semanticSearch = async (
  embedding: number[],
  topK: number,
  maxDistance: number,
  requestID: number
) => {
  if (documentDBPromise === null) {
    throw Error('documentDB is null');
  }
  const documentDB = await documentDBPromise;
  const { keys, distances } = await hnswIndex.query(embedding, topK);

  // Batched query documents from indexedDB
  const results = await documentDB.bulkGet(keys);
  const documents: string[] = [];
  const documentDistances: number[] = [];

  for (const [i, r] of results.entries()) {
    if (r !== undefined && distances[i] <= maxDistance) {
      documents.push(r.text);
      documentDistances.push(distances[i]);
    }
  }

  // await new Promise<void>(resolve => {
  //   setTimeout(resolve, 5000);
  // });

  const message: MememoWorkerMessage = {
    command: 'finishSemanticSearch',
    payload: {
      embedding,
      topK,
      requestID,
      maxDistance,
      documents,
      documentDistances
    }
  };
  postMessage(message);
};
