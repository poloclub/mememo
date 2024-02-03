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

export type MememoWorkerMessage =
  | {
      command: 'startLoadData';
      payload: {
        /** NDJSON data url */
        url: string;
      };
    }
  | {
      command: 'transferLoadData';
      payload: {
        isFirstBatch: boolean;
        isLastBatch: boolean;
        points: DocumentRecord[];
        loadedPointCount: number;
      };
    };

const DEV_MODE = import.meta.env.DEV;
const POINT_THRESHOLD = 100;

let pendingDataPoints: DocumentRecord[] = [];
let loadedPointCount = 0;
let sentPointCount = 0;

let lastDrawnPoints: DocumentRecord[] | null = null;

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

      const url = e.data.payload.url;
      startLoadCompressedData(url);
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
 */
const startLoadCompressedData = (url: string) => {
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
        processPointStream(point);
      }
    }
  });
};

/**
 * Process one data point
 * @param point Loaded data point
 */
const processPointStream = (point: DocumentRecordStreamData) => {
  const promptPoint: DocumentRecord = {
    text: point[0],
    embedding: point[1],
    id: loadedPointCount
  };

  pendingDataPoints.push(promptPoint);
  loadedPointCount += 1;

  // Notify the main thread if we have load enough data
  if (pendingDataPoints.length >= POINT_THRESHOLD) {
    const result: MememoWorkerMessage = {
      command: 'transferLoadData',
      payload: {
        isFirstBatch: lastDrawnPoints === null,
        isLastBatch: false,
        points: pendingDataPoints,
        loadedPointCount
      }
    };
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
      points: pendingDataPoints,
      loadedPointCount
    }
  };
  postMessage(result);

  sentPointCount += pendingDataPoints.length;
  lastDrawnPoints = pendingDataPoints.slice();
  pendingDataPoints = [];
};
