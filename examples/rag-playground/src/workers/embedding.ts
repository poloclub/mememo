import { pipeline, env } from '@xenova/transformers';
import type { FeatureExtractionPipeline } from '@xenova/transformers';

env.allowRemoteModels = false;

export enum EmbeddingModel {
  gteSmall = 'gte-small'
}

export type EmbeddingWorkerMessage =
  | {
      command: 'startExtractEmbedding';
      payload: {
        requestID: string;
        sentences: string[];
        model: EmbeddingModel;
        detail: string;
        windowURL: string;
      };
    }
  | {
      command: 'finishExtractEmbedding';
      payload: {
        requestID: string;
        sentences: string[];
        model: EmbeddingModel;
        detail: string;
        embeddings: number[][];
      };
    }
  | {
      command: 'error';
      payload: {
        requestID: string;
        originalCommand: string;
        message: string;
      };
    };

// Initialize the models
const extractorMap = new Map<
  EmbeddingModel,
  Promise<FeatureExtractionPipeline>
>();

/**
 * Helper function to handle calls from the main thread
 * @param e Message event
 */
self.onmessage = (e: MessageEvent<EmbeddingWorkerMessage>) => {
  switch (e.data.command) {
    case 'startExtractEmbedding': {
      const { model, sentences, requestID, detail, windowURL } = e.data.payload;
      startExtractEmbedding(
        model,
        sentences,
        requestID,
        detail,
        windowURL
      ).then(
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
 * Extract embedding from the input text
 * @param model Embedding model
 * @param text Input text
 */
export const startExtractEmbedding = async (
  model: EmbeddingModel,
  sentences: string[],
  requestID: string,
  detail: string,
  windowURL: string
) => {
  try {
    // Specify a custom location for TF models (need to use absolute path here)
    env.localModelPath = `${windowURL}models/`;

    if (!extractorMap.has(model)) {
      extractorMap.set(model, pipeline('feature-extraction', model));
    }
    const extractor = await extractorMap.get(model)!;
    const output = await extractor(sentences, {
      pooling: 'mean',
      normalize: true
    });

    const embeddings: number[][] = [];
    const flattenEmbedding: number[] = Array.from<number>(
      output.data as Float32Array
    );

    // Un-flatten the embedding output
    for (let i = 0; i < output.dims[0]; i++) {
      const curRow = flattenEmbedding.slice(
        i * output.dims[1],
        (i + 1) * output.dims[1]
      );
      embeddings.push(curRow);
    }

    // Send result to the main thread
    const message: EmbeddingWorkerMessage = {
      command: 'finishExtractEmbedding',
      payload: {
        model,
        requestID,
        detail,
        sentences,
        embeddings
      }
    };
    postMessage(message);
  } catch (error) {
    console.error(error);
    // Send error to the main thread
    const message: EmbeddingWorkerMessage = {
      command: 'error',
      payload: {
        message: `Failed to extract embeddings with error: ${Error}`,
        originalCommand: 'startExtractEmbedding',
        requestID
      }
    };
    postMessage(message);
  }
};
