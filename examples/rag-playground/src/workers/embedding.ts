import { pipeline } from '@xenova/transformers';
import type { FeatureExtractionPipeline } from '@xenova/transformers';

export enum EmbeddingModel {
  gteSmall = 'gte-small'
}

export type EmbeddingWorkerMessage =
  | {
      command: 'startExtractEmbedding';
      payload: {
        requestID: string;
        text: string;
        model: EmbeddingModel;
        detail: string;
      };
    }
  | {
      command: 'finishExtractEmbedding';
      payload: {
        requestID: string;
        text: string;
        model: EmbeddingModel;
        detail: string;
        embedding: number[];
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
const extractors: Record<EmbeddingModel, Promise<FeatureExtractionPipeline>> = {
  'gte-small': pipeline('feature-extraction', 'gte-small')
};

/**
 * Extract embedding from the input text
 * @param model Embedding model
 * @param text Input text
 */
export const getEmbedding = async (
  model: EmbeddingModel,
  text: string,
  requestID: string,
  detail: string
) => {
  try {
    const extractor = await extractors[model];
    const sentences = [text];
    const output = await extractor(sentences, {
      pooling: 'mean',
      normalize: true
    });

    const embedding = Array.from<number>(output.data as Float32Array);

    // Send result to the main thread
    const message: EmbeddingWorkerMessage = {
      command: 'finishExtractEmbedding',
      payload: {
        model,
        requestID,
        detail,
        text,
        embedding
      }
    };
    postMessage(message);
  } catch (error) {
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
