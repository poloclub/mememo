import { LitElement, css, unsafeCSS, html, PropertyValues } from 'lit';
import { customElement, property, state, query } from 'lit/decorators.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';
import { EmbeddingModel } from '../../workers/embedding';
import type { EmbeddingWorkerMessage } from '../../workers/embedding';
import type { MememoTextViewer } from '../text-viewer/text-viewer';

import '../query-box/query-box';
import '../text-viewer/text-viewer';

import componentCSS from './playground.css?inline';
import EmbeddingWorkerInline from '../../workers/embedding?worker&inline';

interface DatasetInfo {
  dataURL: string;
  indexURL: string;
  datasetName: string;
  datasetNameDisplay: string;
}

enum Dataset {
  Arxiv = 'arxiv'
}

const datasets: Record<Dataset, DatasetInfo> = {
  [Dataset.Arxiv]: {
    dataURL: '/data/ml-arxiv-papers-1000.ndjson.gzip',
    indexURL: '/data/ml-arxiv-papers-1000-index.json',
    datasetName: 'ml-arxiv-papers',
    datasetNameDisplay: 'ML arXiv Abstracts (1k)'
  }
};

/**
 * Playground element.
 *
 */
@customElement('mememo-playground')
export class MememoPlayground extends LitElement {
  //==========================================================================||
  //                              Class Properties                            ||
  //==========================================================================||
  userQuery = '';
  embeddingWorker: Worker;
  embeddingWorkerRequestCount = 0;
  get embeddingWorkerRequestID() {
    this.embeddingWorkerRequestCount++;
    return `prompt-panel-${this.embeddingWorkerRequestCount}`;
  }

  @state()
  topK = 10;

  @query('mememo-text-viewer')
  textViewerComponent: MememoTextViewer | undefined | null;

  //==========================================================================||
  //                             Lifecycle Methods                            ||
  //==========================================================================||
  constructor() {
    super();

    this.embeddingWorker = new EmbeddingWorkerInline();
    this.embeddingWorker.addEventListener(
      'message',
      (e: MessageEvent<EmbeddingWorkerMessage>) => {
        this.embeddingWorkerMessageHandler(e);
      }
    );
  }

  /**
   * This method is called before new DOM is updated and rendered
   * @param changedProperties Property that has been changed
   */
  willUpdate(changedProperties: PropertyValues<this>) {}

  //==========================================================================||
  //                              Custom Methods                              ||
  //==========================================================================||
  async initData() {}

  /**
   * Extract embeddings for the input sentences
   * @param sentences Input sentences
   */
  getEmbedding(sentences: string[]) {
    const message: EmbeddingWorkerMessage = {
      command: 'startExtractEmbedding',
      payload: {
        detail: '',
        requestID: this.embeddingWorkerRequestID,
        model: EmbeddingModel.gteSmall,
        sentences: sentences
      }
    };
    this.embeddingWorker.postMessage(message);
  }

  /**
   * Use k-nearest neighbor to find semantically similar documents
   * @param embedding Embeddings of the user query
   */
  semanticSearch(embedding: number[]) {
    if (!this.textViewerComponent) {
      throw Error('textViewerComponent is not initialized.');
    }

    this.textViewerComponent.semanticSearch(embedding, this.topK, 0.5);
  }

  /**
   * Augment the prompt using relevant documents
   * @param relevantDocuments Documents that are relevant to the user query
   */
  compilePrompt(relevantDocuments: string[]) {
    console.log(relevantDocuments);
  }

  //==========================================================================||
  //                              Event Handlers                              ||
  //==========================================================================||
  userQueryRunClickHandler(e: CustomEvent<string>) {
    this.userQuery = e.detail;

    // Extract embeddings for the user query
    this.getEmbedding([this.userQuery]);
  }

  embeddingWorkerMessageHandler(e: MessageEvent<EmbeddingWorkerMessage>) {
    switch (e.data.command) {
      case 'finishExtractEmbedding': {
        const { embeddings } = e.data.payload;
        // Start semantic search using the embedding
        this.semanticSearch(embeddings[0]);
        break;
      }

      case 'error': {
        console.error('Worker error: ', e.data.payload.message);
        break;
      }

      default: {
        console.error('Worker: unknown message', e.data.command);
        break;
      }
    }
  }

  //==========================================================================||
  //                             Private Helpers                              ||
  //==========================================================================||

  //==========================================================================||
  //                           Templates and Styles                           ||
  //==========================================================================||
  render() {
    return html`
      <div class="playground">
        <div class="container container-input">
          <mememo-query-box
            @runButtonClicked=${(e: CustomEvent<string>) =>
              this.userQueryRunClickHandler(e)}
          ></mememo-query-box>
        </div>

        <div class="container container-search">
          <div class="search-box">MeMemo Search</div>
        </div>

        <div class="container container-text">
          <mememo-text-viewer
            dataURL=${datasets['arxiv'].dataURL}
            indexURL=${datasets['arxiv'].indexURL}
            datasetName=${datasets['arxiv'].datasetName}
            datasetNameDisplay=${datasets['arxiv'].datasetNameDisplay}
          ></mememo-text-viewer>
        </div>

        <div class="container container-prompt">Prompt</div>

        <div class="container container-model">
          <div class="model-box">GPT 3.5</div>
        </div>

        <div class="container container-output">Output</div>

        <div class="flow horizontal-flow input-text">
          <div class="background">
            <span class="line-loader hidden"></span>
            <div class="start-rectangle"></div>
            <div class="end-triangle"></div>
          </div>
        </div>

        <div class="flow horizontal-flow text-prompt">
          <div class="background">
            <span class="line-loader hidden"></span>
            <div class="start-rectangle"></div>
            <div class="end-triangle"></div>
          </div>
        </div>

        <div class="flow vertical-flow input-prompt">
          <div class="background">
            <span class="line-loader hidden"></span>
            <div class="start-rectangle"></div>
            <div class="end-triangle"></div>
          </div>
        </div>

        <div class="flow vertical-flow prompt-output">
          <div class="background">
            <span class="line-loader hidden"></span>
            <div class="start-rectangle"></div>
            <div class="end-triangle"></div>
          </div>
        </div>
      </div>
    `;
  }

  static styles = [
    css`
      ${unsafeCSS(componentCSS)}
    `
  ];
}

declare global {
  interface HTMLElementTagNameMap {
    'mememo-playground': MememoPlayground;
  }
}
