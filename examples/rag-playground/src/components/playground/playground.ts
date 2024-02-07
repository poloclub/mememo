import { LitElement, css, unsafeCSS, html, PropertyValues } from 'lit';
import { customElement, property, state, query } from 'lit/decorators.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';
import { EmbeddingModel } from '../../workers/embedding';
import {
  UserConfigManager,
  UserConfig,
  SupportedRemoteModel,
  SupportedLocalModel,
  supportedModelReverseLookup,
  ModelFamily
} from './user-config';
import { textGenGpt } from '../../llms/gpt';
import { textGenMememo } from '../../llms/mememo-gen';
import { textGenGemini } from '../../llms/gemini';
import TextGenLocalWorkerInline from '../../llms/web-llm?worker&inline';

import type { TextGenMessage } from '../../llms/gpt';
import type { EmbeddingWorkerMessage } from '../../workers/embedding';
import type { MememoTextViewer } from '../text-viewer/text-viewer';
import type { TextGenLocalWorkerMessage } from '../../llms/web-llm';

import '../query-box/query-box';
import '../prompt-box/prompt-box';
import '../text-viewer/text-viewer';

import componentCSS from './playground.css?inline';
import EmbeddingWorkerInline from '../../workers/embedding?worker&inline';
import promptTemplatesJSON from '../../config/promptTemplates.json';

interface DatasetInfo {
  dataURL: string;
  indexURL: string;
  datasetName: string;
  datasetNameDisplay: string;
}

enum Dataset {
  Arxiv = 'arxiv'
}

const promptTemplate = promptTemplatesJSON as Record<Dataset, string>;

const datasets: Record<Dataset, DatasetInfo> = {
  [Dataset.Arxiv]: {
    dataURL: '/data/ml-arxiv-papers-1000.ndjson.gzip',
    indexURL: '/data/ml-arxiv-papers-1000-index.json',
    datasetName: 'ml-arxiv-papers',
    datasetNameDisplay: 'ML arXiv Abstracts (1k)'
  }
};

const DEV_MODE = import.meta.env.DEV;
const USE_CACHE = true && DEV_MODE;

/**
 * Playground element.
 *
 */
@customElement('mememo-playground')
export class MememoPlayground extends LitElement {
  //==========================================================================||
  //                              Class Properties                            ||
  //==========================================================================||
  @state()
  userQuery = '';

  @state()
  relevantDocuments: string[] = [];

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

  @state()
  userConfigManager: UserConfigManager;

  @state()
  userConfig!: UserConfig;

  @property({ attribute: false })
  textGenLocalWorker: Worker;
  textGenLocalWorkerResolve = (
    value: TextGenMessage | PromiseLike<TextGenMessage>
  ) => {};

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

    // Initialize the local llm worker
    this.textGenLocalWorker = new TextGenLocalWorkerInline();

    // Set up the user config store
    const updateUserConfig = (userConfig: UserConfig) => {
      this.userConfig = userConfig;
    };
    this.userConfigManager = new UserConfigManager(updateUserConfig);
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
  /**
   * Start extracting embeddings form the user query
   * @param e Event
   */
  userQueryRunClickHandler(e: CustomEvent<string>) {
    this.userQuery = e.detail;

    // Extract embeddings for the user query
    this.getEmbedding([this.userQuery]);
  }

  /**
   * Run the prompt using external AI services or local LLM
   * @param e Event
   */
  promptRunClickHandler(e: CustomEvent<string>) {
    const prompt = e.detail;

    // Run the prompt
    this._runPrompt(prompt);
  }

  semanticSearchFinishedHandler(e: CustomEvent<string[]>) {
    this.relevantDocuments = e.detail;
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
  /**
   * Run the given prompt using the preferred model
   * @returns A promise of the prompt inference
   */
  _runPrompt(curPrompt: string, temperature = 0.2) {
    let runRequest: Promise<TextGenMessage>;

    switch (this.userConfig.preferredLLM) {
      case SupportedRemoteModel['gpt-3.5']: {
        runRequest = textGenGpt(
          this.userConfig.llmAPIKeys[ModelFamily.openAI],
          'text-gen',
          curPrompt,
          temperature,
          'gpt-3.5-turbo',
          USE_CACHE
        );
        break;
      }

      case SupportedRemoteModel['gpt-4']: {
        runRequest = textGenGpt(
          this.userConfig.llmAPIKeys[ModelFamily.openAI],
          'text-gen',
          curPrompt,
          temperature,
          'gpt-4-1106-preview',
          USE_CACHE
        );
        break;
      }

      case SupportedRemoteModel['gemini-pro']: {
        runRequest = textGenGemini(
          this.userConfig.llmAPIKeys[ModelFamily.google],
          'text-gen',
          curPrompt,
          temperature,
          USE_CACHE
        );
        break;
      }

      // case SupportedLocalModel['mistral-7b-v0.2']:
      // case SupportedLocalModel['gpt-2']:
      case SupportedLocalModel['phi-2']:
      case SupportedLocalModel['llama-2-7b']:
      case SupportedLocalModel['tinyllama-1.1b']: {
        runRequest = new Promise<TextGenMessage>(resolve => {
          this.textGenLocalWorkerResolve = resolve;
        });
        const message: TextGenLocalWorkerMessage = {
          command: 'startTextGen',
          payload: {
            apiKey: '',
            prompt: curPrompt,
            requestID: '',
            temperature: temperature
          }
        };
        this.textGenLocalWorker.postMessage(message);
        break;
      }

      case SupportedRemoteModel['gpt-3.5-free']: {
        runRequest = textGenMememo(
          'text-gen',
          curPrompt,
          temperature,
          'gpt-3.5-free',
          USE_CACHE
        );
        break;
      }

      default: {
        console.error('Unknown case ', this.userConfig.preferredLLM);
        runRequest = textGenMememo(
          'text-gen',
          curPrompt,
          temperature,
          'gpt-3.5-free',
          USE_CACHE
        );
      }
    }

    runRequest.then(
      message => {
        switch (message.command) {
          case 'finishTextGen': {
            // Success
            if (DEV_MODE) {
              console.info(
                `Finished running prompt with [${this.userConfig.preferredLLM}]`
              );
              console.info(message.payload.result);
            }

            const output = message.payload.result;
            break;
          }

          case 'error': {
            console.error(message.payload.message);
          }
        }
      },
      () => {}
    );
  }

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
            @semanticSearchFinished=${(e: CustomEvent<string[]>) =>
              this.semanticSearchFinishedHandler(e)}
          ></mememo-text-viewer>
        </div>

        <div class="container container-prompt">
          <mememo-prompt-box
            template=${promptTemplate[Dataset.Arxiv]}
            userQuery=${this.userQuery}
            .relevantDocuments=${this.relevantDocuments}
            @runButtonClicked=${(e: CustomEvent<string>) => {
              this.promptRunClickHandler(e);
            }}
          ></mememo-prompt-box>
        </div>

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
