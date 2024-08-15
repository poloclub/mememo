import { LitElement, css, unsafeCSS, html, PropertyValues } from 'lit';
import { customElement, property, state, query } from 'lit/decorators.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';
import { ifDefined } from 'lit/directives/if-defined.js';
import { EmbeddingModel } from '../../workers/embedding';
import d3 from '../../utils/d3-import';
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
import { textGenDeepSeek } from '../../llms/deepseek';
import { promptTemplates } from '../../config/promptTemplates';

import type { TextGenMessage } from '../../llms/gpt';
import type { EmbeddingWorkerMessage } from '../../workers/embedding';
import type { MememoTextViewer } from '../text-viewer/text-viewer';
import type { MememoPromptBox } from '../prompt-box/prompt-box';
import type { MememoQueryBox } from '../query-box/query-box';
import type { TextGenLocalWorkerMessage } from '../../llms/web-llm';
import type { NightjarToast } from '../toast/toast';

import '../query-box/query-box';
import '../prompt-box/prompt-box';
import '../output-box/output-box';
import '../text-viewer/text-viewer';
import '../panel-setting/panel-setting';
import '../toast/toast';

import componentCSS from './playground.css?inline';
import TextGenLocalWorkerInline from '../../llms/web-llm?worker&inline';
import EmbeddingWorkerInline from '../../workers/embedding?worker';
import gearIcon from '../../images/icon-gear.svg?raw';

const REMOTE_ENDPOINT = 'https://pub-4eccf317e01e4aa3a5caa9991c8b1e2a.r2.dev/';
const STORE_ENDPOINT = import.meta.env.DEV ? '/data/' : REMOTE_ENDPOINT;

interface DatasetInfo {
  dataURL: string;
  indexURL?: string;
  datasetName: string;
  datasetNameDisplay: string;
}

export enum Dataset {
  arXiv1k = 'arxiv-1k',
  arXiv10k = 'arxiv-10k',
  arXiv120k = 'arxiv-120k',
  DiffusionDB10k = 'diffusiondb-10k',
  DiffusionDB100k = 'diffusiondb-100k',
  DiffusionDB500k = 'diffusiondb-500k',
  DiffusionDB1m = 'diffusiondb-1m',
  accident3k = 'accident-3k'
}

enum Arrow {
  Search = 'search',
  Input = 'input',
  Document = 'document',
  Output = 'output'
}

const promptTemplate = promptTemplates as Record<Dataset, string>;

const datasets: Record<Dataset, DatasetInfo> = {
  [Dataset.arXiv1k]: {
    indexURL: STORE_ENDPOINT + 'ml-arxiv-papers-index-1k.json.gzip',
    dataURL: STORE_ENDPOINT + 'ml-arxiv-papers-1k.ndjson.gzip',
    datasetName: 'ml-arxiv-papers-1k',
    datasetNameDisplay: 'ML arXiv Abstracts (1k)'
  },

  [Dataset.arXiv10k]: {
    indexURL: STORE_ENDPOINT + 'ml-arxiv-papers-index-10k.json.gzip',
    dataURL: STORE_ENDPOINT + 'ml-arxiv-papers-10k.ndjson.gzip',
    datasetName: 'ml-arxiv-papers-10k',
    datasetNameDisplay: 'ML arXiv Abstracts (10k)'
  },

  [Dataset.arXiv120k]: {
    indexURL: REMOTE_ENDPOINT + 'ml-arxiv-papers-index-120k.json.gzip',
    dataURL: REMOTE_ENDPOINT + 'ml-arxiv-papers-120k.ndjson.gzip',
    datasetName: 'ml-arxiv-papers-120k',
    datasetNameDisplay: 'ML arXiv Abstracts (120k)'
  },

  [Dataset.DiffusionDB10k]: {
    indexURL: REMOTE_ENDPOINT + 'diffusiondb-prompt-index-10k.json.gzip',
    dataURL: REMOTE_ENDPOINT + 'diffusiondb-prompt-10k.ndjson.gzip',
    datasetName: 'diffusiondb-prompts-10k',
    datasetNameDisplay: 'DiffusionDB Prompts (10k)'
  },

  [Dataset.DiffusionDB100k]: {
    indexURL: REMOTE_ENDPOINT + 'diffusiondb-prompt-index-100k.json.gzip',
    dataURL: REMOTE_ENDPOINT + 'diffusiondb-prompt-100k.ndjson.gzip',
    datasetName: 'diffusiondb-prompts-100k',
    datasetNameDisplay: 'DiffusionDB Prompts (100k)'
  },

  [Dataset.DiffusionDB500k]: {
    indexURL: REMOTE_ENDPOINT + 'diffusiondb-prompt-index-500k.json.gzip',
    dataURL: REMOTE_ENDPOINT + 'diffusiondb-prompt-500k.ndjson.gzip',
    datasetName: 'diffusiondb-prompts-500k',
    datasetNameDisplay: 'DiffusionDB Prompts (500k)'
  },

  [Dataset.DiffusionDB1m]: {
    indexURL: REMOTE_ENDPOINT + 'diffusiondb-prompt-index-1m.json.gzip',
    dataURL: REMOTE_ENDPOINT + 'diffusiondb-prompt-1m.ndjson.gzip',
    datasetName: 'diffusiondb-prompts-1m',
    datasetNameDisplay: 'DiffusionDB Prompts (1M)'
  },

  [Dataset.accident3k]: {
    indexURL: STORE_ENDPOINT + 'accident-index-3k.json.gzip',
    dataURL: STORE_ENDPOINT + 'accident-3k.ndjson.gzip',
    datasetName: 'accidents-3k',
    datasetNameDisplay: 'AI Accidents (3k)'
  }
};

const DEV_MODE = import.meta.env.DEV;
const USE_CACHE = false && DEV_MODE;
const FORMATTER = d3.format(',');

/**
 * Playground element.
 *
 */
@customElement('mememo-playground')
export class MememoPlayground extends LitElement {
  //==========================================================================||
  //                              Class Properties                            ||
  //==========================================================================||
  @property()
  curDataset!: Dataset;

  @state()
  userQuery = '';

  @state()
  relevantDocuments: string[] = [];

  @state()
  llmOutput = '';

  embeddingWorker: Worker;
  embeddingWorkerRequestCount = 0;
  get embeddingWorkerRequestID() {
    this.embeddingWorkerRequestCount++;
    return `prompt-panel-${this.embeddingWorkerRequestCount}`;
  }

  @state()
  topK = 5;

  @state()
  efSearch = 100;

  @state()
  isSearching = false;

  @state()
  searchRunTime: [number, number] | null = null;
  searchStartTime = 0;
  encodeFinishTime = 0;

  @state()
  isRunningLLM = false;

  @state()
  LLMRunTime: number | null = null;
  LLMStartTime = 0;

  @query('mememo-text-viewer')
  textViewerComponent: MememoTextViewer | undefined | null;

  @query('mememo-query-box')
  queryBoxComponent: MememoQueryBox | undefined | null;

  @query('mememo-prompt-box')
  promptBoxComponent: MememoPromptBox | undefined | null;

  @query('.container-input')
  containerInputElement: HTMLElement | undefined | null;

  @query('.container-prompt')
  containerPromptElement: HTMLElement | undefined | null;

  @query('.container-output')
  containerOutputElement: HTMLElement | undefined | null;

  @query('dialog')
  dialogElement: HTMLDialogElement | undefined;

  @state()
  userConfigManager: UserConfigManager;

  @state()
  userConfig!: UserConfig;

  @state()
  toastMessage = '';

  @state()
  toastType: 'success' | 'warning' | 'error' = 'success';

  @query('nightjar-toast')
  toastComponent: NightjarToast | undefined | null;

  @property({ attribute: false })
  textGenLocalWorker: Worker;
  textGenLocalWorkerResolve = (
    value: TextGenMessage | PromiseLike<TextGenMessage>
  ) => {};

  arrowElements: Record<Arrow, HTMLElement | null> = {
    [Arrow.Document]: null,
    [Arrow.Input]: null,
    [Arrow.Search]: null,
    [Arrow.Output]: null
  };

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
    this.textGenLocalWorker.addEventListener(
      'message',
      (e: MessageEvent<TextGenMessage>) => {
        this.textGenLocalWorkerMessageHandler(e);
      }
    );

    // Set up the user config store
    const updateUserConfig = (userConfig: UserConfig) => {
      this.userConfig = userConfig;
    };
    this.userConfigManager = new UserConfigManager(updateUserConfig);
  }

  firstUpdated() {
    if (!this.shadowRoot) {
      throw Error('No shadow root.');
    }

    // Track the arrow elements
    this.arrowElements = {
      [Arrow.Document]: this.shadowRoot.querySelector(
        '.flow.text-prompt'
      ) as HTMLElement,
      [Arrow.Input]: this.shadowRoot.querySelector(
        '.flow.input-prompt'
      ) as HTMLElement,
      [Arrow.Search]: this.shadowRoot.querySelector(
        '.flow.input-text'
      ) as HTMLElement,
      [Arrow.Output]: this.shadowRoot.querySelector(
        '.flow.prompt-output'
      ) as HTMLElement
    };
  }

  /**
   * This method is called before new DOM is updated and rendered
   * @param changedProperties Property that has been changed
   */
  willUpdate(changedProperties: PropertyValues<this>) {}

  disconnectedCallback(): void {
    // Need to terminal workers to avoid aysnc write to indexedDB
    this.embeddingWorker.terminate();
    this.textGenLocalWorker.terminate();
    this.textViewerComponent?.mememoWorker.terminate();
  }

  //==========================================================================||
  //                              Custom Methods                              ||
  //==========================================================================||
  async initData() {}

  /**
   * Extract embeddings for the input sentences
   * @param sentences Input sentences
   */
  getEmbedding(sentences: string[]) {
    this.searchRunTime = null;
    this.searchStartTime = Date.now();
    this.isSearching = true;

    const message: EmbeddingWorkerMessage = {
      command: 'startExtractEmbedding',
      payload: {
        detail: '',
        requestID: this.embeddingWorkerRequestID,
        model: EmbeddingModel.gteSmall,
        sentences: sentences,
        windowURL: window.location.href
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

    // Loading the arrow
    this.arrowElements[Arrow.Search]?.classList.add('running');

    this.textViewerComponent.semanticSearch(
      embedding,
      this.topK,
      0.5,
      this.efSearch
    );
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

    // Activate arrows
    this.arrowElements[Arrow.Input]?.classList.add('activated');
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

    this.searchRunTime = [
      this.encodeFinishTime - this.searchStartTime,
      Date.now() - this.searchStartTime
    ];
    this.isSearching = false;

    // Activate arrows
    this.arrowElements[Arrow.Search]?.classList.remove('running');
    this.arrowElements[Arrow.Search]?.classList.add('activated');
    this.arrowElements[Arrow.Document]?.classList.add('activated');
  }

  embeddingWorkerMessageHandler(e: MessageEvent<EmbeddingWorkerMessage>) {
    switch (e.data.command) {
      case 'finishExtractEmbedding': {
        const { embeddings } = e.data.payload;
        // Start semantic search using the embedding
        this.encodeFinishTime = Date.now();
        this.semanticSearch(embeddings[0]);
        break;
      }

      case 'error': {
        console.error('Worker error: ', JSON.stringify(e.data.payload.message));
        break;
      }

      default: {
        console.error('Worker: unknown message', e.data.command);
        break;
      }
    }
  }

  /**
   * Validate input and update mememo search parameters
   * @param e Input event
   * @param parameter Type of the parameter
   */
  parameterInputChanged(e: InputEvent, parameter: 'efSearch' | 'top-k') {
    const element = e.currentTarget as HTMLInputElement;

    if (parameter === 'efSearch') {
      const value = parseInt(element.value);
      if (value > 0) {
        this.efSearch = value;
      } else {
        this.efSearch = 100;
      }
      element.value = String(this.efSearch);
    }

    if (parameter === 'top-k') {
      const value = parseInt(element.value);
      if (value > 0) {
        this.topK = value;
      } else {
        this.topK = 5;
      }
      element.value = String(this.topK);
    }
  }

  /**
   * Event handler for the text gen local worker
   * @param e Text gen message
   */
  textGenLocalWorkerMessageHandler(e: MessageEvent<TextGenLocalWorkerMessage>) {
    switch (e.data.command) {
      case 'finishTextGen': {
        const message: TextGenMessage = {
          command: 'finishTextGen',
          payload: e.data.payload
        };
        this.textGenLocalWorkerResolve(message);
        break;
      }

      case 'progressLoadModel': {
        break;
      }

      case 'finishLoadModel': {
        break;
      }

      case 'error': {
        const message: TextGenMessage = {
          command: 'error',
          payload: e.data.payload
        };
        this.textGenLocalWorkerResolve(message);
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
  _reComputeMaxHeights(curElement: 'input' | 'prompt') {
    if (
      !this.shadowRoot ||
      !this.queryBoxComponent ||
      !this.promptBoxComponent ||
      !this.containerInputElement ||
      !this.containerPromptElement
    )
      throw Error('No shadow root.');

    const playground = this.shadowRoot.querySelector(
      '.playground'
    ) as HTMLElement;

    const playgroundBBox = playground.getBoundingClientRect();
    const outputBBoxMinHeigh = 82;
    const playgroundPadding = 175;

    if (curElement === 'prompt') {
      // The user is resizing prompt
      const userBBox = this.containerInputElement.getBoundingClientRect();
      const curMaxHeight =
        playgroundBBox.height -
        userBBox.height -
        outputBBoxMinHeigh -
        playgroundPadding;
      this.promptBoxComponent!.setTextareaMaxHeight(curMaxHeight);

      const otherMaxHeight =
        playgroundBBox.height -
        curMaxHeight -
        outputBBoxMinHeigh -
        playgroundPadding;
      this.queryBoxComponent!.setTextareaMaxHeight(otherMaxHeight);
    } else if (curElement === 'input') {
      // The user is resizing user
      const promptBBox = this.containerPromptElement.getBoundingClientRect();
      const curMaxHeight =
        playgroundBBox.height -
        promptBBox.height -
        outputBBoxMinHeigh -
        playgroundPadding;
      this.queryBoxComponent!.setTextareaMaxHeight(curMaxHeight);

      const otherMaxHeight =
        playgroundBBox.height -
        curMaxHeight -
        outputBBoxMinHeigh -
        playgroundPadding;
      this.promptBoxComponent!.setTextareaMaxHeight(otherMaxHeight);
    }
  }

  _deactivateAllArrows() {
    for (const key of Object.values(Arrow)) {
      this.arrowElements[key]?.classList.remove('activated');
    }
  }

  /**
   * Run the given prompt using the preferred model
   * @returns A promise of the prompt inference
   */
  _runPrompt(curPrompt: string, temperature = 0.2) {
    let runRequest: Promise<TextGenMessage>;
    this.llmOutput = '';
    this.LLMRunTime = null;
    this.LLMStartTime = Date.now();
    this.isRunningLLM = true;

    // Show a loader
    this.isRunningLLM = true;
    this.arrowElements[Arrow.Output]?.classList.add('running');

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

      case SupportedRemoteModel['deepseek-chat']: {
        runRequest = textGenDeepSeek(
          this.userConfig.llmAPIKeys[ModelFamily.deepseek],
          'text-gen',
          curPrompt,
          temperature,
          'deepseek-chat',
          USE_CACHE
        );
        break;
      }

      case SupportedRemoteModel['deepseek-coder']: {
        runRequest = textGenDeepSeek(
          this.userConfig.llmAPIKeys[ModelFamily.deepseek],
          'text-gen',
          curPrompt,
          temperature,
          'deepseek-coder',
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
      case SupportedLocalModel['gemma-2b']:
      case SupportedLocalModel['phi-2']:
      case SupportedLocalModel['phi-3']:
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

            // await new Promise<void>(resolve => {
            //   setTimeout(resolve, 5000);
            // });

            this.llmOutput = message.payload.result;
            this.LLMRunTime = Date.now() - this.LLMStartTime;
            this.isRunningLLM = false;

            // Activate arrows
            this.arrowElements[Arrow.Output]?.classList.remove('running');
            this.arrowElements[Arrow.Output]?.classList.add('activated');
            break;
          }

          case 'error': {
            console.error(message.payload.message);
            this.toastMessage =
              'Fail to run this LLM. Try it again later or use a different model.';
            this.toastType = 'error';
            this.toastComponent?.show();

            this.LLMRunTime = null;
            this.isRunningLLM = false;

            // Activate arrows
            this.arrowElements[Arrow.Output]?.classList.remove('running');
          }
        }
      },
      () => {}
    );
  }

  dialogClicked(e: MouseEvent) {
    if (e.target === this.dialogElement) {
      this.dialogElement.close();
    }
  }

  //==========================================================================||
  //                           Templates and Styles                           ||
  //==========================================================================||
  render() {
    return html`
      <div class="playground">
        <div class="toast-container">
          <nightjar-toast
            message=${this.toastMessage}
            type=${this.toastType}
          ></nightjar-toast>
        </div>

        <div class="container container-input">
          <mememo-query-box
            curDataset=${this.curDataset}
            @runButtonClicked=${(e: CustomEvent<string>) =>
              this.userQueryRunClickHandler(e)}
            @queryEdited=${() => this._deactivateAllArrows()}
            @needUpdateMaxHeight=${() => this._reComputeMaxHeights('input')}
          ></mememo-query-box>
        </div>

        <div class="container container-search">
          <div class="search-box">
            <div class="search-time-info" ?is-hidden=${true}>
              <span class="row"
                >encode:
                ${this.searchRunTime
                  ? FORMATTER(this.searchRunTime[0])
                  : ''}ms</span
              >
              <span class="row"
                >retrieval:
                ${this.searchRunTime
                  ? FORMATTER(this.searchRunTime[1])
                  : ''}ms</span
              >
            </div>

            <div class="search-loader" ?is-hidden=${!this.isSearching}>
              <div class=" loader-container">
                <div class="circle-loader"></div>
              </div>
            </div>

            <div class="search-top-info">
              <span class="row"
                >ef-search =
                <input
                  id="input-efSearch"
                  type="text"
                  value="100"
                  @change=${(e: InputEvent) =>
                    this.parameterInputChanged(e, 'efSearch')}
              /></span>
              <span class="row"
                >top-k =
                <input
                  id="input-top-k"
                  type="text"
                  value="5"
                  @change=${(e: InputEvent) =>
                    this.parameterInputChanged(e, 'top-k')}
              /></span>
            </div>
            <div class="header">MeMemo Search</div>
          </div>
        </div>

        <div class="container container-text">
          <mememo-text-viewer
            dataURL=${datasets[this.curDataset].dataURL}
            indexURL=${ifDefined(datasets[this.curDataset].indexURL)}
            datasetName=${datasets[this.curDataset].datasetName}
            datasetNameDisplay=${datasets[this.curDataset].datasetNameDisplay}
            @semanticSearchFinished=${(e: CustomEvent<string[]>) =>
              this.semanticSearchFinishedHandler(e)}
          ></mememo-text-viewer>
        </div>

        <div class="container container-prompt">
          <mememo-prompt-box
            template=${promptTemplate[this.curDataset]}
            userQuery=${this.userQuery}
            .relevantDocuments=${this.relevantDocuments}
            @runButtonClicked=${(e: CustomEvent<string>) => {
              this.promptRunClickHandler(e);
            }}
            @promptEdited=${() => this._deactivateAllArrows()}
            @needUpdateMaxHeight=${() => this._reComputeMaxHeights('prompt')}
          ></mememo-prompt-box>
        </div>

        <div class="container container-model">
          <div class="model-box">
            <div class="header">${this.userConfig.preferredLLM}</div>
            <div class="button-group">
              <button
                @click=${() => {
                  this.dialogElement?.showModal();
                }}
              >
                <span class="svg-icon">${unsafeHTML(gearIcon)}</span>
                change
              </button>
            </div>

            <div class="model-loader" ?is-hidden=${!this.isRunningLLM}>
              <div class=" loader-container">
                <div class="circle-loader"></div>
              </div>
            </div>

            <div class="model-time-info" ?is-hidden=${true}>
              ${this.LLMRunTime === null ? '' : FORMATTER(this.LLMRunTime)}ms
            </div>
          </div>
        </div>

        <div class="container container-output">
          <mememo-output-box llmOutput=${this.llmOutput}></mememo-output-box>
        </div>

        <div class="flow horizontal-flow user input-text">
          <div class="background">
            <span class="line-loader"></span>
            <div class="start-rectangle"></div>
            <div class="end-triangle"></div>
          </div>
        </div>

        <div class="flow horizontal-flow context text-prompt">
          <div class="background">
            <span class="line-loader"></span>
            <div class="start-rectangle"></div>
            <div class="end-triangle"></div>
          </div>
        </div>

        <div class="flow vertical-flow user input-prompt">
          <div class="background">
            <span class="line-loader"></span>
            <div class="start-rectangle"></div>
            <div class="end-triangle"></div>
          </div>
        </div>

        <div class="flow vertical-flow user-context prompt-output">
          <div class="background">
            <span class="line-loader"></span>
            <div class="start-rectangle"></div>
            <div class="end-triangle"></div>
          </div>
        </div>

        <dialog
          class="setting-dialog"
          @click=${(e: MouseEvent) => this.dialogClicked(e)}
        >
          <mememo-panel-setting
            .userConfigManager=${this.userConfigManager}
            .userConfig=${this.userConfig}
            .textGenLocalWorker=${this.textGenLocalWorker}
            ?is-shown=${true}
            @closeClicked=${() => {
              this.dialogElement?.close();
            }}
          ></mememo-panel-setting>
        </dialog>
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
