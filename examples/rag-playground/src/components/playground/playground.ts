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
import type { MememoPromptBox } from '../prompt-box/prompt-box';
import type { MememoQueryBox } from '../query-box/query-box';
import type { TextGenLocalWorkerMessage } from '../../llms/web-llm';

import '../query-box/query-box';
import '../prompt-box/prompt-box';
import '../output-box/output-box';
import '../text-viewer/text-viewer';

import componentCSS from './playground.css?inline';
import EmbeddingWorkerInline from '../../workers/embedding?worker&inline';
import promptTemplatesJSON from '../../config/promptTemplates.json';
import logoIcon from '../../images/icon-logo.svg?raw';

interface DatasetInfo {
  dataURL: string;
  indexURL: string;
  datasetName: string;
  datasetNameDisplay: string;
}

enum Dataset {
  Arxiv = 'arxiv'
}

enum Arrow {
  Search = 'search',
  Input = 'input',
  Document = 'document',
  Output = 'output'
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

  @state()
  llmOutput = '';

  embeddingWorker: Worker;
  embeddingWorkerRequestCount = 0;
  get embeddingWorkerRequestID() {
    this.embeddingWorkerRequestCount++;
    return `prompt-panel-${this.embeddingWorkerRequestCount}`;
  }

  @state()
  topK = 10;

  @state()
  maxDistance = 0.25;

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

  @state()
  userConfigManager: UserConfigManager;

  @state()
  userConfig!: UserConfig;

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

    // Loading the arrow
    this.arrowElements[Arrow.Search]?.classList.add('running');

    this.textViewerComponent.semanticSearch(embedding, this.topK, 0.5);
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

  /**
   * Validate input and update mememo search parameters
   * @param e Input event
   * @param parameter Type of the parameter
   */
  parameterInputChanged(e: InputEvent, parameter: 'distance' | 'top-k') {
    const element = e.currentTarget as HTMLInputElement;

    if (parameter === 'distance') {
      const value = parseFloat(element.value);
      if (value > 0 && value < 1) {
        this.maxDistance = value;
      } else {
        this.maxDistance = 0.25;
      }
      element.value = String(this.maxDistance);
    }

    if (parameter === 'top-k') {
      const value = parseInt(element.value);
      if (value > 0) {
        this.topK = value;
      } else {
        this.topK = 10;
      }
      element.value = String(this.topK);
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

    // Show a loader
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

            // await new Promise<void>(resolve => {
            //   setTimeout(resolve, 5000);
            // });

            this.llmOutput = message.payload.result;

            // Activate arrows
            this.arrowElements[Arrow.Output]?.classList.remove('running');
            this.arrowElements[Arrow.Output]?.classList.add('activated');
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
            @queryEdited=${() => this._deactivateAllArrows()}
            @needUpdateMaxHeight=${() => this._reComputeMaxHeights('input')}
          ></mememo-query-box>
        </div>

        <div class="container container-search">
          <div class="search-box">
            <div class="search-top-info">
              <span class="row"
                >distance &lt;
                <input
                  id="input-distance"
                  type="text"
                  value="0.25"
                  @change=${(e: InputEvent) =>
                    this.parameterInputChanged(e, 'distance')}
              /></span>
              <span class="row"
                >top-k =
                <input
                  id="input-top-k"
                  type="text"
                  value="10"
                  @change=${(e: InputEvent) =>
                    this.parameterInputChanged(e, 'top-k')}
              /></span>
            </div>
            <div class="header">MeMemo Search</div>
          </div>
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
            @promptEdited=${() => this._deactivateAllArrows()}
            @needUpdateMaxHeight=${() => this._reComputeMaxHeights('prompt')}
          ></mememo-prompt-box>
        </div>

        <div class="container container-model">
          <div class="model-box">
            <div class="header">GPT 3.5</div>
          </div>
        </div>

        <div class="container container-output">
          <mememo-output-box llmOutput=${this.llmOutput}></mememo-output-box>
        </div>

        <div class="flow horizontal-flow input-text">
          <div class="background">
            <span class="line-loader"></span>
            <div class="start-rectangle"></div>
            <div class="end-triangle"></div>
          </div>
        </div>

        <div class="flow horizontal-flow text-prompt">
          <div class="background">
            <span class="line-loader"></span>
            <div class="start-rectangle"></div>
            <div class="end-triangle"></div>
          </div>
        </div>

        <div class="flow vertical-flow input-prompt">
          <div class="background">
            <span class="line-loader"></span>
            <div class="start-rectangle"></div>
            <div class="end-triangle"></div>
          </div>
        </div>

        <div class="flow vertical-flow prompt-output">
          <div class="background">
            <span class="line-loader"></span>
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
