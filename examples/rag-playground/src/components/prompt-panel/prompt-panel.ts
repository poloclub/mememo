import { LitElement, css, unsafeCSS, html, PropertyValues } from 'lit';
import { customElement, property, state, query } from 'lit/decorators.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';
import { EmbeddingModel } from '../../workers/embedding';

import type { EmbeddingWorkerMessage } from '../../workers/embedding';

import componentCSS from './prompt-panel.css?inline';
import EmbeddingWorkerInline from '../../workers/embedding?worker&inline';

/**
 * Prompt panel element.
 *
 */
@customElement('mememo-prompt-panel')
export class MememoPromptPanel extends LitElement {
  //==========================================================================||
  //                              Class Properties                            ||
  //==========================================================================||
  embeddingWorker: Worker;

  embeddingWorkerRequestCount = 0;

  get embeddingWorkerRequestID() {
    this.embeddingWorkerRequestCount++;
    return `prompt-panel-${this.embeddingWorkerRequestCount}`;
  }

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

  firstUpdated() {
    this.getEmbedding();
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

  //==========================================================================||
  //                              Event Handlers                              ||
  //==========================================================================||

  embeddingWorkerMessageHandler(e: MessageEvent<EmbeddingWorkerMessage>) {
    switch (e.data.command) {
      case 'finishExtractEmbedding': {
        const embeddings = e.data.payload.embeddings;
        console.log(embeddings);
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
    return html` <div class="prompt-panel">Prompt panel</div> `;
  }

  static styles = [
    css`
      ${unsafeCSS(componentCSS)}
    `
  ];
}

declare global {
  interface HTMLElementTagNameMap {
    'mememo-prompt-panel': MememoPromptPanel;
  }
}
