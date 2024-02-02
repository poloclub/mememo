import { LitElement, css, unsafeCSS, html, PropertyValues } from 'lit';
import { customElement, property, state, query } from 'lit/decorators.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';
import { pipeline } from '@xenova/transformers';

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

  //==========================================================================||
  //                             Lifecycle Methods                            ||
  //==========================================================================||
  constructor() {
    super();
    this.embeddingWorker = new EmbeddingWorkerInline();
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

  async getEmbedding() {}

  //==========================================================================||
  //                              Event Handlers                              ||
  //==========================================================================||

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
