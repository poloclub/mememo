import { LitElement, css, unsafeCSS, html, PropertyValues } from 'lit';
import { customElement, property, state, query } from 'lit/decorators.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';

import componentCSS from './query-box.css?inline';
import searchIcon from '../../images/icon-search.svg?raw';
import refreshIcon from '../../images/icon-refresh2.svg?raw';
import playIcon from '../../images/icon-play.svg?raw';

/**
 * Query box element.
 *
 */
@customElement('mememo-query-box')
export class MememoQueryBox extends LitElement {
  //==========================================================================||
  //                              Class Properties                            ||
  //==========================================================================||
  userQuery: string;
  defaultQuery: string;

  //==========================================================================||
  //                             Lifecycle Methods                            ||
  //==========================================================================||
  constructor() {
    super();
    this.defaultQuery =
      'What are some ways to integrate information retrieval into machine learning?';
    this.userQuery = this.defaultQuery;
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

  setTextareaMaxHeight(maxHeight: number) {
    if (!this.shadowRoot) return;
    const textarea = this.shadowRoot.querySelector('textarea') as HTMLElement;
    textarea.style.setProperty('max-height', `${maxHeight - 50}px`);
  }

  //==========================================================================||
  //                              Event Handlers                              ||
  //==========================================================================||
  textareaInput(e: InputEvent) {
    const textareaElement = e.currentTarget as HTMLTextAreaElement;
    this.userQuery = textareaElement.value;
  }

  runButtonClicked() {
    // Notify the parent to run the user query
    const event = new CustomEvent('runButtonClicked', {
      bubbles: true,
      composed: true,
      detail: this.userQuery
    });
    this.dispatchEvent(event);
  }

  //==========================================================================||
  //                             Private Helpers                              ||
  //==========================================================================||

  //==========================================================================||
  //                           Templates and Styles                           ||
  //==========================================================================||
  render() {
    return html`
      <div class="query-box">
        <div class="header">
          <span class="text">User Query</span>

          <div class="button-group">
            <button>
              <span class="svg-icon">${unsafeHTML(refreshIcon)}</span>
              random
            </button>

            <button @click=${() => this.runButtonClicked()}>
              <span class="svg-icon">${unsafeHTML(playIcon)}</span>
              run
            </button>
          </div>
        </div>
        <textarea rows="2" @input=${(e: InputEvent) => this.textareaInput(e)}>
${this.userQuery}</textarea
        >
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
    'mememo-query-box': MememoQueryBox;
  }
}
