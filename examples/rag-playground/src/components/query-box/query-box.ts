import { LitElement, css, unsafeCSS, html, PropertyValues } from 'lit';
import { customElement, property, state, query } from 'lit/decorators.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';
import { Dataset } from '../playground/playground';
import { userQueries } from '../../config/userQueries';
import d3 from '../../utils/d3-import';

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
  @property({ type: String })
  curDataset: Dataset | undefined;

  @state()
  userQuery: string;

  //==========================================================================||
  //                             Lifecycle Methods                            ||
  //==========================================================================||
  constructor() {
    super();
    this.userQuery = this.defaultQuery;
  }

  firstUpdated() {
    if (!this.shadowRoot) {
      throw Error('no shadow root!');
    }

    // Tell parent to recompute the max size of different elements
    const textareaElement = this.shadowRoot.querySelector('textarea');
    textareaElement?.addEventListener('mousedown', () => {
      const event = new Event('needUpdateMaxHeight', {
        bubbles: true,
        composed: true
      });
      this.dispatchEvent(event);
    });
  }

  /**
   * This method is called before new DOM is updated and rendered
   * @param changedProperties Property that has been changed
   */
  willUpdate(changedProperties: PropertyValues<this>) {
    if (changedProperties.has('curDataset') && this.curDataset) {
      this.userQuery = userQueries[this.curDataset][0];
    }
  }

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
  randomButtonClicked() {
    if (this.curDataset) {
      const allQueries = userQueries[this.curDataset];
      const i = d3.randomInt(allQueries.length)();
      this.userQuery = allQueries[i];

      // Notify the parent
      const event = new Event('queryEdited', {
        bubbles: true,
        composed: true
      });
      this.dispatchEvent(event);
    }
  }

  textareaInput(e: InputEvent) {
    const textareaElement = e.currentTarget as HTMLTextAreaElement;
    this.userQuery = textareaElement.value;

    // Notify the parent
    const event = new Event('queryEdited', {
      bubbles: true,
      composed: true
    });
    this.dispatchEvent(event);
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
            <button @click=${() => this.randomButtonClicked()}>
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
