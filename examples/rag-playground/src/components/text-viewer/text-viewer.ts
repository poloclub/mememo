import { LitElement, css, unsafeCSS, html, PropertyValues } from 'lit';
import { customElement, property, state, query } from 'lit/decorators.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';

import type { MememoWorkerMessage } from '../../workers/mememo-worker';

// Assets
import componentCSS from './text-viewer.css?inline';
import searchIcon from '../../images/icon-search.svg?raw';
import crossIcon from '../../images/icon-cross-thick.svg?raw';
import crossSmallIcon from '../../images/icon-cross.svg?raw';

import MememoWorkerInline from '../../workers/mememo-worker?worker&inline';
import paperDataJSON from '../../../notebooks/ml-arxiv-papers-1000.json';
const paperData = paperDataJSON as string[];

/**
 * Text viewer element.
 */
@customElement('mememo-text-viewer')
export class MememoTextViewer extends LitElement {
  //==========================================================================||
  //                              Class Properties                            ||
  //==========================================================================||
  @state()
  clickedItemIndexes: number[] = [];

  @state()
  shownItems: string[] = paperData.slice(0, 100);

  @state()
  showSearchBarCancelButton = false;

  loaderWorker: Worker;

  //==========================================================================||
  //                             Lifecycle Methods                            ||
  //==========================================================================||
  constructor() {
    super();

    this.loaderWorker = new MememoWorkerInline();
    this.loaderWorker.addEventListener(
      'message',
      (e: MessageEvent<MememoWorkerMessage>) =>
        this.loaderWorkerMessageHandler(e)
    );

    const message: MememoWorkerMessage = {
      command: 'startLoadData',
      payload: {
        url: '/data/ml-arxiv-papers-1000.ndjson'
      }
    };
    this.loaderWorker.postMessage(message);
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

  //==========================================================================||
  //                              Event Handlers                              ||
  //==========================================================================||
  searchBarEntered(e: InputEvent) {}

  showSearchBarCancelButtonClicked() {}

  loaderWorkerMessageHandler(e: MessageEvent<MememoWorkerMessage>) {
    switch (e.data.command) {
      case 'transferLoadData': {
        console.log(e.data.payload);
        break;
      }

      default: {
        console.error(`Unknown command ${e.data.command}`);
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
    // Compile the item list
    let items = html``;
    for (const [i, text] of this.shownItems.entries()) {
      items = html`${items}
        <div
          class="item"
          ?clamp-line=${!this.clickedItemIndexes.includes(i)}
          @click=${() => {
            if (this.clickedItemIndexes.includes(i)) {
              this.clickedItemIndexes = this.clickedItemIndexes.filter(
                d => d !== i
              );
            } else {
              this.clickedItemIndexes = [...this.clickedItemIndexes, i];
            }
          }}
        >
          ${text}
        </div> `;
    }

    return html`
      <div class="text-viewer">
        <div class="header-bar">
          <div class="header">MeMemo Database</div>
          <div class="description">1000 arXiv abstracts</div>
        </div>

        <div class="search-bar-container">
          <div class="search-bar">
            <span class="icon-container">
              <span class="svg-icon search">${unsafeHTML(searchIcon)}</span>
            </span>

            <input
              id="search-bar-input"
              type="text"
              name="search-bar-input"
              @input=${(e: InputEvent) => this.searchBarEntered(e)}
              placeholder="Search local documents"
            />

            <span
              class="icon-container"
              @click=${() => this.showSearchBarCancelButtonClicked()}
              ?is-hidden=${!this.showSearchBarCancelButton}
            >
              <span class="svg-icon cross">${unsafeHTML(crossIcon)}</span>
            </span>
          </div>
        </div>

        <div class="content-list">${items}</div>
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
    'mememo-text-viewer': MememoTextViewer;
  }
}
