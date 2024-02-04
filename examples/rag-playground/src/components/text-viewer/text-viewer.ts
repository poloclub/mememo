import { LitElement, css, unsafeCSS, html, PropertyValues } from 'lit';
import { customElement, property, state, query } from 'lit/decorators.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';

import type { MememoWorkerMessage } from '../../workers/mememo-worker';

// Assets
import componentCSS from './text-viewer.css?inline';
import MememoWorkerInline from '../../workers/mememo-worker?worker&inline';
import searchIcon from '../../images/icon-search.svg?raw';
import crossIcon from '../../images/icon-cross-thick.svg?raw';
import crossSmallIcon from '../../images/icon-cross.svg?raw';

const MAX_DOCUMENTS_IN_MEMORY = 1000;
const DOCUMENT_INCREMENT = 100;

/**
 * Text viewer element.
 */
@customElement('mememo-text-viewer')
export class MememoTextViewer extends LitElement {
  //==========================================================================||
  //                              Class Properties                            ||
  //==========================================================================||
  @property({ type: String })
  dataURL: string | undefined;

  @property({ type: String })
  datasetName = 'my-dataset';

  @property({ type: String })
  documentName = '';

  @state()
  clickedItemIndexes: number[] = [];

  documents: string[] = [];
  curDocumentCap = 100;

  @state()
  documentCount = 0;

  @state()
  shownDocuments: string[] = [];

  @state()
  isFiltered = false;

  @state()
  showSearchBarCancelButton = false;

  mememoWorker: Worker;
  mememoFinishedLoading: Promise<void>;

  @state()
  isMememoFinishedLoading = false;
  markMememoFinishedLoading = () => {};

  lexicalSearchRequestCount = 0;
  get lexicalSearchRequestID() {
    return this.lexicalSearchRequestCount++;
  }

  //==========================================================================||
  //                             Lifecycle Methods                            ||
  //==========================================================================||
  constructor() {
    super();

    this.mememoFinishedLoading = new Promise<void>(resolve => {
      this.markMememoFinishedLoading = resolve;
    });

    this.mememoWorker = new MememoWorkerInline();
    this.mememoWorker.addEventListener(
      'message',
      (e: MessageEvent<MememoWorkerMessage>) =>
        this.loaderWorkerMessageHandler(e)
    );
  }

  firstUpdated() {
    this.mememoFinishedLoading.then(() => {
      const message: MememoWorkerMessage = {
        command: 'startLexicalSearch',
        payload: {
          query: 'human',
          limit: 10,
          requestID: this.lexicalSearchRequestID
        }
      };
      this.mememoWorker.postMessage(message);
    });

    console.log(this.dataURL);
  }

  /**
   * This method is called before new DOM is updated and rendered
   * @param changedProperties Property that has been changed
   */
  willUpdate(changedProperties: PropertyValues<this>) {
    if (
      changedProperties.has('dataURL') &&
      changedProperties.get('dataURL') === undefined
    ) {
      this.initData();
    }
  }

  //==========================================================================||
  //                              Custom Methods                              ||
  //==========================================================================||
  initData() {
    if (this.dataURL === undefined) {
      throw Error('dataURL is undefined');
    }
    const message: MememoWorkerMessage = {
      command: 'startLoadData',
      payload: {
        url: this.dataURL,
        datasetName: this.datasetName
      }
    };
    this.mememoWorker.postMessage(message);
  }

  //==========================================================================||
  //                              Event Handlers                              ||
  //==========================================================================||
  searchBarEntered(e: InputEvent) {}

  showSearchBarCancelButtonClicked() {}

  loaderWorkerMessageHandler(e: MessageEvent<MememoWorkerMessage>) {
    switch (e.data.command) {
      case 'transferLoadData': {
        const documents = e.data.payload.documents;
        this.documentCount += documents.length;

        // Load some documents in the memory in the main thread
        if (this.documents.length < MAX_DOCUMENTS_IN_MEMORY) {
          for (const document of documents) {
            this.documents.push(document);
          }
        }

        // Add data to the shown list
        if (this.shownDocuments.length < this.curDocumentCap) {
          this.shownDocuments = [...this.shownDocuments, ...documents];
        }

        if (e.data.payload.isLastBatch) {
          // Mark the loading has completed
          this.markMememoFinishedLoading();
          this.isMememoFinishedLoading = true;
        }
        break;
      }

      case 'finishLexicalSearch': {
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
    for (const [i, text] of this.shownDocuments.entries()) {
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

    // Configure search bar icon (loading => show loader, finished => search icon)
    let searchBarIcon = html`<span class="svg-icon search"
      >${unsafeHTML(searchIcon)}</span
    >`;

    if (!this.isMememoFinishedLoading) {
      searchBarIcon = html` <div class="search-bar-loader">
        <div class="loader-container">
          <div class="circle-loader"></div>
        </div>
      </div>`;
    }

    return html`
      <div class="text-viewer">
        <div class="header-bar">
          <div class="header">MeMemo Database</div>
          <div class="description">${this.documentCount} arXiv abstracts</div>
        </div>

        <div class="search-bar-container">
          <div class="search-bar">
            <span class="icon-container"> ${searchBarIcon} </span>

            <input
              id="search-bar-input"
              type="text"
              name="search-bar-input"
              ?disabled=${!this.isMememoFinishedLoading}
              @input=${(e: InputEvent) => this.searchBarEntered(e)}
              placeholder=${this.isMememoFinishedLoading
                ? 'Search local documents'
                : 'Loading documents and embeddings...'}
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

        <div class="content-list">
          ${items}
          <div
            class="item add-more-button"
            ?is-hidden=${this.documents.length === this.shownDocuments.length}
          >
            Show More
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
    'mememo-text-viewer': MememoTextViewer;
  }
}
