import { LitElement, css, unsafeCSS, html, PropertyValues } from 'lit';
import { customElement, property, state, query } from 'lit/decorators.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';
import d3 from '../../utils/d3-import';

import type { MememoWorkerMessage } from '../../workers/mememo-worker';

// Assets
import componentCSS from './text-viewer.css?inline';
import MememoWorkerInline from '../../workers/mememo-worker?worker&inline';
import searchIcon from '../../images/icon-search.svg?raw';
import crossIcon from '../../images/icon-cross-thick.svg?raw';
import crossSmallIcon from '../../images/icon-cross.svg?raw';

const MAX_DOCUMENTS_IN_MEMORY = 1000;
const DOCUMENT_INCREMENT = 100;
const LEXICAL_SEARCH_LIMIT = 2000;
const numberFormatter = d3.format(',');

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
  datasetNameDisplay = '';

  @state()
  clickedItemIndexes: number[] = [];

  documents: string[] = [];

  /** Current document source (total documents or filtered documents) */
  curDocuments: string[] = this.documents;
  shownDocumentCap = 100;

  @state()
  documentCount = 0;

  @state()
  shownDocuments: string[] = [];

  @state()
  isFiltered = false;

  @state()
  isSearchScrolled = false;

  @state()
  showSearchBarCancelButton = false;

  isSearching = false;
  pendingQuery: string | null = null;

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
    this.initData();
  }

  /**
   * This method is called before new DOM is updated and rendered
   * @param changedProperties Property that has been changed
   */
  willUpdate(changedProperties: PropertyValues<this>) {}

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
  searchBarEntered(e: InputEvent) {
    const inputElement = e.currentTarget as HTMLInputElement;
    const query = inputElement.value;

    if (query.length === 0) {
      this.showSearchBarCancelButton = false;
      this.curDocuments = this.documents;
      this.shownDocuments = this.documents.slice(0, this.shownDocumentCap);
      this.isFiltered = false;
    } else {
      // Show the cancel button
      this.showSearchBarCancelButton = true;

      // Start lexical search
      if (this.isSearching) {
        this.pendingQuery = query;
      } else {
        const message: MememoWorkerMessage = {
          command: 'startLexicalSearch',
          payload: {
            query: query,
            limit: LEXICAL_SEARCH_LIMIT,
            requestID: this.lexicalSearchRequestID
          }
        };
        this.mememoWorker.postMessage(message);
      }
    }
  }

  showSearchBarCancelButtonClicked() {
    const inputElement = this.shadowRoot!.querySelector(
      '#search-bar-input'
    ) as HTMLInputElement;
    inputElement.value = '';
    this.showSearchBarCancelButton = false;
    this.curDocuments = this.documents;
    this.shownDocuments = this.documents.slice(0, this.shownDocumentCap);
    this.isFiltered = false;
  }

  async showMoreButtonClicked() {
    if (this.shadowRoot === null) {
      throw Error('shadowRoot is null');
    }

    // Need to manually track the scroll position and set it after updating the list
    const contentList = this.shadowRoot.querySelector(
      '.content-list'
    ) as HTMLElement;
    const scrollTop = contentList.scrollTop;

    // Update the list
    this.shownDocumentCap += DOCUMENT_INCREMENT;
    this.shownDocuments = this.curDocuments.slice(0, this.shownDocumentCap);

    await this.updateComplete;
    contentList.scrollTop = scrollTop;
  }

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
        if (this.shownDocuments.length < this.shownDocumentCap) {
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
        this.isSearching = false;

        // Update the shown documents
        this.isFiltered = true;
        this.curDocuments = e.data.payload.results;
        this.shownDocuments = this.curDocuments.slice(0, this.shownDocumentCap);

        // Start a new search if there is a pending query
        if (this.pendingQuery !== null) {
          const message: MememoWorkerMessage = {
            command: 'startLexicalSearch',
            payload: {
              query: this.pendingQuery,
              limit: LEXICAL_SEARCH_LIMIT,
              requestID: this.lexicalSearchRequestID
            }
          };
          this.mememoWorker.postMessage(message);
          this.pendingQuery = null;
        }

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

    // Compile the count label
    let countLabel = html` <div class="count-label">
      ${numberFormatter(this.documentCount)} documents
    </div>`;

    if (this.isFiltered) {
      if (this.curDocuments.length < LEXICAL_SEARCH_LIMIT) {
        countLabel = html` <div class="count-label">
          ${numberFormatter(this.curDocuments.length)} search results
        </div>`;
      } else {
        countLabel = html` <div class="count-label">
          ${numberFormatter(this.curDocuments.length)}+ search results
        </div>`;
      }
    }

    return html`
      <div class="text-viewer">
        <div class="header-bar">
          <div class="header">MeMemo Database</div>
          <div class="description">${this.datasetNameDisplay}</div>
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

        <div class="list-container">
          <div class="header-gap" ?is-hidden=${!this.isSearchScrolled}></div>

          <div
            class="content-list"
            @scroll=${(e: Event) => {
              this.isSearchScrolled = (e.target as HTMLElement).scrollTop > 0;
            }}
          >
            ${countLabel} ${items}
            <div
              class="item add-more-button"
              @click=${() => this.showMoreButtonClicked()}
              ?is-hidden=${this.curDocuments.length ===
              this.shownDocuments.length}
            >
              Show More
            </div>
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
