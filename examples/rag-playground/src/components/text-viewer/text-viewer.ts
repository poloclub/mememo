import { LitElement, css, unsafeCSS, html, PropertyValues } from 'lit';
import { customElement, property, state, query } from 'lit/decorators.js';
import { unsafeHTML, UnsafeHTMLDirective } from 'lit/directives/unsafe-html.js';
import { DirectiveResult } from 'lit/directive.js';
import { downloadJSON, round } from '@xiaohk/utils';
import d3 from '../../utils/d3-import';

import type { MememoWorkerMessage } from '../../workers/mememo-worker';

// Assets
import componentCSS from './text-viewer.css?inline';
import searchIcon from '../../images/icon-search.svg?raw';
import crossIcon from '../../images/icon-cross-thick.svg?raw';
import downloadIcon from '../../images/icon-download.svg?raw';
import crossSmallIcon from '../../images/icon-cross.svg?raw';

import MememoWorkerInline from '../../workers/mememo-worker?worker';

const MAX_DOCUMENTS_IN_MEMORY = 1000;
const DOCUMENT_INCREMENT = 100;
const LEXICAL_SEARCH_LIMIT = 2000;
const numberFormatter = d3.format(',');

const startLoadingTime = Date.now();
const loadingTimes: number[] = [];
const TRACK_LOADING_TIME = false;

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
  indexURL: string | undefined;

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
  shownDocumentDistances: number[] = [];

  @state()
  isFiltered = false;

  @state()
  isSearchScrolled = false;

  @state()
  showSearchBarCancelButton = false;

  isSearching = false;
  pendingQuery: string | null = null;

  @state()
  curQuery: string | null = null;

  mememoWorker: Worker;
  mememoFinishedLoading: Promise<void>;

  @state()
  isMememoFinishedLoading = false;
  markMememoFinishedLoading = () => {};

  lexicalSearchRequestCount = 0;
  get lexicalSearchRequestID() {
    return this.lexicalSearchRequestCount++;
  }

  semanticSearchRequestCount = 0;
  get semanticSearchRequestID() {
    return this.semanticSearchRequestCount++;
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
        indexURL: this.indexURL,
        datasetName: this.datasetName
      }
    };
    this.mememoWorker.postMessage(message);
  }

  /**
   * Retrieve relevant documents using MeMemo
   * @param embedding Input embedding
   * @param topK Top k relevant documents to retrieve
   * @param maxDistance Distance threshold for relevance
   */
  semanticSearch(embedding: number[], topK: number, maxDistance: number) {
    const message: MememoWorkerMessage = {
      command: 'startSemanticSearch',
      payload: {
        embedding,
        requestID: this.semanticSearchRequestID,
        topK,
        maxDistance
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
      this.curQuery = null;
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
    this.curQuery = null;
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

        // Track the loading time
        if (TRACK_LOADING_TIME) {
          const now = Date.now();
          loadingTimes.push(now - startLoadingTime);
        }

        if (e.data.payload.isLastBatch) {
          // Mark the loading has completed
          this.markMememoFinishedLoading();
          this.isMememoFinishedLoading = true;

          if (TRACK_LOADING_TIME) {
            // Download the loading times
            downloadJSON(loadingTimes, undefined, 'loading-times.json');

            // Export the index to a json file
            const message: MememoWorkerMessage = {
              command: 'startExportIndex',
              payload: { requestID: 0 }
            };
            this.mememoWorker.postMessage(message);
          }
        }
        break;
      }

      case 'finishLexicalSearch': {
        this.isSearching = false;

        // Update the shown documents
        this.curQuery = e.data.payload.query;
        this.isFiltered = true;
        this.curDocuments = this._formatSearchResults(e.data.payload.results);
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

      case 'finishSemanticSearch': {
        const { documents, documentDistances, embedding } = e.data.payload;

        // Update the shown documents
        this.curQuery = null;
        this.isFiltered = true;
        this.curDocuments = documents;
        this.showSearchBarCancelButton = true;
        this.shownDocuments = this.curDocuments.slice(0, this.shownDocumentCap);
        this.shownDocumentDistances = documentDistances.slice(
          0,
          this.shownDocumentCap
        );

        // Send the documents back to the parent
        const event = new CustomEvent<string[]>('semanticSearchFinished', {
          bubbles: true,
          composed: true,
          detail: documents
        });
        this.dispatchEvent(event);
        break;
      }

      case 'finishExportIndex': {
        const indexJSON = e.data.payload.indexJSON;

        // Download a compressed file
        compressTextGzip(JSON.stringify(indexJSON)).then(
          value => {
            downloadBlob(value, 'mememo-index.json.gzip');
          },
          () => {}
        );

        break;
      }

      default: {
        console.error(`Unknown command ${e.data.command}`);
      }
    }
  }

  /**
   * Export and download the created index
   */
  downloadButtonClicked() {
    // Export the index to a json file
    if (this.isMememoFinishedLoading) {
      const message: MememoWorkerMessage = {
        command: 'startExportIndex',
        payload: { requestID: 0 }
      };
      this.mememoWorker.postMessage(message);
    }
  }

  //==========================================================================||
  //                             Private Helpers                              ||
  //==========================================================================||
  /**
   * Format the search results to highlight matches
   * @param results Current search results
   */
  _formatSearchResults = (results: string[]) => {
    if (this.curQuery === null) {
      throw Error('curQuery is null');
    }

    // Function to escape special characters for regular expression
    const escapeRegExp = (string: string): string => {
      return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
    };

    const formattedResults: string[] = [];

    for (const result of results) {
      // Try to avoid XSS attack
      if (result.includes('iframe')) continue;
      if (result.includes('<script')) continue;

      // Escape special characters in the query
      const escapedQuery = escapeRegExp(this.curQuery);

      // Create a regular expression to find all occurrences of the query (case-insensitive)
      const regex = new RegExp(`(${escapedQuery})`, 'gi');

      // Replace all occurrences of the query in the result with <em>query</em>
      const newResult = result.replace(regex, '<em>$1</em>');

      formattedResults.push(newResult);
    }

    return formattedResults;
  };

  //==========================================================================||
  //                           Templates and Styles                           ||
  //==========================================================================||
  render() {
    // Compile the item list
    let items = html``;
    for (const [i, text] of this.shownDocuments.entries()) {
      let itemText: DirectiveResult<typeof UnsafeHTMLDirective> | string = text;
      if (this.curQuery !== null) {
        itemText = unsafeHTML(text);
      }
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
          <span
            class="distance-overlay"
            ?is-hidden=${!this.isFiltered || this.curQuery !== null}
            >${i < this.shownDocumentDistances.length
              ? round(this.shownDocumentDistances[i], 4)
              : ''}</span
          >
          ${itemText}
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
      const name =
        this.curQuery === null
          ? 'semantic search results'
          : 'lexical search results';

      if (this.curDocuments.length < LEXICAL_SEARCH_LIMIT) {
        countLabel = html` <div class="count-label">
          ${numberFormatter(this.curDocuments.length)} ${name}
        </div>`;
      } else {
        countLabel = html` <div class="count-label">
          ${numberFormatter(this.curDocuments.length)}+ ${name}
        </div>`;
      }
    }

    return html`
      <div class="text-viewer">
        <div class="header-bar">
          <div class="header">MeMemo Database</div>
          <div class="description">${this.datasetNameDisplay}</div>
          <div class="button-group">
            <button @click=${() => this.downloadButtonClicked()}>
              <span class="svg-icon">${unsafeHTML(downloadIcon)}</span>
            </button>
          </div>
        </div>

        <div class="search-bar-container">
          <div class="search-bar">
            <span class="icon-container"> ${searchBarIcon} </span>

            <input
              id="search-bar-input"
              type="text"
              name="search-bar-input"
              ?disabled=${!this.isMememoFinishedLoading ||
              (this.isFiltered && this.curQuery === null)}
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

            <div
              class="semantic-search-info"
              ?is-hidden=${!this.isFiltered || this.curQuery !== null}
            >
              Documents similar to user query
            </div>
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

const downloadBlob = (blob: Blob, filename: string) => {
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
};

const compressTextGzip = async (text: string): Promise<Blob> => {
  // Create a stream from the text
  const textBlob = new Blob([text], { type: 'text/plain' });
  const textStream = textBlob.stream();

  // Create a GZIP CompressionStream
  const gzipStream = new CompressionStream('gzip');

  // Pipe the text stream through the gzip compressor
  const compressedStream = textStream.pipeThrough(gzipStream);

  // Convert the compressed stream back to a Blob
  const compressedBlob = await new Response(compressedStream).blob();

  return compressedBlob;
};
