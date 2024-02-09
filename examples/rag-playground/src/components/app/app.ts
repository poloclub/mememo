import { LitElement, css, unsafeCSS, html, PropertyValues } from 'lit';
import { customElement, property, state, query } from 'lit/decorators.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';

import '../playground/playground';
import componentCSS from './app.css?inline';
import logoIcon from '../../images/icon-logo.svg?raw';
import { Dataset } from '../playground/playground';

/**
 * App element.
 *
 */
@customElement('mememo-rag-playground')
export class MememoRagPlayground extends LitElement {
  //==========================================================================||
  //                              Class Properties                            ||
  //==========================================================================||
  @state()
  curDataset: Dataset;

  //==========================================================================||
  //                             Lifecycle Methods                            ||
  //==========================================================================||
  constructor() {
    super();
    this.curDataset = Dataset.arXiv1k;
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
  tabButtonClicked(dataset: Dataset) {
    if (dataset !== this.curDataset) {
      this.curDataset = dataset;
    }
  }

  //==========================================================================||
  //                             Private Helpers                              ||
  //==========================================================================||

  //==========================================================================||
  //                           Templates and Styles                           ||
  //==========================================================================||
  render() {
    // Compile the playground
    let playground = html``;

    switch (this.curDataset) {
      case Dataset.arXiv1k: {
        playground = html`<mememo-playground
          curDataset=${this.curDataset}
        ></mememo-playground>`;
        break;
      }

      case Dataset.arXiv10k: {
        playground = html`<mememo-playground
          curDataset=${this.curDataset}
        ></mememo-playground>`;
        break;
      }

      case Dataset.arXiv120k: {
        playground = html`<mememo-playground
          curDataset=${this.curDataset}
        ></mememo-playground>`;
        break;
      }

      case Dataset.DiffusionDB10k: {
        playground = html`<mememo-playground
          curDataset=${this.curDataset}
        ></mememo-playground>`;
        break;
      }

      case Dataset.DiffusionDB100k: {
        playground = html`<mememo-playground
          curDataset=${this.curDataset}
        ></mememo-playground>`;
        break;
      }

      case Dataset.DiffusionDB1m: {
        playground = html`<mememo-playground
          curDataset=${this.curDataset}
        ></mememo-playground>`;
        break;
      }

      case Dataset.accident3k: {
        playground = html`<mememo-playground
          curDataset=${this.curDataset}
        ></mememo-playground>`;
        break;
      }

      default: {
        playground = html`<mememo-playground
          curDataset=${Dataset.arXiv10k}
        ></mememo-playground>`;
      }
    }

    return html`
      <div class="page">
        <div class="main-app">
          <div class="text-left"></div>

          <div class="app-wrapper">
            <div class="app-title">
              <div class="title-left">
                <div class="app-icon"></div>
                <div class="app-info">
                  <div class="app-name">RAG Playground</div>
                  <div class="app-tagline">
                    <span
                      >Prototype RAG applications in your browser! Powered
                      by</span
                    >
                    <a
                      class="mememo-logo"
                      href="https://github.com/poloclub/mememo"
                      target="_blank"
                    >
                      <span class="svg-icon">${unsafeHTML(logoIcon)}</span>
                      <span>MeMemo</span>
                    </a>
                  </div>
                </div>
              </div>
            </div>

            ${playground}

            <div class="app-tabs">
              <div class="tab">
                ML arXiv Abstracts
                <button
                  ?selected=${this.curDataset === Dataset.arXiv1k}
                  @click=${() => this.tabButtonClicked(Dataset.arXiv1k)}
                >
                  1k
                </button>
                <button
                  ?selected=${this.curDataset === Dataset.arXiv10k}
                  @click=${() => this.tabButtonClicked(Dataset.arXiv10k)}
                >
                  10k
                </button>
                <button
                  ?selected=${this.curDataset === Dataset.arXiv120k}
                  @click=${() => this.tabButtonClicked(Dataset.arXiv120k)}
                >
                  120k
                </button>
              </div>

              <div class="splitter"></div>

              <div class="tab">
                DiffusionDB Prompts
                <button
                  ?selected=${this.curDataset === Dataset.DiffusionDB10k}
                  @click=${() => this.tabButtonClicked(Dataset.DiffusionDB10k)}
                >
                  10k
                </button>
                <button
                  ?selected=${this.curDataset === Dataset.DiffusionDB100k}
                  @click=${() => this.tabButtonClicked(Dataset.DiffusionDB100k)}
                >
                  100k
                </button>
                <button
                  ?selected=${this.curDataset === Dataset.DiffusionDB500k}
                  @click=${() => this.tabButtonClicked(Dataset.DiffusionDB500k)}
                >
                  500k
                </button>
                <button
                  ?selected=${this.curDataset === Dataset.DiffusionDB1m}
                  @click=${() => this.tabButtonClicked(Dataset.DiffusionDB1m)}
                >
                  1M
                </button>
              </div>

              <div class="splitter"></div>

              <div class="tab">
                AI Accident Reports
                <button
                  ?selected=${this.curDataset === Dataset.accident3k}
                  @click=${() => this.tabButtonClicked(Dataset.accident3k)}
                >
                  3k
                </button>
              </div>
            </div>
          </div>

          <div class="text-right"></div>
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
    'mememo-rag-playground': MememoRagPlayground;
  }
}
