import { LitElement, css, unsafeCSS, html, PropertyValues } from 'lit';
import { customElement, property, state, query } from 'lit/decorators.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';

import '../query-box/query-box';
import '../text-viewer/text-viewer';

import componentCSS from './playground.css?inline';

interface DatasetInfo {
  dataURL: string;
  datasetName: string;
  datasetNameDisplay: string;
}

enum Dataset {
  Arxiv = 'arxiv'
}

const datasets: Record<Dataset, DatasetInfo> = {
  [Dataset.Arxiv]: {
    dataURL: '/data/ml-arxiv-papers-1000.ndjson.gzip',
    datasetName: 'ml-arxiv-papers',
    datasetNameDisplay: 'ML arXiv Abstracts (1k)'
  }
};

/**
 * Playground element.
 *
 */
@customElement('mememo-playground')
export class MememoPlayground extends LitElement {
  //==========================================================================||
  //                              Class Properties                            ||
  //==========================================================================||

  //==========================================================================||
  //                             Lifecycle Methods                            ||
  //==========================================================================||
  constructor() {
    super();
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

  //==========================================================================||
  //                             Private Helpers                              ||
  //==========================================================================||

  //==========================================================================||
  //                           Templates and Styles                           ||
  //==========================================================================||
  render() {
    return html`
      <div class="playground">
        <div class="container container-input">
          <mememo-query-box></mememo-query-box>
        </div>

        <div class="container container-search">
          <div class="search-box">MeMemo Search</div>
        </div>

        <div class="container container-text">
          <mememo-text-viewer
            dataURL=${datasets['arxiv'].dataURL}
            datasetName=${datasets['arxiv'].datasetName}
            datasetNameDisplay=${datasets['arxiv'].datasetNameDisplay}
          ></mememo-text-viewer>
        </div>

        <div class="container container-prompt">Prompt</div>

        <div class="container container-model">
          <div class="model-box">GPT 3.5</div>
        </div>

        <div class="container container-output">Output</div>

        <div class="flow horizontal-flow input-text">
          <div class="background">
            <span class="line-loader hidden"></span>
            <div class="start-rectangle"></div>
            <div class="end-triangle"></div>
          </div>
        </div>

        <div class="flow horizontal-flow text-prompt">
          <div class="background">
            <span class="line-loader hidden"></span>
            <div class="start-rectangle"></div>
            <div class="end-triangle"></div>
          </div>
        </div>

        <div class="flow vertical-flow input-prompt">
          <div class="background">
            <span class="line-loader hidden"></span>
            <div class="start-rectangle"></div>
            <div class="end-triangle"></div>
          </div>
        </div>

        <div class="flow vertical-flow prompt-output">
          <div class="background">
            <span class="line-loader hidden"></span>
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
