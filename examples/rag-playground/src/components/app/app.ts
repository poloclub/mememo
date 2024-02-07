import { LitElement, css, unsafeCSS, html, PropertyValues } from 'lit';
import { customElement, property, state, query } from 'lit/decorators.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';

import '../playground/playground';
import componentCSS from './app.css?inline';
import logoIcon from '../../images/icon-logo.svg?raw';

/**
 * App element.
 *
 */
@customElement('mememo-rag-playground')
export class MememoRagPlayground extends LitElement {
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

            <mememo-playground></mememo-playground>
            <div class="app-tabs"></div>
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
