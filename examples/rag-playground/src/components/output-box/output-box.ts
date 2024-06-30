import { LitElement, css, unsafeCSS, html, PropertyValues } from 'lit';
import { customElement, property, state, query } from 'lit/decorators.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';
import d3 from '../../utils/d3-import';

import componentCSS from './output-box.css?inline';
import searchIcon from '../../images/icon-search.svg?raw';
import expandIcon from '../../images/icon-expand.svg?raw';
import playIcon from '../../images/icon-play.svg?raw';

const numberFormatter = d3.format(',');

/**
 * Output box element.
 *
 */
@customElement('mememo-output-box')
export class MememoOutputBox extends LitElement {
  //==========================================================================||
  //                              Class Properties                            ||
  //==========================================================================||
  @property({ type: String })
  llmOutput: string | undefined;

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
  willUpdate(changedProperties: PropertyValues<this>) {
    console.log(this.llmOutput);
  }

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
      <div class="output-box">
        <div class="header">
          <div class="text-group">
            <span class="text">LLM Output</span>
          </div>

          <div class="button-group">
            <button>
              <span class="svg-icon">${unsafeHTML(expandIcon)}</span>
              view
            </button>
          </div>
        </div>
        <div rows="5" class="output-container">${this.llmOutput}
          <div class="placeholder" ?is-hidden=${this.llmOutput !== ''}>
            Click the run buttons above to see LLM's output.
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
    'mememo-output-box': MememoOutputBox;
  }
}
