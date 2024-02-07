import { LitElement, css, unsafeCSS, html, PropertyValues } from 'lit';
import { customElement, property, state, query } from 'lit/decorators.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';
import { encode } from 'gpt-tokenizer/model/gpt-3.5-turbo';
import d3 from '../../utils/d3-import';

import componentCSS from './prompt-box.css?inline';
import searchIcon from '../../images/icon-search.svg?raw';
import expandIcon from '../../images/icon-expand.svg?raw';
import playIcon from '../../images/icon-play.svg?raw';

const numberFormatter = d3.format(',');

/**
 * Prompt box element.
 *
 */
@customElement('mememo-prompt-box')
export class MememoPromptBox extends LitElement {
  //==========================================================================||
  //                              Class Properties                            ||
  //==========================================================================||
  @property({ type: String })
  template: string | undefined;

  @property({ type: String })
  userQuery: string | undefined;

  @property({ attribute: false })
  relevantDocuments: string[] | undefined;

  @state()
  prompt = '';

  @state()
  tokenCount = 0;

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
    if (
      changedProperties.has('template') ||
      changedProperties.has('userQuery') ||
      changedProperties.has('relevantDocuments')
    ) {
      this.updatePrompt();
    }
  }

  //==========================================================================||
  //                              Custom Methods                              ||
  //==========================================================================||
  async initData() {}

  /**
   * Recompile the prompt using template and provided information.
   */
  updatePrompt() {
    if (this.template === undefined) return;

    let prompt = this.template;

    if (this.userQuery !== undefined) {
      prompt = prompt.replace('{{user}}', this.userQuery);
    }

    if (this.relevantDocuments !== undefined) {
      const documents = this.relevantDocuments.join('\n');
      prompt = prompt.replace('{{context}}', documents);
    }

    this.prompt = prompt;
    this.tokenCount = encode(prompt).length;
  }

  //==========================================================================||
  //                              Event Handlers                              ||
  //==========================================================================||
  textareaInput(e: InputEvent) {
    const textareaElement = e.currentTarget as HTMLTextAreaElement;
    this.template = textareaElement.value;
  }

  runButtonClicked() {
    // Notify the parent to run the user query
    const event = new CustomEvent('runButtonClicked', {
      bubbles: true,
      composed: true,
      detail: this.prompt
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
      <div class="prompt-box">
        <div class="header">
          <div class="text-group">
            <span class="text">Retrieval Augmented Prompt</span>

            <span class="token-count" ?is-oversized=${this.tokenCount > 8000}
              >${numberFormatter(this.tokenCount)} tokens</span
            >
          </div>

          <div class="button-group">
            <button @click=${() => this.runButtonClicked()}>
              <span class="svg-icon">${unsafeHTML(playIcon)}</span>
              run
            </button>

            <button>
              <span class="svg-icon">${unsafeHTML(expandIcon)}</span>
              view
            </button>
          </div>
        </div>
        <textarea rows="5" @input=${(e: InputEvent) => this.textareaInput(e)}>
${this.prompt}</textarea
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
    'mememo-prompt-box': MememoPromptBox;
  }
}
