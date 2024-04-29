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
  promptHTML = '';

  @state()
  tokenCount = 0;

  //==========================================================================||
  //                             Lifecycle Methods                            ||
  //==========================================================================||
  constructor() {
    super();
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
    if (
      changedProperties.has('template') ||
      changedProperties.has('userQuery') ||
      changedProperties.has('relevantDocuments')
    ) {
      this.updatePrompt();

      // If the update is triggered by a relevant document update, we also
      // run the compiled prompt
      if (this.relevantDocuments && this.relevantDocuments.length > 0) {
        this.runButtonClicked();
      }
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
    let promptHTMLString = this.template.replace(/</g, '&lt;');
    promptHTMLString = promptHTMLString.replace(/>/g, '&gt;');
    promptHTMLString = promptHTMLString.replace(/\n/g, '<br>');
    promptHTMLString = promptHTMLString.replace(
      /{{user}}/g,
      '<span class="user-query">{{user}}</span>'
    );
    promptHTMLString = promptHTMLString.replace(
      /{{context}}/g,
      '<span class="context-query">{{context}}</span>'
    );

    if (this.userQuery !== undefined && this.userQuery !== '') {
      prompt = prompt.replace('{{user}}', this.userQuery);
      promptHTMLString = promptHTMLString.replace('{{user}}', this.userQuery);
    }

    if (
      this.relevantDocuments !== undefined &&
      this.relevantDocuments.length > 0
    ) {
      const documents = this.relevantDocuments.join('\n');
      prompt = prompt.replace('{{context}}', documents);
      promptHTMLString = promptHTMLString.replace('{{context}}', documents);
    }

    this.prompt = prompt;
    this.promptHTML = promptHTMLString;
    this.tokenCount = encode(prompt).length;
  }

  setTextareaMaxHeight(maxHeight: number) {
    if (!this.shadowRoot) return;
    const textarea = this.shadowRoot.querySelector('textarea') as HTMLElement;
    textarea.style.setProperty('max-height', `${maxHeight - 50}px`);
  }

  //==========================================================================||
  //                              Event Handlers                              ||
  //==========================================================================||
  textareaInput(e: InputEvent) {
    const textareaElement = e.currentTarget as HTMLTextAreaElement;
    this.prompt = textareaElement.innerText;

    // Notify the parent
    const event = new Event('promptEdited', {
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
            <button>
              <span class="svg-icon">${unsafeHTML(expandIcon)}</span>
              view
            </button>

            <button @click=${() => this.runButtonClicked()}>
              <span class="svg-icon">${unsafeHTML(playIcon)}</span>
              run
            </button>
          </div>
        </div>
        <div
          class="input-box"
          contenteditable
          @input=${(e: InputEvent) => this.textareaInput(e)}
        >
          ${unsafeHTML(this.promptHTML)}
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
    'mememo-prompt-box': MememoPromptBox;
  }
}
