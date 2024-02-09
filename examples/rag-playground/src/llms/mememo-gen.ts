import type { TextGenMessage } from './gpt';
import type { ChatCompletion } from '../types/gpt-types';
import {
  SupportedRemoteModel,
  SupportedLocalModel
} from '../components/playground/user-config';

export type PromptRunSuccessResponse = {
  command: 'finishTextGen';
  completion: ChatCompletion;
  payload: {
    result: string;
    fullPrompt: string;
    detail: string;
  };
};

export type PromptRunErrorResponse = {
  message: string;
};

export interface PromptRunPostBody {
  prompt: string;
  temperature: number;
  model: keyof typeof SupportedRemoteModel | keyof typeof SupportedLocalModel;
}

const ENDPOINT =
  'https://aq079yrw81.execute-api.us-east-1.amazonaws.com/prod/run';

/**
 * Use mememo to generate text based on a given prompt
 * @param requestID Worker request ID
 * @param prompt Prompt prefix
 * @param inputText Input text
 * @param temperature Model temperature
 * @param userID User ID
 * @param model The model to use
 * @param detail Extra string information to include (will be returned)
 */
export const textGenMememo = async (
  requestID: string,
  prompt: string,
  temperature: number,
  model: keyof typeof SupportedRemoteModel | keyof typeof SupportedLocalModel,
  useCache: boolean = false,
  detail: string = ''
): Promise<TextGenMessage> => {
  // Check if the model output is cached
  const cachedValue = localStorage.getItem('[mememo]' + prompt);
  if (useCache && cachedValue !== null) {
    console.log('Use cached output (text gen)');
    await new Promise(resolve => setTimeout(resolve, 1000));
    const message: TextGenMessage = {
      command: 'finishTextGen',
      payload: {
        requestID: '',
        apiKey: '',
        result: cachedValue,
        prompt: prompt,
        detail: detail
      }
    };
    return message;
  }

  // Run the prompt through mememo API
  const body: PromptRunPostBody = {
    prompt,
    temperature,
    model
  };

  const url = new URL(ENDPOINT);
  url.searchParams.append('type', 'run');

  const requestOptions: RequestInit = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    credentials: 'include',
    body: JSON.stringify(body)
  };

  try {
    const response = await fetch(url.toString(), requestOptions);
    const data = (await response.json()) as
      | PromptRunSuccessResponse
      | PromptRunErrorResponse;
    if (response.status !== 200) {
      // Throw the error to the main thread
      const errorData = data as PromptRunErrorResponse;
      const message: TextGenMessage = {
        command: 'error',
        payload: {
          requestID: requestID,
          originalCommand: 'startTextGen',
          message: errorData.message
        }
      };
      return message;
    }

    const successData = data as PromptRunSuccessResponse;
    // Send back the data to the main thread
    const message: TextGenMessage = {
      command: 'finishTextGen',
      payload: {
        requestID: '',
        apiKey: '',
        result: successData.payload.result,
        prompt: successData.payload.fullPrompt,
        detail: detail
      }
    };

    // Also cache the model output
    if (useCache) {
      if (localStorage.getItem('[mememo]' + prompt) === null) {
        localStorage.setItem('[mememo]' + prompt, successData.payload.result);
      }
    }

    return message;
  } catch (error) {
    // Throw the error to the main thread
    const message: TextGenMessage = {
      command: 'error',
      payload: {
        requestID: requestID,
        originalCommand: 'startTextGen',
        message: error as string
      }
    };
    return message;
  }
};
