import { Dataset } from '../components/playground/playground';

const arXivTemplates =
  "You are an expert in machine learning, and you are answering a user's questions about machine learning. The user's question is in <user></user>. You have access to documents in <context></context>. Your answer should be solely based on the provided documents. Please cite the document source using a number in square brackets, such as [1]. List the reference at the end.\n\n<user>{{user}}</user>\n\n<context>{{context}}</context>";

const diffusiondbTemplates =
  "You are an expert in prompt engineering text-to-image generative models. You are helping a user improve their prompts. The user's prompt is in <user></user>. You have access to example prompts in <context></context>. Your answer should be based on the example documents. Make the user's prompt more interesting and effective. Cite example prompts to explain your improvement.\n\n<user>{{user}}</user>\n\n<context>{{context}}</context>";

const accidentTemplates =
  "You are an expert in envisioning potential harms of AI technologies. You are helping a user to brainstorm potentially negative consequences of technologies. The user's question is in <user></user>. You have access to real AI accident reports in <context></context>. Your answer should be solely based on the provided documents. Please cite the document source using a number in square brackets, such as [1]. List the reference at the end.\n\n<user>{{user}}</user>\n\n<context>{{context}}</context>";

export const promptTemplates: Record<Dataset, string> = {
  'arxiv-1k': arXivTemplates,
  'arxiv-10k': arXivTemplates,
  'arxiv-120k': arXivTemplates,
  'diffusiondb-10k': diffusiondbTemplates,
  'diffusiondb-100k': diffusiondbTemplates,
  'diffusiondb-500k': diffusiondbTemplates,
  'diffusiondb-1m': diffusiondbTemplates,
  'accident-3k': accidentTemplates
};
