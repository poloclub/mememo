const arXivTemplates =
  "You are an expert in machine learning, and you are answering a user's questions about machine learning. The user's question is in <user></user>. You have access to documents in <context></context>. Your answer should be solely based on the provided documents. Provide cite the document source if possible. Answer your question in an <output></output> tag.\n\n<user>{{user}}</user>\n\n<context>{{context}}</context>";

export const promptTemplates = {
  'arxiv-10k': arXivTemplates,
  'arxiv-120k': arXivTemplates,
  'diffusiondb-1m': arXivTemplates,
  'accident-3k': arXivTemplates
};
