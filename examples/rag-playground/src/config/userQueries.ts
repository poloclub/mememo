import { Dataset } from '../components/playground/playground';

const arXivQueries = [
  'How to integrate information retrieval into machine learning systems?',
  'What are the best practices for preprocessing data for machine learning models?',
  'How can we improve the interpretability of deep learning models?',
  'What are effective methods for dealing with imbalanced datasets in classification tasks?',
  'How can transfer learning be applied to small datasets in a specific domain?',
  'What are the challenges and solutions for real-time machine learning predictions?',
  'How can unsupervised learning techniques improve data understanding in unlabeled datasets?',
  'What are the key factors in choosing between traditional machine learning algorithms and deep learning approaches?',
  'How can machine learning be used to enhance cybersecurity defenses?',
  'What are the ethical considerations in developing and deploying machine learning models?',
  'How can reinforcement learning be applied to optimize decision-making processes in business?',
  'What are the current limitations of natural language processing models, and how can they be addressed?',
  'How can machine learning contribute to personalized medicine and healthcare?',
  'What strategies can be used to reduce overfitting in complex machine learning models?',
  'How can machine learning algorithms be adapted for energy-efficient computing?',
  'What are the implications of federated learning for privacy-preserving machine learning?',
  'How can anomaly detection be improved in machine learning models for fraud detection?',
  'What role does feature engineering play in improving the performance of machine learning models?',
  'How can machine learning be integrated with blockchain technology for enhanced security?',
  'What are the challenges in scaling machine learning models for large-scale applications?',
  'How can machine learning techniques be applied to predict and mitigate the effects of climate change?'
];

const diffusiondbQueries: string[] = [
  'A cute brown dog jumping around',
  'White dog wandering through a dense forest',
  'Red apple resting on an aged wooden table',
  'Sunny field dotted with vibrant sunflowers',
  'Blue boat gently floating on a tranquil lake',
  'Green frog perched on a lily pad, eyeing its surroundings',
  'Brown horse galloping across a wide-open meadow',
  'Gray cat curled up, napping on a cozy windowsill',
  'Black bicycle leaning against a graffiti-covered city wall',
  'Orange pumpkin nestled among leaves in a patch',
  "Pink flamingo standing gracefully at the water's edge",
  'Purple butterfly fluttering around a fragrant lavender bush',
  'Goldfish swimming circles in a sparkling fishbowl',
  'Rainbow arching majestically over a lush mountain landscape',
  'Silver key lying atop an ancient, leather-bound book',
  'Colorful parrot perched in the dense canopy of a tropical jungle',
  'Snowman adorned with a carrot nose in a snowy field',
  'Red and white lighthouse standing guard by the rocky coastline',
  'Blue and yellow beach umbrella casting a cool shadow on sandy shores',
  'Chocolate cupcake topped with a rich, creamy swirl on a festive plate',
  "Pair of red sneakers ready at the start line of a running track'"
];

const accidentQueries: string[] = [
  'What are some potential harms associated with facial recognition systems?',
  'How could a loan approval ML system inadvertently reinforce existing societal biases?',
  'What risks are involved with an ML-based healthcare diagnosis tool providing incorrect treatment recommendations?',
  'How might a job applicant screening ML system discriminate against certain candidates based on non-job-related characteristics?',
  'What if an autonomous driving ML system fails to correctly interpret road signs or pedestrian signals in different weather conditions?',
  'Could an ML content recommendation system on social media promote harmful or extremist content?',
  'What are the privacy concerns with an ML-powered personal assistant listening to and analyzing private conversations?',
  'How might an ML-powered stock trading system contribute to market instability or unfair trading advantages?',
  'What if an ML-based wildfire prediction system provides inaccurate forecasts, leading to insufficient disaster preparedness?',
  'How could an ML model used for educational content personalization create learning gaps or reinforce stereotypes?',
  'What are the implications of an ML system for energy grid management malfunctioning and causing widespread power outages?',
  'How might an ML-powered emotion detection system in advertising manipulate consumer behavior unethically?',
  'What if an ML model used in wildlife conservation misidentifies species and leads to inappropriate conservation efforts?',
  'Could an ML-based sentencing recommendation tool in the legal system perpetuate racial or gender biases?',
  'What are the potential consequences of an ML-driven agricultural advice system providing incorrect farming guidance?',
  "How might an ML-based air traffic control system's malfunction lead to flight safety risks?",
  'What if an ML algorithm for detecting online bullying overlooks certain forms of harassment, leaving victims unprotected?',
  "How could an ML-powered credit scoring system unfairly lower someone's credit score based on non-financial personal data?",
  'What if an ML-based disaster response system prioritizes resources in a way that discriminates against certain communities?',
  'How might an ML system designed to filter out fake news inadvertently censor legitimate information or viewpoints?'
];

export const userQueries: Record<Dataset, string[]> = {
  'arxiv-1k': arXivQueries,
  'arxiv-10k': arXivQueries,
  'arxiv-120k': arXivQueries,
  'diffusiondb-10k': diffusiondbQueries,
  'diffusiondb-100k': diffusiondbQueries,
  'diffusiondb-500k': diffusiondbQueries,
  'diffusiondb-1m': diffusiondbQueries,
  'accident-3k': accidentQueries
};
