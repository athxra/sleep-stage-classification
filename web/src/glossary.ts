/** Plain-language definitions shown by <InfoTip term="..."/>. */
export const GLOSSARY: Record<string, { title: string; text: string }> = {
  macroF1: {
    title: 'Macro-F1',
    text: 'F1 balances precision (how often a predicted stage is right) and recall (how many true epochs of a stage are found). Macro-F1 averages it over the five stages equally, so rare stages like N1 count as much as common ones like N2.',
  },
  kappa: {
    title: 'Cohen’s κ (kappa)',
    text: 'Agreement between the model and the expert after removing agreement expected by chance. 0 = chance, 1 = perfect. 0.61–0.80 is “substantial”, the range human scorers typically reach with each other.',
  },
  accuracy: {
    title: 'Accuracy',
    text: 'Share of epochs where the model matches the expert. It looks high on imbalanced data, so compare it with the majority-class baseline: the score of always guessing the most common stage.',
  },
  baseline: {
    title: 'Majority-class baseline',
    text: 'The accuracy you would get by always predicting the most frequent stage. A useful model must beat it clearly.',
  },
  epoch: {
    title: 'Epoch',
    text: 'A 30-second window of EEG, the standard unit sleep technicians score. A night has roughly 900–1,000 epochs of sleep.',
  },
  hypnogram: {
    title: 'Hypnogram',
    text: 'A timeline of sleep stages across the night. Rows run from Wake at the top to deep N3 sleep at the bottom, the layout clinicians use.',
  },
  hypnodensity: {
    title: 'Hypnodensity',
    text: 'The model’s probability for each stage at every epoch, stacked to 100%. Solid bands mean a confident model; mixed colours show uncertainty, usually at stage transitions.',
  },
  subjectIndependent: {
    title: 'Subject-independent evaluation',
    text: 'Test people are never seen during training or model selection, and both nights of a person stay in the same split. This measures how the model works on a new person, not on nights it has memorised.',
  },
  crossValidation: {
    title: '5-fold cross-validation',
    text: 'The 20 subjects are split into five groups. Each group is the test set once while the model trains on the others, so every subject is tested exactly once and results come with a spread (± std).',
  },
  transition: {
    title: 'Stage transitions',
    text: 'Epochs next to a change in the expert’s scoring. They often contain two stages at once, so they are harder for both models and human scorers.',
  },
  bandPower: {
    title: 'Band power',
    text: 'How the epoch’s energy splits across EEG rhythms: delta (0.5–4 Hz, deep sleep), theta (4–8 Hz, light sleep), alpha (8–12 Hz, relaxed wake), sigma (12–15 Hz, N2 spindles), beta (15–30 Hz, alert wake).',
  },
  spectrum: {
    title: 'Power spectrum',
    text: 'Power at each frequency (Welch method, 4-second windows). Peaks reveal rhythms: a delta hump for N3, a sigma bump for spindles in N2, an alpha peak for relaxed wake.',
  },
  confidence: {
    title: 'Confidence',
    text: 'The model’s probability for its chosen stage. Low confidence usually marks ambiguous or transitional epochs.',
  },
  macs: {
    title: 'MACs',
    text: 'Multiply-accumulate operations: a hardware-independent measure of compute. Fewer MACs means less energy, which matters for battery-powered wearables.',
  },
  sleepEfficiency: {
    title: 'Sleep efficiency',
    text: 'Total sleep time divided by time in bed. Healthy adults usually exceed 85%.',
  },
  waso: {
    title: 'Wake after sleep onset (WASO)',
    text: 'Minutes spent awake between first falling asleep and the final awakening.',
  },
}
