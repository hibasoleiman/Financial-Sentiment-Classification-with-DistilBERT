# Financial Sentiment Classification with DistilBERT

Classifying financial news sentences as **positive**, **negative** or **neutral** from an investor's point of view, comparing a classical baseline (TF-IDF + Naive Bayes) with a pre-trained Transformer (DistilBERT), first frozen and then fine-tuned.

> Course project, M2 MoSIG (Université Grenoble Alpes), NLP with Transformers.

## Results

| Model | Accuracy | Macro F1 | Negative-class recall |
|---|---|---|---|
| Always predict "neutral" (majority baseline) | 0.621 | ~0.26 | 0.00 |
| TF-IDF + Naive Bayes (100 features) | 0.738 | 0.600 | — |
| DistilBERT, frozen encoder (linear head only) | 0.735 | ~0.47 | 0.01 |
| **DistilBERT, fully fine-tuned** | **0.944** | **0.931** | **0.93** |

Per-class F1 of the fine-tuned model: neutral 0.963, positive 0.908, negative 0.923.

## Dataset

[Financial PhraseBank v1.0](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10) (Malo et al., 2014): sentences from financial news, each labeled by 5–8 annotators with a background in finance. This project uses `Sentences_75Agree.txt` (sentences with at least 75% annotator agreement):

- **3,453 sentences**, format `sentence@label`, encoding `iso-8859-1`
- **Imbalanced classes:** neutral 62%, positive 26%, negative 12%
- **Stratified 80/20 split:** 2,762 train / 691 test, with the same class proportions in both splits

The dataset is not included in this repository. Download it from the link above and place it in `FinancialPhraseBank-v1.0/`.

## Method

### 1. Baseline: TF-IDF + Naive Bayes
- Sentences are turned into TF-IDF vectors, with the vocabulary fitted on the training set only to avoid data leakage.
- A multinomial Naive Bayes classifier is trained for vocabulary sizes from 5 to 1,000 words. Test F1 plateaus around 0.68–0.69 from about 200 words.

![F1 score vs. number of TF-IDF features](images/f1_vs_max_features.png)

### 2. Tokenization and sequence length
- `distilbert-base-uncased` WordPiece tokenizer (vocabulary of 30,522 tokens).
- Sentence length analysis: mean 30 tokens, median 28, 99th percentile 69, max 150.
- `MAX_LEN` reduced from 512 to **128**. This truncates only 1 of 3,453 sentences and makes sequences 4× shorter, so training is much faster, since attention cost grows quadratically with length.

![Distribution of sentence lengths in tokens](images/token_length_distribution.png)

### 3. Model
DistilBERT encoder (6 layers, hidden size 768, ~66M parameters), followed by dropout and a linear layer (768 → 3) applied to the `[CLS]` token representation. Trained with cross-entropy loss.

### 4. Two training setups
| | Frozen encoder | Fine-tuned |
|---|---|---|
| Trainable parameters | 2,307 (head only) | ~66M (all) |
| Epochs | 16 | 4 |
| Batch size | 8 | 16 |
| Learning rate | 5e-5 | 2e-5 |
| Warm-up steps | — | 70 |
| Weight decay | 0.01 | 0.01 |

## Key findings

- **Accuracy is misleading on imbalanced data.** The frozen model's 73.5% accuracy looks acceptable, but it is only 11 points above always predicting "neutral". Macro F1 (~0.47) reveals the problem.
- **The frozen encoder fails on the negative class.** The confusion matrix shows only 1 of 84 negative test sentences classified correctly; 58 were predicted *positive*. Financial positive and negative sentences share the same structure ("profit rose/fell to X from Y"), and general-purpose `[CLS]` representations do not capture the direction of the change.

  ![Confusion matrix of the frozen-encoder model](images/confusion_matrix_frozen.png)

- **Fine-tuning fixes it.** Letting the encoder adapt raises accuracy to 94.4%, macro F1 to 0.93 and negative recall to 0.93, without any explicit rebalancing technique.

### Example predictions (fine-tuned model)

| Sentence | Prediction | Probability |
|---|---|---|
| "CocaCola saw its share price dropping of more than 25% this semester." | negative | 0.997 |
| "Despite most of the company's sales are taking place in China, the shareholders decided not to relocate the production there." | neutral | 0.999 |
| "This year's profits quintuplicated with respect to the last year." | positive | 0.727 |

The lower confidence on the third sentence likely comes from the rare word "quintuplicated", which the tokenizer splits into sub-words and which does not appear in the training data.

## Limitations

- Hyperparameters were compared on the test set. A separate validation set or cross-validation would give a more reliable estimate.
- Results come from a single training run; variance across random seeds was not measured.
- Small, single-source dataset: performance on other kinds of financial text is not guaranteed.

## Possible extensions

- Compare with domain-specific models such as FinBERT.
- Class-weighted loss to handle the imbalance explicitly.
- Error analysis of the remaining misclassifications.

## How to run

The notebook was developed on Google Colab with a GPU.

```bash
pip install torch transformers tokenizers scikit-learn pandas numpy matplotlib seaborn
```

1. Download Financial PhraseBank and place it in `FinancialPhraseBank-v1.0/`.
2. Open `Bert_project.ipynb` and run the cells in order. On Colab, select *Runtime → Change runtime type → GPU*.

Tested with `transformers` 5.18. Notes for recent versions: use `padding='max_length'` instead of `pad_to_max_length`, set the TensorBoard log folder with the `TENSORBOARD_LOGGING_DIR` environment variable instead of `logging_dir`, and use `warmup_steps` instead of `warmup_ratio`.

## Tech stack

Python · PyTorch · HuggingFace Transformers · scikit-learn · pandas · matplotlib / seaborn · Google Colab

## Author

**Heba Sleiman**, M2 MoSIG (AI specialization), Université Grenoble Alpes
[GitHub](https://github.com/hibasoleiman) · [LinkedIn](https://linkedin.com/in/hiba-sleiman-39646422a)
