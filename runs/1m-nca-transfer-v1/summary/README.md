# One-seed exploratory NCA → TinyStories transfer

Model: 1,081,177 trainable parameters; {'vocab_size': 50257, 'context_length': 256, 'layers': 4, 'heads': 2, 'embedding_dim': 20, 'dropout': 0.0, 'output_vocab_size': None, 'tie_embeddings': True}.

NCA budget: 19,715,325 usable labels per regime. TinyStories: 5,000,000 training tokens per condition.

| Condition | Final validation loss | Final perplexity |
| --- | ---: | ---: |
| downstream/scratch | 5.19832 | 180.97 |
| downstream/nca_20_30 | 4.92979 | 138.35 |
| downstream/nca_30_40 | 5.04776 | 155.67 |
| downstream/nca_40_50 | 4.83565 | 125.92 |
| downstream/nca_50_plus | 4.94507 | 140.48 |

See summary.json for milestones, throughput, runtime, and loss differences. This one-seed run does not establish statistical significance.
