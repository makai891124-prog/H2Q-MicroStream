# Training Compare: sta_v2 vs binary_sta(packbits)

- device: cuda:0
- config: {'steps': 40, 'batch_size': 4, 'seq_len': 128, 'dim': 128, 'layers': 4, 'lr': 0.0003, 'corpus': 'data/open_corpus/open_corpus.txt'}

## sta_v2
- avg_loss: 4.728309
- last_loss: 3.905694
- tokens_per_sec: 20206.85
- peak_vram_mb: 39.03
- readability: 0.9438

## binary_sta(packbits)
- avg_loss: 4.500698
- last_loss: 3.851679
- tokens_per_sec: 44241.91
- peak_vram_mb: 32.88
- readability: 0.7875

## Delta
- loss_last_binary_minus_sta: -0.054015
- tokens_per_sec_binary_over_sta: 2.1895x
- peak_vram_mb_binary_minus_sta: -6.15
- readability_binary_minus_sta: -0.1563

## Notes
- This is a short controlled training experiment for directional comparison.
- For final model decision, rerun with longer steps and fixed seeds over multiple trials.
