## Additional research

Model performances where evaluated when applying only strict local attention. For this, a custom local attention head was implemented that strictly masks the receptive field at every position to a given window size. This approach is in contrast to [the current approach](https://github.com/lucidrains/local-attention).

|       Model                |     Local attention    |     Training Time  |     ROC AUC  |     PR AUC  |
|----------------------------|------------------------|--------------------|--------------|-------------|
|     203 window size        |   8/8 heads per layer  |   ~20h             |   0.999      |   0.781     |
|     600 window size        |   8/8 heads per layer  |   ~44h             |   0.999      |   0.809     |
|     Transformer L (paper)  |   5/8 heads per layer  |   ~6h              |   0.999      |   0.829     |



