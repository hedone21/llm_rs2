```
=====================================================================
  resil-mode-c-repeated vs resil-baseline
  Directive(seq=1, [KvEvictH2o(0.50)]) at token 64; Directive(seq=2, [KvEvictH2o(0.50)]) at token 128; Directive(seq=3, [KvEvictH2o(0.50)]) at token 192; Directive(seq=4, [KvEvictH2o(0.50)]) at token 256; Directive(seq=5, [KvEvictH2o(0.50)]) at token 384  |  h2o
=====================================================================

  -- Speed -------------------------------------------------------
  Avg TBT:           102.6ms -> 88.3ms   (-13.9%)
  Avg Forward:       100.8ms -> 86.4ms   (-14.3%)
  Throttle:          0ms total
  Throughput:        9.7 -> 11.3 t/s   (+16.2%)

  -- Quality ------------------------------------------------------
  First Divergent Token:   120 / 511
  Exact Match Rate:        0.250  (128/511)
  Suffix EMR (post-FDT):   0.020  (8/391)
  ROUGE-L F1:              0.376
  BLEU-4:                  0.295
  Top-K Overlap (avg):     0.269
  Top-K Overlap (pre-FDT): 0.988
  Top-K Overlap (post-FDT):0.049

  -- Resources ----------------------------------------------------
  Evictions:          5 (1331 tokens removed)
  Cache Utilization:  0.124  (253/2048)
  RSS Start:          3018.1MB -> 3018.1MB
  RSS End:            3049.6MB -> 3015.9MB

=====================================================================
```
