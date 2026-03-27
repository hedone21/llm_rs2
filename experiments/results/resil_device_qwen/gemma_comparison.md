```
=====================================================================
  resil-mode-c-repeated vs resil-baseline
  Directive(seq=1, [KvEvictH2o(0.50)]) at token 64; Directive(seq=2, [KvEvictH2o(0.50)]) at token 128; Directive(seq=3, [KvEvictH2o(0.50)]) at token 192; Directive(seq=4, [KvEvictH2o(0.50)]) at token 256; Directive(seq=5, [KvEvictH2o(0.50)]) at token 384  |  h2o
=====================================================================

  -- Speed -------------------------------------------------------
  Avg TBT:           62.1ms -> 70.4ms   (+13.5%)
  Avg Forward:       60.2ms -> 68.0ms   (+13.0%)
  Throttle:          0ms total
  Throughput:        16.1 -> 14.2 t/s   (-11.9%)

  -- Quality ------------------------------------------------------
  First Divergent Token:   69 / 511
  Exact Match Rate:        0.172  (88/511)
  Suffix EMR (post-FDT):   0.043  (19/442)
  ROUGE-L F1:              0.288
  BLEU-4:                  0.198
  Top-K Overlap (avg):     0.535
  Top-K Overlap (pre-FDT): 0.990
  Top-K Overlap (post-FDT):0.464

  -- Resources ----------------------------------------------------
  Evictions:          5 (1348 tokens removed)
  Cache Utilization:  0.124  (254/2048)
  RSS Start:          2266.2MB -> 2265.9MB
  RSS End:            2250.3MB -> 2218.4MB

=====================================================================
```
