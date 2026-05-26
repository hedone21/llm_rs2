| ID | OP (GGML / IDL) | Qwen tier | shape hint | **CPU** | **L.cpp.GPU** (Adreno) | **L.cpp.HTP0** (Hexagon) | **Ours-NPU** |
|---:|---|:--:|---|:--:|:--:|:--:|:--:|
|  6 | **RMS_NORM** / RMS_NORM | A | [1536] | ✓ | △ 76% | △ 52% | ✓ |
|  4 | **MUL_MAT** / MUL_MAT | A | Q4_0 1536/8960/151936 | △ 81% | △ 52% | △ 25% | ✗ |
| 14 | **ROPE** / ROPE | A | head_dim=128 | ✓ | ✓ | △ 22% | ✗ |
| 15 | **FLASH_ATTN_EXT** / FLASH_ATTN_EXT | A | hs=128 nh=12 nkv=2 | ✓ | △ 54% | ✗ | ✗ |
| 17 | **GET_ROWS** / GET_ROWS | A | vocab=151936 | ✓ | △ 12% | △ 8% | ✗ |
|  7 | **SILU** / UNARY_SILU | A | [8960] | ✓ | △ 25% | △ 25% | ✗ |
|  0 | **MUL** / MUL | A | [8960] | ✓ | ✓ | △ 73% | ✗ |
|  1 | **ADD** / ADD | A | [1536] | ✓ | ✓ | △ 73% | ✗ |
| 12 | **SOFT_MAX** / SOFTMAX | B | nh=12 nctx | ✓ | ✓ | △ 50% | ✗ |
| 18 | **SCALE** / SCALE | B | fused | ✓ | ✓ | ✓ | ✗ |
| 19 | **CPY** / CPY | B | dtype | △ 74% | △ 10% | △ 9% | ✗ |
| 16 | **SET_ROWS** / SET_ROWS | B | KV scatter | △ 72% | △ 14% | △ 7% | ✗ |
|  9 | **SWIGLU** / GLU_SWIGLU | D | fused alt | ✓ | ✓ | △ 25% | ✗ |
|  2 | **SUB** / SUB | E |  | ✓ | ✓ | △ 73% | ✗ |
|  3 | **DIV** / DIV | E |  | ✓ | ✓ | △ 73% | ✗ |
|  5 | **MUL_MAT_ID** / MUL_MAT_ID | E | MoE | ✓ | △ 35% | △ 35% | ✗ |
|  8 | **GELU** / UNARY_GELU | E |  | ✓ | △ 25% | △ 25% | ✗ |
| 10 | **SWIGLU_OAI** / GLU_SWIGLU_OAI | E |  | ✓ | ✓ | △ 50% | ✗ |
| 11 | **GEGLU** / GLU_GEGLU | E |  | ✓ | ✓ | △ 25% | ✗ |
| 13 | **ADD_ID** / ADD_ID | E | MoE | ✓ | ✓ | ✓ | ✗ |
| 20 | **ARGSORT** / ARGSORT | E | host | ✓ | △ 40% | △ 74% | ✗ |
| 21 | **SQR** / SQR | E |  | ✓ | ✓ | △ 50% | ✗ |
| 22 | **SQRT** / SQRT | E |  | ✓ | ✓ | △ 50% | ✗ |
| 23 | **SUM_ROWS** / SUM_ROWS | E | RMSN fused | ✓ | △ 40% | △ 40% | ✗ |
| 24 | **SSM_CONV** / SSM_CONV | E |  | ✓ | ✓ | ✓ | ✗ |
