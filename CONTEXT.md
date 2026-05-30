# llm.rs — KV/Weight 캐시 관리 컨텍스트

decode 루프가 도는 동안 KV·weight 캐시를 *어떤 형태로 저장*하고 *어떻게 관리*하는가에 관한 용어집. 두 직교 축 — **저장 형태(Format)** 와 **관리 동작(Stage)** — 과, 그 둘이 그 위에서 실행되는 **실행 바탕(device)** 을 다룬다. device는 축이 아니라 두 축이 올라앉는 무대다.

## Language

### 두 축

**Format 축 (format axis)**:
"데이터를 어떤 바이트 형태로 저장하고, 그 위에서 attention을 어떻게 계산하는가"의 축. 멤버는 **Format** impl로 표현된다 (noun).

**Stage 축 (stage axis)**:
"decode가 도는 동안 KV·weight·system state에 어떤 관리 동작을 끼워 넣는가"의 축. 멤버는 **Stage**로 표현된다 (verb). eviction·swap·quantize 등.

**직교성 (orthogonality)**:
두 축은 서로 독립이다. M개 Format × N개 Stage의 조합 비용이 M×N이 아니라 **M+N**이 되도록 한다 (조합 폭발 방지).

**실행 바탕은 축이 아니다 (device)**:
두 축이 *그 위에서 실행되는* 물리 자원(연산기·메모리)은 별도의 "축"이 아니라 **바탕**이다. device를 축으로 보면 Format·Stage와 곱해지는 것처럼 오해되지만, device가 바뀌어도(GPU→CPU) Format은 그대로 따라간다 — KIVI는 GPU든 CPU든 KIVI다. 즉 device는 곱이 아니라 *위치*다. 상세는 아래 «실행 바탕 — device».

### 저장 형태 — noun

**Format**:
KV 또는 weight 데이터가 *저장된 바이트 형태*와 그 형태로 연산하는 방식을 캡슐화한 단위. "어떻게 생겼나"라는 명사(noun)에 해당한다. 한 구체적 형태가 곧 하나의 Format이다 — 예: `Standard`(F32/F16/Q4_0 연속), `KIVI`(Q2 packed + F32 residual). 새 Format을 더하면 새 바이트 레이아웃 + 전용 커널(**paired kernel**)이 따라온다 — 무겁지만 격리된 확장.
_Note_: 정밀도(F32/F16/Q4_0)는 한 Format *안의* 파라미터지 Format을 가르는 축이 아니다 (그래서 "Precision"이라 부르지 않는다). `Standard` 하나가 세 정밀도를 품고, Q4_0(양자화)도 `Standard`에 속한다 — Format을 가르는 축은 바이트 레이아웃 + paired kernel이다.
_Avoid_: Layer (저장 형태에는 쓰지 않는다 — "Layer"는 transformer layer 전용. → Flagged ambiguities). Paradigm (Format과 동의어라 폐기).

**KVCacheFormat**:
여러 KV Format을 추상화하는 base trait. 호출자에게 geometry(위치·용량)·mutation(쓰기·compact)·attention만 노출하고, dtype·codebook·정밀도 같은 내부는 숨긴다 (Format-agnostic). 현 코드의 `KVCacheOps`가 이 역할이며 명칭 정리 예정.

**WeightFormat**:
모델 가중치의 여러 Format을 추상화하는 base trait. KVCacheFormat과 대칭이며, dispatch 모드(Full/Skip/Partition)만 노출한다.

**Paired kernel**:
한 Format의 바이트 레이아웃을 읽어 attention을 수행하는 전용 연산 커널. 저장 형태가 다르면 읽는 커널도 달라야 하므로, 새 Format 추가에 따라오는 환원 불가능한 비용이다.

### 관리 동작 — verb

**Stage**:
decode 루프가 도는 동안 매 단계(phase)마다 끼어들어 KV·weight·system state를 변경하는 cross-cutting 동작. "무엇을 하나"라는 동사(verb)에 해당한다. 대부분의 Stage는 저장 형태를 바꾸지 않고 Format의 mutation primitive(예: `compact`)만 호출하므로 가볍게 확장된다 (eviction·swap·quantize). 일부 Stage는 **실행 바탕(device)을 제어**한다 (예: switch — 연산기를 GPU↔CPU 전환). 어느 경우든 *표현(Format)은 안 바꾼다*. 실행 순서는 통합자 책임, 안전(crash-free)은 프레임워크 책임.
_Avoid_: Handler, hook — 같은 개념의 현 코드 구현 명칭일 뿐, 도메인 용어로는 Stage.

**EvictionPolicy** (eviction Stage 내부 규칙):
eviction Stage가 *어느 토큰을 버릴지* 고를 때 참조하는 규칙. Stage 자체가 아니라 한 Stage(eviction)가 감싸 실행하는 부품이며, 모든 Stage가 갖는 건 아니다 (swap·quantize Stage엔 없음). 예: `Sliding`(최근 N), `H2O`(heavy hitter + 최근), `D2O`(버릴 토큰을 병합).

### 실행 바탕 — device

**device (실행 바탕)**:
Format·Stage가 *그 위에서 실행되는* 물리 자원. 두 직교 하위차원으로 분해된다 — **compute**(연산기 = backend: CPU NEON / Adreno OpenCL / Jetson CUDA)와 **data**(메모리 = memory allocator). 둘은 직교다: UMA(ARM SoC)에서는 여러 backend가 한 memory를 공유하고(연산기만 바뀜), discrete GPU에서는 backend마다 별도 memory(VRAM↔RAM 이동)다. 그래서 device를 (backend, memory) 1:1 페어로 묶지 않는다.

**Fabric** (이름 미확정 — v2 §3 참조):
device 자원을 담는 런타임 객체. 내부에 backend 레지스트리 ⊥ memory 레지스트리를 분리 보유하고, `resolve(target)`이 "이 backend로 가려면 어느 memory?"의 UMA/discrete 분기를 한 곳에 가둔다. Stage는 이 객체를 register 시점에 보관하고(`Arc`, interior mutability), device를 바꾸는 Stage(switch)가 그 내부 활성 backend를 mutate한다.

**switch** (device 제어 Stage):
실행 바탕을 바꾸는 Stage. 연산기를 GPU↔CPU 전환하고(필요시 KV를 새 backend로 migrate), 표현(Format)은 안 바꾼다. 안전한 경계(prefill→decode 등)에서만 실행된다. 별도 축이 아니라 [Stage](#관리-동작--verb)의 한 종류다.

**partition** (WeightFormat dispatch 모드):
한 layer의 forward를 여러 backend에 동시 분산하는 것. 별도 축이 아니라 **WeightFormat의 dispatch 모드**(Full / Skip / Partition)이며, 분산 대상 backend는 Fabric에서 받는다. 즉 partition = WeightFormat dispatch(Format 축) × companion backend(실행 바탕)의 곱이다.

### Stage가 Format을 잡는 방식 — handle 3종

Stage가 자신이 관리할 Format을 보유하는 참조의 정적 타입. 순서 없는 3개 메뉴이며, 4번째는 없다.

**Base-trait-handle**:
`Arc<dyn KVCacheFormat>` 형태. 이를 든 Stage는 어느 Format인지 *몰라야* 한다. 예: position만 보는 Sliding.

**Concrete-handle**:
`Arc<StandardFormat>`처럼 특정 Format 타입을 든 형태. 그 타입을 *아는 것이 정상*이다(그게 concrete를 든다는 의미). 예: 원본 K를 직접 읽어야 하는 D2O.

**Capability-handle**:
`Arc<dyn SomeCapability>` 형태. 이종 Format을 가로지르는 *능력*을 추상화한다. 소비자가 둘 이상일 때만 만든다(하나뿐이면 가설적 seam이라 만들지 않는다).

## Flagged ambiguities

- **Layer는 transformer layer 전용**: "Layer"(`LlamaLayer`/`TransformerLayer`, 16개 디코더 블록, `LayerSlot`·`layer_idx` 등)는 *모델 구조*만 가리킨다. KV·weight의 *저장 형태*는 절대 "Layer"라 부르지 않고 **Format**(`KVCacheFormat`/`WeightFormat`)이라 부른다. (이전엔 둘 다 "Layer"여서 충돌 — Format으로 분리해 해소.)
- **Stage vs Handler/hook**: 도메인 용어는 **Stage**. 현 코드의 `CachePressureHandler` + 여러 decode hook trait은 같은 개념의 과도기 구현이다.
- **Eviction은 동작이지 형태가 아니다**: eviction은 토큰을 버리는 *동작*(verb)이라 **Stage**(Stage 축)다. KIVI 양자화는 바이트 형태를 바꾸는 *저장 형태*(noun)라 **Format**(Format 축)다. "KIVI도 양자화라는 동작이니 Stage 아니냐"는 흔한 혼동 — KIVI는 전용 paired kernel을 동반하므로 Format이다.
- **device는 축이 아니라 바탕**: 저장 형태(Format)·관리 동작(Stage)은 직교 *축*(곱해진다)이지만, 연산기·메모리(device)는 두 축이 *그 위에서 실행되는 바탕*이다. switch(device 전환)는 새 축이 아니라 device를 바꾸는 **Stage**이고, partition(multi-device 분산)은 **WeightFormat dispatch 모드**다. "device를 3번째 축으로" 라는 충동은 기각됐다 — device가 바뀌어도 Format은 따라가므로 곱이 아니라 위치다 (2026-05-30 grill 결정).

## Example dialogue

> **Dev**: KIVI를 추가하려는데, Sliding이나 H2O처럼 Stage로 넣으면 되나요?
> **Expert**: 아니요. Sliding·H2O는 "어느 토큰을 버리나"라는 **동작(Stage)** 이라 저장 형태를 안 건드려요. KIVI는 KV를 Q2로 *다시 깔고* 그 위에서 attention하는 **저장 형태(Format)** 예요. 그래서 KVCacheFormat impl + paired kernel이 따라옵니다.
> **Dev**: 그럼 KIVI Format 위에서 Sliding을 돌릴 수도 있나요?
> **Expert**: 네. 그게 두 축이 **직교**한다는 뜻이에요. KIVI(Format 축) 위에서 Sliding(Stage 축)이 `compact`만 호출하면 됩니다. Sliding은 그 밑이 Q2인지 F32인지 몰라요 — **base-trait-handle**을 들었으니까요.
> **Dev**: 그 "Format"이 transformer layer랑 다른 건가요?
> **Expert**: 완전히 달라요. transformer **layer**는 모델 디코더 블록(16개)이고, **Format**은 그 layer들의 KV/weight가 *어떤 바이트로 저장되나*예요. "layer 16번의 KV를 KIVI Format으로 저장"처럼 둘이 같이 쓰입니다.
> **Dev**: D2O도 base-trait-handle인가요?
> **Expert**: D2O는 원본 K를 직접 읽어 병합해야 해서 **concrete-handle**(`Arc<StandardFormat>`)을 듭니다. 특정 Format을 아는 게 정상이에요 — 그게 concrete를 든다는 의미니까요.
