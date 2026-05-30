# llm.rs — KV/Weight 캐시 관리 컨텍스트

decode 루프가 도는 동안 KV·weight 캐시를 *어떤 형태로 저장*하고 *어떻게 관리*하는가에 관한 용어집. 두 직교 축 — **저장 형태(Format)** 와 **관리 동작(Stage)** — 을 중심으로 한다.

## Language

### 두 축

**Storage 축 (storage axis)**:
"데이터를 어떤 바이트 형태로 저장하고, 그 위에서 attention을 어떻게 계산하는가"의 축. 멤버는 **Format** impl로 표현된다.

**Policy 축 (policy axis)**:
"이미 저장된 토큰 중 어느 것을 버리거나 병합하는가"의 축. 멤버는 **Stage**(특히 **EvictionPolicy**)로 표현된다.

**직교성 (orthogonality)**:
두 축은 서로 독립이다. M개 저장 형태 × N개 관리 정책의 조합 비용이 M×N이 아니라 **M+N**이 되도록 한다 (조합 폭발 방지).

### 저장 형태 — noun

**Format**:
KV 또는 weight 데이터가 *저장된 바이트 형태*와 그 형태로 연산하는 방식을 캡슐화한 단위. "어떻게 생겼나"라는 명사(noun)에 해당한다. 새 Format을 더하면 새 바이트 레이아웃 + 전용 커널(**paired kernel**)이 따라온다 — 무겁지만 격리된 확장.
_Avoid_: Layer (저장 형태에는 쓰지 않는다 — "Layer"는 transformer layer 전용. → Flagged ambiguities).

**KVCacheFormat**:
KV 캐시의 한 저장 **paradigm**을 캡슐화하는 base trait. 호출자에게 geometry(위치·용량)·mutation(쓰기·compact)·attention만 노출하고, dtype·codebook·정밀도 같은 내부는 숨긴다 (storage-format-agnostic). 현 코드의 `KVCacheOps`가 이 역할이며 명칭 정리 예정.

**WeightFormat**:
모델 가중치의 한 저장 paradigm을 캡슐화하는 base trait. KVCacheFormat과 대칭이며, dispatch 모드(Full/Skip/Partition)만 노출한다.

**Paradigm**:
하나의 구체적 저장 형태. 예: `Standard`(F32/F16/Q4_0 연속), `KIVI`(Q2 packed + F32 residual). KVCacheFormat impl 하나가 paradigm 하나다.

**Paired kernel**:
한 Format paradigm의 바이트 레이아웃을 읽어 attention을 수행하는 전용 연산 커널. 저장 형태가 다르면 읽는 커널도 달라야 하므로, 새 Format 추가에 따라오는 환원 불가능한 비용이다.

### 관리 동작 — verb

**Stage**:
decode 루프가 도는 동안 매 단계(phase)마다 끼어들어 KV·weight·system state를 변경하는 cross-cutting 동작. "무엇을 하나"라는 동사(verb)에 해당한다. 저장 형태를 바꾸지 않고 Format의 mutation primitive(예: `compact`)만 호출하므로 가볍게 확장된다. 실행 순서는 통합자 책임, 안전(crash-free)은 프레임워크 책임.
_Avoid_: Handler, hook — 같은 개념의 현 코드 구현 명칭일 뿐, 도메인 용어로는 Stage.

**EvictionPolicy**:
"어느 토큰을 버릴지" 결정하는 규칙. Policy 축의 한 종류이며 Stage가 이를 감싸 실행한다. 예: `Sliding`(최근 N), `H2O`(heavy hitter + 최근), `D2O`(버릴 토큰을 병합).

### Stage가 Format을 잡는 방식 — handle 3종

Stage가 자신이 관리할 Format을 보유하는 참조의 정적 타입. 순서 없는 3개 메뉴이며, 4번째는 없다.

**Base-trait-handle**:
`Arc<dyn KVCacheFormat>` 형태. 든 Stage는 paradigm을 *몰라야* 한다. 예: position만 보는 Sliding.

**Concrete-handle**:
`Arc<StandardFormat>`처럼 특정 paradigm 타입을 든 형태. 그 타입을 *아는 것이 정상*이다(그게 concrete를 든다는 의미). 예: 원본 K를 직접 읽어야 하는 D2O.

**Capability-handle**:
`Arc<dyn SomeCapability>` 형태. 이종 Format을 가로지르는 *능력*을 추상화한다. 소비자가 둘 이상일 때만 만든다(하나뿐이면 가설적 seam이라 만들지 않는다).

## Flagged ambiguities

- **Layer는 transformer layer 전용**: "Layer"(`LlamaLayer`/`TransformerLayer`, 16개 디코더 블록, `LayerSlot`·`layer_idx` 등)는 *모델 구조*만 가리킨다. KV·weight의 *저장 형태*는 절대 "Layer"라 부르지 않고 **Format**(`KVCacheFormat`/`WeightFormat`)이라 부른다. (이전엔 둘 다 "Layer"여서 충돌 — Format으로 분리해 해소.)
- **Stage vs Handler/hook**: 도메인 용어는 **Stage**. 현 코드의 `CachePressureHandler` + 여러 decode hook trait은 같은 개념의 과도기 구현이다.
- **Eviction은 동작이지 형태가 아니다**: eviction은 토큰을 버리는 *동작*(verb)이라 **Stage**(Policy 축)다. KIVI 양자화는 바이트 형태를 바꾸는 *저장 형태*(noun)라 **Format**(Storage 축)다. "KIVI도 양자화라는 동작이니 Stage 아니냐"는 흔한 혼동 — KIVI는 전용 paired kernel을 동반하므로 Format이다.

## Example dialogue

> **Dev**: KIVI를 추가하려는데, Sliding이나 H2O처럼 Stage로 넣으면 되나요?
> **Expert**: 아니요. Sliding·H2O는 "어느 토큰을 버리나"라는 **동작(Stage)** 이라 저장 형태를 안 건드려요. KIVI는 KV를 Q2로 *다시 깔고* 그 위에서 attention하는 **저장 형태(Format)** 예요. 그래서 KVCacheFormat impl + paired kernel이 따라옵니다.
> **Dev**: 그럼 KIVI Format 위에서 Sliding을 돌릴 수도 있나요?
> **Expert**: 네. 그게 두 축이 **직교**한다는 뜻이에요. KIVI(Storage 축) 위에서 Sliding(Policy 축)이 `compact`만 호출하면 됩니다. Sliding은 그 밑이 Q2인지 F32인지 몰라요 — **base-trait-handle**을 들었으니까요.
> **Dev**: 그 "Format"이 transformer layer랑 다른 건가요?
> **Expert**: 완전히 달라요. transformer **layer**는 모델 디코더 블록(16개)이고, **Format**은 그 layer들의 KV/weight가 *어떤 바이트로 저장되나*예요. "layer 16번의 KV를 KIVI Format으로 저장"처럼 둘이 같이 쓰입니다.
> **Dev**: D2O도 base-trait-handle인가요?
> **Expert**: D2O는 원본 K를 직접 읽어 병합해야 해서 **concrete-handle**(`Arc<StandardFormat>`)을 듭니다. 특정 paradigm을 아는 게 정상이에요 — 그게 concrete를 든다는 의미니까요.
