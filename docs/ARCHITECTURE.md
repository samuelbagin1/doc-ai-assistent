# Architektúra riešenia

## Ciele a hranice

Systém je navrhnutý ako lokálna CLI aplikácia s jasnými portmi pre dokumentové
úložisko, vektory, embeddingy, LLM a web. Jadro nepozná konkrétny UI framework.
To umožňuje neskôr doplniť REST API alebo webové UI bez zmeny rozhodovacieho grafu.

## UML komponentový diagram

```mermaid
classDiagram
    class AssistantCLI {
      +run()
      +command(text)
    }
    class DocumentService {
      +add(path) DocumentRecord
      +delete(id) DocumentRecord
      +list() DocumentRecord[]
    }
    class DocumentStore {
      +prepare(path) chunks
      +commit(record)
      +delete_record(id)
    }
    class SectionAwareChunker {
      +split(pages) Chunk[]
    }
    class QdrantVectorStore {
      +add(chunks)
      +search(query, k=5)
      +delete_document(id)
    }
    class RAGWorkflow {
      +ask(question) AssistantAnswer
    }
    class DeepSeekRAGModel {
      +draft(question, evidence)
      +rewrite_query(...)
    }
    class JevVerifier {
      +verify(question, draft, evidence) Verification
    }
    class ModelCallGate {
      +call(operation, provider)
    }
    class OpenAIWebSearch {
      +search(question) AssistantAnswer
    }
    class MetricsStore {
      +record(...)
      +feedback(...)
      +summary()
    }

    AssistantCLI --> DocumentService
    AssistantCLI --> RAGWorkflow
    AssistantCLI --> MetricsStore
    DocumentService --> DocumentStore
    DocumentService --> SectionAwareChunker
    DocumentService --> QdrantVectorStore
    RAGWorkflow --> QdrantVectorStore
    RAGWorkflow --> DeepSeekRAGModel
    RAGWorkflow --> JevVerifier
    RAGWorkflow --> OpenAIWebSearch
    DeepSeekRAGModel --> ModelCallGate
    JevVerifier --> ModelCallGate
    OpenAIWebSearch --> ModelCallGate
```

## UML sekvenčný diagram odpovede

```mermaid
sequenceDiagram
    actor U as Používateľ
    participant C as CLI
    participant G as LangGraph
    participant Q as Qdrant
    participant D as DeepSeek
    participant J as Jev / TypeSafe
    participant W as OpenAI Web Search

    U->>C: otázka + Enter
    C->>G: ask(otázka)
    G->>Q: similarity search, k=5
    Q-->>G: chunky + skóre + metadata
    G->>D: draft(otázka, chunky)
    D-->>G: odpoveď + atómové tvrdenia + chunk_id citácie
    G->>J: verify(otázka, tvrdenia, citované chunky)
    J-->>G: podpora tvrdení, úplnosť, relevancia
    alt dôkazy sú dostatočné
        G-->>C: odpoveď + dokumentové zdroje
    else prvý pokus nestačí
        G->>D: rewrite_query(chýbajúci dôkaz)
        D-->>G: doplňujúci query
        G->>Q: druhý similarity search
    else interné pokusy vyčerpané a web povolený
        G->>W: Responses API + web_search
        W-->>G: odpoveď + URL citácie
        G-->>C: webová odpoveď alebo abstencia
    else bez spoľahlivých zdrojov
        G-->>C: abstencia + dôvod
    end
    C-->>U: odpoveď, confidence, čas, zdroje
```

## UML stavový diagram

```mermaid
stateDiagram-v2
    [*] --> Retrieve
    Retrieve --> Draft
    Draft --> Verify
    Verify --> Answer: faithful && sufficient && confidence >= threshold
    Verify --> Rewrite: attempt < max_attempts
    Rewrite --> Retrieve
    Verify --> WebSearch: attempts exhausted && web enabled
    Verify --> Abstain: attempts exhausted && web disabled
    WebSearch --> WebAnswer: citations && confidence >= threshold
    WebSearch --> Abstain: no citations / low confidence
    Answer --> [*]
    WebAnswer --> [*]
    Abstain --> [*]
```

## UML deployment diagram

```mermaid
flowchart TB
    subgraph Workstation[Používateľská stanica]
      CLI[Python CLI]
      FILES[(files/)]
      SQLITE[(metrics.sqlite3)]
      QLOCAL[(Embedded Qdrant)]
      CLI --- FILES
      CLI --- SQLITE
      CLI --- QLOCAL
    end
    CLI -->|TLS, API key| DS[DeepSeek API]
    CLI -->|TLS, API key| TS[TypeSafe Jev API]
    CLI -->|TLS, API key| OAI[OpenAI Embeddings + Responses Web Search]
    CLI -. iba OCR_MODE=hosted .-> FC[Firecrawl Parse]
```

## Externé systémy a API

Provider adaptéry sú na okraji systému. DeepSeek používa OpenAI-compatible Chat
Completions s JSON režimom. Jev používa TypeSafe Python SDK: pre každé tvrdenie
vracia typovanú voľbu `supports`/`contradicts`/`says_nothing` a pre dostatočnosť,
relevanciu a úplnosť numerické odpovede. Jev neposkytuje finálny text odpovede.
OpenAI web používa Responses API s explicitným nástrojom
`{"type":"web_search"}` a zbiera URL anotácie. Embedding adapter možno vymeniť za
lokálny SentenceTransformer.

Jeden `ModelCallGate` obmedzuje súbežnosť volaní modelových API na 1. Retry
schéma je 10 s, potom 60 s; čakanie prebieha mimo semafora. Rešpektuje sa dlhší
`Retry-After`. Trvalé chyby kreditu/kvóty sa neopakujú a interné retry SDK sú
vypnuté. Po poslednej chybe sa výnimka propaguje do CLI; nie je to verifikovaná
odpoveď ani tichá abstencia.

Jev kontroluje dokumentovú vetvu. Webový fallback má URL citácie, ale bez
obsahu webových stránok nevykonáva claim-level kontrolu Jev. Pre citlivé nasadenie
treba web vypnúť alebo doplniť načítanie a overenie citovaných stránok.

Pre budúce podnikové API treba pridať samostatný, úzko typovaný tool adapter:

1. pevné meno operácie, napr. `get_invoice_status`,
2. JSON Schema vstupu a výstupu,
3. pevný hostname a metódu; žiadna modelom zadaná URL,
4. tenant/user authorization pred volaním,
5. timeout, retry iba pre idempotentné operácie a circuit breaker,
6. audit event bez tajomstiev a citlivého payloadu,
7. výsledok zaradiť medzi dôkazy a znovu overiť.

Zápisové operácie musia mať human confirmation a idempotency key. Aktuálna verzia
vykonáva iba čítacie externé volania.

## Kľúčové návrhové rozhodnutia

- **LangGraph namiesto voľného agenta:** povolené prechody, retry a náklady sú
  deterministické.
- **Oddelený generator a verifier:** DeepSeek navrhuje odpoveď, Jev nezávisle
  klasifikuje citované tvrdenia a dostatočnosť dôkazov.
- **Citácie cez ID:** model neprodukuje názov súboru ani stranu; iba vyberá ID a
  aplikácia z neho vytvorí referenciu.
- **Tenant filter pri každom čítaní a mazaní:** ochrana nesmie existovať iba v UI.
- **Abstencia je úspešný výsledok:** nejde o exception ani neúspech aplikácie.
