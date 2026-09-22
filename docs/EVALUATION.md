# Testovanie kvality a prevádzkový monitoring

## Golden dataset

Každý príklad má obsahovať:

```json
{
  "question": "...",
  "gold_answer": "...",
  "gold_document_ids": ["..."],
  "gold_pages": [3, 8],
  "answerable": true,
  "required_claims": ["..."],
  "forbidden_claims": ["..."]
}
```

Aspoň 20–30 % otázok má byť zámerne nezodpovedateľných. Pridanie parafrázy,
preklepy, multi-hop otázky, konfliktné verzie dokumentov, tabuľky, skeny, prompt
injection, citlivé dokumenty iného tenantu a časovo premenlivé otázky.

## Vhodné verejné datasety

| Dataset | Čo testuje | Prečo je vhodný |
|---|---|---|
| [HotpotQA](https://hotpotqa.github.io/) | multi-hop retrieval | otázky vyžadujú viac dokumentov a obsahujú supporting facts |
| [QASPER](https://allenai.org/data/qasper) | dlhé vedecké dokumenty | odpovede aj evidence z celých článkov, vrátane nezodpovedateľných otázok |
| [RAGBench](https://huggingface.co/datasets/rungalileo/ragbench) | end-to-end RAG | anotácie pre relevance, utilization a completeness |
| [CRAG](https://github.com/facebookresearch/CRAG) | robustný RAG | dynamické a faktografické otázky, retrieval/generation hodnotenie |
| [TriviaQA na Kaggle](https://www.kaggle.com/datasets/programmerrdai/triviaqa) | QA s viacerými evidence dokumentmi | približne šesť dôkazových dokumentov na otázku |
| [CoQA na Kaggle](https://www.kaggle.com/datasets/jeromeblanchet/conversational-question-answering-dataset-coqa) | konverzačný kontext | nadväzujúce otázky nad pasážou |
| [SubjQA na Kaggle](https://www.kaggle.com/datasets/arashnic/subjqa-question-answering-dataset) | subjektívne/nezodpovedateľné otázky | test odmietnutia a rozdielu fakt verzus názor |

Verejné datasety rýchle overenie: finálne rozhodnutie musí používať vlastný doménový
golden set, lebo jazyk, štruktúra dokumentov a riziko sa líšia.

## Metriky

### Retrieval

- Recall@5 a Hit@5 voči gold chunkom/stranám,
- Mean Reciprocal Rank a normalized Discounted Cumulative Gain@5,
- document recall a page recall,
- podiel otázok bez výsledku nad minimálnym skóre.

### Odpoveď a citácie

- answer exact match/F1 pre krátke odpovede,
- semantická správnosť s kalibrovaným LLM judge + ľudská vzorka,
- faithfulness/groundedness po jednotlivých tvrdeniach,
- citation precision a citation recall,
- completeness: či nechýba podstatný gold claim,
- unsupported claim rate.
- presnosť Jev triedenia `supports`/`contradicts`/`says_nothing` po tvrdeniach,
- podiel odpovedí, kde zoznam tvrdení nepokrýva celý výsledný text.

### Abstencia

- precision abstencie: koľko odmietnutí bolo naozaj nezodpovedateľných,
- recall abstencie: koľko nezodpovedateľných otázok systém správne odmietol,
- selective accuracy a risk-coverage curve,
- Expected Calibration Error a Brier score confidence.

### Prevádzka

- p50/p95/p99 celkovej latencie a latencie každého uzla,
- input/output tokeny, embedding tokeny a cena na otázku,
- počet retrieval retry, web fallback rate a abstention rate,
- chyby a rate limits podľa providera,
- počet opakovaní po 10/60 s a čas strávený čakaním na API,
- objem Qdrantu, ingest throughput a OCR failure rate.

## Ablácie a optimalizácia

Použitie rovnakého zmrazeného test setu a deaktivovanie rôznych uzlov (snažiť sa zrýchliť systém, znížiť náklady, ...):

| Experiment | Zmena | Hypotéza |
|---|---|---|
| A0 | plný graf | kontrolná konfigurácia |
| A1 | bez query rewrite | nižšia cena/čas, možný pokles multi-hop recall |
| A2 | verifier iba pri nízkom retrieval score | menej LLM volaní, riziko nepodložených odpovedí |
| A3 | Jev verzus iný verifier alebo vypnutá kontrola | cena a latencia oproti unsupported claim rate |
| A4 | lokálne embeddingy | súkromie a cena vs. recall v slovenčine |
| A5 | hybrid dense + BM25 | lepšie presné termíny/čísla za vyššiu zložitosť |
| A6 | reranker po top-20 | vyšší Recall@5 za dodatočnú latenciu |
| A7 | chunk 800/1400/2200 | optimum kontextu, recall a ceny |
| A8 | web fallback vypnutý | bezpečnosť a cena vs. coverage |


## Prompt/model experimenty

- verziovanie prompt šablóny hashom,
- pri GPT-5.6 Luna meranie kvality pri `reasoning_effort=low` oproti `medium`;
  Jev používa typované pravdepodobnostné verdikty,
- každú kombináciu model + prompt + retriever,
- prahy Jev kalibrovať osobitne na slovenskom validačnom sete a sledovať nesprávne potvrdené citácie aj zbytočné abstencie,

## Monitoring feedbacku a budúci ML triage

`/good` a `/bad` ukladajú štítok k interakcii. Negatívne chaty patria do review
fronty spolu s nízkym confidence, web fallbackom, vysokou latenciou a vysokým
počtom tokenov.

Po nazbieraní a **ľudskom overení** dostatočného množstva príkladov možno vytrénovať jednoduchú logistickú regresiu alebo gradient boosting na predikciu priority review. Vstupy: retrieval skóre, rozdiel top1–top2, citation coverage, verifier skóre, route, retry count, latencia a tokeny. Text otázky/odpovede radšej neukladať ako feature bez privacy posúdenia/potvrdenia. 

V produkcii a so súhlasom spracovania údajov, vytvorenie modelu na triedenie ekosentimentu, kde by sa na základe interkacie poúživateľa (vyjadrovanie a použitie určitých slov) triedili chaty (RandomForestClassificator), na neskoršie vyhodnocovanie a úpravu systému.
