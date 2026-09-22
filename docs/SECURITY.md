# Bezpečnosť a oprávnenia

## Threat model

Chrániť obsah dokumentov, API kľúče, identitu používateľa, integritu citácií a
externé systémy. Hlavné riziká sú prompt injection v dokumente, cross-tenant únik,
SSRF cez všeobecný HTTP tool, neautorizované mazanie, exfiltrácia pri OCR/web volaní,
škodlivé alebo komprimované dokumenty a citlivé dáta v logoch.

## Odporúčané oprávnenia

Neviem presné využitie systému. Pri obyčajnom chatovaciom asistentovy na webe netreba používateľov, ale ak by systém bol využivaný ako internŷ nástroj alebo nástroj s využitím profilov, tak by bolo nasadené použitie OIDC/OAuth2 a tieto roly:

| Rola | Čítať/QA | Importovať | Mazať vlastné | Mazať všetky | Meniť nástroje | Audit |
|---|---:|---:|---:|---:|---:|---:|
| `reader` | áno | nie | nie | nie | nie | nie |
| `editor` | áno | áno | áno | nie | nie | nie |
| `auditor` | áno | nie | nie | nie | nie | áno |
| `admin` | áno | áno | áno | áno | áno | áno |

Každý dokument má `tenant_id`, voliteľne `owner_id` a klasifikáciu. Qdrant filter
musí byť zostavený serverom z overeného tokenu, nikdy z voľného modelového výstupu.
Pri citlivých kolekciách je bezpečnejšia samostatná collection alebo samostatný
Qdrant projekt pre každý tenant.

## Bezpečnostné kontroly ingestu

- whitelist prípon aj kontrola skutočného formátu podľa signatúry,
- limit veľkosti, počtu strán, rozbalených bajtov a času spracovania,
- antivírus pred parsovaním v produkcii (check dokumentu, ako pri .docx dokumentoch s obsahujúcim skriptom),
- šifrovanie disku a objektového úložiska,
- kontrola duplicity cez SHA-256,
- retention politika a overiteľné zmazanie súboru, vektorov, cache a záloh.

## Prompt injection

Text dokumentu je **nedôveryhodný údaj**, nie inštrukcia. System prompt musí jasne
oddeliť otázku a dôkazy. Model nesmie podľa textu chunku meniť oprávnenia, volať
nástroj ani prezradiť iný dokument. Tool selection sa povoľuje iba policy vrstvou.
Jev porovnáva každé tvrdenie iba s chunkmi, ktoré návrh skutočne cituje; chýbajúce
alebo cudzie `chunk_id` sa odmietne ešte pred externým volaním. Ide o kontrolu
podloženosti, nie o náhradu autorizácie, ochrany pred prompt injection ani ľudskej
revízie citlivých odpovedí.
Do eval datasetu patria útoky typu „ignoruj pravidlá“, falošné citácie a pokusy
vyžiadať dokument iného tenantu.

## Tajomstvá a sieť

- lokálne `.env` iba pre vývoj; produkčne Vault/KMS/secret manager,
- samostatné service accounts a minimálne scopes,
- rotácia, expirácia a audit použitia kľúčov,
- outbound allowlist na konkrétne API hostnames,
- TypeSafe AI je ďalší externý príjemca otázky, odpovede a citovaných chunkov; pred použitím s citlivými dokumentmi posúďte prenos dát a zmluvné podmienky. T.z. využitie lokálneho riešenia Laya (Jev alternatíva), využitie lokálnych malých LLM modelov (ak to infraštruktúra a produkcia dovoľuje alebo ak sa narába s citlivými údajmi) a asi nevyhnutné využitie modelov na európskych serveroch (consent of processing data)
- TLS verifikácia, timeouty, maximálna odpoveď a rate limiting,
- PII redakcia pred externým API, ak to právny základ vyžaduje,
- zákaz logovania promptov a dokumentov v defaultnom režime.

## Audit

Pri nasadení s možnosťou využia používateľov.

Zaznamenávanie kto, kedy a v akom tenante importoval, čítal alebo zmazal dokument,
ktorý nástroj bol volaný, model/version, hash prompt šablóny a ID citovaných chunkov.
Audit log má byť append-only a oddelený od aplikačných logov. Samotný text dokumentu
do auditu nepatrí.

## Medzery prototypu

CLI identita z `.env` nie je autentifikácia. Embedded Qdrant ani lokálny manifest
neposkytujú serverové RBAC. Pred sieťovým alebo multi-user nasadením preto treba
doplniť API gateway, overenie tokenov, databázové politiky, šifrovanie a centrálny
audit.
