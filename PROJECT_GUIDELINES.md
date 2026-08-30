# Project 2 — CORES for Monocular Depth Estimation

## 1. Obiettivo

Il progetto studia se le risposte dei layer convoluzionali di una rete per **Monocular Depth Estimation (MDE)** permettono di distinguere immagini **in-distribution (ID)** da immagini **out-of-distribution (OOD)**.

La configurazione principale proposta è:

- **NYU Depth v2**: dataset ID, composto da scene indoor;
- **KITTI**: dataset OOD, composto da scene stradali outdoor;
- **FastDepth** o un'altra rete convoluzionale leggera: modello MDE;
- **CORES**: metodo per produrre uno score OOD dalle attivazioni convoluzionali;
- **AUROC e FPR95**: metriche principali per OOD detection;
- **RMSE, AbsRel, δ1, δ2, δ3**: metriche per depth estimation.

Il contributo sperimentale sarà il confronto tra score estratti da layer diversi e una loro aggregazione multi-layer.

## 2. Requisiti del bando

Il progetto deve:

- essere implementato in Python con PyTorch;
- supportare l'esecuzione su GPU;
- essere eseguibile senza errori;
- includere preparazione dei dataset, rete, training ed evaluation;
- addestrare una rete MDE leggera su dati ID;
- integrare CORES in una rete convoluzionale;
- valutare separatamente qualità depth e capacità di rilevare OOD;
- includere uno studio di ablation su layer e configurazioni convoluzionali;
- essere accompagnato da dataset o relativi link, README e presentazione;
- contenere un contributo originale e scelte sperimentali motivate.

Il confronto fra due o più modelli è incoraggiato, ma non dichiarato obbligatorio. Prima di investire tempo in un secondo modello, è opportuno confermare con la docente che una variante della stessa architettura sia sufficiente per l'ablation richiesta.

## 3. Scope consigliato

### Ambiente di esecuzione

Il progetto sarà sviluppato in modalità **Kaggle-first**:

- il deliverable eseguibile principale sarà un notebook Jupyter (`.ipynb`);
- il notebook dovrà funzionare su una sessione Kaggle con acceleratore GPU T4 x2;
- la baseline userà inizialmente una sola T4 (`cuda:0`), senza distributed training;
- non dovrà dipendere da percorsi locali Windows;
- input e dataset Kaggle saranno cercati sotto `/kaggle/input`;
- checkpoint, log, grafici e risultati saranno scritti sotto `/kaggle/working`;
- l'eventuale portabilità verso Google Colab sarà mantenuta tramite una piccola configurazione dei percorsi, senza duplicare il notebook;
- ogni installazione di dipendenze dovrà essere dichiarata nelle prime celle e mantenuta al minimo;
- il notebook dovrà poter ripartire da checkpoint per tollerare i limiti temporali delle sessioni cloud.

Quando utile, il codice riutilizzabile potrà essere organizzato in moduli Python generati o importati dal notebook, ma l'intero esperimento dovrà restare avviabile dal notebook principale.

### Minimum Viable Project

La versione minima completa deve includere:

1. un modello FastDepth funzionante;
2. training o fine-tuning su NYU Depth v2;
3. valutazione depth sul validation/test split di NYU;
4. preparazione di immagini KITTI compatibili con l'input del modello;
5. estrazione delle attivazioni da almeno tre layer convoluzionali;
6. implementazione verificata dello score CORES;
7. valutazione ID/OOD con AUROC e FPR95;
8. confronto early/middle/late layer;
9. aggregazione multi-layer come contributo originale;
10. risultati riproducibili tramite configurazioni e seed.

### Estensioni, solo dopo il completamento del minimo

- confronto con una seconda architettura MDE;
- confronto tra modello più superficiale e più profondo;
- score aggregato con pesi appresi esclusivamente su un validation set;
- OOD sintetico o corruption benchmark;
- analisi della relazione fra errore depth e score OOD;
- visualizzazioni delle feature e delle attivazioni estreme.

Non ampliare lo scope finché la pipeline minima non produce una tabella completa di risultati.

## 4. Domande di ricerca

Le domande principali sono:

1. CORES distingue scene indoor NYU da scene outdoor KITTI usando una rete MDE?
2. Quali layer producono gli score OOD più discriminativi?
3. L'aggregazione di più livelli migliora AUROC e FPR95 rispetto al miglior singolo layer?
4. Esiste una relazione tra degrado della stima depth e score OOD?
5. La profondità o configurazione della rete modifica la qualità del segnale OOD?

## 5. Protocollo sperimentale

### Dataset e split

- Usare split ufficiali o pubblicamente riconosciuti.
- Separare rigorosamente training, validation e test.
- Non selezionare soglie o pesi sul test set.
- Documentare versione, provenienza, licenza e procedura di download.
- Mantenere un manifest con percorsi e split, senza committare dataset pesanti in Git.
- Per KITTI, distinguere tra immagini utilizzate solamente per OOD detection e immagini dotate di ground truth depth.

### Preprocessing

Il preprocessing condiviso deve rendere gli input dimensionalmente compatibili senza eliminare le differenze di dominio che il progetto vuole misurare.

Documentare sempre:

- resize e crop;
- normalizzazione RGB;
- intervallo e unità della profondità;
- gestione dei pixel senza depth valida;
- eventuale allineamento di scala;
- differenze di camera e campo visivo tra NYU e KITTI.

Non confrontare direttamente metriche depth fra NYU e KITTI senza dichiarare protocollo, crop, range e gestione della scala.

### Training MDE

- Salvare configurazione, seed e checkpoint migliore.
- Selezionare il checkpoint sul validation set.
- Registrare loss, metriche, tempo e memoria GPU.
- Prevedere una modalità rapida su subset per testare il codice.
- Prevedere resume da checkpoint.
- Non modificare contemporaneamente data pipeline, loss e architettura durante il debugging.

### Integrazione CORES

- Consultare il paper CORES e verificare la definizione esatta dello score prima dell'implementazione definitiva.
- Usare forward hook o un feature extractor esplicito.
- Assegnare nomi stabili ai layer osservati.
- Salvare statistiche aggregate, non feature map complete, quando possibile.
- Evitare qualsiasi contaminazione del test set nella calibrazione dello score.
- Dichiarare chiaramente ogni adattamento di CORES dal classification setting al dense prediction setting.

### Aggregazione multi-layer

Baseline da confrontare:

- score early;
- score middle;
- score late;
- media degli score normalizzati;
- media pesata, se i pesi sono determinati sul validation set.

Poiché layer differenti possono produrre scale diverse, normalizzare gli score usando esclusivamente statistiche del training o validation ID.

## 6. Metriche

### OOD detection

- **AUROC**: più alto è migliore;
- **FPR95**: più basso è migliore;
- opzionale: AUPR-IN e AUPR-OUT;
- intervalli di confidenza tramite bootstrap, se il tempo lo consente.

Prima di calcolare le metriche, fissare esplicitamente se uno score alto significhi ID oppure OOD.

### Depth estimation

- RMSE;
- AbsRel;
- δ1, δ2 e δ3;
- opzionale: SqRel e log-RMSE.

Le metriche devono ignorare pixel invalidi e rispettare il protocollo specifico del dataset.

### Efficienza

Per ogni configurazione registrare almeno:

- numero di parametri;
- tempo medio di inferenza;
- picco di memoria GPU, se disponibile;
- costo aggiuntivo introdotto da CORES.

## 7. Ablation study

Tabella minima:

| Configurazione | Layer | Aggregazione | AUROC ↑ | FPR95 ↓ | Costo aggiuntivo |
|---|---|---|---:|---:|---:|
| FastDepth | Early | Nessuna | 0.98967 | 0.044 | Non misurato |
| FastDepth | Middle | Nessuna | 0.99990 | 0.001 | Non misurato |
| FastDepth | Late | Nessuna | 0.90275 | 0.466 | Non misurato |
| FastDepth | E+M+L+D | Media | 0.99974 | 0.001 | Non misurato |
| FastDepth | E+M+L+D | Pesata sintetica | 0.99974 | 0.001 | Non misurato |

Per coprire l'impatto della profondità del modello, aggiungere almeno una variante controllata, per esempio un encoder alternativo o una configurazione ridotta. Tutto il resto del protocollo deve restare invariato.

## 8. Struttura consigliata del repository

```text
project/
├── README.md
├── PROJECT_GUIDELINES.md
├── requirements.txt
├── configs/
│   ├── fastdepth_nyu.yaml
│   └── ood_cores.yaml
├── notebooks/
│   └── project_2_cores_mde_kaggle.ipynb
├── src/
│   ├── data/
│   │   ├── nyu.py
│   │   ├── kitti.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── fastdepth.py
│   │   └── feature_hooks.py
│   ├── cores/
│   │   ├── scoring.py
│   │   └── aggregation.py
│   ├── metrics/
│   │   ├── depth.py
│   │   └── ood.py
│   ├── train.py
│   ├── evaluate_depth.py
│   └── evaluate_ood.py
├── scripts/
│   ├── prepare_nyu.ps1
│   └── prepare_kitti.ps1
├── tests/
├── outputs/
└── presentation/
```

Dataset, checkpoint e output grandi devono essere esclusi tramite `.gitignore`. Il notebook finale deve richiamare il codice modulare, non duplicarne integralmente l'implementazione.

### Struttura del notebook Kaggle

Il notebook principale seguirà quest'ordine:

1. titolo, obiettivi e modalità d'uso;
2. installazione/import delle dipendenze;
3. configurazione globale e rilevamento Kaggle/Colab;
4. seed, device e riproducibilità;
5. individuazione e verifica dei dataset;
6. utility e visualizzazione dei campioni;
7. dataset e DataLoader NYU/KITTI;
8. modello MDE;
9. loss e metriche depth;
10. training o caricamento checkpoint;
11. valutazione depth;
12. estrazione delle risposte convoluzionali;
13. implementazione CORES;
14. valutazione OOD;
15. ablation multi-layer e architetturale;
16. grafici, tabelle e salvataggio dei risultati;
17. conclusioni, limiti e istruzioni per riprodurre l'esperimento.

Ogni sezione dovrà essere eseguibile in ordine dall'inizio. Le celle costose dovranno supportare flag come `QUICK_MODE`, `TRAIN_MODEL` e `LOAD_CHECKPOINT`.

## 9. Qualità e riproducibilità

- Configurazioni fuori dal codice tramite YAML o argomenti CLI.
- Seed per Python, NumPy e PyTorch.
- Controlli espliciti di shape, dtype, device e valori non finiti.
- Test unitari per metriche depth, metriche OOD e aggregazione.
- Smoke test end-to-end su pochi campioni.
- Logging di commit, configurazione e versione delle dipendenze.
- Comandi completi nel README per preparazione dati, training ed evaluation.
- Risultati rigenerabili senza modifiche manuali ai file sorgente.
- Nessun percorso assoluto specifico del computer locale.
- Configurazione centralizzata dei percorsi Kaggle e Colab.
- Celle di training idempotenti o protette da flag espliciti.
- Output persistenti esportabili come Kaggle Dataset o scaricabili come archivio.

## 10. Divisione del lavoro per due persone

### Persona A — Depth estimation

- loader e preprocessing NYU/KITTI;
- modello FastDepth;
- training, checkpoint e metriche depth;
- profiling della rete.

### Persona B — OOD detection

- studio e implementazione CORES;
- feature extraction e scelta dei layer;
- metriche AUROC/FPR95;
- aggregazione multi-layer e ablation.

### Attività condivise

- definizione del protocollo;
- revisione incrociata del codice;
- interpretazione dei risultati;
- README, relazione e presentazione;
- preparazione delle risposte per la discussione orale.

Entrambe le persone devono saper spiegare l'intera pipeline.

## 11. Scaletta di lavoro

### Stato aggiornato

- Completati: training FastDepth su NYU, checkpoint ripristinabili, metriche depth
  ID, implementazione CORES, valutazione NYU/KITTI, ablation per layer e
  componente, aggregazione multi-layer, controlli, stabilità, risultati grezzi,
  otto grafici, relazione e contenuto della presentazione.
- Completata nella versione Kaggle 17 la valutazione depth OOD su KITTI 0–80 m.
- Limite dichiarato: una sola architettura depth addestrata; il confronto di due
  o più modelli è incoraggiato dalla traccia ma non obbligatorio.

### Fase 0 — Definizione e verifica dello scope

- [x] Leggere integralmente paper CORES e FastDepth.
- [ ] Confermare con la docente lo scope e l'ablation architetturale.
- [ ] Stabilire hardware disponibile e budget temporale.
- [ ] Scegliere split, risoluzione e subset iniziale.
- [x] Creare repository e notebook Kaggle riproducibile.
- [x] Definire la configurazione portabile Kaggle/Colab.
- [x] Verificare disponibilità GPU, spazio disco e versioni preinstallate su Kaggle.

**Deliverable:** protocollo di una pagina e repository eseguibile.

### Fase 1 — Baseline depth

- [x] Implementare i loader NYU e KITTI.
- [x] Visualizzare immagini, depth map e maschere valide.
- [x] Implementare/importare FastDepth citando correttamente la fonte.
- [x] Eseguire overfit su un batch come controllo.
- [x] Completare un training breve su subset.
- [x] Verificare le metriche depth con test sintetici.

**Deliverable:** checkpoint baseline e tabella depth su NYU validation.

### Fase 2 — Baseline CORES

- [x] Identificare layer early, middle e late.
- [x] Estrarre le attivazioni con shape documentate.
- [x] Implementare lo score single-layer secondo il paper.
- [x] Verificare lo score su tensori controllati.
- [x] Generare distribuzioni degli score per NYU e KITTI.
- [x] Calcolare AUROC e FPR95.

**Deliverable:** prima curva ROC e tabella per layer.

### Fase 3 — Contributo multi-layer

- [x] Normalizzare gli score senza usare il test set.
- [x] Implementare media semplice.
- [x] Implementare eventuale aggregazione pesata.
- [x] Confrontare singoli layer e aggregazioni.
- [x] Analizzare stabilità rispetto a seed e quantità di dati.

**Deliverable:** ablation multi-layer completa.

### Fase 4 — Ablation architetturale

- [ ] Scegliere una variante leggera e motivata.
- [ ] Eseguire lo stesso protocollo sperimentale.
- [ ] Misurare parametri, latenza e memoria.
- [ ] Confrontare qualità depth e capacità OOD.

**Deliverable:** tabella efficienza–accuratezza e risposta alla quinta domanda di ricerca.

### Fase 5 — Consolidamento

- [x] Ripetere gli esperimenti definitivi con configurazioni congelate.
- [x] Salvare risultati grezzi in CSV/JSON.
- [x] Generare grafici tramite script riproducibili.
- [x] Fare smoke test su un ambiente pulito.
- [x] Completare README e notebook demo.

**Deliverable:** repository riproducibile e risultati finali.

### Fase 6 — Presentazione

- [x] Descrivere problema, ipotesi e contributo.
- [x] Spiegare CORES senza affidarsi al codice.
- [x] Mostrare pipeline, protocollo e principali risultati.
- [x] Discutere failure case e limiti del confronto NYU/KITTI.
- [ ] Dichiarare contributi individuali e fonti esterne.
- [ ] Provare la presentazione e preparare domande tecniche.

**Deliverable:** slide finali e demo verificata.

## 12. Ordine di priorità

In caso di poco tempo:

1. correttezza e riproducibilità della baseline MDE;
2. implementazione fedele e verificata di CORES;
3. metriche ID/OOD corrette;
4. ablation per layer;
5. aggregazione multi-layer;
6. variante architetturale;
7. estensioni e visualizzazioni aggiuntive.

## 13. Definition of Done

Il progetto è pronto quando:

- un nuovo utente può configurarlo seguendo il README;
- training ed evaluation possono essere eseguiti in ordine dal notebook Kaggle;
- il notebook funziona senza percorsi o dipendenze specifiche del PC locale;
- il codice funziona sia in modalità rapida sia sull'esperimento completo;
- tutti i risultati nelle slide provengono da file salvati e tracciabili;
- split e calibrazione non presentano data leakage;
- sono disponibili metriche depth e OOD;
- l'ablation confronta almeno tre layer e un'aggregazione;
- il contributo originale e i suoi limiti sono formulati chiaramente;
- entrambi i componenti del gruppo sanno motivare ogni scelta importante.

## 14. Rischi principali

| Rischio | Mitigazione |
|---|---|
| Dataset molto grandi | Partire con subset e manifest; scaricare il completo solo quando serve |
| Training lento | Risoluzione ridotta, mixed precision, checkpoint e modalità quick |
| Metriche KITTI scorrette | Usare protocollo, crop e maschere documentati |
| Data leakage nella calibrazione | Separare train/validation/test e congelare le soglie prima del test |
| Score CORES ambiguo | Riprodurre prima la formula del paper con test controllati |
| Feature map troppo pesanti | Calcolare statistiche online senza salvare tensori completi |
| Scope eccessivo | Bloccare le estensioni fino alla tabella minima completa |
| Codice esterno o generato non compreso | Citare le fonti, riscrivere/adattare consapevolmente e revisionare insieme |
