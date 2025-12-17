# Plant-Aid-KBS

**Sistema Diagnostico Intelligente per la Cura delle Piante**
*Progetto di Ingegneria della Conoscenza - A.A. 2024/2025*

---

## 📋 Descrizione

Plant-Aid-KBS è un **sistema ibrido basato sulla conoscenza (KBS)** che integra tecniche simboliche e statistiche per diagnosticare malattie delle piante e fornire raccomandazioni di cura.

Il sistema combina quattro moduli principali:

1.  **Ontologia OWL** (`kbs_engine/ontology_manager.py`): Rappresentazione formale della conoscenza botanica (piante, malattie, sintomi).
2.  **Regole Datalog** (`kbs_engine/datalog_engine.py`): Formalizzazione della conoscenza euristica degli esperti per l'inferenza simbolica.
3.  **Modello SVM** (`ml_engine/svm_model.py`): Classificazione supervisionata per l'analisi statistica dei sintomi.
4.  **Rete Bayesiana** (`ml_engine/bn_model.py`): Gestione dell'incertezza e inferenza probabilistica.

L'intero sistema è orchestrato da `main_cli.py`, che aggrega i risultati dei diversi moduli per fornire una diagnosi finale robusta e spiegabile.

La cartella `docs/` contiene la documentazione formale `doc_progetto.docx`, utile alla consegna finale.

La cartella `data/` contiene i dati generati dall'escuzione dei moduli implementati.
<br>

## 🚀 Guida all'Installazione e Esecuzione

Questa guida descrive i passaggi necessari per configurare l'ambiente, generare i modelli di machine learning (che non sono tracciati da Git) ed eseguire il sistema diagnostico.

### Passo 1: Clonare la Repository

Clona la repository sulla tua macchina locale:

```bash
git clone https://github.com/Donato2403/Plant-Aid-KBS.git
cd Plant-Aid-KBS
```

### Passo 2: Configurare l'Ambiente Virtuale

Crea l'ambiente virtuale (assicurati che python si riferisca a Python 3.8+).

```bash
python -m venv venv
```

Attiva l'ambiente.

```bash
# Su Windows (cmd):
.\venv\Scripts\activate
```

```bash
# Su macOS/Linux (bash):
source venv/bin/activate
```

### Passo 3: Installare le Dipendenze necessarie

Le dipendenze principali includono **owlready2**, **clingo**, **scikit-learn**, **pgmpy** e **click**

```bash
pip install -r requirements.txt
```

### Passo 4: Generazione dei Modelli (Cruciale)

I file dei modelli addestrati (**.pkl**, **.bif**) non sono inclusi nella repository e devono essere generati prima di poter eseguire il programma principale.

Esegui i seguenti script di addestramento dalla cartella radice del progetto:

1. Genera il modello SVM: Questo script addestra il classificatore SVM (usando un set di dati sintetico interno) e salva `svm_model.pkl` e        `svm_transformer.pkl` nella cartella `data/`. (Output atteso: Report di classificazione SVM e messaggi di salvataggio)

```bash
python ml_engine/svm_model.py
```

2. Genera la Rete Bayesiana: Questo script addestra la Rete Bayesiana utilizzando `data/training_data.csv`
e salva `bn_model.bif` nella cartella `data/`. (Output atteso: Log di addestramento della BN e messaggi di salvataggio)

```bash
python ml_engine/bn_model.py
``` 

Al termine di questo passaggio, la cartella `data/` conterrà tutti i file necessari per l'esecuzione.

### Passo 5: Eseguire il Sistema Diagnostico

Ora che l'ambiente è configurato e i modelli sono stati generati, è possibile avviare l'interfaccia CLI principale:

```bash
python main_cli.py diagnosi
``` 
Il programma avvierà un'interfaccia interattiva. Segui le istruzioni a schermo:

#### 1. Selezionare la pianta:
   
```bash
    1. Seleziona la pianta da diagnosticare: 
       [1] Olivo
       [2] Rosa
       [3] Basilico
    Inserisci il numero della pianta:
``` 
Inserisci un singolo numero (es. 1) e premi *Invio*.

#### 2. Selezionare i sintomi osservati (selezione multipla):

```bash
    2. Seleziona i sintomi osservati:
       [1] Macchie circolari grigie...
       ...
       [9] Avvizzimento completo della pianta
       [0] Termina selezione sintomi
    Inserisci un numero (Sintomi scelti: 0) o 0 per continuare:
``` 
Questa è una selezione multipla. Devi inserire un numero di sintomo (es. 2) e premere *Invio*.
Il programma ti chiederà un altro numero. Continua a inserire i numeri per tutti i sintomi che osservi.
Quando hai finito di aggiungere sintomi, inserisci *0* e premi *Invio* per passare allo step successivo.

#### 3. Selezionare la stagione corrente:
 
```bash
    3. Seleziona la stagione corrente (per Datalog):
       [1] Primavera
       ...
    Inserisci il numero della stagione:
```  
Inserisci un singolo numero (es. 1) e premi *Invio*.

#### 4. Conferma:
```bash
    Riepilogo Input Selezionato:
      Pianta:    Olivo
      Stagione:  Primavera
      Sintomi:   ['Ingiallimento delle foglie', ...]
    Procedere con la diagnosi? [Y/n]:
```  
Inserisci *Y* e premi *Invio* per avviare l'analisi ibrida.

Al termine dell'analisi, il sistema fornirà un report diagnostico ibrido completo, aggregando i risultati di Datalog, SVM e Rete Bayesiana, e arricchendo il risultato con i trattamenti recuperati dall'ontologia.

<br>

## 🔬 Flusso di Esecuzione Dettagliato
Quando esegui il comando, il sistema segue un processo articolato in 6 fasi principali.

### FASE 1: Inizializzazione e Caricamento dei Moduli
Appena avvii lo script, la classe *SistemaDiagnostico* viene istanziata. Questa fase carica in memoria tutti i componenti del KBS:

- **Caricamento Ontologia:** Il *GestoreOntologia* viene inizializzato e legge il file `data/plant_care.owl`, mappando tutte le classi (Piante, Malattie), le proprietà e gli individui (Occhio di Pavone, Trattamento Rame, ecc.).

- **Inizializzazione Datalog:** Il *MotoreDatalog* viene inizializzato, caricando l'intera base di conoscenza simbolica (le regole euristiche) come una stringa *Python*, pronta per essere usata da *Clingo*.

- **Caricamento SVM:** Il *ClassificatoreSVM* carica i file `data/svm_model.pkl` (il modello addestrato) e `data/svm_transformer.pkl` (lo scaler e il codificatore delle etichette).

- **Caricamento Rete Bayesiana:** La *ReteBayesiana* carica il file `data/bn_model.bif`, che contiene la struttura della rete e le Tabelle di Probabilità Condizionata (CPD) apprese.

### FASE 2: Raccolta dell'Input Utente
Il sistema entra nella funzione *accogli_input_utente()*:

- Mostra le opzioni per la pianta, i sintomi e la stagione.

- L'utente inserisce le sue scelte (es. "1" per Olivo, "1", "2", "3" per i sintomi, "1" per Primavera).

- L'input viene memorizzato in un dizionario (es. *{'pianta': 'olivo', 'sintomi': ['macchie_circolari_grigie', ...], 'stagione': 'primavera'}*).

### FASE 3: Esecuzione Parallela dei Motori Diagnostici
Il flusso chiama *esegui_diagnosi_completa()*, che interroga i tre motori diagnostici:

#### 1. Inferenza Simbolica (Datalog):

- Vengono asseriti i fatti raccolti (es. *pianta_tipo(olivo).*, *sintomo_presente(macchie_circolari_grigie)*., *stagione_corrente(primavera)*.).

- Il *MotoreDatalog* esegue *diagnosi_completa_integrata()*, che combina le regole e i fatti, lanciando il solver *Clingo*.

- Il solver deriva le conclusioni, ad esempio *diagnosi_finale(occhio_pavone, olivo, critica)*.

- **Output:** Un dizionario con le diagnosi simboliche e la loro confidenza (es. *{'Occhio di Pavone': 1.0}*).

#### 2. Classificazione Statistica (SVM):

- L'input utente viene convertito in un vettore di feature numerico (es. *[1, 1, 1, 0, ..., 1, 0, 0]*).

- *ClassificatoreSVM.predici()* usa questo vettore per calcolare le probabilità di classificazione.

- **Output:** Un oggetto *RisultatoClassificazione* con la malattia più probabile e la sua confidenza (es. *malattia_predetta='Occhio di Pavone', confidenza=0.98*).

#### 3. Analisi Probabilistica (Rete Bayesiana):

- *ReteBayesiana.esegui_inferenza()* riceve la lista dei soli sintomi (es. *['macchie_circolari_grigie', ...]*).

- Costruisce l'evidenza (es. *{'macchie_circolari_grigie': '1', 'tumori_rami': '0', ...}*).

- Esegue la query sulla rete per calcolare le probabilità a posteriori per la variabile malattia.

- **Output:** Un dizionario con le probabilità per tutte le malattie (es. *{'occhio_pavone': 0.97, 'rogna_olivo': 0.01, ...}*).

### FASE 4: Aggregazione e Spiegazione
Questa è la logica ibrida centrale. La funzione *_aggrega_risultati()*:

- Cicla su tutte le malattie conosciute.

- Per ognuna, estrae i punteggi dai tre motori (Datalog, SVM, BN).

- Calcola un "Fattore di Confidenza" finale usando la media pesata (es. BN * 0.5 + Datalog * 0.3 + SVM * 0.2).

- Ordina i risultati e identifica la *diagnosi_top*.

- Conserva i punteggi individuali per la spiegazione.

### Fase 5: Arricchimento con Ontologia
- Il sistema ora sa qual è la diagnosi più probabile (es. *occhio_pavone*).

- Converte il nome canonico (*occhio_pavone*) nel nome dell'individuo nell'ontologia (*Occhio_di_Pavone*).

- Chiama *gestore_ontologia.ottieni_info_malattia('Occhio_di_Pavone')*.

- Il gestore interroga l'ontologia OWL per trovare la descrizione, la gravità e, soprattutto, le istanze collegate tramite la proprietà *richiede_trattamento*.

### Fase 6: Presentazione del Report Finale
La funzione *stampa_report_diagnosi()* riceve il report finale aggregato e arricchito e stampa a schermo le tre sezioni:

- **Diagnosi Ibrida:** La malattia più probabile e il fattore di confidenza.

- **Spiegazione:** I punteggi individuali dei tre motori che hanno portato a quel risultato.

- **Dettagli (dall'Ontologia):** Descrizione, gravità e trattamenti consigliati.
  

---

## 👥 Autori

**Donato Cancellara**  
Corso di Ingegneria della Conoscenza  
Università degli Studi di Bari Aldo Moro  
Dipartimento di Informatica


## 🔗 Riferimenti

### Riferimenti Tecnici del Progetto

- [Owlready2 Documentation](https://owlready2.readthedocs.io/)
- [Clingo User Guide](https://www.google.com/search?q=https://potassco.org/clingo/user-guide/)
- [pgmpy Documentation](https://pgmpy.org/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- La base di conoscenza simbolica (ASP/Datalog) è stata sviluppata con il supporto di *Google Gemini Pro* per la formalizzazione delle euristiche diagnostiche a partire dalle fonti di dominio.

### Riferimenti Dominio Botanico (Fonti Esterne)

La conoscenza del dominio è stata modellata basandosi su fonti esperte di botanica e fitopatologia:

##### Nomenclatura e Tassonomia
- [CABI Plant Sciences](https://www.google.com/search?q=https://www.cabidigitallibrary.org/product/QC)
- [International Plant Names Index (IPNI)](https://www.ipni.org/)
- [Plants of the World Online (POWO)](https://powo.science.kew.org/)
- [World Flora Online (WFO)](https://wfoplantlist.org/)

##### Malattie della Rosa
- [Coltivazione Biologica - Malattie Rose](https://www.coltivazionebiologica.it/malattie-delle-rose)
- [Rose Barni - Malattie Fungine](https://www.rosebarni.it/malattia-fungine-piante-quali-sono-i-migliori-trattamenti-preventivi)

##### Malattie dell'Olivo
- [Regione Veneto - Malattie Olivo](https://www.regione.veneto.it/web/fitosanitario/malattie-olivo)
- [Orto da Coltivare - Problemi Olivo](https://www.google.com/search?q=https://www.ortodacoltivare.it/guide/problemi-ulivo)

##### Malattie del Basilico
- [Orto da Coltivare - Fusarium Basilico](https://www.google.com/search?q=https://www.ortodacoltivare.it/difesa/malattie/fusarium-basilico)
- [AgroNotizie - Basilico](https://agronotizie.imagelinenetwork.com/colture/basilico/257)