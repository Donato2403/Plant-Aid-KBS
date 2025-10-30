"""
Modulo per il modello Bayesiano del sistema Plant-Aid-KBS

Questo modulo fornisce funzionalità per:
- Definizione e addestramento di una Rete Bayesiana per la diagnosi
- Gestione dell'incertezza attraverso probabilità condizionate (CPD)
- Inferenza probabilistica tramite Variable Elimination
- Salvataggio e caricamento del modello addestrato

Versione: 2.0 - Ragionamento sotto Incertezza (CORRETTO)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
import traceback
warnings.filterwarnings('ignore')

# Import pgmpy per le Reti Bayesiane
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import BIFWriter, BIFReader


class ReteBayesiana:
    """
    Rete Bayesiana per la diagnosi di malattie delle piante sotto incertezza.
    
    Implementa un modello probabilistico grafico che rappresenta le dipendenze
    condizionate tra sintomi e malattie, consentendo inferenza probabilistica
    anche con evidenza parziale o incerta.
    """
    
    # Lista completa dei sintomi supportati (coerente con SVM e Datalog)
    SINTOMI_SUPPORTATI = [
        "macchie_circolari_grigie",
        "ingiallimento_foglie",
        "caduta_foglie",
        "tumori_rami",
        "macchie_bruno_nerastre_frutti",
        "macchie_nere_foglie",
        "muffa_biancastra",
        "annerimento_gambo",
        "avvizzimento_pianta"
    ]
    
    # Malattie supportate (formato snake_case, coerente con gli altri moduli)
    MALATTIE_SUPPORTATE = [
        "occhio_pavone",
        "rogna_olivo",
        "lebbra_olivo",
        "ticchiolatura_rosa",
        "oidio_rosa",
        "peronospora_rosa",
        "peronospora_basilico",
        "fusarium_basilico"
    ]
    
    # Mapping per output human-readable
    MAPPING_MALATTIE = {
        "occhio_pavone": "Occhio di Pavone",
        "rogna_olivo": "Rogna dell'Olivo",
        "lebbra_olivo": "Lebbra dell'Olivo",
        "ticchiolatura_rosa": "Ticchiolatura della Rosa",
        "oidio_rosa": "Oidio della Rosa",
        "peronospora_rosa": "Peronospora della Rosa",
        "peronospora_basilico": "Peronospora del Basilico",
        "fusarium_basilico": "Fusarium del Basilico"
    }
    
    def __init__(self, percorso_modello: str = "data/bn_model.bif"):
        """
        Inizializza la Rete Bayesiana.
        
        Args:
            percorso_modello: Percorso del file per salvare/caricare il modello
        """
        self.percorso_modello = Path(percorso_modello)
        self.modello = None
        self.inferenza = None
        self.struttura_definita = False
        self.stati_sintomi = None  # Memorizza gli stati delle variabili sintomo
        
        print("[INFO] Rete Bayesiana inizializzata")
    
    def _prepara_dati(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara i dati dal formato CSV al formato richiesto da pgmpy.
        
        Il CSV ha i sintomi in una singola stringa separata da virgole.
        Questo metodo crea una colonna binaria (0/1) per ogni sintomo.
        
        Args:
            df: DataFrame con colonna 'sintomi' (stringa) e 'malattia'
            
        Returns:
            DataFrame trasformato con colonne separate per ogni sintomo
        """
        print("[INFO] Preparazione dati per addestramento...")
        
        # Crea una copia del DataFrame
        df_trasformato = df.copy()
        
        # Verifica presenza colonna 'sintomi'
        if 'sintomi' not in df.columns:
            raise ValueError("Il DataFrame deve contenere la colonna 'sintomi'")
        
        # Trasforma la colonna 'sintomi' in colonne separate
        # Usa get_dummies con il separatore virgola
        sintomi_espansi = df['sintomi'].str.get_dummies(sep=',')
        
        # Rimuove spazi dai nomi delle colonne
        sintomi_espansi.columns = sintomi_espansi.columns.str.strip()
        
        # Assicura che tutti i sintomi supportati siano presenti
        for sintomo in self.SINTOMI_SUPPORTATI:
            if sintomo not in sintomi_espansi.columns:
                sintomi_espansi[sintomo] = 0
        
        # Mantiene solo i sintomi supportati nell'ordine corretto
        sintomi_espansi = sintomi_espansi[self.SINTOMI_SUPPORTATI]
        
        # Combina con la colonna malattia
        df_trasformato = pd.concat([
            df[['malattia']].reset_index(drop=True),
            sintomi_espansi.reset_index(drop=True)
        ], axis=1)
        
        print(f"[INFO] Dati preparati: {len(df_trasformato)} esempi, "
              f"{len(self.SINTOMI_SUPPORTATI)} sintomi")
        
        return df_trasformato
    
    def _definisci_struttura(self) -> List[Tuple[str, str]]:
        """
        Definisce la struttura della Rete Bayesiana.
        
        Per questo prototipo, si usa una struttura semplice:
        - Un nodo 'malattia' (genitore)
        - Nodi sintomo (figli) dipendenti dalla malattia
        
        Struttura: malattia -> sintomo_i per ogni sintomo
        
        Returns:
            Lista di archi (genitore, figlio)
        """
        archi = []
        
        # Ogni sintomo dipende dalla malattia
        for sintomo in self.SINTOMI_SUPPORTATI:
            archi.append(('malattia', sintomo))
        
        print(f"[INFO] Struttura definita: {len(archi)} archi")
        print(f"       Genitore: malattia")
        print(f"       Figli: {len(self.SINTOMI_SUPPORTATI)} sintomi")
        
        return archi
    
    def addestra(
        self,
        percorso_dati: str = "data/training_data.csv",
        metodo: str = "bayes",
        pseudo_conteggio: float = 1.0
    ) -> None:
        """
        Addestra la Rete Bayesiana sui dati forniti.
        
        Args:
            percorso_dati: Percorso del file CSV con i dati di training
            metodo: Metodo di apprendimento ('bayes' o 'mle')
            pseudo_conteggio: Pseudo-conteggio per smoothing (Laplace)
        """
        print("\n=== ADDESTRAMENTO RETE BAYESIANA ===\n")
        
        # Carica i dati
        try:
            dati_raw = pd.read_csv(percorso_dati)
            print(f"[INFO] Dati caricati da: {percorso_dati}")
            print(f"       Esempi: {len(dati_raw)}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"File dati non trovato: {percorso_dati}\n"
                "Assicurati di aver generato il dataset con lo script SVM"
            )
        
        # Prepara i dati
        dati_preparati = self._prepara_dati(dati_raw)
        
        # Definisce la struttura della rete
        archi = self._definisci_struttura()
        
        # Crea il modello Bayesiano
        self.modello = BayesianNetwork(archi)
        self.struttura_definita = True
        
        print("\n[INFO] Apprendimento parametri (CPD)...")
        
        # Apprende i parametri (CPD - Tabelle di Probabilità Condizionata)
        if metodo == "bayes":
            # Usa Bayesian Estimator con pseudo-conteggi (smoothing)
            estimatore = BayesianEstimator(
                self.modello,
                dati_preparati
            )
            
            # Stima le CPD per ogni nodo
            for nodo in self.modello.nodes():
                cpd = estimatore.estimate_cpd(
                    nodo,
                    prior_type='dirichlet',
                    pseudo_counts=pseudo_conteggio
                )
                self.modello.add_cpds(cpd)
        
        elif metodo == "mle":
            # Usa Maximum Likelihood Estimator
            self.modello.fit(
                dati_preparati,
                estimator=MaximumLikelihoodEstimator
            )
        
        else:
            raise ValueError(f"Metodo non supportato: {metodo}. "
                           "Usa 'bayes' o 'mle'")
        
        # Verifica la validità del modello
        if self.modello.check_model():
            print("[INFO] ✓ Modello valido e consistente")
        else:
            print("[WARNING] Modello potrebbe avere inconsistenze")
        
        # Mostra statistiche CPD
        print(f"\n[INFO] CPD apprese per {len(self.modello.nodes())} nodi")
        print(f"       Nodi: malattia + {len(self.SINTOMI_SUPPORTATI)} sintomi")
        
        # Memorizza gli stati delle variabili sintomo per l'inferenza
        self._carica_stati_sintomi()
        
        # Inizializza il motore di inferenza
        self.inferenza = VariableElimination(self.modello)
        
        print("\n[INFO] ✓ Addestramento completato con successo")
    
    def _carica_stati_sintomi(self) -> None:
        """
        Carica e memorizza i nomi degli stati delle variabili sintomo dalla CPD.
        Questo è necessario per preparare correttamente l'evidenza durante l'inferenza.
        """
        # Prende la CPD del primo sintomo come riferimento
        primo_sintomo = self.SINTOMI_SUPPORTATI[0]
        cpd_sintomo = self.modello.get_cpds(primo_sintomo)
        
        # Ottiene i nomi degli stati (es. [0, 1] oppure ["False", "True"])
        stati = cpd_sintomo.state_names[primo_sintomo]
        self.stati_sintomi = stati
        
        print(f"[INFO] Stati sintomi rilevati: {stati}")
    
    def salva_modello(self) -> None:
        """
        Salva il modello addestrato su disco in formato BIF.
        
        Il formato BIF (Bayesian Interchange Format) è uno standard
        per la rappresentazione di Reti Bayesiane.
        """
        if self.modello is None:
            raise RuntimeError(
                "Impossibile salvare: il modello non è stato addestrato.\n"
                "Chiama prima il metodo 'addestra()'"
            )
        
        # Crea la directory se non esiste
        self.percorso_modello.parent.mkdir(parents=True, exist_ok=True)
        
        # Salva in formato BIF
        writer = BIFWriter(self.modello)
        writer.write_bif(str(self.percorso_modello))
        
        print(f"[INFO] ✓ Modello salvato in: {self.percorso_modello}")
    
    def carica_modello(self) -> None:
        """
        Carica un modello precedentemente salvato da disco.
        """
        if not self.percorso_modello.exists():
            raise FileNotFoundError(
                f"File modello non trovato: {self.percorso_modello}\n"
                "Addestra prima il modello con 'addestra()' e salvalo con 'salva_modello()'"
            )
        
        # Carica il modello da file BIF
        reader = BIFReader(str(self.percorso_modello))
        self.modello = reader.get_model()
        
        # Carica gli stati dei sintomi
        self._carica_stati_sintomi()
        
        # Inizializza il motore di inferenza
        self.inferenza = VariableElimination(self.modello)
        
        print(f"[INFO] ✓ Modello caricato da: {self.percorso_modello}")
        print(f"       Nodi: {len(self.modello.nodes())}")
        print(f"       Archi: {len(self.modello.edges())}")
    
    def esegui_inferenza(
        self,
        lista_sintomi_input: List[str],
        mostra_dettagli: bool = False
    ) -> Dict[str, float]:
        """
        Esegue l'inferenza probabilistica per calcolare le probabilità
        a posteriori delle malattie, dati i sintomi osservati.
        
        Args:
            lista_sintomi_input: Lista dei sintomi osservati
            mostra_dettagli: Se True, mostra dettagli dell'inferenza
            
        Returns:
            Dizionario {malattia: probabilità} ordinato per probabilità decrescente
        """
        # Assicura che il modello sia caricato
        if self.modello is None:
            print("[INFO] Modello non caricato, caricamento in corso...")
            self.carica_modello()
        
        if mostra_dettagli:
            print("\n=== INFERENZA BAYESIANA ===")
            print(f"Sintomi osservati: {lista_sintomi_input}")
        
        # Verifica gli stati dei sintomi
        if self.stati_sintomi is None:
            self._carica_stati_sintomi()
        
        # Crea mapping corretto per evidenza basato sugli stati effettivi
        # Gli stati possono essere [0, 1] oppure ["False", "True"] o altri
        # Usiamo il primo come "assente" e il secondo come "presente"
        stato_assente = self.stati_sintomi[0]
        stato_presente = self.stati_sintomi[1]
        
        if mostra_dettagli:
            print(f"Stati sintomi: assente={stato_assente}, presente={stato_presente}")
        
        # Prepara l'evidenza: tutti i sintomi con il corretto nome di stato
        evidenza = {}
        for sintomo in self.SINTOMI_SUPPORTATI:
            if sintomo in lista_sintomi_input:
                evidenza[sintomo] = stato_presente
            else:
                evidenza[sintomo] = stato_assente
        
        if mostra_dettagli:
            print(f"\nEvidenza preparata: {sum(1 for v in evidenza.values() if v == stato_presente)} sintomi attivi")
            print(f"Esempio evidenza: {list(evidenza.items())[:2]}")
        
        # Esegue la query per la variabile 'malattia'
        try:
            risultato_query = self.inferenza.query(
                variables=['malattia'],
                evidence=evidenza,
                show_progress=False
            )
        except Exception as e:
            print(f"\n[ERRORE] Inferenza fallita!")
            print(f"Tipo errore: {type(e).__name__}")
            print(f"Messaggio: {str(e)}")
            if mostra_dettagli:
                print("\nTraceback completo:")
                traceback.print_exc()
            # Ritorna distribuzione uniforme in caso di errore
            return {
                malattia: 1.0 / len(self.MALATTIE_SUPPORTATE)
                for malattia in self.MALATTIE_SUPPORTATE
            }
        
        # Estrae le probabilità per ogni malattia
        probabilita_malattie = {}
        
        # Ottiene i valori dalla distribuzione
        valori_malattie = risultato_query.values
        stati_malattie = risultato_query.state_names['malattia']
        
        for i, malattia in enumerate(stati_malattie):
            probabilita_malattie[malattia] = float(valori_malattie[i])
        
        # Ordina per probabilità decrescente
        probabilita_ordinate = dict(
            sorted(
                probabilita_malattie.items(),
                key=lambda x: x[1],
                reverse=True
            )
        )
        
        if mostra_dettagli:
            print("\n=== RISULTATI INFERENZA ===")
            for malattia, prob in probabilita_ordinate.items():
                nome_leggibile = self.MAPPING_MALATTIE.get(malattia, malattia)
                print(f"{nome_leggibile:30s}: {prob:.4f} ({prob*100:.2f}%)")
        
        return probabilita_ordinate
    
    def inferenza_multipla(
        self,
        casi_test: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """
        Esegue inferenza su più casi contemporaneamente.
        
        Args:
            casi_test: Lista di dizionari con 'sintomi' e opzionalmente 'malattia_vera'
            
        Returns:
            Lista di risultati con predizioni e probabilità
        """
        risultati = []
        
        for i, caso in enumerate(casi_test):
            sintomi = caso.get('sintomi', [])
            malattia_vera = caso.get('malattia_vera', None)
            
            # Esegue inferenza
            prob_malattie = self.esegui_inferenza(sintomi, mostra_dettagli=False)
            
            # Predizione: malattia con probabilità massima
            malattia_predetta = max(prob_malattie, key=prob_malattie.get)
            confidenza = prob_malattie[malattia_predetta]
            
            risultato = {
                'caso_id': i + 1,
                'sintomi': sintomi,
                'malattia_predetta': malattia_predetta,
                'confidenza': confidenza,
                'probabilita_tutte': prob_malattie
            }
            
            if malattia_vera:
                risultato['malattia_vera'] = malattia_vera
                risultato['corretto'] = (malattia_predetta == malattia_vera)
            
            risultati.append(risultato)
        
        return risultati
    
    def valuta_modello(
        self,
        percorso_test: str = "data/test_data.csv"
    ) -> Dict[str, float]:
        """
        Valuta le prestazioni del modello su un dataset di test.
        
        Args:
            percorso_test: Percorso del file CSV con i dati di test
            
        Returns:
            Dizionario con metriche di valutazione
        """
        print("\n=== VALUTAZIONE MODELLO ===\n")
        
        # Carica dati di test
        try:
            dati_test = pd.read_csv(percorso_test)
        except FileNotFoundError:
            print(f"[WARNING] File test non trovato: {percorso_test}")
            return {}
        
        # Prepara casi di test
        casi_test = []
        for _, riga in dati_test.iterrows():
            sintomi_lista = riga['sintomi'].split(',')
            sintomi_lista = [s.strip() for s in sintomi_lista]
            
            casi_test.append({
                'sintomi': sintomi_lista,
                'malattia_vera': riga['malattia']
            })
        
        # Esegue inferenza multipla
        risultati = self.inferenza_multipla(casi_test)
        
        # Calcola metriche
        corretti = sum(1 for r in risultati if r.get('corretto', False))
        totali = len(risultati)
        accuracy = corretti / totali if totali > 0 else 0.0
        
        confidenza_media = np.mean([r['confidenza'] for r in risultati])
        
        metriche = {
            'accuracy': accuracy,
            'corretti': corretti,
            'totali': totali,
            'confidenza_media': confidenza_media
        }
        
        print(f"Accuracy: {accuracy:.3f} ({corretti}/{totali})")
        print(f"Confidenza Media: {confidenza_media:.3f}")
        
        return metriche
    
    def mostra_cpd(self, variabile: str) -> None:
        """
        Mostra la tabella di probabilità condizionata (CPD) per una variabile.
        
        Args:
            variabile: Nome della variabile (es. 'malattia' o un sintomo)
        """
        if self.modello is None:
            raise RuntimeError("Modello non caricato")
        
        try:
            cpd = self.modello.get_cpds(variabile)
            print(f"\n=== CPD per '{variabile}' ===")
            print(cpd)
        except Exception as e:
            print(f"[ERRORE] Impossibile mostrare CPD: {e}")
    
    def ottieni_statistiche_modello(self) -> Dict[str, any]:
        """
        Restituisce statistiche sul modello addestrato.
        
        Returns:
            Dizionario con statistiche del modello
        """
        if self.modello is None:
            return {"errore": "Modello non addestrato"}
        
        statistiche = {
            'numero_nodi': len(self.modello.nodes()),
            'numero_archi': len(self.modello.edges()),
            'nodi': list(self.modello.nodes()),
            'numero_cpd': len(self.modello.get_cpds()),
            'numero_sintomi': len(self.SINTOMI_SUPPORTATI),
            'numero_malattie': len(self.MALATTIE_SUPPORTATE),
            'stati_sintomi': self.stati_sintomi
        }
        
        return statistiche


# ================================================================
# FUNZIONI DI UTILITÀ
# ================================================================

def genera_dataset_sintetico_bn(
    percorso_output: str = "data/training_data.csv",
    n_esempi: int = 200
) -> pd.DataFrame:
    """
    Genera un dataset sintetico per test della Rete Bayesiana.
    
    Questo è un dataset di esempio con pattern realistici
    sintomo-malattia per il testing.
    
    Args:
        percorso_output: Dove salvare il CSV
        n_esempi: Numero di esempi da generare
        
    Returns:
        DataFrame generato
    """
    np.random.seed(42)
    
    # Pattern sintomo-malattia (probabilità condizionate)
    patterns = {
        "occhio_pavone": {
            'sintomi': ['macchie_circolari_grigie', 'ingiallimento_foglie', 'caduta_foglie'],
            'prob': 0.85
        },
        "fusarium_basilico": {
            'sintomi': ['annerimento_gambo', 'avvizzimento_pianta'],
            'prob': 0.9
        },
        "oidio_rosa": {
            'sintomi': ['muffa_biancastra'],
            'prob': 0.95
        },
        "ticchiolatura_rosa": {
            'sintomi': ['macchie_nere_foglie', 'ingiallimento_foglie', 'caduta_foglie'],
            'prob': 0.8
        }
    }
    
    dati = []
    malattie = list(patterns.keys())
    esempi_per_malattia = n_esempi // len(malattie)
    
    print(f"Generazione dataset con {esempi_per_malattia} esempi per ognuna delle {len(malattie)} malattie...")
    
    for malattia in malattie:
        pattern = patterns[malattia]
        sintomi_tipici = pattern['sintomi']
        prob_presenza = pattern['prob']
        
        for _ in range(esempi_per_malattia):
            sintomi_caso = []
            
            # Aggiungi sintomi tipici con alta probabilità
            for sintomo in sintomi_tipici:
                if np.random.random() < prob_presenza:
                    sintomi_caso.append(sintomo)
            
            # Aggiungi rumore: sintomi casuali con bassa probabilità
            altri_sintomi = [s for s in ReteBayesiana.SINTOMI_SUPPORTATI
                           if s not in sintomi_tipici]
            for sintomo in altri_sintomi:
                if np.random.random() < 0.1:  # 10% rumore
                    sintomi_caso.append(sintomo)
            
            # Assicura almeno un sintomo
            if not sintomi_caso:
                sintomi_caso = [np.random.choice(sintomi_tipici)]
            
            dati.append({
                'malattia': malattia,  # SNAKE_CASE!
                'sintomi': ','.join(sintomi_caso)
            })
    
    df = pd.DataFrame(dati)
    
    # Salva su disco
    Path(percorso_output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(percorso_output, index=False)
    print(f"[INFO] Dataset sintetico generato: {percorso_output}")
    print(f"       {len(df)} esempi, {len(malattie)} malattie")
    
    return df


# ================================================================
# ESEMPIO DI UTILIZZO
# ================================================================

if __name__ == "__main__":
    print("=== RETE BAYESIANA - PLANT-AID-KBS ===\n")
    
    # ============================================================
    # TEST 1: Generazione Dataset Sintetico
    # ============================================================
    print("TEST 1: Generazione Dataset Sintetico")
    print("=" * 60)
    
    # Verifica se esiste già il dataset
    percorso_dataset = "data/training_data.csv"
    if not Path(percorso_dataset).exists():
        print("\n[INFO] Dataset non trovato, generazione in corso...")
        genera_dataset_sintetico_bn(percorso_dataset, n_esempi=300)
    else:
        print(f"\n[INFO] Dataset già esistente: {percorso_dataset}")
    
    # ============================================================
    # TEST 2: Addestramento Rete Bayesiana
    # ============================================================
    print("\n\nTEST 2: Addestramento Rete Bayesiana")
    print("=" * 60)
    
    # Crea e addestra la rete
    rete = ReteBayesiana(percorso_modello="data/bn_model.bif")
    
    rete.addestra(
        percorso_dati=percorso_dataset,
        metodo="bayes",
        pseudo_conteggio=1.0  # Smoothing di Laplace
    )
    
    # Mostra statistiche
    stats = rete.ottieni_statistiche_modello()
    print(f"\n=== STATISTICHE MODELLO ===")
    print(f"Nodi totali: {stats['numero_nodi']}")
    print(f"Archi totali: {stats['numero_archi']}")
    print(f"CPD apprese: {stats['numero_cpd']}")
    print(f"Stati sintomi: {stats['stati_sintomi']}")
    
    # ============================================================
    # TEST 3: Salvataggio Modello
    # ============================================================
    print("\n\nTEST 3: Salvataggio Modello")
    print("=" * 60)
    
    rete.salva_modello()
    
    # ============================================================
    # TEST 4: Caricamento Modello
    # ============================================================
    print("\n\nTEST 4: Caricamento Modello")
    print("=" * 60)
    
    rete_caricata = ReteBayesiana(percorso_modello="data/bn_model.bif")
    rete_caricata.carica_modello()
    
    # ============================================================
    # TEST 5: Inferenza - Caso Occhio di Pavone
    # ============================================================
    print("\n\nTEST 5: Inferenza - Caso Occhio di Pavone")
    print("=" * 60)
    
    sintomi_test = [
        'macchie_circolari_grigie',
        'ingiallimento_foglie',
        'caduta_foglie'
    ]
    
    risultato = rete_caricata.esegui_inferenza(
        sintomi_test,
        mostra_dettagli=True
    )
    
    # ============================================================
    # TEST 6: Inferenza - Caso Fusarium
    # ============================================================
    print("\n\nTEST 6: Inferenza - Caso Fusarium del Basilico")
    print("=" * 60)
    
    sintomi_fusarium = [
        'annerimento_gambo',
        'avvizzimento_pianta'
    ]
    
    risultato_fusarium = rete_caricata.esegui_inferenza(
        sintomi_fusarium,
        mostra_dettagli=True
    )
    
    # ============================================================
    # TEST 7: Inferenza Multipla
    # ============================================================
    print("\n\nTEST 7: Inferenza Multipla")
    print("=" * 60)
    
    casi_multipli = [
        {
            'sintomi': ['macchie_nere_foglie', 'caduta_foglie'],
            'malattia_vera': 'ticchiolatura_rosa'
        },
        {
            'sintomi': ['muffa_biancastra'],
            'malattia_vera': 'oidio_rosa'
        },
        {
            'sintomi': ['annerimento_gambo'],
            'malattia_vera': 'fusarium_basilico'
        }
    ]
    
    risultati_multipli = rete_caricata.inferenza_multipla(casi_multipli)
    
    print("\nRisultati inferenza multipla:")
    for ris in risultati_multipli:
        nome_leggibile = rete_caricata.MAPPING_MALATTIE.get(ris['malattia_predetta'], ris['malattia_predetta'])
        print(f"\nCaso {ris['caso_id']}:")
        print(f"  Sintomi: {ris['sintomi']}")
        if ris.get('malattia_vera'):
            nome_vera = rete_caricata.MAPPING_MALATTIE.get(ris['malattia_vera'], ris['malattia_vera'])
            print(f"  Vera: {nome_vera}")
        print(f"  Predetta: {nome_leggibile}")
        print(f"  Confidenza: {ris['confidenza']:.3f}")
        print(f"  Corretto: {ris.get('corretto', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETATI CON SUCCESSO!")
    print("=" * 60)
