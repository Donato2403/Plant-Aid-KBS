"""
Modulo per il classificatore SVM del sistema Plant-Aid-KBS

Questo modulo fornisce funzionalità per:
- Addestramento di un modello SVM multi-classe per la classificazione delle malattie
- Preprocessing e trasformazione delle feature
- Salvataggio e caricamento del modello addestrato
- Predizione con livelli di confidenza
- Valutazione delle prestazioni

Versione: 1.0 - Apprendimento Supervisionato
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import joblib
from dataclasses import dataclass

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RisultatoClassificazione:
    """
    Rappresenta il risultato di una classificazione SVM.
    
    Attributes:
        malattia_predetta: Nome della malattia predetta
        confidenza: Livello di confidenza della predizione (0.0-1.0)
        probabilita_classi: Dizionario con probabilità per ogni classe
        feature_utilizzate: Feature utilizzate per la predizione
    """
    malattia_predetta: str
    confidenza: float
    probabilita_classi: Dict[str, float]
    feature_utilizzate: List[str]


class ClassificatoreSVM:
    """
    Classificatore SVM per la diagnosi di malattie delle piante.
    
    Implementa un modello di Support Vector Machine con kernel RBF
    per la classificazione multi-classe delle malattie in base ai sintomi
    e alle caratteristiche osservate.
    """
    
    # Percorsi predefiniti per il salvataggio dei modelli
    PERCORSO_MODELLO_DEFAULT = Path("data/svm_model.pkl")
    PERCORSO_TRASFORMATORE_DEFAULT = Path("data/svm_transformer.pkl")
    
    # Mapping centralizzato malattie (coerente con Datalog)
    MALATTIE_SUPPORTATE = [
        "Occhio di Pavone",
        "Rogna dell'Olivo",
        "Lebbra dell'Olivo",
        "Ticchiolatura della Rosa",
        "Oidio della Rosa",
        "Peronospora della Rosa",
        "Peronospora del Basilico",
        "Fusarium del Basilico"
    ]
    
    # Feature utilizzate per la classificazione
    FEATURE_SINTOMI = [
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
    
    FEATURE_AMBIENTALI = [
        "umidita_alta",
        "temperatura_mite",
        "piogge_recenti",
        "ristagno_idrico"
    ]
    
    FEATURE_PIANTA = [
        "pianta_olivo",
        "pianta_rosa",
        "pianta_basilico"
    ]
    
    def __init__(
        self,
        kernel: str = 'rbf',
        C: float = 1.0,
        gamma: str = 'scale',
        probabilita: bool = True,
        random_state: int = 42
    ):
        """
        Inizializza il classificatore SVM.
        
        Args:
            kernel: Tipo di kernel ('linear', 'rbf', 'poly')
            C: Parametro di regolarizzazione
            gamma: Coefficiente del kernel
            probabilita: Se True, abilita le stime di probabilità
            random_state: Seed per la riproducibilità
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probabilita = probabilita
        self.random_state = random_state
        
        # Inizializza modello e trasformatori
        self.modello = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=self.probabilita,
            random_state=self.random_state,
            class_weight='balanced'  # Gestisce classi sbilanciate
        )
        
        self.scaler = StandardScaler()
        self.codificatore_etichette = LabelEncoder()
        
        # Stato del modello
        self.addestrato = False
        self.feature_nomi = None
        self.classi_ = None
        self.metriche_addestramento = {}
        
    def _prepara_feature(
        self, 
        dati: pd.DataFrame,
        addestramento: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepara e trasforma le feature per l'addestramento o la predizione.
        
        Args:
            dati: DataFrame con le feature
            addestramento: Se True, è in fase di addestramento
            
        Returns:
            Tupla (X, y) dove X sono le feature trasformate e y le etichette
        """
        # Costruisce la lista completa delle feature
        tutte_feature = (
            self.FEATURE_SINTOMI + 
            self.FEATURE_AMBIENTALI + 
            self.FEATURE_PIANTA
        )
        
        # Verifica che tutte le feature richieste siano presenti
        feature_mancanti = set(tutte_feature) - set(dati.columns)
        if feature_mancanti and 'malattia' not in feature_mancanti:
            print(f"[WARNING] Feature mancanti: {feature_mancanti}")
            # Aggiunge colonne mancanti con valore 0
            for feat in feature_mancanti:
                if feat in tutte_feature:
                    dati[feat] = 0
        
        # Estrae le feature
        X = dati[tutte_feature].values
        
        # Normalizzazione
        if addestramento:
            X = self.scaler.fit_transform(X)
            self.feature_nomi = tutte_feature
        else:
            X = self.scaler.transform(X)
        
        # Estrae le etichette se in fase di addestramento
        y = None
        if 'malattia' in dati.columns:
            if addestramento:
                y = self.codificatore_etichette.fit_transform(dati['malattia'])
                self.classi_ = self.codificatore_etichette.classes_
            else:
                y = self.codificatore_etichette.transform(dati['malattia'])
        
        return X, y
    
    def addestra(
        self,
        dati_addestramento: pd.DataFrame,
        validazione_incrociata: bool = True,
        n_fold: int = 5,
        ottimizzazione_iperparametri: bool = False
    ) -> Dict[str, Any]:
        """
        Addestra il modello SVM sui dati forniti.
        
        Args:
            dati_addestramento: DataFrame con colonne per sintomi, 
                               condizioni ambientali, tipo pianta e 'malattia'
            validazione_incrociata: Se True, esegue cross-validation
            n_fold: Numero di fold per la cross-validation
            ottimizzazione_iperparametri: Se True, ottimizza C e gamma
            
        Returns:
            Dizionario con metriche di addestramento
        """
        print("=== ADDESTRAMENTO CLASSIFICATORE SVM ===\n")
        
        # Verifica presenza colonna target
        if 'malattia' not in dati_addestramento.columns:
            raise ValueError("Il DataFrame deve contenere la colonna 'malattia'")
        
        # Prepara i dati
        X, y = self._prepara_feature(dati_addestramento, addestramento=True)
        
        print(f"Dimensione dataset: {X.shape[0]} esempi, {X.shape[1]} feature")
        print(f"Classi: {list(self.classi_)}\n")
        
        # Ottimizzazione iperparametri (opzionale)
        if ottimizzazione_iperparametri:
            print("Ottimizzazione iperparametri in corso...")
            self._ottimizza_iperparametri(X, y)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=self.random_state,
            stratify=y
        )
        
        # Addestramento
        print("Addestramento del modello SVM...")
        self.modello.fit(X_train, y_train)
        self.addestrato = True
        
        # Validazione incrociata
        if validazione_incrociata:
            print(f"\nEsecuzione {n_fold}-Fold Cross Validation...")
            scores_cv = cross_val_score(
                self.modello, 
                X_train, 
                y_train, 
                cv=n_fold,
                scoring='accuracy'
            )
            print(f"Accuracy CV: {scores_cv.mean():.3f} (+/- {scores_cv.std():.3f})")
        
        # Valutazione su test set
        y_pred_train = self.modello.predict(X_train)
        y_pred_test = self.modello.predict(X_test)
        
        # Calcola metriche
        metriche = {
            'accuracy_train': accuracy_score(y_train, y_pred_train),
            'accuracy_test': accuracy_score(y_test, y_pred_test),
            'f1_score_test': f1_score(y_test, y_pred_test, average='weighted'),
            'precision_test': precision_score(y_test, y_pred_test, average='weighted'),
            'recall_test': recall_score(y_test, y_pred_test, average='weighted'),
            'n_support_vectors': self.modello.n_support_.sum(),
            'n_esempi_train': X_train.shape[0],
            'n_esempi_test': X_test.shape[0]
        }
        
        if validazione_incrociata:
            metriche['cv_accuracy_mean'] = scores_cv.mean()
            metriche['cv_accuracy_std'] = scores_cv.std()
        
        self.metriche_addestramento = metriche
        
        # Stampa report
        print("\n=== METRICHE DI ADDESTRAMENTO ===")
        print(f"Accuracy Training: {metriche['accuracy_train']:.3f}")
        print(f"Accuracy Test: {metriche['accuracy_test']:.3f}")
        print(f"F1-Score Test: {metriche['f1_score_test']:.3f}")
        print(f"Precision Test: {metriche['precision_test']:.3f}")
        print(f"Recall Test: {metriche['recall_test']:.3f}")
        print(f"Vettori di Supporto: {metriche['n_support_vectors']}")
        
        print("\n=== REPORT DI CLASSIFICAZIONE (Test Set) ===")
        nomi_classi = self.codificatore_etichette.inverse_transform(
            np.unique(y_test)
        )
        print(classification_report(
            y_test, 
            y_pred_test, 
            target_names=nomi_classi,
            digits=3
        ))
        
        return metriche
    
    def _ottimizza_iperparametri(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ottimizza gli iperparametri C e gamma tramite Grid Search.
        
        Args:
            X: Feature di addestramento
            y: Etichette di addestramento
        """
        griglia_parametri = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
        
        grid_search = GridSearchCV(
            SVC(kernel=self.kernel, probability=self.probabilita, random_state=self.random_state),
            griglia_parametri,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        print(f"Migliori parametri trovati: {grid_search.best_params_}")
        print(f"Miglior score CV: {grid_search.best_score_:.3f}\n")
        
        # Aggiorna il modello con i parametri ottimali
        self.C = grid_search.best_params_['C']
        self.gamma = grid_search.best_params_['gamma']
        self.modello = grid_search.best_estimator_
    
    def predici(
        self,
        feature_input: Dict[str, int]
    ) -> RisultatoClassificazione:
        """
        Predice la malattia in base alle feature fornite.
        
        Args:
            feature_input: Dizionario con feature booleane (0/1)
                          Es: {'macchie_circolari_grigie': 1, 'ingiallimento_foglie': 1, ...}
        
        Returns:
            RisultatoClassificazione con malattia predetta e confidenza
        """
        if not self.addestrato:
            raise RuntimeError("Il modello deve essere addestrato prima di predire")
        
        # Crea DataFrame da feature input
        tutte_feature = (
            self.FEATURE_SINTOMI + 
            self.FEATURE_AMBIENTALI + 
            self.FEATURE_PIANTA
        )
        
        # Completa le feature mancanti con 0
        dati_input = {feat: feature_input.get(feat, 0) for feat in tutte_feature}
        df_input = pd.DataFrame([dati_input])
        
        # Prepara le feature
        X, _ = self._prepara_feature(df_input, addestramento=False)
        
        # Predizione
        classe_predetta_idx = self.modello.predict(X)[0]
        malattia_predetta = self.codificatore_etichette.inverse_transform([classe_predetta_idx])[0]
        
        # Probabilità (se abilitata)
        if self.probabilita:
            probabilita = self.modello.predict_proba(X)[0]
            confidenza = probabilita[classe_predetta_idx]
            
            # Crea dizionario probabilità per tutte le classi
            probabilita_classi = {
                self.codificatore_etichette.inverse_transform([i])[0]: float(prob)
                for i, prob in enumerate(probabilita)
            }
        else:
            # Se la probabilità non è abilitata, usa la decision function
            decision = self.modello.decision_function(X)[0]
            confidenza = float(np.max(np.abs(decision)))
            probabilita_classi = {}
        
        # Feature effettivamente utilizzate (quelle con valore 1)
        feature_utilizzate = [
            feat for feat, val in feature_input.items() if val == 1
        ]
        
        return RisultatoClassificazione(
            malattia_predetta=malattia_predetta,
            confidenza=confidenza,
            probabilita_classi=probabilita_classi,
            feature_utilizzate=feature_utilizzate
        )
    
    def predici_batch(
        self,
        dati: pd.DataFrame
    ) -> List[RisultatoClassificazione]:
        """
        Predice le malattie per un batch di esempi.
        
        Args:
            dati: DataFrame con le feature
            
        Returns:
            Lista di RisultatoClassificazione
        """
        risultati = []
        
        for _, riga in dati.iterrows():
            feature_dict = riga.to_dict()
            # Rimuove eventuale colonna 'malattia'
            feature_dict.pop('malattia', None)
            
            risultato = self.predici(feature_dict)
            risultati.append(risultato)
        
        return risultati
    
    def valuta(self, dati_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Valuta le prestazioni del modello su un dataset di test.
        
        Args:
            dati_test: DataFrame con feature e colonna 'malattia'
            
        Returns:
            Dizionario con metriche di valutazione
        """
        if not self.addestrato:
            raise RuntimeError("Il modello deve essere addestrato prima della valutazione")
        
        if 'malattia' not in dati_test.columns:
            raise ValueError("Il DataFrame di test deve contenere la colonna 'malattia'")
        
        # Prepara i dati
        X_test, y_test = self._prepara_feature(dati_test, addestramento=False)
        
        # Predizioni
        y_pred = self.modello.predict(X_test)
        
        # Calcola metriche
        metriche = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(
                y_test, 
                y_pred, 
                target_names=self.classi_,
                output_dict=True
            )
        }
        
        print("\n=== METRICHE DI VALUTAZIONE ===")
        print(f"Accuracy: {metriche['accuracy']:.3f}")
        print(f"F1-Score: {metriche['f1_score']:.3f}")
        print(f"Precision: {metriche['precision']:.3f}")
        print(f"Recall: {metriche['recall']:.3f}")
        
        print("\n=== REPORT DI CLASSIFICAZIONE ===")
        print(classification_report(y_test, y_pred, target_names=self.classi_, digits=3))
        
        return metriche
    
    def salva_modello(
        self,
        percorso_modello: Optional[Path] = None,
        percorso_trasformatore: Optional[Path] = None
    ) -> None:
        """
        Salva il modello addestrato e i trasformatori su disco.
        
        Args:
            percorso_modello: Percorso per salvare il modello SVM
            percorso_trasformatore: Percorso per salvare scaler e encoder
        """
        if not self.addestrato:
            raise RuntimeError("Il modello deve essere addestrato prima di salvarlo")
        
        # Usa percorsi di default se non specificati
        percorso_modello = percorso_modello or self.PERCORSO_MODELLO_DEFAULT
        percorso_trasformatore = percorso_trasformatore or self.PERCORSO_TRASFORMATORE_DEFAULT
        
        # Crea directory se non esiste
        percorso_modello.parent.mkdir(parents=True, exist_ok=True)
        percorso_trasformatore.parent.mkdir(parents=True, exist_ok=True)
        
        # Salva modello
        joblib.dump(self.modello, percorso_modello)
        print(f"[INFO] Modello salvato in: {percorso_modello}")
        
        # Salva trasformatori
        trasformatori = {
            'scaler': self.scaler,
            'codificatore_etichette': self.codificatore_etichette,
            'feature_nomi': self.feature_nomi,
            'classi': self.classi_,
            'metriche_addestramento': self.metriche_addestramento,
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma
        }
        joblib.dump(trasformatori, percorso_trasformatore)
        print(f"[INFO] Trasformatori salvati in: {percorso_trasformatore}")
    
    def carica_modello(
        self,
        percorso_modello: Optional[Path] = None,
        percorso_trasformatore: Optional[Path] = None
    ) -> None:
        """
        Carica un modello addestrato e i trasformatori da disco.
        
        Args:
            percorso_modello: Percorso del modello SVM salvato
            percorso_trasformatore: Percorso dei trasformatori salvati
        """
        # Usa percorsi di default se non specificati
        percorso_modello = percorso_modello or self.PERCORSO_MODELLO_DEFAULT
        percorso_trasformatore = percorso_trasformatore or self.PERCORSO_TRASFORMATORE_DEFAULT
        
        # Verifica esistenza file
        if not percorso_modello.exists():
            raise FileNotFoundError(f"Modello non trovato: {percorso_modello}")
        if not percorso_trasformatore.exists():
            raise FileNotFoundError(f"Trasformatori non trovati: {percorso_trasformatore}")
        
        # Carica modello
        self.modello = joblib.load(percorso_modello)
        print(f"[INFO] Modello caricato da: {percorso_modello}")
        
        # Carica trasformatori
        trasformatori = joblib.load(percorso_trasformatore)
        self.scaler = trasformatori['scaler']
        self.codificatore_etichette = trasformatori['codificatore_etichette']
        self.feature_nomi = trasformatori['feature_nomi']
        self.classi_ = trasformatori['classi']
        self.metriche_addestramento = trasformatori.get('metriche_addestramento', {})
        self.kernel = trasformatori.get('kernel', 'rbf')
        self.C = trasformatori.get('C', 1.0)
        self.gamma = trasformatori.get('gamma', 'scale')
        
        self.addestrato = True
        print(f"[INFO] Trasformatori caricati da: {percorso_trasformatore}")
        print(f"[INFO] Classi supportate: {list(self.classi_)}")
    
    def ottieni_feature_importanti(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Restituisce le feature più importanti per la classificazione.
        
        NOTA: Per SVM con kernel non lineare, l'importanza delle feature
        non è direttamente calcolabile. Questo metodo fornisce una stima
        basata sui coefficienti del modello (solo per kernel lineare).
        
        Args:
            top_n: Numero di feature da restituire
            
        Returns:
            Lista di tuple (nome_feature, importanza)
        """
        if not self.addestrato:
            raise RuntimeError("Il modello deve essere addestrato")
        
        if self.kernel == 'linear':
            # Per kernel lineare, usa i coefficienti
            coefficienti = np.abs(self.modello.coef_).mean(axis=0)
            feature_importance = list(zip(self.feature_nomi, coefficienti))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            return feature_importance[:top_n]
        else:
            print("[WARNING] L'importanza delle feature è disponibile solo per kernel='linear'")
            return []


# ================================================================
# FUNZIONI DI UTILITÀ
# ================================================================

def crea_dataset_sintetico(
    n_esempi: int = 200,
    seed: int = 42
) -> pd.DataFrame:
    """
    Crea un dataset sintetico per test e demo.
    
    Args:
        n_esempi: Numero di esempi da generare
        seed: Seed per la riproducibilità
        
    Returns:
        DataFrame con esempi sintetici
    """
    np.random.seed(seed)
    
    malattie_patterns = {
        "Occhio di Pavone": {
            'macchie_circolari_grigie': 0.9,
            'ingiallimento_foglie': 0.8,
            'caduta_foglie': 0.7,
            'pianta_olivo': 1.0,
            'umidita_alta': 0.8
        },
        "Fusarium del Basilico": {
            'annerimento_gambo': 0.95,
            'avvizzimento_pianta': 0.85,
            'pianta_basilico': 1.0,
            'umidita_alta': 0.7,
            'ristagno_idrico': 0.6
        },
        "Oidio della Rosa": {
            'muffa_biancastra': 0.95,
            'pianta_rosa': 1.0,
            'umidita_alta': 0.6,
            'temperatura_mite': 0.7
        },
        "Ticchiolatura della Rosa": {
            'macchie_nere_foglie': 0.9,
            'ingiallimento_foglie': 0.7,
            'caduta_foglie': 0.6,
            'pianta_rosa': 1.0,
            'piogge_recenti': 0.7
        }
    }
    
    esempi = []
    malattie = list(malattie_patterns.keys())
    esempi_per_malattia = n_esempi // len(malattie)
    
    tutte_feature = (
        ClassificatoreSVM.FEATURE_SINTOMI +
        ClassificatoreSVM.FEATURE_AMBIENTALI +
        ClassificatoreSVM.FEATURE_PIANTA
    )
    
    for malattia in malattie:
        pattern = malattie_patterns[malattia]
        
        for _ in range(esempi_per_malattia):
            esempio = {'malattia': malattia}
            
            for feat in tutte_feature:
                if feat in pattern:
                    # Genera valore binario con probabilità dal pattern
                    esempio[feat] = int(np.random.random() < pattern[feat])
                else:
                    # Rumore casuale per altre feature
                    esempio[feat] = int(np.random.random() < 0.1)
            
            esempi.append(esempio)
    
    return pd.DataFrame(esempi)


def genera_report_completo(
    classificatore: ClassificatoreSVM,
    dati_test: pd.DataFrame
) -> str:
    """
    Genera un report testuale completo delle prestazioni del modello.
    
    Args:
        classificatore: Classificatore addestrato
        dati_test: Dati per la valutazione
        
    Returns:
        Stringa con il report formattato
    """
    metriche = classificatore.valuta(dati_test)
    
    report = f"""
{'='*60}
REPORT COMPLETO CLASSIFICATORE SVM - PLANT-AID-KBS
{'='*60}

CONFIGURAZIONE MODELLO:
- Kernel: {classificatore.kernel}
- C (regolarizzazione): {classificatore.C}
- Gamma: {classificatore.gamma}
- Vettori di Supporto: {classificatore.metriche_addestramento.get('n_support_vectors', 'N/A')}

METRICHE GLOBALI:
- Accuracy: {metriche['accuracy']:.3f}
- F1-Score (weighted): {metriche['f1_score']:.3f}
- Precision (weighted): {metriche['precision']:.3f}
- Recall (weighted): {metriche['recall']:.3f}

METRICHE PER CLASSE:
"""
    
    for classe, metr in metriche['classification_report'].items():
        if isinstance(metr, dict):
            report += f"\n{classe}:\n"
            report += f"  - Precision: {metr['precision']:.3f}\n"
            report += f"  - Recall: {metr['recall']:.3f}\n"
            report += f"  - F1-Score: {metr['f1-score']:.3f}\n"
            report += f"  - Support: {metr['support']}\n"
    
    report += f"\n{'='*60}\n"
    
    return report


# ================================================================
# ESEMPIO DI UTILIZZO
# ================================================================

if __name__ == "__main__":
    print("=== CLASSIFICATORE SVM - PLANT-AID-KBS ===\n")
    
    # ============================================================
    # TEST 1: Addestramento con Dataset Sintetico
    # ============================================================
    print("TEST 1: Generazione Dataset e Addestramento")
    print("=" * 60)
    
    # Genera dataset sintetico
    print("\nGenerazione dataset sintetico...")
    dataset = crea_dataset_sintetico(n_esempi=400, seed=42)
    print(f"Dataset generato: {len(dataset)} esempi")
    print(f"\nDistribuzione classi:")
    print(dataset['malattia'].value_counts())
    
    # Crea e addestra classificatore
    classificatore = ClassificatoreSVM(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        probabilita=True
    )
    
    metriche = classificatore.addestra(
        dataset,
        validazione_incrociata=True,
        n_fold=5,
        ottimizzazione_iperparametri=False
    )
    
    # ============================================================
    # TEST 2: Predizione Singola
    # ============================================================
    print("\n\nTEST 2: Predizione Singola - Occhio di Pavone")
    print("=" * 60)
    
    feature_test = {
        'macchie_circolari_grigie': 1,
        'ingiallimento_foglie': 1,
        'caduta_foglie': 1,
        'pianta_olivo': 1,
        'umidita_alta': 1,
        'temperatura_mite': 1,
        'piogge_recenti': 1
    }
    
    risultato = classificatore.predici(feature_test)
    
    print(f"\nMalattia Predetta: {risultato.malattia_predetta}")
    print(f"Confidenza: {risultato.confidenza:.3f}")
    print(f"\nProbabilità per tutte le classi:")
    for malattia, prob in sorted(
        risultato.probabilita_classi.items(), 
        key=lambda x: x[1], 
        reverse=True
    ):
        print(f"  {malattia}: {prob:.3f}")
    
    # ============================================================
    # TEST 3: Salvataggio e Caricamento
    # ============================================================
    print("\n\nTEST 3: Salvataggio e Caricamento Modello")
    print("=" * 60)
    
    # Salva
    print("\nSalvataggio modello...")
    classificatore.salva_modello()
    
    # Carica
    print("\nCaricamento modello...")
    classificatore_caricato = ClassificatoreSVM()
    classificatore_caricato.carica_modello()
    
    # Test predizione con modello caricato
    print("\nTest predizione con modello caricato...")
    risultato_caricato = classificatore_caricato.predici(feature_test)
    print(f"Malattia Predetta: {risultato_caricato.malattia_predetta}")
    print(f"Confidenza: {risultato_caricato.confidenza:.3f}")
    
    # ============================================================
    # TEST 4: Predizione Batch
    # ============================================================
    print("\n\nTEST 4: Predizione Batch")
    print("=" * 60)
    
    # Crea mini-batch di test
    test_batch = dataset.sample(n=5, random_state=42)
    risultati_batch = classificatore.predici_batch(test_batch)
    
    print("\nRisultati predizioni batch:")
    for i, (_, riga) in enumerate(test_batch.iterrows()):
        print(f"\nEsempio {i+1}:")
        print(f"  Ground Truth: {riga['malattia']}")
        print(f"  Predetto: {risultati_batch[i].malattia_predetta}")
        print(f"  Confidenza: {risultati_batch[i].confidenza:.3f}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETATI")
    print("=" * 60)
