#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================================================
Plant-Aid-KBS - Sistema Diagnostico Intelligente per la Cura delle Piante
=============================================================================

File Principale (main_cli.py)
Orchestratore del sistema ibrido.

Questo script funge da interfaccia a riga di comando (CLI) per l'utente
e implementa il flusso diagnostico descritto nella proposta di progetto:
1. Raccoglie l'input dell'utente.
2. Esegue l'inferenza simbolica (Datalog).
3. Esegue la classificazione statistica (SVM).
4. Esegue l'analisi probabilistica (Rete Bayesiana).
5. Aggrega i risultati e li presenta all'utente con una spiegazione.

Autore: Donato Cancellara (come da README)
Integrazione: Gemini
"""

import click  # Per un'interfaccia CLI pulita
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional

# --- Importazione dei Moduli del Progetto ---
# Assicura che la directory radice sia nel path
sys.path.append(str(Path(__file__).resolve().parent))

# 1. Motore di Conoscenza Formale (Ontologia)
from kbs_engine.ontology_manager import GestoreOntologia
# 2. Motore di Inferenza Simbolica (Regole)
from kbs_engine.datalog_engine import MotoreDatalog
# 3. Modello Statistico (Classificazione)
from ml_engine.svm_model import ClassificatoreSVM, RisultatoClassificazione
# 4. Modello Probabilistico (Incertezza)
from ml_engine.bn_model import ReteBayesiana


# --- Costanti e Mappature ---

# Percorsi ai modelli addestrati e alla base di conoscenza
PATH_ONTOLOGIA = Path("data/plant_care.owl")
PATH_SVM_MODELLO = Path("data/svm_model.pkl")
PATH_SVM_TRASFORMATORE = Path("data/svm_transformer.pkl")
PATH_BN_MODELLO = Path("data/bn_model.bif")

# Pesi per l'aggregazione dei risultati (come da proposta )
# Diamo più peso alla rete Bayesiana per la gestione dell'incertezza
# e al Datalog per la conoscenza esperta.
PESO_RETE_BAYESIANA = 0.5
PESO_MOTORE_DATALOG = 0.3
PESO_CLASSIFICATORE_SVM = 0.2

# Mappatura Piante (per CLI)
PIANTE_DISPONIBILI = {
    "1": {"nome_leggibile": "Olivo", "codice": "olivo"},
    "2": {"nome_leggibile": "Rosa", "codice": "rosa"},
    "3": {"nome_leggibile": "Basilico", "codice": "basilico"},
}

# Mappatura Sintomi (per CLI)
# Basata su FEATURE_SINTOMI (SVM) e SINTOMI_SUPPORTATI (BN)
SINTOMI_DISPONIBILI = {
    "1": {"nome_leggibile": "Macchie circolari grigie (su foglie)", "codice": "macchie_circolari_grigie"},
    "2": {"nome_leggibile": "Ingiallimento delle foglie", "codice": "ingiallimento_foglie"},
    "3": {"nome_leggibile": "Caduta prematura delle foglie", "codice": "caduta_foglie"},
    "4": {"nome_leggibile": "Tumori o escrescenze (su rami/fusto)", "codice": "tumori_rami"},
    "5": {"nome_leggibile": "Macchie bruno-nerastre (su frutti)", "codice": "macchie_bruno_nerastre_frutti"},
    "6": {"nome_leggibile": "Macchie nere (su foglie)", "codice": "macchie_nere_foglie"},
    "7": {"nome_leggibile": "Muffa bianca e polverosa", "codice": "muffa_biancastra"},
    "8": {"nome_leggibile": "Annerimento del gambo (alla base)", "codice": "annerimento_gambo"},
    "9": {"nome_leggibile": "Avvizzimento completo della pianta", "codice": "avvizzimento_pianta"},
}

# Mappatura Stagioni (per Datalog)
STAGIONI_DISPONIBILI = {
    "1": {"nome_leggibile": "Primavera", "codice": "primavera"},
    "2": {"nome_leggibile": "Estate", "codice": "estate"},
    "3": {"nome_leggibile": "Autunno", "codice": "autunno"},
    "4": {"nome_leggibile": "Inverno (mite)", "codice": "inverno_mite"},
}

# Mappatura cruciale per normalizzare i nomi delle malattie tra i moduli
# Datalog/BN (snake_case) -> Ontologia (CamelCase)
MAPPING_CANONICO_TO_ONTOLOGIA = {
    "occhio_pavone": "Occhio_di_Pavone",
    "rogna_olivo": "Rogna_Olivo",
    "lebbra_olivo": "Lebbra_Olivo",
    "ticchiolatura_rosa": "Ticchiolatura_Rosa",
    "oidio_rosa": "Oidio_Rosa",
    "peronospora_rosa": "Peronospora_Rosa",
    "peronospora_basilico": "Peronospora_Basilico",
    "fusarium_basilico": "Fusarium_Basilico"
}


class SistemaDiagnostico:
    """
    Classe orchestratore che carica e gestisce tutti i moduli del KBS.
    """

    def __init__(self):
        """Inizializza e carica tutti i sottomoduli."""
        click.echo(click.style("Avvio del Sistema Diagnostico Ibrido...", fg="cyan"))
        self.modelli_caricati = False
        
        try:
            # 1. Carica Ontologia (Conoscenza statica)
            self.gestore_ontologia = GestoreOntologia(str(PATH_ONTOLOGIA))
            click.echo(f"  [1/4] ✓ Gestore Ontologia caricato ({PATH_ONTOLOGIA.name})")

            # 2. Inizializza Motore Datalog (Regole euristiche)
            self.motore_datalog = MotoreDatalog()
            click.echo("  [2/4] ✓ Motore Datalog (Regole) inizializzato")

            # 3. Carica Classificatore SVM (Modello statistico)
            self.classificatore_svm = ClassificatoreSVM()
            self.classificatore_svm.carica_modello(
                percorso_modello=PATH_SVM_MODELLO,
                percorso_trasformatore=PATH_SVM_TRASFORMATORE
            )
            click.echo(f"  [3/4] ✓ Classificatore SVM caricato ({PATH_SVM_MODELLO.name})")

            # 4. Carica Rete Bayesiana (Modello probabilistico)
            self.rete_bayesiana = ReteBayesiana(str(PATH_BN_MODELLO))
            self.rete_bayesiana.carica_modello()
            click.echo(f"  [4/4] ✓ Rete Bayesiana caricata ({PATH_BN_MODELLO.name})")

            click.echo(click.style("\nPlant-Aid-KBS pronto per la diagnosi.", bold=True, fg="green"))
            self.modelli_caricati = True

        except FileNotFoundError as e:
            click.echo(click.style(f"\n[ERRORE FATALE] File modello non trovato!", fg="red", bold=True))
            click.echo(f"Dettagli: {e}")
            click.echo("Assicurati di aver eseguito gli script di addestramento per SVM e BN e che i file .owl siano presenti.")
            sys.exit(1)
        except Exception as e:
            click.echo(click.style(f"\n[ERRORE FATALE] Errore imprevisto durante il caricamento.", fg="red", bold=True))
            click.echo(f"Dettagli: {e}")
            sys.exit(1)

    def _mappa_input_per_svm(self, sintomi_codice: List[str], pianta_codice: str) -> Dict[str, int]:
        """
        Converte l'input dell'utente nel formato dizionario richiesto dal
        ClassificatoreSVM.predici()
        """
        feature_input = {}
        
        # 1. Imposta tutte le feature dei sintomi a 0 (assente)
        for sintomo in self.classificatore_svm.FEATURE_SINTOMI:
            feature_input[sintomo] = 0
            
        # 2. Imposta i sintomi selezionati a 1 (presente)
        for sintomo_sel in sintomi_codice:
            if sintomo_sel in feature_input:
                feature_input[sintomo_sel] = 1
        
        # 3. Imposta le feature della pianta (one-hot encoding)
        feature_input["pianta_olivo"] = 1 if pianta_codice == "olivo" else 0
        feature_input["pianta_rosa"] = 1 if pianta_codice == "rosa" else 0
        feature_input["pianta_basilico"] = 1 if pianta_codice == "basilico" else 0
        
        # (Opzionale: aggiunge feature ambientali se fossero raccolte)
        for amb in self.classificatore_svm.FEATURE_AMBIENTALI:
             feature_input[amb] = 0 # Non gestite in questa CLI
        
        return feature_input

    def esegui_diagnosi_completa(
        self, 
        pianta_codice: str, 
        sintomi_codice: List[str], 
        stagione_codice: str
    ) -> Dict[str, Any]:
        """
        Orchestra l'intero flusso diagnostico ibrido.
        """
        if not self.modelli_caricati:
            raise RuntimeError("Modelli non caricati correttamente.")

        click.echo("\n" + "="*70)
        click.echo(click.style("AVVIO FLUSSO DIAGNOSTICO IBRIDO...", fg="cyan", bold=True))
        click.echo("="*70)

        # --- 1. INFERENZA SIMBOLICA (DATALOG) ---
        click.echo("[1/3] Esecuzione Motore Datalog (Inferenza simbolica)...")
        self.motore_datalog.azzera_fatti()
        self.motore_datalog.imposta_tipo_pianta(pianta_codice)
        self.motore_datalog.imposta_stagione(stagione_codice)
        for s in sintomi_codice:
            self.motore_datalog.aggiungi_sintomo_osservato(s)
        
        # 'diagnosi_completa_integrata' restituisce diagnosi già filtrate
        risultati_datalog = self.motore_datalog.diagnosi_completa_integrata()

        # --- 2. CLASSIFICAZIONE STATISTICA (SVM) ---
        click.echo("[2/3] Esecuzione Classificatore SVM (Classificazione statistica)...")
        feature_svm = self._mappa_input_per_svm(sintomi_codice, pianta_codice)
        risultato_svm = self.classificatore_svm.predici(feature_svm)
        
        # --- 3. ANALISI PROBABILISTICA (RETE BAYESIANA) ---
        click.echo("[3/3] Esecuzione Rete Bayesiana (Analisi probabilistica)...")
        # La Rete Bayesiana usa solo i sintomi come evidenza
        risultati_bn = self.rete_bayesiana.esegui_inferenza(sintomi_codice)
        
        click.echo(click.style("...Tutti i moduli hanno completato l'analisi.", fg="cyan"))
        
        # --- 4. AGGREGAZIONE E SPIEGAZIONE  ---
        report_finale = self._aggrega_risultati(
            risultati_datalog, 
            risultato_svm, 
            risultati_bn
        )
        
        # --- 5. ARRICCHIMENTO CON ONTOLOGIA ---
        malattia_top_canonica = report_finale["diagnosi_top"].get("nome_canonico")
        
        if malattia_top_canonica:
            # Converte il nome canonico (es. occhio_pavone) nel nome dell'Ontologia (es. Occhio_di_Pavone)
            nome_ontologia = MAPPING_CANONICO_TO_ONTOLOGIA.get(malattia_top_canonica)
            if nome_ontologia:
                info_ontologia = self.gestore_ontologia.ottieni_info_malattia(nome_ontologia)
                report_finale["info_ontologia"] = info_ontologia
            else:
                report_finale["info_ontologia"] = None
        else:
             report_finale["info_ontologia"] = None

        return report_finale

    def _aggrega_risultati(
        self,
        risultati_datalog: Dict[str, Any],
        risultato_svm: RisultatoClassificazione,
        risultati_bn: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Motore di aggregazione.
        Combina i risultati dei tre moduli in un unico "Fattore di Confidenza"
        e prepara la spiegazione.
        """
        
        # Mappe per normalizzare i nomi
        mappa_leggibile_a_canonico = self.motore_datalog.MAPPING_MALATTIE_LEGGIBILE_TO_ASP
        mappa_canonico_a_leggibile = self.motore_datalog.MAPPING_MALATTIE_ASP_TO_LEGGIBILE
        
        diagnosi_aggregate = {}
        
        # Normalizza l'output SVM
        # 'malattia_predetta' è in formato leggibile (es. "Occhio di Pavone")
        svm_pred_canonica = mappa_leggibile_a_canonico.get(risultato_svm.malattia_predetta)
        svm_confidenza = risultato_svm.confidenza

        # Normalizza l'output Datalog
        # 'risultati_datalog['diagnosi']' è una lista di dizionari
        datalog_confidenze = {}
        for diag in risultati_datalog.get('diagnosi', []):
            # 'malattia' è in formato leggibile (es. "Occhio di Pavone")
            nome_canonico = mappa_leggibile_a_canonico.get(diag.get('malattia'))
            if nome_canonico:
                datalog_confidenze[nome_canonico] = diag.get('confidenza', 0.0)

        # I risultati della Rete Bayesiana (risultati_bn) sono già la nostra fonte canonica
        # (es. {'occhio_pavone': 0.8, ...})

        # Cicla su tutte le malattie possibili (definite dalla BN/Datalog)
        for nome_canonico, nome_leggibile in mappa_canonico_a_leggibile.items():
            
            # Estrae punteggi da ciascun modulo
            prob_bn = risultati_bn.get(nome_canonico, 0.0)
            conf_datalog = datalog_confidenze.get(nome_canonico, 0.0)
            
            # L'SVM dà confidenza solo alla sua predizione top
            conf_svm = svm_confidenza if nome_canonico == svm_pred_canonica else 0.0
            
            # Calcola Fattore di Confidenza Aggregato (Ponderato)
            confidenza_finale = (
                (prob_bn * PESO_RETE_BAYESIANA) +
                (conf_datalog * PESO_MOTORE_DATALOG) +
                (conf_svm * PESO_CLASSIFICATORE_SVM)
            )
            
            diagnosi_aggregate[nome_canonico] = {
                "nome_canonico": nome_canonico,
                "nome_leggibile": nome_leggibile,
                "confidenza_finale": confidenza_finale,
                "spiegazione": {
                    "Rete Bayesiana": prob_bn,
                    "Motore Datalog": conf_datalog,
                    "Classificatore SVM": conf_svm
                }
            }
            
        # Trova la diagnosi migliore
        if not diagnosi_aggregate:
            return {"diagnosi_top": None, "tutte_diagnosi": []}
            
        lista_diagnosi = sorted(
            diagnosi_aggregate.values(), 
            key=lambda x: x["confidenza_finale"], 
            reverse=True
        )
        
        return {
            "diagnosi_top": lista_diagnosi[0],
            "tutte_diagnosi": lista_diagnosi
        }


# --- Funzioni della CLI ---

def stampa_titolo():
    """Stampa il titolo del programma."""
    click.echo("\n" + "*"*70)
    click.echo(click.style("                 Plant-Aid-KBS: Sistema Diagnostico Ibrido", bold=True, fg="green"))
    click.echo(click.style("     Progetto di Ingegneria della Conoscenza - A.A. 2024/2025", fg="green"))
    click.echo("*"*70 + "\n")

def raccogli_input_utente() -> Optional[Dict[str, Any]]:
    """Guida l'utente nella selezione di pianta, sintomi e stagione."""
    
    # 1. Selezione Pianta
    click.echo(click.style("1. Seleziona la pianta da diagnosticare:", bold=True, fg="yellow"))
    prompt_piante = "\n".join([f"  [{k}] {v['nome_leggibile']}" for k, v in PIANTE_DISPONIBILI.items()])
    click.echo(prompt_piante)
    scelta_pianta_idx = click.prompt("Inserisci il numero della pianta", type=click.Choice(PIANTE_DISPONIBILI.keys()), show_choices=False)
    pianta_scelta = PIANTE_DISPONIBILI[scelta_pianta_idx]
    
    # 2. Selezione Sintomi (Multi-selezione)
    click.echo(click.style("\n2. Seleziona i sintomi osservati:", bold=True, fg="yellow"))
    prompt_sintomi = "\n".join([f"  [{k}] {v['nome_leggibile']}" for k, v in SINTOMI_DISPONIBILI.items()])
    click.echo(prompt_sintomi)
    click.echo(click.style("  [0] Termina selezione sintomi", bold=True))
    
    sintomi_scelti_idx = set()
    while True:
        scelta_sintomo_idx = click.prompt(
            f"Inserisci un numero (Sintomi scelti: {len(sintomi_scelti_idx)}) o 0 per continuare", 
            type=click.Choice(['0'] + list(SINTOMI_DISPONIBILI.keys())),
            show_choices=False
        )
        if scelta_sintomo_idx == '0':
            if not sintomi_scelti_idx:
                click.echo(click.style("Attenzione: nessun sintomo selezionato. Riprova.", fg="red"))
                continue
            break
        sintomi_scelti_idx.add(scelta_sintomo_idx)

    sintomi_scelti = [SINTOMI_DISPONIBILI[idx] for idx in sintomi_scelti_idx]
    
    # 3. Selezione Stagione
    click.echo(click.style("\n3. Seleziona la stagione corrente (per Datalog):", bold=True, fg="yellow"))
    prompt_stagioni = "\n".join([f"  [{k}] {v['nome_leggibile']}" for k, v in STAGIONI_DISPONIBILI.items()])
    click.echo(prompt_stagioni)
    scelta_stagione_idx = click.prompt("Inserisci il numero della stagione", type=click.Choice(STAGIONI_DISPONIBILI.keys()), show_choices=False)
    stagione_scelta = STAGIONI_DISPONIBILI[scelta_stagione_idx]

    # Riepilogo Input
    click.echo("\n" + "-"*40)
    click.echo(click.style("Riepilogo Input Selezionato:", underline=True))
    click.echo(f"  Pianta:    {pianta_scelta['nome_leggibile']}")
    click.echo(f"  Stagione:  {stagione_scelta['nome_leggibile']}")
    click.echo(f"  Sintomi:   {[s['nome_leggibile'] for s in sintomi_scelti]}")
    click.echo("-"*40)
    
    if not click.confirm(click.style("Procedere con la diagnosi?", fg="bright_blue"), default=True):
        click.echo("Diagnosi annullata dall'utente.")
        return None

    return {
        "pianta": pianta_scelta,
        "sintomi": sintomi_scelti,
        "stagione": stagione_scelta
    }

def stampa_report_diagnosi(report: Dict[str, Any]):
    """Formatta e stampa il report finale aggregato."""
    
    diagnosi_top = report.get("diagnosi_top")
    
    if not diagnosi_top or diagnosi_top.get("confidenza_finale", 0.0) < 0.1: # Soglia minima
        click.echo("\n" + "="*70)
        click.echo(click.style("RAPPORTO DIAGNOSTICO FINALE", bold=True, fg="red"))
        click.echo("="*70)
        click.echo(click.style("Non è stato possibile formulare una diagnosi robusta.", fg="red"))
        click.echo("I sintomi forniti non corrispondono in modo sufficiente a nessuna malattia nota nel sistema.")
        return

    click.echo("\n" + "="*70)
    click.echo(click.style("RAPPORTO DIAGNOSTICO FINALE", bold=True, fg="green"))
    click.echo("="*70)
    
    # --- Sezione 1: Diagnosi Finale ---
    click.echo(click.style("\n--- 1. Diagnosi Ibrida Aggregata ---", bold=True, fg="yellow"))
    click.echo(f"  Malattia più probabile: {click.style(diagnosi_top['nome_leggibile'], bold=True, fg='bright_green')}")
    click.echo(f"  Fattore di Confidenza:  {click.style(f"{diagnosi_top['confidenza_finale']:.1%}", bold=True, fg='bright_green')}")

    # --- Sezione 2: Spiegazione del Ragionamento  ---
    click.echo(click.style("\n--- 2. Spiegazione (Pesi dei Moduli) ---", bold=True, fg="yellow"))
    spiegazione = diagnosi_top['spiegazione']
    click.echo(f"  - Rete Bayesiana (Probabilità):   {spiegazione['Rete Bayesiana']:.1%} (Peso: {PESO_RETE_BAYESIANA*100}%)")
    click.echo(f"  - Motore Datalog (Conf. Euristica): {spiegazione['Motore Datalog']:.1%} (Peso: {PESO_MOTORE_DATALOG*100}%)")
    click.echo(f"  - Classificatore SVM (Predizione):  {spiegazione['Classificatore SVM']:.1%} (Peso: {PESO_CLASSIFICATORE_SVM*100}%)")
    
    # --- Sezione 3: Dettagli dalla Base di Conoscenza (Ontologia) ---
    info_ontologia = report.get("info_ontologia")
    if info_ontologia:
        click.echo(click.style("\n--- 3. Dettagli dalla Base di Conoscenza (Ontologia) ---", bold=True, fg="yellow"))
        click.echo(click.style("\n  Descrizione Malattia:", underline=True))
        click.echo(f"  {info_ontologia.get('descrizione', 'N/A')}")
        click.echo(f"  (Nome Scientifico: {info_ontologia.get('nome_scientifico', 'N/A')})")
        click.echo(f"\n  Gravità stimata: {info_ontologia.get('gravita', 'N/A')} / 5")
        click.echo(f"  Periodo di attività: {info_ontologia.get('periodo_attivo', 'N/A')}")
        
        click.echo(click.style("\n  Trattamenti Consigliati:", underline=True))
        trattamenti = info_ontologia.get('trattamenti', [])
        if trattamenti:
            for t in trattamenti:
                click.echo(f"  - {t.get('nome').replace('_', ' ')}: {t.get('descrizione', 'N/A')}")
                if t.get('dosaggio'):
                    click.echo(f"    (Dosaggio: {t.get('dosaggio')})")
        else:
            click.echo("  Nessun trattamento specifico trovato nell'ontologia.")
    
    click.echo("\n" + "="*70)
    click.echo(click.style("Avvertenza: Questa è una diagnosi automatica. Consultare sempre un agronomo.", fg="cyan"))
    click.echo("="*70)


# --- Definizione Comandi CLI ---

@click.group()
def cli():
    """
    Plant-Aid-KBS: Sistema Diagnostico Ibrido per la Cura delle Piante.
    """
    stampa_titolo()
    pass


@cli.command(name="diagnosi", help="Avvia una nuova sessione di diagnosi interattiva.")
def avvia_diagnosi():
    """
    Comando principale per eseguire una diagnosi interattiva.
    """
    try:
        # 1. Inizializza il sistema (carica i modelli)
        sistema = SistemaDiagnostico()
        
        # 2. Raccogli input
        input_utente = raccogli_input_utente()
        
        if input_utente is None:
            return # L'utente ha annullato

        # 3. Esegui flusso ibrido
        report = sistema.esegui_diagnosi_completa(
            pianta_codice=input_utente["pianta"]["codice"],
            sintomi_codice=[s["codice"] for s in input_utente["sintomi"]],
            stagione_codice=input_utente["stagione"]["codice"]
        )
        
        # 4. Stampa il report finale
        stampa_report_diagnosi(report)

    except Exception as e:
        click.echo(click.style(f"\n[ERRORE NON GESTITO] Il programma è terminato in modo anomalo.", fg="red", bold=True))
        click.echo(f"Dettagli: {e}")
        import traceback
        traceback.print_exc()


# --- Punto di Ingresso Principale ---

if __name__ == "__main__":
    cli()