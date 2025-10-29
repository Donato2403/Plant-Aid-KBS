"""
Modulo per il motore di inferenza basato su regole Datalog/ASP del sistema Plant-Aid-KBS

Questo modulo fornisce funzionalita per:
- Definire regole diagnostiche in formato Datalog/ASP
- Eseguire inferenza backward chaining
- Derivare ipotesi diagnostiche da sintomi osservati
- Integrare conoscenza euristica esperta

Versione: 2.0 - Avanzata con logiche ambientali
"""

import clingo
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class IpotesiDiagnostica:
    """
    Rappresenta un'ipotesi diagnostica derivata dal motore Datalog.
    
    Attributes:
        malattia: Nome della malattia ipotizzata
        pianta: Tipo di pianta interessata
        confidenza: Livello di confidenza (0.0-1.0)
        regole_attivate: Lista di regole che hanno portato a questa ipotesi
        sintomi_corrispondenti: Sintomi che hanno attivato le regole
    """
    malattia: str
    pianta: str
    confidenza: float
    regole_attivate: List[str]
    sintomi_corrispondenti: List[str]


class MotoreDatalog:
    """
    Motore di inferenza basato su regole Datalog/ASP usando Clingo.
    
    Implementa ragionamento simbolico per la diagnosi di malattie delle piante
    attraverso regole logiche che formalizzano la conoscenza esperta.
    """
    
    # Mapping centralizzato malattia leggibile <-> ASP
    MAPPING_MALATTIE_ASP_TO_LEGGIBILE = {
        "occhio_pavone": "Occhio di Pavone",
        "rogna_olivo": "Rogna dell'Olivo",
        "lebbra_olivo": "Lebbra dell'Olivo",
        "ticchiolatura_rosa": "Ticchiolatura della Rosa",
        "oidio_rosa": "Oidio della Rosa",
        "peronospora_rosa": "Peronospora della Rosa",
        "peronospora_basilico": "Peronospora del Basilico",
        "fusarium_basilico": "Fusarium del Basilico"
    }
    
    # Mapping inverso (leggibile -> ASP)
    MAPPING_MALATTIE_LEGGIBILE_TO_ASP = {
        v: k for k, v in MAPPING_MALATTIE_ASP_TO_LEGGIBILE.items()
    }
    
    def __init__(self):
        """Inizializza il motore Datalog con la base di conoscenza."""
        self.regole_base = self._carica_regole_diagnostiche()
        self.fatti_correnti = []
        
    def _carica_regole_diagnostiche(self) -> str:
        """
        Carica le regole diagnostiche in formato ASP (Answer Set Programming).
        
        Le regole codificano la conoscenza esperta sulle relazioni tra:
        - Sintomi osservabili
        - Malattie delle piante
        - Condizioni ambientali e fattori di rischio
        
        Returns:
            Stringa contenente tutte le regole ASP
        """
        
        regole = """
% ================================================================
% BASE DI CONOSCENZA DATALOG AVANZATA - PLANT-AID-KBS
% Sistema diagnostico per malattie di Olivo, Rosa e Basilico
% Versione 2.0 - Con logiche ambientali e correlazioni complesse
% ================================================================

% ----------------------------------------------------------------
% DEFINIZIONE TIPI DI PIANTE
% ----------------------------------------------------------------
tipo_pianta(olivo).
tipo_pianta(rosa).
tipo_pianta(basilico).

% ----------------------------------------------------------------
% DEFINIZIONE CATEGORIE DI SINTOMI
% ----------------------------------------------------------------
categoria_sintomo(visivo_foglie).
categoria_sintomo(visivo_fusto).
categoria_sintomo(visivo_frutti).
categoria_sintomo(strutturale).
categoria_sintomo(fisiologico).

% ----------------------------------------------------------------
% RELAZIONI SINTOMO-MALATTIA COMPLETE (sintomo_relato)
% ----------------------------------------------------------------

% Occhio di Pavone
sintomo_relato(occhio_pavone, macchie_circolari_grigie).
sintomo_relato(occhio_pavone, ingiallimento_foglie).
sintomo_relato(occhio_pavone, caduta_foglie).

% Rogna dell'Olivo
sintomo_relato(rogna_olivo, tumori_rami).

% Lebbra dell'Olivo
sintomo_relato(lebbra_olivo, macchie_bruno_nerastre_frutti).
sintomo_relato(lebbra_olivo, ingiallimento_foglie).
sintomo_relato(lebbra_olivo, caduta_foglie).

% Ticchiolatura della Rosa
sintomo_relato(ticchiolatura_rosa, macchie_nere_foglie).
sintomo_relato(ticchiolatura_rosa, ingiallimento_foglie).
sintomo_relato(ticchiolatura_rosa, caduta_foglie).

% Oidio della Rosa
sintomo_relato(oidio_rosa, muffa_biancastra).

% Peronospora della Rosa
sintomo_relato(peronospora_rosa, ingiallimento_foglie).
sintomo_relato(peronospora_rosa, caduta_foglie).

% Peronospora del Basilico
sintomo_relato(peronospora_basilico, ingiallimento_foglie).
sintomo_relato(peronospora_basilico, caduta_foglie).

% Fusarium del Basilico
sintomo_relato(fusarium_basilico, annerimento_gambo).
sintomo_relato(fusarium_basilico, avvizzimento_pianta).

% ----------------------------------------------------------------
% FATTORI AMBIENTALI E CONDIZIONI FAVOREVOLI
% (Basati sulle fonti botaniche reali)
% ----------------------------------------------------------------

% Umidita favorevole
richiede_umidita_alta(occhio_pavone).
richiede_umidita_alta(lebbra_olivo).
richiede_umidita_alta(ticchiolatura_rosa).
richiede_umidita_alta(oidio_rosa).
richiede_umidita_alta(peronospora_rosa).
richiede_umidita_alta(peronospora_basilico).
richiede_umidita_alta(fusarium_basilico).
richiede_umidita_alta(rogna_olivo).

% Periodi dell'anno di attivita (stagioni)
attivo_in(occhio_pavone, primavera).
attivo_in(occhio_pavone, autunno).
attivo_in(occhio_pavone, inverno_mite).

attivo_in(rogna_olivo, tutto_anno).

attivo_in(lebbra_olivo, autunno).

attivo_in(ticchiolatura_rosa, primavera).
attivo_in(ticchiolatura_rosa, fine_estate).

attivo_in(oidio_rosa, primavera).
attivo_in(oidio_rosa, estate).

attivo_in(peronospora_rosa, primavera).

attivo_in(peronospora_basilico, primavera).
attivo_in(peronospora_basilico, estate).

attivo_in(fusarium_basilico, primavera).
attivo_in(fusarium_basilico, estate).

% Condizioni climatiche specifiche
favorita_da_pioggia(occhio_pavone).
favorita_da_pioggia(lebbra_olivo).
favorita_da_pioggia(ticchiolatura_rosa).
favorita_da_pioggia(peronospora_rosa).
favorita_da_pioggia(peronospora_basilico).

favorita_da_ristagno_idrico(fusarium_basilico).
favorita_da_ristagno_idrico(peronospora_basilico).

% ----------------------------------------------------------------
% REGOLE DIAGNOSTICHE BASE PER OLIVO
% ----------------------------------------------------------------

% Regola 1: Occhio di Pavone dell'Olivo (confidenza alta)
malattia_probabile(occhio_pavone, olivo, alta) :-
    sintomo_presente(macchie_circolari_grigie),
    sintomo_presente(ingiallimento_foglie),
    pianta_tipo(olivo).

% Regola 2: Occhio di Pavone (confidenza media se manca un sintomo)
malattia_probabile(occhio_pavone, olivo, media) :-
    sintomo_presente(macchie_circolari_grigie),
    pianta_tipo(olivo),
    not sintomo_presente(ingiallimento_foglie).

% Regola 3: Rogna dell'Olivo
malattia_probabile(rogna_olivo, olivo, alta) :-
    sintomo_presente(tumori_rami),
    pianta_tipo(olivo).

% Regola 4: Lebbra dell'Olivo
malattia_probabile(lebbra_olivo, olivo, alta) :-
    sintomo_presente(macchie_bruno_nerastre_frutti),
    pianta_tipo(olivo).

% Regola 5: Lebbra con sintomi secondari (confidenza molto alta)
malattia_probabile(lebbra_olivo, olivo, molto_alta) :-
    sintomo_presente(macchie_bruno_nerastre_frutti),
    sintomo_presente(ingiallimento_foglie),
    sintomo_presente(caduta_foglie),
    pianta_tipo(olivo).

% ----------------------------------------------------------------
% REGOLE DIAGNOSTICHE BASE PER ROSA
% ----------------------------------------------------------------

% Regola 6: Ticchiolatura della Rosa
malattia_probabile(ticchiolatura_rosa, rosa, alta) :-
    sintomo_presente(macchie_nere_foglie),
    pianta_tipo(rosa).

% Regola 7: Ticchiolatura con defogliazione (confidenza molto alta)
malattia_probabile(ticchiolatura_rosa, rosa, molto_alta) :-
    sintomo_presente(macchie_nere_foglie),
    sintomo_presente(ingiallimento_foglie),
    sintomo_presente(caduta_foglie),
    pianta_tipo(rosa).

% Regola 8: Oidio della Rosa
malattia_probabile(oidio_rosa, rosa, alta) :-
    sintomo_presente(muffa_biancastra),
    pianta_tipo(rosa).

% Regola 9: Peronospora della Rosa (diagnosi differenziale)
malattia_probabile(peronospora_rosa, rosa, alta) :-
    sintomo_presente(ingiallimento_foglie),
    sintomo_presente(caduta_foglie),
    pianta_tipo(rosa),
    not sintomo_presente(macchie_nere_foglie),
    not sintomo_presente(muffa_biancastra).

% ----------------------------------------------------------------
% REGOLE DIAGNOSTICHE BASE PER BASILICO
% ----------------------------------------------------------------

% Regola 10: Peronospora del Basilico
malattia_probabile(peronospora_basilico, basilico, alta) :-
    sintomo_presente(ingiallimento_foglie),
    pianta_tipo(basilico),
    not sintomo_presente(annerimento_gambo).

% Regola 11: Peronospora con caduta foglie
malattia_probabile(peronospora_basilico, basilico, molto_alta) :-
    sintomo_presente(ingiallimento_foglie),
    sintomo_presente(caduta_foglie),
    pianta_tipo(basilico),
    not sintomo_presente(annerimento_gambo).

% Regola 12: Fusarium del Basilico (confidenza molto alta)
malattia_probabile(fusarium_basilico, basilico, molto_alta) :-
    sintomo_presente(annerimento_gambo),
    pianta_tipo(basilico).

% Regola 13: Fusarium con avvizzimento completo (confidenza critica)
malattia_probabile(fusarium_basilico, basilico, critica) :-
    sintomo_presente(annerimento_gambo),
    sintomo_presente(avvizzimento_pianta),
    pianta_tipo(basilico).

% ----------------------------------------------------------------
% REGOLE AMBIENTALI AVANZATE - MODULAZIONE CONFIDENZA
% ----------------------------------------------------------------

% Aumenta confidenza se condizioni ambientali favorevoli
diagnosi_potenziata(Malattia, Pianta, molto_alta) :-
    malattia_probabile(Malattia, Pianta, alta),
    condizione_ambiente(umidita_alta),
    richiede_umidita_alta(Malattia).

diagnosi_potenziata(Malattia, Pianta, critica) :-
    malattia_probabile(Malattia, Pianta, molto_alta),
    condizione_ambiente(umidita_alta),
    condizione_ambiente(temperatura_mite),
    richiede_umidita_alta(Malattia).

% Condizioni meteo favorevoli
rischio_aumentato(Malattia) :-
    favorita_da_pioggia(Malattia),
    condizione_ambiente(piogge_recenti).

rischio_aumentato(Malattia) :-
    favorita_da_ristagno_idrico(Malattia),
    condizione_ambiente(ristagno_idrico).

% Periodo stagionale favorevole
periodo_favorevole_ora(Malattia) :-
    attivo_in(Malattia, Stagione),
    stagione_corrente(Stagione).

% Riduce confidenza se periodo non favorevole
diagnosi_ridotta(Malattia, Pianta, media) :-
    malattia_probabile(Malattia, Pianta, alta),
    not periodo_favorevole_ora(Malattia).

% ----------------------------------------------------------------    
% CORRELAZIONI SINTOMO-MALATTIA COMPLESSE
% ----------------------------------------------------------------

% Sintomi chiave discriminanti per diagnosi differenziale
sintomo_chiave_discriminante(occhio_pavone, macchie_circolari_grigie).
sintomo_chiave_discriminante(ticchiolatura_rosa, macchie_nere_foglie).
sintomo_chiave_discriminante(oidio_rosa, muffa_biancastra).
sintomo_chiave_discriminante(fusarium_basilico, annerimento_gambo).
sintomo_chiave_discriminante(rogna_olivo, tumori_rami).
sintomo_chiave_discriminante(lebbra_olivo, macchie_bruno_nerastre_frutti).

% Predicato ausiliario: c'e' un sintomo chiave mancante?
c_manca_sintomo_chiave(Malattia) :- 
    sintomo_chiave_discriminante(Malattia, S), 
    not sintomo_presente(S).

% Diagnosi forte se tutti i sintomi chiave presenti
diagnosi_forte(Malattia, Pianta) :-
    malattia_probabile(Malattia, Pianta, _),
    not c_manca_sintomo_chiave(Malattia).

% Confidenza critica se 3+ sintomi correlati presenti E diagnosi forte
diagnosi_completa(Malattia, Pianta, critica) :-
    malattia_probabile(Malattia, Pianta, _),
    #count { S : sintomo_presente(S), sintomo_relato(Malattia, S) } >= 3,
    diagnosi_forte(Malattia, Pianta).

% ----------------------------------------------------------------
% REGOLE DI ESCLUSIONE (DIAGNOSI DIFFERENZIALE)
% ----------------------------------------------------------------

% Esclusioni basate su sintomi incompatibili
esclude_malattia(ticchiolatura_rosa) :-
    sintomo_presente(muffa_biancastra).

esclude_malattia(oidio_rosa) :-
    sintomo_presente(macchie_nere_foglie).

esclude_malattia(peronospora_basilico) :-
    sintomo_presente(annerimento_gambo).

esclude_malattia(fusarium_basilico) :-
    sintomo_presente(macchie_nere_foglie).

% ----------------------------------------------------------------
% DIAGNOSI FINALE INTEGRATA
% ----------------------------------------------------------------

% Diagnosi valida: malattia probabile non esclusa
diagnosi_valida(Malattia, Pianta, Confidenza) :-
    malattia_probabile(Malattia, Pianta, Confidenza),
    not esclude_malattia(Malattia).

% Diagnosi finale potenziata da fattori ambientali
diagnosi_finale(Malattia, Pianta, Confidenza_Potenziata) :-
    diagnosi_potenziata(Malattia, Pianta, Confidenza_Potenziata),
    not esclude_malattia(Malattia).

% Diagnosi finale da correlazioni complete
diagnosi_finale(Malattia, Pianta, critica) :-
    diagnosi_completa(Malattia, Pianta, critica),
    not esclude_malattia(Malattia).

% Diagnosi finale standard (se non potenziata)
diagnosi_finale(Malattia, Pianta, Confidenza) :-
    diagnosi_valida(Malattia, Pianta, Confidenza),
    not diagnosi_potenziata(Malattia, Pianta, _),
    not diagnosi_completa(Malattia, Pianta, _).

% ----------------------------------------------------------------
% REGOLE DI GRAVITA
% ----------------------------------------------------------------

gravita_malattia(Malattia, critica) :-
    malattia_probabile(Malattia, _, _),
    sintomo_presente(avvizzimento_pianta).

gravita_malattia(Malattia, critica) :-
    malattia_probabile(Malattia, _, _),
    sintomo_presente(annerimento_gambo).

gravita_malattia(Malattia, alta) :-
    malattia_probabile(Malattia, _, _),
    sintomo_presente(caduta_foglie),
    #count { S : sintomo_presente(S), sintomo_relato(Malattia, S) } >= 2.

% ----------------------------------------------------------------
% REGOLE DI TRATTAMENTO CONSIGLIATO
% ----------------------------------------------------------------

% Trattamenti per malattie fungine dell'olivo
trattamento_consigliato(occhio_pavone, rame).
trattamento_consigliato(lebbra_olivo, rame).

% Trattamenti per malattie batteriche
trattamento_consigliato(rogna_olivo, potatura_parti_infette).
trattamento_consigliato(rogna_olivo, rame).

% Trattamenti per malattie della rosa
trattamento_consigliato(ticchiolatura_rosa, rame).
trattamento_consigliato(oidio_rosa, zolfo).
trattamento_consigliato(peronospora_rosa, rame).

% Trattamenti per malattie del basilico
trattamento_consigliato(peronospora_basilico, bicarbonato_potassio).
trattamento_consigliato(fusarium_basilico, potatura_parti_infette).

% Trattamento finale associato alla diagnosi finale
trattamento_finale(Trattamento, Malattia) :-
    diagnosi_finale(Malattia, _, _),
    trattamento_consigliato(Malattia, Trattamento).

% Trattamento finale anche per diagnosi valida
trattamento_finale(Trattamento, Malattia) :-
    diagnosi_valida(Malattia, _, _),
    trattamento_consigliato(Malattia, Trattamento),
    not diagnosi_finale(Malattia, _, _).

% ----------------------------------------------------------------
% REGOLE DI PREVENZIONE
% ----------------------------------------------------------------

prevenzione_generica(malattie_fungine, evitare_ristagni_acqua).
prevenzione_generica(malattie_fungine, potatura_aerazione).
prevenzione_generica(malattie_fungine, trattamenti_preventivi_rame).

prevenzione_specifica(Malattia, evitare_irrigazione_fogliare) :-
    richiede_umidita_alta(Malattia).

prevenzione_specifica(fusarium_basilico, drenaggio_terreno).
prevenzione_specifica(peronospora_basilico, drenaggio_terreno).

% ----------------------------------------------------------------
% REGOLE AUSILIARIE DI SUPPORTO
% ----------------------------------------------------------------

% Malattia fungina
malattia_fungina(occhio_pavone).
malattia_fungina(lebbra_olivo).
malattia_fungina(ticchiolatura_rosa).
malattia_fungina(oidio_rosa).
malattia_fungina(peronospora_rosa).
malattia_fungina(peronospora_basilico).
malattia_fungina(fusarium_basilico).

% Malattia batterica
malattia_batterica(rogna_olivo).

% Rischio elevato malattie fungine
rischio_alto_funghi :-
    condizione_ambiente(umidita_alta),
    condizione_ambiente(temperatura_mite).

% ================================================================
% FINE BASE DI CONOSCENZA AVANZATA
% ================================================================

#show gravita_malattia/2.
#show diagnosi_finale/3.
#show diagnosi_valida/3.
#show diagnosi_potenziata/3.
#show malattia_probabile/3.
#show trattamento_finale/2.
#show trattamento_consigliato/2.
#show rischio_aumentato/1.
#show diagnosi_completa/3.
#show diagnosi_forte/2.
"""
        
        return regole
    
    def _converti_sintomo_in_predicato(self, sintomo: str) -> str:
        """
        Converte il nome di un sintomo in formato predicato ASP.
        
        Args:
            sintomo: Nome del sintomo
        
        Returns:
            Predicato ASP
        """
        predicato = sintomo.lower()
        return predicato
    
    def _converti_pianta_in_predicato(self, pianta: str) -> str:
        """
        Converte il nome di una pianta in formato predicato ASP.
        
        Args:
            pianta: Nome della pianta
        
        Returns:
            Predicato ASP
        """
        pianta_lower = pianta.lower()
        
        # Mapping nomi ontologia -> predicati Datalog
        mapping = {
            "olivo": "olivo",
            "rosa": "rosa",
            "basilico": "basilico"
        }
        
        # Cerca corrispondenza parziale
        for chiave, valore in mapping.items():
            if chiave in pianta_lower:
                return valore
        
        # Default: prende la prima parola
        return pianta_lower.split("_")[0]
    
    def _converti_confidenza(self, livello_asp: str) -> float:
        """
        Converte i livelli di confidenza ASP in valori numerici.
        
        Args:
            livello_asp: Livello simbolico (bassa, media, alta, molto_alta, critica)
        
        Returns:
            Valore numerico tra 0.0 e 1.0
        """
        mapping_confidenza = {
            "bassa": 0.3,
            "media": 0.5,
            "alta": 0.7,
            "molto_alta": 0.9,
            "critica": 1.0
        }
        
        return mapping_confidenza.get(livello_asp, 0.5)
    
    def _estrai_nome_malattia_leggibile(self, nome_asp: str) -> str:
        """
        Converte il nome della malattia da formato ASP a formato leggibile.
        
        Args:
            nome_asp: Nome in formato ASP
        
        Returns:
            Nome leggibile
        """
        return self.MAPPING_MALATTIE_ASP_TO_LEGGIBILE.get(
            nome_asp, 
            nome_asp.replace("_", " ").title()
        )
    
    def _converti_nome_malattia_in_asp(self, nome_leggibile: str) -> str:
        """
        Converte il nome della malattia da formato leggibile a formato ASP.
        
        Args:
            nome_leggibile: Nome leggibile
        
        Returns:
            Nome ASP
        """
        return self.MAPPING_MALATTIE_LEGGIBILE_TO_ASP.get(
            nome_leggibile,
            nome_leggibile.lower().replace(" ", "_").replace("'", "").replace("dell", "").replace("del", "")
        )
    
    def aggiungi_sintomo_osservato(self, sintomo: str) -> None:
        """
        Aggiunge un sintomo osservato alla base di fatti corrente.
        
        Args:
            sintomo: Nome del sintomo osservato
        """
        predicato = self._converti_sintomo_in_predicato(sintomo)
        fatto = f"sintomo_presente({predicato})."
        
        if fatto not in self.fatti_correnti:
            self.fatti_correnti.append(fatto)
    
    def imposta_tipo_pianta(self, pianta: str) -> None:
        """
        Imposta il tipo di pianta che si sta diagnosticando.
        
        Args:
            pianta: Nome del tipo di pianta
        """
        predicato = self._converti_pianta_in_predicato(pianta)
        fatto = f"pianta_tipo({predicato})."
        
        # Rimuove eventuali dichiarazioni precedenti di tipo pianta
        self.fatti_correnti = [f for f in self.fatti_correnti if not f.startswith("pianta_tipo(")]
        self.fatti_correnti.append(fatto)
    
    def aggiungi_condizione_ambiente(self, condizione: str) -> None:
        """
        Aggiunge una condizione ambientale alla base di fatti.
        
        Args:
            condizione: Nome della condizione (umidita_alta, temperatura_mite, 
                       piogge_recenti, ristagno_idrico)
        """
        fatto = f"condizione_ambiente({condizione})."
        
        if fatto not in self.fatti_correnti:
            self.fatti_correnti.append(fatto)
    
    def imposta_stagione(self, stagione: str) -> None:
        """
        Imposta la stagione corrente per modulare le diagnosi.
        
        Args:
            stagione: Nome della stagione (primavera, estate, autunno, 
                     inverno_mite, tutto_anno, fine_estate)
        """
        # Rimuove eventuali dichiarazioni precedenti di stagione
        self.fatti_correnti = [f for f in self.fatti_correnti 
                              if not f.startswith("stagione_corrente(")]
        
        fatto = f"stagione_corrente({stagione})."
        self.fatti_correnti.append(fatto)
    
    def azzera_fatti(self) -> None:
        """Rimuove tutti i fatti correnti (sintomi e tipo pianta)."""
        self.fatti_correnti = []
    
    def esegui_inferenza(self) -> List[IpotesiDiagnostica]:
        """
        Esegue l'inferenza Datalog per derivare diagnosi dai sintomi osservati.
        
        Utilizza Clingo per calcolare gli answer sets e derivare tutte le
        malattie probabili compatibili con i sintomi forniti.
        
        Returns:
            Lista di ipotesi diagnostiche ordinate per confidenza decrescente
        """
        # Costruisce il programma ASP completo
        programma_completo = self.regole_base + "\n" + "\n".join(self.fatti_correnti)
        
        # Crea il controller Clingo
        controllo = clingo.Control()
        
        # Aggiunge il programma
        controllo.add("base", [], programma_completo)
        
        # Grounding (istanziazione delle regole)
        controllo.ground([("base", [])])
        
        # Risoluzione e raccolta risultati
        ipotesi_trovate = []
        
        def estrai_modello(modello):
            """Callback per estrarre i risultati da ogni answer set."""
            for simbolo in modello.symbols(shown=True):
                # Cerca predicati diagnosi_valida(Malattia, Pianta, Confidenza)
                if simbolo.name == "diagnosi_valida" and len(simbolo.arguments) == 3:
                    malattia_asp = str(simbolo.arguments[0])
                    pianta_asp = str(simbolo.arguments[1])
                    confidenza_asp = str(simbolo.arguments[2])
                    
                    # Converti i valori
                    malattia_leggibile = self._estrai_nome_malattia_leggibile(malattia_asp)
                    pianta_leggibile = pianta_asp.capitalize()
                    confidenza_numerica = self._converti_confidenza(confidenza_asp)
                    
                    # Estrae sintomi che hanno portato a questa diagnosi
                    sintomi_usati = []
                    for fatto in self.fatti_correnti:
                        if fatto.startswith("sintomo_presente("):
                            match = re.search(r'sintomo_presente\((.+?)\)', fatto)
                            if match:
                                sintomi_usati.append(match.group(1))
                    
                    # Crea ipotesi
                    ipotesi = IpotesiDiagnostica(
                        malattia=malattia_leggibile,
                        pianta=pianta_leggibile,
                        confidenza=confidenza_numerica,
                        regole_attivate=[f"Regola Datalog per {malattia_asp}"],
                        sintomi_corrispondenti=sintomi_usati
                    )
                    
                    ipotesi_trovate.append(ipotesi)
        
        # Esegui il solving
        controllo.solve(on_model=estrai_modello)
        
        # Ordina per confidenza decrescente
        ipotesi_trovate.sort(key=lambda x: x.confidenza, reverse=True)
        
        return ipotesi_trovate
    
    def ottieni_diagnosi_finale(self) -> List[IpotesiDiagnostica]:
        """
        Esegue l'inferenza avanzata per ottenere diagnosi finali integrate
        con fattori ambientali e correlazioni complesse.
        
        Returns:
            Lista di ipotesi diagnostiche finali ordinate per confidenza
        """
        # Costruisce il programma ASP completo
        programma_completo = self.regole_base + "\n" + "\n".join(self.fatti_correnti)
        
        # Crea il controller Clingo
        controllo = clingo.Control()
        controllo.add("base", [], programma_completo)
        controllo.ground([("base", [])])
        
        # Risoluzione e raccolta risultati
        ipotesi_trovate = []
        
        def estrai_diagnosi_finali(modello):
            """Callback per estrarre diagnosi finali."""
            for simbolo in modello.symbols(shown=True):
                # Cerca predicati diagnosi_finale(Malattia, Pianta, Confidenza)
                if simbolo.name == "diagnosi_finale" and len(simbolo.arguments) == 3:
                    malattia_asp = str(simbolo.arguments[0])
                    pianta_asp = str(simbolo.arguments[1])
                    confidenza_asp = str(simbolo.arguments[2])
                    
                    # Converti i valori
                    malattia_leggibile = self._estrai_nome_malattia_leggibile(malattia_asp)
                    pianta_leggibile = pianta_asp.capitalize()
                    confidenza_numerica = self._converti_confidenza(confidenza_asp)
                    
                    # Estrae sintomi che hanno portato a questa diagnosi
                    sintomi_usati = []
                    for fatto in self.fatti_correnti:
                        if fatto.startswith("sintomo_presente("):
                            match = re.search(r'sintomo_presente\((.+?)\)', fatto)
                            if match:
                                sintomi_usati.append(match.group(1))
                    
                    # Determina regole attivate
                    regole = [f"Regola Datalog avanzata per {malattia_asp}"]
                    if any("condizione_ambiente" in f for f in self.fatti_correnti):
                        regole.append("Modulazione ambientale applicata")
                    if any("stagione_corrente" in f for f in self.fatti_correnti):
                        regole.append("Fattore stagionale considerato")
                    
                    # Crea ipotesi
                    ipotesi = IpotesiDiagnostica(
                        malattia=malattia_leggibile,
                        pianta=pianta_leggibile,
                        confidenza=confidenza_numerica,
                        regole_attivate=regole,
                        sintomi_corrispondenti=sintomi_usati
                    )
                    
                    ipotesi_trovate.append(ipotesi)
        
        # Esegui il solving
        controllo.solve(on_model=estrai_diagnosi_finali)
        
        # Ordina per confidenza decrescente
        ipotesi_trovate.sort(key=lambda x: x.confidenza, reverse=True)
        
        return ipotesi_trovate
    
    def ottieni_trattamenti(self, malattia: str) -> List[str]:
        """
        Ottiene i trattamenti consigliati per una malattia.
        
        Args:
            malattia: Nome della malattia (formato leggibile)
        
        Returns:
            Lista di trattamenti consigliati
        """
        # Converti nome malattia in formato ASP
        malattia_asp = self._converti_nome_malattia_in_asp(malattia)
        
        # Costruisce query per trattamenti
        query = f"""
{self.regole_base}
malattia_diagnosticata({malattia_asp}).
"""
        
        controllo = clingo.Control()
        controllo.add("base", [], query)
        controllo.ground([("base", [])])
        
        trattamenti = []
        
        def estrai_trattamenti(modello):
            for simbolo in modello.symbols(shown=True):
                if simbolo.name == "trattamento_consigliato" and len(simbolo.arguments) == 2:
                    if str(simbolo.arguments[0]) == malattia_asp:
                        trattamento = str(simbolo.arguments[1])
                        # Converti in formato leggibile
                        tratt_leggibile = trattamento.replace("_", " ").title()
                        trattamenti.append(tratt_leggibile)
        
        controllo.solve(on_model=estrai_trattamenti)
        
        return trattamenti
    
    def ottieni_trattamenti_finali(self, malattia: str) -> List[str]:
        """
        Ottiene i trattamenti finali per una malattia diagnosticata.
        
        Args:
            malattia: Nome della malattia (formato leggibile)
        
        Returns:
            Lista di trattamenti finali consigliati
        """
        # Converti nome malattia in formato ASP
        malattia_asp = self._converti_nome_malattia_in_asp(malattia)
        
        # Costruisce programma con diagnosi
        programma_completo = self.regole_base + "\n"
        programma_completo += "\n".join(self.fatti_correnti) + "\n"
        programma_completo += f"malattia_diagnosticata({malattia_asp}).\n"
        
        controllo = clingo.Control()
        controllo.add("base", [], programma_completo)
        controllo.ground([("base", [])])
        
        trattamenti = set()
        
        def estrai_trattamenti(modello):
            for simbolo in modello.symbols(shown=True):
                if simbolo.name == "trattamento_finale" and len(simbolo.arguments) == 2:
                    trattamento = str(simbolo.arguments[0])
                    # Converti in formato leggibile
                    tratt_leggibile = trattamento.replace("_", " ").title()
                    trattamenti.add(tratt_leggibile)
        
        controllo.solve(on_model=estrai_trattamenti)
        
        return list(trattamenti)
    
    def ottieni_gravita(self, malattia: str) -> str:
        """
        Restituisce il livello di gravità più alto dedotto da Clingo.
        """
        malattia_asp = self._converti_nome_malattia_in_asp(malattia)
        
        programma_completo = self.regole_base + "\n"
        programma_completo += "\n".join(self.fatti_correnti) + "\n"
        
        controllo = clingo.Control()
        controllo.add("base", [], programma_completo)
        controllo.ground([("base", [])])

        livelli_gravita_trovati = set()
        
        def estrai_gravita(modello):
            for simbolo in modello.symbols(shown=True):
                if simbolo.name == "gravita_malattia" and len(simbolo.arguments) == 2:
                    mal = str(simbolo.arguments[0])
                    if mal == malattia_asp:
                        livello = str(simbolo.arguments[1])
                        livelli_gravita_trovati.add(livello)
        
        controllo.solve(on_model=estrai_gravita)

        # DEBUG: stampa tutti i livelli di gravità trovati
        print(f"[DEBUG] Livelli gravita trovati per {malattia_asp}: {livelli_gravita_trovati}")

        livelli_ordine = {"bassa": 1, "media": 2, "alta": 3, "critica": 4}
        gravita_max = "media"
        max_val = livelli_ordine[gravita_max]
        
        for livello in livelli_gravita_trovati:
            val_attuale = livelli_ordine.get(livello, 0)
            if val_attuale > max_val:
                max_val = val_attuale
                gravita_max = livello
        
        return gravita_max
    
    def diagnosi_completa_integrata(self) -> Dict[str, any]:
        """
        Esegue una diagnosi completa integrando tutte le funzionalita avanzate.
        
        Returns:
            Dizionario con diagnosi, trattamenti, gravita e raccomandazioni
        """
        # Ottieni diagnosi finali
        diagnosi = self.ottieni_diagnosi_finale()
        
        # Se non ci sono diagnosi finali, usa diagnosi standard
        if not diagnosi:
            diagnosi = self.esegui_inferenza()
        
        risultato = {
            "diagnosi": [],
            "condizioni_ambientali": [],
            "stagione": None
        }
        
        # Estrai condizioni ambientali dai fatti
        for fatto in self.fatti_correnti:
            if fatto.startswith("condizione_ambiente("):
                match = re.search(r'condizione_ambiente\((.+?)\)', fatto)
                if match:
                    risultato["condizioni_ambientali"].append(match.group(1))
            elif fatto.startswith("stagione_corrente("):
                match = re.search(r'stagione_corrente\((.+?)\)', fatto)
                if match:
                    risultato["stagione"] = match.group(1)
        
        # Per ogni diagnosi, ottieni informazioni complete
        for ipo in diagnosi:
            trattamenti = self.ottieni_trattamenti_finali(ipo.malattia)
            gravita = self.ottieni_gravita(ipo.malattia)
            
            diagnosi_completa = {
                "malattia": ipo.malattia,
                "pianta": ipo.pianta,
                "confidenza": ipo.confidenza,
                "sintomi": ipo.sintomi_corrispondenti,
                "regole_attivate": ipo.regole_attivate,
                "trattamenti": trattamenti,
                "gravita": gravita if gravita else "media"
            }
            
            risultato["diagnosi"].append(diagnosi_completa)
        
        return risultato
    
    def stampa_regole(self) -> None:
        """Stampa tutte le regole diagnostiche caricate."""
        print("\n=== REGOLE DIAGNOSTICHE DATALOG ===")
        print(self.regole_base)
    
    def stampa_fatti_correnti(self) -> None:
        """Stampa i fatti attualmente asseriti (sintomi e tipo pianta)."""
        print("\n=== FATTI CORRENTI ===")
        if not self.fatti_correnti:
            print("Nessun fatto asserito")
        else:
            for fatto in self.fatti_correnti:
                print(f"  {fatto}")


# ================================================================
# FUNZIONI DI UTILITA
# ================================================================

def crea_report_diagnosi(ipotesi: IpotesiDiagnostica) -> str:
    """
    Genera un report testuale dettagliato per un'ipotesi diagnostica.
    
    Args:
        ipotesi: Ipotesi diagnostica da descrivere
    
    Returns:
        Stringa con il report formattato
    """
    report = f"""
--- IPOTESI DIAGNOSTICA ---
Malattia: {ipotesi.malattia}
Pianta: {ipotesi.pianta}
Confidenza: {ipotesi.confidenza:.1%}

Sintomi corrispondenti:
"""
    for sintomo in ipotesi.sintomi_corrispondenti:
        report += f"  - {sintomo.replace('_', ' ').title()}\n"
    
    report += f"\nRegole attivate: {len(ipotesi.regole_attivate)}\n"
    
    return report


# ================================================================
# ESEMPIO DI UTILIZZO
# ================================================================

if __name__ == "__main__":
    print("=== MOTORE DATALOG AVANZATO - PLANT-AID-KBS ===\n")
    
    # Crea il motore
    motore = MotoreDatalog()
    
    # ============================================================
    # TEST 1: Occhio di Pavone con Fattori Ambientali
    # ============================================================
    print("TEST 1: Occhio di Pavone - Diagnosi Potenziata")
    print("=" * 60)
    
    motore.azzera_fatti()
    motore.imposta_tipo_pianta("Olivo")
    motore.aggiungi_sintomo_osservato("Macchie_Circolari_Grigie")
    motore.aggiungi_sintomo_osservato("Ingiallimento_Foglie")
    motore.aggiungi_sintomo_osservato("Caduta_Foglie")
    
    # Aggiungi condizioni ambientali
    motore.aggiungi_condizione_ambiente("umidita_alta")
    motore.aggiungi_condizione_ambiente("temperatura_mite")
    motore.aggiungi_condizione_ambiente("piogge_recenti")
    motore.imposta_stagione("primavera")
    
    print("\nFatti asseriti:")
    motore.stampa_fatti_correnti()
    
    # Diagnosi completa integrata
    risultato = motore.diagnosi_completa_integrata()
    
    print(f"\n--- RISULTATO DIAGNOSI COMPLETA ---")
    print(f"Stagione: {risultato['stagione']}")
    print(f"Condizioni ambientali: {', '.join(risultato['condizioni_ambientali'])}")
    print(f"\nNumero diagnosi: {len(risultato['diagnosi'])}\n")
    
    for i, diag in enumerate(risultato['diagnosi'], 1):
        print(f"{i}. {diag['malattia']}")
        print(f"   Pianta: {diag['pianta']}")
        print(f"   Confidenza: {diag['confidenza']:.1%}")
        print(f"   Gravita: {diag['gravita'].upper()}")
        print(f"   Sintomi rilevati: {', '.join(diag['sintomi'])}")
        print(f"   Trattamenti: {', '.join(diag['trattamenti'])}")
        print()
    
    # ============================================================
    # TEST 2: Fusarium Basilico - Condizioni Critiche
    # ============================================================
    print("\n\nTEST 2: Fusarium Basilico - Condizioni Critiche")
    print("=" * 60)
    
    motore.azzera_fatti()
    motore.imposta_tipo_pianta("Basilico")
    motore.aggiungi_sintomo_osservato("Annerimento_Gambo")
    motore.aggiungi_sintomo_osservato("Avvizzimento_Pianta")
    
    # Condizioni ambientali favorevoli al fungo
    motore.aggiungi_condizione_ambiente("umidita_alta")
    motore.aggiungi_condizione_ambiente("temperatura_mite")
    motore.aggiungi_condizione_ambiente("ristagno_idrico")
    motore.imposta_stagione("estate")
    
    risultato = motore.diagnosi_completa_integrata()
    
    print(f"\n--- RISULTATO DIAGNOSI COMPLETA ---")
    print(f"Stagione: {risultato['stagione']}")
    print(f"Condizioni ambientali: {', '.join(risultato['condizioni_ambientali'])}")
    
    for diag in risultato['diagnosi']:
        print(f"\nMALATTIA: {diag['malattia']}")
        print(f"Confidenza: {diag['confidenza']:.1%}")
        print(f"Gravita: {diag['gravita'].upper()}")
        print(f"\nTrattamenti consigliati:")
        for tratt in diag['trattamenti']:
            print(f"  - {tratt}")
    
    # ============================================================
    # TEST 3: Oidio Rosa - Diagnosi Standard
    # ============================================================
    print("\n\nTEST 3: Oidio Rosa - Diagnosi Standard")
    print("=" * 60)
    
    motore.azzera_fatti()
    motore.imposta_tipo_pianta("Rosa")
    motore.aggiungi_sintomo_osservato("Muffa_Biancastra")
    
    # Nessuna condizione ambientale specificata
    risultato = motore.diagnosi_completa_integrata()
    
    for diag in risultato['diagnosi']:
        print(f"\nMALATTIA: {diag['malattia']}")
        print(f"Confidenza: {diag['confidenza']:.1%}")
        print(f"Gravita: {diag['gravita']}")
        print(f"Sintomi: {', '.join(diag['sintomi'])}")
        print(f"Trattamenti: {', '.join(diag['trattamenti'])}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETATI")
