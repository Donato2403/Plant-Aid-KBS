"""
Modulo per il motore di inferenza basato su regole Datalog/ASP del sistema Plant-Aid-KBS

Questo modulo fornisce funzionalita per:
- Definire regole diagnostiche in formato Datalog/ASP
- Eseguire inferenza backward chaining
- Derivare ipotesi diagnostiche da sintomi osservati
- Integrare conoscenza euristica esperta
"""

import clingo
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class IpotesiDiagnostica:
    """
    Rappresenta unipotesi diagnostica derivata dal motore Datalog.

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
% BASE DI CONOSCENZA DATALOG - PLANT-AID-KBS
% Sistema diagnostico per malattie di Olivo, Rosa e Basilico
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
% REGOLE DIAGNOSTICHE PER OLIVO
% ----------------------------------------------------------------

% Regola 1: Occhio di Pavone dell'Olivo
% Sintomi caratteristici: macchie circolari grigie, ingiallimento, caduta foglie
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
% Sintomo caratteristico: tumori su rami
malattia_probabile(rogna_olivo, olivo, alta) :-
    sintomo_presente(tumori_rami),
    pianta_tipo(olivo).

% Regola 4: Lebbra dell'Olivo
% Sintomi: macchie bruno-nerastre sui frutti
malattia_probabile(lebbra_olivo, olivo, alta) :-
    sintomo_presente(macchie_bruno_nerastre_frutti),
    pianta_tipo(olivo).

% Regola 5: Lebbra con sintomi secondari
malattia_probabile(lebbra_olivo, olivo, molto_alta) :-
    sintomo_presente(macchie_bruno_nerastre_frutti),
    sintomo_presente(ingiallimento_foglie),
    sintomo_presente(caduta_foglie),
    pianta_tipo(olivo).

% ----------------------------------------------------------------
% REGOLE DIAGNOSTICHE PER ROSA
% ----------------------------------------------------------------

% Regola 6: Ticchiolatura della Rosa (Macchia nera)
malattia_probabile(ticchiolatura_rosa, rosa, alta) :-
    sintomo_presente(macchie_nere_foglie),
    pianta_tipo(rosa).

% Regola 7: Ticchiolatura con defogliazione
malattia_probabile(ticchiolatura_rosa, rosa, molto_alta) :-
    sintomo_presente(macchie_nere_foglie),
    sintomo_presente(ingiallimento_foglie),
    sintomo_presente(caduta_foglie),
    pianta_tipo(rosa).

% Regola 8: Oidio della Rosa (Mal bianco)
malattia_probabile(oidio_rosa, rosa, alta) :-
    sintomo_presente(muffa_biancastra),
    pianta_tipo(rosa).

% Regola 9: Peronospora della Rosa
malattia_probabile(peronospora_rosa, rosa, alta) :-
    sintomo_presente(ingiallimento_foglie),
    sintomo_presente(caduta_foglie),
    pianta_tipo(rosa),
    not sintomo_presente(macchie_nere_foglie),
    not sintomo_presente(muffa_biancastra).

% ----------------------------------------------------------------
% REGOLE DIAGNOSTICHE PER BASILICO
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

% Regola 12: Fusarium del Basilico (Gamba nera)
malattia_probabile(fusarium_basilico, basilico, molto_alta) :-
    sintomo_presente(annerimento_gambo),
    pianta_tipo(basilico).

% Regola 13: Fusarium con avvizzimento completo
malattia_probabile(fusarium_basilico, basilico, critica) :-
    sintomo_presente(annerimento_gambo),
    sintomo_presente(avvizzimento_pianta),
    pianta_tipo(basilico).

% ----------------------------------------------------------------
% REGOLE DI FATTORI AGGRAVANTI
% ----------------------------------------------------------------

% Fattori ambientali che aumentano il rischio di malattie fungine
rischio_aumentato(malattie_fungine) :-
    condizione_ambiente(umidita_alta),
    condizione_ambiente(temperatura_mite).

% Periodo dell'anno favorevole a malattie
periodo_favorevole(primavera, malattie_fungine).
periodo_favorevole(autunno, malattie_fungine).

% ----------------------------------------------------------------
% REGOLE DI ESCLUSIONE (DIAGNOSI DIFFERENZIALE)
% ----------------------------------------------------------------

% Se presente muffa biancastra, escludere ticchiolatura
esclude_malattia(ticchiolatura_rosa) :-
    sintomo_presente(muffa_biancastra).

% Se presente annerimento gambo, escludere peronospora
esclude_malattia(peronospora_basilico) :-
    sintomo_presente(annerimento_gambo).

% ----------------------------------------------------------------
% REGOLE DI GRAVITA
% ----------------------------------------------------------------

% Malattia grave se colpisce strutture essenziali
gravita_alta(Malattia) :-
    malattia_probabile(Malattia, _, _),
    sintomo_presente(avvizzimento_pianta).

gravita_alta(Malattia) :-
    malattia_probabile(Malattia, _, _),
    sintomo_presente(annerimento_gambo).

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

% ----------------------------------------------------------------
% REGOLE DI PREVENZIONE
% ----------------------------------------------------------------

% Misure preventive per malattie fungine
prevenzione(malattie_fungine, evitare_ristagni_acqua).
prevenzione(malattie_fungine, potatura_aerazione).
prevenzione(malattie_fungine, trattamenti_preventivi_rame).

% ----------------------------------------------------------------
% REGOLE AUSILIARIE DI RAGIONAMENTO
% ----------------------------------------------------------------

% Una diagnosi e valida se la malattia e probabile e non e esclusa
diagnosi_valida(Malattia, Pianta, Confidenza) :-
    malattia_probabile(Malattia, Pianta, Confidenza),
    not esclude_malattia(Malattia).

% Conteggio sintomi corrispondenti per una malattia
% (utilizzato per calcolare confidenza)
sintomi_corrispondenti(Malattia, N) :-
    malattia_probabile(Malattia, _, _),
    N = #count{S : sintomo_presente(S), sintomo_relato(Malattia, S)}.

% ================================================================
% FINE BASE DI CONOSCENZA
% ================================================================
"""

        return regole

    def _converti_sintomo_in_predicato(self, sintomo: str) -> str:
        """
        Converte il nome di un sintomo in formato predicato ASP.

        Args:
            sintomo: Nome del sintomo (es. "Macchie_Circolari_Grigie")

        Returns:
            Predicato ASP (es. "macchie_circolari_grigie")
        """
        # Rimuove underscore e converte in minuscolo
        predicato = sintomo.lower()
        return predicato

    def _converti_pianta_in_predicato(self, pianta: str) -> str:
        """
        Converte il nome di una pianta in formato predicato ASP.

        Args:
            pianta: Nome della pianta (es. "Olivo_Europeo", "Rosa_Ornamentale")

        Returns:
            Predicato ASP (es. "olivo", "rosa")
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
            nome_asp: Nome in formato ASP (es. "occhio_pavone")

        Returns:
            Nome leggibile (es. "Occhio di Pavone")
        """
        mapping_nomi = {
            "occhio_pavone": "Occhio di Pavone",
            "rogna_olivo": "Rogna dell'Olivo",
            "lebbra_olivo": "Lebbra dell'Olivo",
            "ticchiolatura_rosa": "Ticchiolatura della Rosa",
            "oidio_rosa": "Oidio della Rosa",
            "peronospora_rosa": "Peronospora della Rosa",
            "peronospora_basilico": "Peronospora del Basilico",
            "fusarium_basilico": "Fusarium del Basilico"
        }

        return mapping_nomi.get(nome_asp, nome_asp.replace("_", " ").title())

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

    def ottieni_trattamenti(self, malattia: str) -> List[str]:
        """
        Ottiene i trattamenti consigliati per una malattia.

        Args:
            malattia: Nome della malattia (formato leggibile)

        Returns:
            Lista di trattamenti consigliati
        """
        # Converti nome malattia in formato ASP
        malattia_asp = malattia.lower().replace(" ", "_").replace("'", "")

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
    Genera un report testuale dettagliato per unipotesi diagnostica.

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
    print("=== TEST MOTORE DATALOG - PLANT-AID-KBS ===\n")

    # Crea il motore
    motore = MotoreDatalog()

    # Test 1: Diagnosi Occhio di Pavone dell'Olivo
    print("TEST 1: Occhio di Pavone dell'Olivo")
    print("-" * 50)

    motore.azzera_fatti()
    motore.imposta_tipo_pianta("Olivo")
    motore.aggiungi_sintomo_osservato("Macchie_Circolari_Grigie")
    motore.aggiungi_sintomo_osservato("Ingiallimento_Foglie")

    motore.stampa_fatti_correnti()

    ipotesi = motore.esegui_inferenza()

    print(f"\nTrovate {len(ipotesi)} ipotesi diagnostiche:\n")
    for i, ipo in enumerate(ipotesi, 1):
        print(f"{i}. {ipo.malattia} - Confidenza: {ipo.confidenza:.1%}")

    # Test 2: Diagnosi Fusarium del Basilico
    print("\n\nTEST 2: Fusarium del Basilico (Gamba Nera)")
    print("-" * 50)

    motore.azzera_fatti()
    motore.imposta_tipo_pianta("Basilico")
    motore.aggiungi_sintomo_osservato("Annerimento_Gambo")
    motore.aggiungi_sintomo_osservato("Avvizzimento_Pianta")

    motore.stampa_fatti_correnti()

    ipotesi = motore.esegui_inferenza()

    print(f"\nTrovate {len(ipotesi)} ipotesi diagnostiche:\n")
    for i, ipo in enumerate(ipotesi, 1):
        print(crea_report_diagnosi(ipo))

    # Test 3: Diagnosi Oidio della Rosa
    print("\n\nTEST 3: Oidio della Rosa")
    print("-" * 50)

    motore.azzera_fatti()
    motore.imposta_tipo_pianta("Rosa")
    motore.aggiungi_sintomo_osservato("Muffa_Biancastra")

    motore.stampa_fatti_correnti()

    ipotesi = motore.esegui_inferenza()

    print(f"\nTrovate {len(ipotesi)} ipotesi diagnostiche:\n")
    for i, ipo in enumerate(ipotesi, 1):
        print(f"{i}. {ipo.malattia} - Confidenza: {ipo.confidenza:.1%}")
        trattamenti = motore.ottieni_trattamenti(ipo.malattia)
        if trattamenti:
            print(f"   Trattamenti: {', '.join(trattamenti)}")
