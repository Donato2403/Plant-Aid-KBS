"""
Modulo per la gestione dell'ontologia OWL del sistema Plant-Aid-KBS

Questo modulo fornisce funzionalità per:
- Creare e caricare ontologie OWL
- Definire classi, proprietà e individui
- Eseguire query sull'ontologia
- Eseguire ragionamento automatico con reasoner
"""

from owlready2 import *
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class GestoreOntologia:
    """
    Classe principale per la gestione dell'ontologia Plant-Aid-KBS.

    Gestisce la creazione, il caricamento e l'interrogazione dell'ontologia
    che rappresenta la conoscenza botanica del dominio (piante, malattie, sintomi).
    """

    def __init__(self, percorso_file: str = "data/plant_care.owl"):
        """
        Inizializza il gestore dell'ontologia.

        Args:
            percorso_file: Percorso del file OWL dell'ontologia
        """
        self.percorso_file = percorso_file
        self.ontologia = None
        self.namespace = None

        # Verifica se il file esiste
        if os.path.exists(percorso_file):
            self.carica_ontologia()
        else:
            self.crea_ontologia_base()

    def crea_ontologia_base(self) -> None:
        """
        Crea l'ontologia di base con classi, proprietà e individui
        per le tre piante di riferimento: Olivo, Rosa, Basilico.
        """
        print("Creazione nuova ontologia Plant-Aid-KBS...")

        # Crea ontologia con IRI
        self.ontologia = get_ontology("http://www.plantaid.org/ontologia/plant_care.owl")

        with self.ontologia:
            # ==========================================
            # DEFINIZIONE CLASSI PRINCIPALI
            # ==========================================

            class Pianta(Thing):
                """Classe base per tutte le piante"""
                pass

            class Malattia(Thing):
                """Classe base per tutte le malattie delle piante"""
                pass

            class Sintomo(Thing):
                """Classe base per tutti i sintomi osservabili"""
                pass

            class Trattamento(Thing):
                """Classe base per i trattamenti e le cure"""
                pass

            class AgentePatogeno(Thing):
                """Classe per gli agenti patogeni (funghi, batteri, virus)"""
                pass

            # ==========================================
            # SOTTOCLASSI DI PIANTA
            # ==========================================

            class Olivo(Pianta):
                """Olea europaea - Pianta di olivo"""
                pass

            class Rosa(Pianta):
                """Rosa spp. - Rosa ornamentale"""
                pass

            class Basilico(Pianta):
                """Ocimum basilicum - Basilico"""
                pass

            # ==========================================
            # SOTTOCLASSI DI AGENTE PATOGENO
            # ==========================================

            class Fungo(AgentePatogeno):
                """Agente patogeno di tipo fungino"""
                pass

            class Batterio(AgentePatogeno):
                """Agente patogeno di tipo batterico"""
                pass

            # ==========================================
            # PROPRIETA' OGGETTO
            # ==========================================

            class colpisce(ObjectProperty):
                """Una malattia colpisce una pianta"""
                domain = [Malattia]
                range = [Pianta]

            class presenta_sintomo(ObjectProperty):
                """Una malattia presenta un determinato sintomo"""
                domain = [Malattia]
                range = [Sintomo]

            class richiede_trattamento(ObjectProperty):
                """Una malattia richiede un trattamento specifico"""
                domain = [Malattia]
                range = [Trattamento]

            class causata_da(ObjectProperty):
                """Una malattia è causata da un agente patogeno"""
                domain = [Malattia]
                range = [AgentePatogeno]

            class suscettibile_a(ObjectProperty):
                """Una pianta è suscettibile a una malattia"""
                domain = [Pianta]
                range = [Malattia]
                inverse_property = colpisce

            # ==========================================
            # PROPRIETA' DATI
            # ==========================================

            class ha_nome_scientifico(DataProperty):
                """Nome scientifico dell'entità"""
                domain = [Pianta | Malattia | AgentePatogeno]
                range = [str]

            class ha_descrizione(DataProperty):
                """Descrizione testuale dell'entità"""
                domain = [Thing]
                range = [str]

            class ha_gravita(DataProperty):
                """Livello di gravità della malattia (1-5)"""
                domain = [Malattia]
                range = [int]

            class ha_periodo_attivo(DataProperty):
                """Periodo dell'anno in cui la malattia è attiva"""
                domain = [Malattia]
                range = [str]

            class ha_dosaggio(DataProperty):
                """Dosaggio del trattamento"""
                domain = [Trattamento]
                range = [str]

            # ==========================================
            # INDIVIDUI: MALATTIE DELL'OLIVO
            # ==========================================

            # Occhio di Pavone
            occhio_pavone = Malattia("Occhio_di_Pavone")
            occhio_pavone.ha_nome_scientifico = ["Spilocaea oleagina"]
            occhio_pavone.ha_descrizione = ["Malattia fungina con macchie circolari grigie e alone giallo sulle foglie"]
            occhio_pavone.ha_gravita = [4]
            occhio_pavone.ha_periodo_attivo = ["Primavera-Autunno"]

            # Rogna dell'Olivo
            rogna_olivo = Malattia("Rogna_Olivo")
            rogna_olivo.ha_nome_scientifico = ["Pseudomonas savastanoi"]
            rogna_olivo.ha_descrizione = ["Malattia batterica con formazione di tumori su rami e tronco"]
            rogna_olivo.ha_gravita = [5]
            rogna_olivo.ha_periodo_attivo = ["Tutto_anno"]

            # Lebbra dell'Olivo
            lebbra_olivo = Malattia("Lebbra_Olivo")
            lebbra_olivo.ha_nome_scientifico = ["Gloeosporium olivarum"]
            lebbra_olivo.ha_descrizione = ["Malattia fungina che colpisce i frutti causando macchie bruno-nerastre"]
            lebbra_olivo.ha_gravita = [4]
            lebbra_olivo.ha_periodo_attivo = ["Autunno"]

            # ==========================================
            # INDIVIDUI: MALATTIE DELLA ROSA
            # ==========================================

            # Ticchiolatura della Rosa
            ticchiolatura_rosa = Malattia("Ticchiolatura_Rosa")
            ticchiolatura_rosa.ha_nome_scientifico = ["Diplocarpon rosae"]
            ticchiolatura_rosa.ha_descrizione = ["Malattia fungina con macchie nere circolari sulle foglie"]
            ticchiolatura_rosa.ha_gravita = [4]
            ticchiolatura_rosa.ha_periodo_attivo = ["Primavera-Autunno"]

            # Oidio della Rosa
            oidio_rosa = Malattia("Oidio_Rosa")
            oidio_rosa.ha_nome_scientifico = ["Sphaeroteca pannosa"]
            oidio_rosa.ha_descrizione = ["Mal bianco, muffa biancastra su foglie e germogli"]
            oidio_rosa.ha_gravita = [3]
            oidio_rosa.ha_periodo_attivo = ["Primavera-Estate"]

            # Peronospora della Rosa
            peronospora_rosa = Malattia("Peronospora_Rosa")
            peronospora_rosa.ha_nome_scientifico = ["Peronospora sparsa"]
            peronospora_rosa.ha_descrizione = ["Malattia fungina con macchie clorotiche e feltro miceliare grigio"]
            peronospora_rosa.ha_gravita = [4]
            peronospora_rosa.ha_periodo_attivo = ["Primavera"]

            # ==========================================
            # INDIVIDUI: MALATTIE DEL BASILICO
            # ==========================================

            # Peronospora del Basilico
            peronospora_basilico = Malattia("Peronospora_Basilico")
            peronospora_basilico.ha_nome_scientifico = ["Peronospora belbahrii"]
            peronospora_basilico.ha_descrizione = ["Malattia fungina con macchie giallastre e muffa grigio-violacea"]
            peronospora_basilico.ha_gravita = [4]
            peronospora_basilico.ha_periodo_attivo = ["Primavera-Estate"]

            # Fusarium del Basilico (Gamba Nera)
            fusarium_basilico = Malattia("Fusarium_Basilico")
            fusarium_basilico.ha_nome_scientifico = ["Fusarium oxysporum"]
            fusarium_basilico.ha_descrizione = ["Annerimento del gambo con avvizzimento della pianta"]
            fusarium_basilico.ha_gravita = [5]
            fusarium_basilico.ha_periodo_attivo = ["Primavera-Estate"]

            # ==========================================
            # INDIVIDUI: SINTOMI
            # ==========================================

            # Sintomi visivi sulle foglie
            macchie_circolari_grigie = Sintomo("Macchie_Circolari_Grigie")
            macchie_circolari_grigie.ha_descrizione = ["Macchie rotonde grigie con alone giallo"]

            macchie_nere_foglie = Sintomo("Macchie_Nere_Foglie")
            macchie_nere_foglie.ha_descrizione = ["Macchie nere circolari su foglie"]

            ingiallimento_foglie = Sintomo("Ingiallimento_Foglie")
            ingiallimento_foglie.ha_descrizione = ["Foglie che diventano gialle"]

            caduta_foglie = Sintomo("Caduta_Foglie")
            caduta_foglie.ha_descrizione = ["Defogliazione prematura"]

            muffa_biancastra = Sintomo("Muffa_Biancastra")
            muffa_biancastra.ha_descrizione = ["Patina bianca polverosa su foglie e germogli"]

            # Sintomi su stelo/rami
            tumori_rami = Sintomo("Tumori_Rami")
            tumori_rami.ha_descrizione = ["Escrescenze tumorali su rami e tronco"]

            annerimento_gambo = Sintomo("Annerimento_Gambo")
            annerimento_gambo.ha_descrizione = ["Gambo che diventa nero dalla base"]

            # Sintomi su frutti
            macchie_bruno_nerastre_frutti = Sintomo("Macchie_Bruno_Nerastre_Frutti")
            macchie_bruno_nerastre_frutti.ha_descrizione = ["Tacche bruno-nerastre su frutti"]

            # Sintomi generali
            avvizzimento_pianta = Sintomo("Avvizzimento_Pianta")
            avvizzimento_pianta.ha_descrizione = ["Pianta che appassisce completamente"]

            # ==========================================
            # INDIVIDUI: TRATTAMENTI
            # ==========================================

            trattamento_rame = Trattamento("Trattamento_Rame")
            trattamento_rame.ha_descrizione = ["Prodotti rameici (idrossido di rame, poltiglia bordolese)"]
            trattamento_rame.ha_dosaggio = ["200g per 100 litri di acqua"]

            trattamento_zolfo = Trattamento("Trattamento_Zolfo")
            trattamento_zolfo.ha_descrizione = ["Zolfo in polvere bagnabile"]
            trattamento_zolfo.ha_dosaggio = ["2g per litro di acqua"]

            trattamento_bicarbonato = Trattamento("Trattamento_Bicarbonato_Potassio")
            trattamento_bicarbonato.ha_descrizione = ["Bicarbonato di potassio in soluzione acquosa"]
            trattamento_bicarbonato.ha_dosaggio = ["5g per litro di acqua"]

            potatura_parti_infette = Trattamento("Potatura_Parti_Infette")
            potatura_parti_infette.ha_descrizione = ["Asportazione meccanica delle parti malate"]

            # ==========================================
            # INDIVIDUI: AGENTI PATOGENI
            # ==========================================

            fungo_spilocaea = Fungo("Spilocaea_oleagina")
            batterio_pseudomonas = Batterio("Pseudomonas_savastanoi")
            fungo_gloeosporium = Fungo("Gloeosporium_olivarum")
            fungo_diplocarpon = Fungo("Diplocarpon_rosae")
            fungo_sphaerotheca = Fungo("Sphaeroteca_pannosa")
            fungo_peronospora_rosa = Fungo("Peronospora_sparsa")
            fungo_peronospora_bas = Fungo("Peronospora_belbahrii")
            fungo_fusarium = Fungo("Fusarium_oxysporum")

            # ==========================================
            # INDIVIDUI: PIANTE
            # ==========================================

            olivo_1 = Olivo("Olivo_Europeo")
            olivo_1.ha_nome_scientifico = ["Olea europaea"]
            olivo_1.ha_descrizione = ["Albero sempreverde tipico del Mediterraneo"]

            rosa_1 = Rosa("Rosa_Ornamentale")
            rosa_1.ha_nome_scientifico = ["Rosa spp."]
            rosa_1.ha_descrizione = ["Arbusto ornamentale con fiori profumati"]

            basilico_1 = Basilico("Basilico_Comune")
            basilico_1.ha_nome_scientifico = ["Ocimum basilicum"]
            basilico_1.ha_descrizione = ["Pianta aromatica annuale"]

            # ==========================================
            # RELAZIONI: Malattie -> Piante
            # ==========================================

            occhio_pavone.colpisce = [olivo_1]
            rogna_olivo.colpisce = [olivo_1]
            lebbra_olivo.colpisce = [olivo_1]

            ticchiolatura_rosa.colpisce = [rosa_1]
            oidio_rosa.colpisce = [rosa_1]
            peronospora_rosa.colpisce = [rosa_1]

            peronospora_basilico.colpisce = [basilico_1]
            fusarium_basilico.colpisce = [basilico_1]

            # ==========================================
            # RELAZIONI: Malattie -> Sintomi
            # ==========================================

            occhio_pavone.presenta_sintomo = [macchie_circolari_grigie, ingiallimento_foglie, caduta_foglie]
            rogna_olivo.presenta_sintomo = [tumori_rami]
            lebbra_olivo.presenta_sintomo = [macchie_bruno_nerastre_frutti, ingiallimento_foglie, caduta_foglie]

            ticchiolatura_rosa.presenta_sintomo = [macchie_nere_foglie, ingiallimento_foglie, caduta_foglie]
            oidio_rosa.presenta_sintomo = [muffa_biancastra]
            peronospora_rosa.presenta_sintomo = [ingiallimento_foglie, caduta_foglie]

            peronospora_basilico.presenta_sintomo = [ingiallimento_foglie, caduta_foglie]
            fusarium_basilico.presenta_sintomo = [annerimento_gambo, avvizzimento_pianta]

            # ==========================================
            # RELAZIONI: Malattie -> Trattamenti
            # ==========================================

            occhio_pavone.richiede_trattamento = [trattamento_rame]
            rogna_olivo.richiede_trattamento = [potatura_parti_infette, trattamento_rame]
            lebbra_olivo.richiede_trattamento = [trattamento_rame]

            ticchiolatura_rosa.richiede_trattamento = [trattamento_rame]
            oidio_rosa.richiede_trattamento = [trattamento_zolfo]
            peronospora_rosa.richiede_trattamento = [trattamento_rame]

            peronospora_basilico.richiede_trattamento = [trattamento_bicarbonato]
            fusarium_basilico.richiede_trattamento = [potatura_parti_infette]

            # ==========================================
            # RELAZIONI: Malattie -> Agenti Patogeni
            # ==========================================

            occhio_pavone.causata_da = [fungo_spilocaea]
            rogna_olivo.causata_da = [batterio_pseudomonas]
            lebbra_olivo.causata_da = [fungo_gloeosporium]
            ticchiolatura_rosa.causata_da = [fungo_diplocarpon]
            oidio_rosa.causata_da = [fungo_sphaerotheca]
            peronospora_rosa.causata_da = [fungo_peronospora_rosa]
            peronospora_basilico.causata_da = [fungo_peronospora_bas]
            fusarium_basilico.causata_da = [fungo_fusarium]

        # Salva l'ontologia
        self.salva_ontologia()
        print(f"Ontologia creata e salvata in: {self.percorso_file}")

    def carica_ontologia(self) -> None:
        """Carica un'ontologia esistente da file."""
        print(f"Caricamento ontologia da: {self.percorso_file}")
        self.ontologia = get_ontology(self.percorso_file).load()
        print("Ontologia caricata con successo")

    def salva_ontologia(self) -> None:
        """Salva l'ontologia su file."""
        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(self.percorso_file), exist_ok=True)
        self.ontologia.save(file=self.percorso_file, format="rdfxml")

    def esegui_reasoner(self, reasoner_type: str = "HermiT") -> None:
        """
        Esegue il reasoner per inferire nuova conoscenza.

        Args:
            reasoner_type: Tipo di reasoner ("HermiT" o "Pellet")
        """
        print(f"Esecuzione reasoner {reasoner_type}...")
        try:
            with self.ontologia:
                if reasoner_type == "HermiT":
                    sync_reasoner_hermit()
                else:
                    sync_reasoner_pellet()
            print("Reasoner eseguito con successo")
        except Exception as e:
            print(f"Errore nell'esecuzione del reasoner: {e}")

    def ottieni_malattie_per_pianta(self, nome_pianta: str) -> List[str]:
        """
        Restituisce tutte le malattie che colpiscono una determinata pianta.

        Args:
            nome_pianta: Nome della pianta (Olivo, Rosa, Basilico)

        Returns:
            Lista di nomi delle malattie
        """
        malattie = []

        # Cerca la classe della pianta
        classe_pianta = self.ontologia.search_one(iri=f"*{nome_pianta}")

        if classe_pianta:
            # Cerca tutte le malattie che colpiscono questa pianta
            for malattia in self.ontologia.Malattia.instances():
                if classe_pianta in malattia.colpisce:
                    malattie.append(malattia.name)

        return malattie

    def ottieni_sintomi_malattia(self, nome_malattia: str) -> List[Dict[str, str]]:
        """
        Restituisce tutti i sintomi associati a una malattia.

        Args:
            nome_malattia: Nome della malattia

        Returns:
            Lista di dizionari con informazioni sui sintomi
        """
        sintomi = []

        # Cerca la malattia
        malattia = self.ontologia.search_one(iri=f"*{nome_malattia}")

        if malattia and hasattr(malattia, 'presenta_sintomo'):
            for sintomo in malattia.presenta_sintomo:
                sintomi.append({
                    "nome": sintomo.name,
                    "descrizione": sintomo.ha_descrizione[0] if sintomo.ha_descrizione else ""
                })

        return sintomi

    def ottieni_trattamenti_malattia(self, nome_malattia: str) -> List[Dict[str, str]]:
        """
        Restituisce tutti i trattamenti per una malattia.

        Args:
            nome_malattia: Nome della malattia

        Returns:
            Lista di dizionari con informazioni sui trattamenti
        """
        trattamenti = []

        # Cerca la malattia
        malattia = self.ontologia.search_one(iri=f"*{nome_malattia}")

        if malattia and hasattr(malattia, 'richiede_trattamento'):
            for trattamento in malattia.richiede_trattamento:
                trattamenti.append({
                    "nome": trattamento.name,
                    "descrizione": trattamento.ha_descrizione[0] if trattamento.ha_descrizione else "",
                    "dosaggio": trattamento.ha_dosaggio[0] if trattamento.ha_dosaggio else ""
                })

        return trattamenti

    def diagnosi_da_sintomi(self, lista_sintomi: List[str]) -> List[Tuple[str, float, str]]:
        """
        Effettua una diagnosi basata sui sintomi osservati.

        Args:
            lista_sintomi: Lista di nomi di sintomi osservati

        Returns:
            Lista di tuple (nome_malattia, confidenza, pianta_colpita)
        """
        risultati = []

        # Per ogni malattia nell'ontologia
        for malattia in self.ontologia.Malattia.instances():
            if hasattr(malattia, 'presenta_sintomo'):
                # Conta quanti sintomi coincidono
                sintomi_malattia = [s.name for s in malattia.presenta_sintomo]
                sintomi_trovati = set(lista_sintomi).intersection(set(sintomi_malattia))

                if sintomi_trovati:
                    # Calcola confidenza come rapporto sintomi trovati / sintomi totali
                    confidenza = len(sintomi_trovati) / len(sintomi_malattia)

                    # Ottieni pianta colpita
                    pianta = malattia.colpisce[0].name if malattia.colpisce else "Sconosciuta"

                    risultati.append((malattia.name, confidenza, pianta))

        # Ordina per confidenza decrescente
        risultati.sort(key=lambda x: x[1], reverse=True)

        return risultati

    def ottieni_info_malattia(self, nome_malattia: str) -> Dict[str, any]:
        """
        Restituisce tutte le informazioni disponibili su una malattia.

        Args:
            nome_malattia: Nome della malattia

        Returns:
            Dizionario con tutte le informazioni
        """
        malattia = self.ontologia.search_one(iri=f"*{nome_malattia}")

        if not malattia:
            return {}

        info = {
            "nome": malattia.name,
            "nome_scientifico": malattia.ha_nome_scientifico[0] if malattia.ha_nome_scientifico else "",
            "descrizione": malattia.ha_descrizione[0] if malattia.ha_descrizione else "",
            "gravita": malattia.ha_gravita[0] if malattia.ha_gravita else 0,
            "periodo_attivo": malattia.ha_periodo_attivo[0] if malattia.ha_periodo_attivo else "",
            "pianta_colpita": malattia.colpisce[0].name if malattia.colpisce else "",
            "sintomi": self.ottieni_sintomi_malattia(nome_malattia),
            "trattamenti": self.ottieni_trattamenti_malattia(nome_malattia)
        }

        return info

    def stampa_statistiche(self) -> None:
        """Stampa statistiche sull'ontologia."""
        print("\n=== STATISTICHE ONTOLOGIA ===")
        print(f"Numero di classi: {len(list(self.ontologia.classes()))}")
        print(f"Numero di individui: {len(list(self.ontologia.individuals()))}")
        print(f"Numero di proprietà oggetto: {len(list(self.ontologia.object_properties()))}")
        print(f"Numero di proprietà dati: {len(list(self.ontologia.data_properties()))}")
        print(f"\nPiante: {len(list(self.ontologia.Pianta.instances()))}")
        print(f"Malattie: {len(list(self.ontologia.Malattia.instances()))}")
        print(f"Sintomi: {len(list(self.ontologia.Sintomo.instances()))}")
        print(f"Trattamenti: {len(list(self.ontologia.Trattamento.instances()))}")


# ==========================================
# ESEMPIO DI UTILIZZO
# ==========================================

if __name__ == "__main__":
    # Crea il gestore dell'ontologia
    gestore = GestoreOntologia()

    # Stampa statistiche
    gestore.stampa_statistiche()

    # Esempio 1: Ottenere malattie dell'olivo
    print("\n=== MALATTIE DELL'OLIVO ===")
    malattie_olivo = gestore.ottieni_malattie_per_pianta("Olivo")
    for malattia in malattie_olivo:
        print(f"- {malattia}")

    # Esempio 2: Ottenere info su una malattia specifica
    print("\n=== INFO OCCHIO DI PAVONE ===")
    info = gestore.ottieni_info_malattia("Occhio_di_Pavone")
    print(f"Nome: {info['nome']}")
    print(f"Nome scientifico: {info['nome_scientifico']}")
    print(f"Descrizione: {info['descrizione']}")
    print(f"Gravità: {info['gravita']}/5")
    print(f"Periodo attivo: {info['periodo_attivo']}")
    print(f"\nSintomi:")
    for sintomo in info['sintomi']:
        print(f"  - {sintomo['nome']}: {sintomo['descrizione']}")
    print(f"\nTrattamenti:")
    for tratt in info['trattamenti']:
        print(f"  - {tratt['nome']}")
        print(f"    Dosaggio: {tratt['dosaggio']}")

    # Esempio 3: Diagnosi da sintomi
    print("\n=== DIAGNOSI DA SINTOMI ===")
    sintomi_osservati = ["Macchie_Circolari_Grigie", "Ingiallimento_Foglie"]
    print(f"Sintomi osservati: {sintomi_osservati}")
    diagnosi = gestore.diagnosi_da_sintomi(sintomi_osservati)
    print("\nPossibili malattie:")
    for nome, confidenza, pianta in diagnosi[:3]:
        print(f"  - {nome} (confidenza: {confidenza:.2%}, pianta: {pianta})")
