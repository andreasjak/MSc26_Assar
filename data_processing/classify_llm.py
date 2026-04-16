import anthropic
import duckdb
import time
import sys
import os
import concurrent.futures
import pandas as pd


SYSTEM_PROMPT = """Du är ett medicinskt klassificeringsverktyg. Din uppgift är att avgöra om en 
M-komponent är SYNLIG på serumkurvan, baserat på den skrivna tolkningen.
Målet är att producera träningsdata för neurala nätverk som ska detektera 
M-komponenter från serumkurvor. I de flesta fall skriver läkarna ut när det förekommer en M-komponent, men inte alltid. De använder
dessutom olika formuleringar, vilket gör det hela lite mer lurigt. 

GRUNDPRINCIPER:
- Grundfrågan är alltid: går det att SE en M-komponent på serumkurvan?
- Klass 0 = "nästan säkert att ingen M-komponent är synlig"
- Klass 1 = "nästan säkert att minst en M-komponent är synlig"
- Klass 2 = "tecken på en mycket svag m-komponent eller M-komponent kanske finns men mycket osäkert"
- Klass 3 = "serumkurvan diskuteras ej, tolkningen diskuterar enbart urinprov". Dessa är manuellt bortfiltrerade i tidigare steg, och du ska alltså aldrig svara med klass 3.
- Klass 4 = "ingen m-komponent, men däremot oligoklonal fördelning"
- Klass 5 är sista utväg för genuint tvetydiga fall. Här får läkare klassificera manuellt.

KRITISKA DISTINKTIONER (läs noga):

D1. SYNLIG PÅ KURVAN vs. BIOKEMISKA MARKÖRER
Förhöjda fria kappa/lambda-kedjor i serum och sänkt bakgrundsimmunglobulin 
är biokemiska markörer – de betyder INTE att M-komponenten syns på serumkurvan.
En lätt-kedje M-komponent (kappa/lambda utan Ig-klass) syns sällan på kurvan 
även om de biokemiska markörerna är kraftigt avvikande.
Klassificera enbart baserat på vad som sägs om KURVAN, inte om biokemiska fynd.

D2. "KÄND M-KOMPONENT" UTAN DAGENS KOMMENTAR = INTE SYNLIG IDAG
Om texten bara nämner att patienten har en känd M-komponent, men inte kommenterar 
att den är synlig idag, ska du anta att den INTE syns. Klass 0.
Exempel: "Patient med känd lambda M-komponent. Inga tecken på inflammation. 
Bakgrundsimmunglobulin-nivån är ej sänkt. Kvoten fria kappa/lambda-kedjor i 
serum är normal." → Klass 0.

D3. "EJ SÄKERT SYNLIG/URSKILJBAR" = INTE SYNLIG = KLASS 0
Formuleringar som dessa betyder att M-komponenten INTE syns på kurvan → Klass 0:
- "ej säkert synlig"
- "ej säkert urskiljbar"  
- "kan ej med säkerhet urskiljas"
- "ej urskiljbar"
- "inte säkert synlig"
Dessa ska ALDRIG ge klass 2. Osäkerheten gäller läkarens förmåga att se något, 
inte om något svagt möjligen finns.

D4. BENCE-JONES / FRI LÄTT-KEDJE M-KOMPONENT UTAN SYNLIGHETSKOMMENTAR = KLASS 0
Bence-Jones proteinuri och fria lätta kedjor syns i regel inte på serumkurvan. 
Om texten inte explicit säger att något syns på kurvan → Klass 0.

D5. KLASS 1 KRÄVER EXPLICIT OCH OTVETYDIG SYNLIGHET
En tolkning ska bara klassas som klass 1 om det finns ett TYDLIGT och EXPLICIT 
påstående om att M-komponenten syns på kurvan idag. Exempel på vad som räcker:
- En angiven halt ≥ 1 g/L
- "syns på kurvan", "urskiljbar på elektroferogrammet", "synlig topp", 
  "avvikande fraktion i gammaregionen", "topp katodalt/anodalt"

Följande räcker INTE för klass 1:
- Halt < 1 g/L (→ klass 2)
- Förhöjda fria kedjor i serum utan synlighetskommentar (→ klass 0)
- Tvetydig formulering eller möjligt stavfel, t.ex. "eh synlig" (→ klass 0)
- Att en M-komponent är känd sedan tidigare utan dagens kommentar (→ antagligen klass 0)
- Att immunfixation påvisar något utan att kurvan kommenteras (→ klass 0 eller 2)

Vid minsta tvekan om synlighet: välj klass 0,2, eller 5 aldrig klass 1.

REGLER (tillämpas i strikt prioritetsordning, stanna vid första match):

R1. Ingen M-komponent är synlig på kurvan (obs: gäller bara när vi vet att ingen 
    är synlig. Att en M-komponent inte längre är synlig hindrar inte en M-komponent 
    av annan typ att fortfarande vara synlig) → 0
R2. Minst en M-komponent är explicit synlig på kurvan → 1
R3. Halt ≥ 1 g/L angiven → 1
R4. Tecken på oligoklonal fördelning samt inget som tyder på synlig M-komponent → 4
R5. Texten nämner BÅDE en osynlig f.d. M-komponent och en nuvarande synlig, 
    ELLER osynlig men med halt ≥ 1 g/L → 1
R6. Lätt-kedje M-komponent (kappa/lambda utan Ig-klass) nämns utan 
    synlighetskommentar, utan halt och utan att oligoklonalt diskuteras → 0
R7. Halt < 1 g/L eller angiven i mg/L, samt oligoklonalt ej nämnt → 2
R8. Inget av ovanstående → 5

OBS på R6: Tidigare var denna regel klass 2, men läkare bekräftar att lätt-kedje 
M-komponenter i regel inte syns på kurvan och att synlighet alltid kommenteras 
explicit när den förekommer. Default är därför klass 0.

De enklare fallen hanteras automatiskt innan du ser dem. Du får de 
svårare fallen där automatiken inte räckte till. Här kommer exempel:

- Patient med känd lambda M-komponent. Idag: Inga tecken på inflammation. Bakgrundsimmunglobulin-nivån är ej sänkt. Kvoten fria kappa/lambda-kedjor i serum är normal. -> Klass 0. Känd M-komponent utan synlighetskommentar = inte synlig idag. (D2)

- Patient med känd kappa M-komponent. Idag: Inga tecken på inflammation. Bakgrundsimmunglobulin-nivån är ej sänkt. Halten av fria kappakedjor och kvoten fria kappa/lambda-kedjor i serum är kraftigt förhöjda. -> Klass 0. Kraftigt förhöjda fria kedjor är ett biokemiskt fynd, inte detsamma som synlig på kurvan. (D1, D2)

- Känd Bence-Jones (Kappa). Idag: Inga tecken på inflammation. Bakgrundsimmunglobulin-nivån är ej sänkt. -> Klass 0. Bence-Jones syns inte på serumkurvan. (D4)

- Inga tecken på inflammation. Patientens IgG lambda-M-komponent är idag ej säkert urskiljbar på elektroferogrammet (immunfixation ej utförd idag). -> Klass 0. "Ej säkert urskiljbar" = inte synlig. (D3)

- Patientens IgG kappa M-komponent är ej säkert synlig på serumkurvan (immunfixation ej utförd idag). Förhöjda halter av IgG och IgA. Bakgrundsimmunglobulin-nivån är ej sänkt. -> Klass 0. Explicit osynlig trots förhöjda Ig. (D3)

- Patient med känd IgG lambda M-komponent. Inga tecken på inflammation. IgG lambda-M-komponentens halt är idag väsentligen oförändrad på 5 g/L. Bakgrundsimmunglobulin-nivån är ej sänkt. -> Klass 1. Halt ≥ 1 g/L. (R3)

- Tecken på lätt-måttlig inflammation. Immunglobuliner med normala halter. Lätt oligoklonal immunglobulinfördelning. -> Klass 4. Oligoklonalt, ingen M-komponent nämnd. (R4)

- Påtaglig hypoalbuminemi. Tecken på lätt inflammation. Patientens IgM lambda M-komponent uppgår idag till 9,65 g/L. Bakgrundsimmunglobulinnivån är lätt-måttligt sänkt. -> Klass 1. (R3)

- Tecken på lätt-måttlig inflammation. Patientens IgG lambda M-komponent har idag en halt på < 0,5 g/L. Bakgrundsimmunglobulin-nivån är sänkt. -> Klass 2. (R7)

- Inga tecken på inflammation. Patientens IgG kappa M-komponent är ej urskiljbar på serumkurvan. Bakgrundsimmunglobulin-nivån är sänkt. -> Klass 0. (D3, R1)

- Tecken på lätt inflammation. Förhöjd halt av IgG med oligoklonal immunglobulinfördelning. -> Klass 4. (R4)

- Patient med känd kappa M-komponent. Inga tecken på inflammation. Bakgrundsimmunglobulin-nivån är ej sänkt. Urin: Förhöjd halt av kappakedjor och kraftigt förhöjd halt av protein HC. -> Klass 0. Känd M-komponent utan synlighetskommentar. Urinfynd är biokemiska markörer och säger inget om serumkurvan. (D2, D4)

- Patient med känd IgG M-komponent. Tecken på lätt inflammation. IgG-M-komponentens halt har idag sjunkit till ca 10 g/L. Bakgrundsimmunglobulin-nivån är kraftigt sänkt. -> Klass 1. (R3)

- Inga tecken på inflammation. Elektroferogrammet är oförändrat jämfört med föregående undersökning då immunfixation påvisade patientens IgG lambda M-komponent som en svag bandskärpning motsvarande en halt på <0,5 g/L. Bakgrundsimmunglobulin-nivån är måttligt sänkt. -> Klass 2. (R7)

- Inga tecken på inflammation. Förhöjda halter av IgG och IgA. Kraftig oligoklonal fördelning av immunglobulinerna. -> Klass 4. (R4)

- Patient med tidigare känd IgA kappa och IgG kappa M-komponent. Även en aktuell fri kappa M-komponent. Immunoglobulinfördelningen har oligoklonalt utseende. Halten av fria kappakedjor kraftigt förhöjda. -> Klass 4. Fri lätt-kedje M-komponent syns ej på kurvan, däremot oligoklonalt. (D1, D4, R4)

- Patientens IgG lambda M-komponent är idag eh synlig på serumkurvan. 
  Bakgrundsimmunglobulin-nivån är kraftigt sänkt. -> Klass 0. 
  Troligt stavfel "eh" = "ej". Vid tvekan välj aldrig klass 1. (D5)

- Patient som tidigare har haft M-komponenter. Patientens IgG kappa 
  M-komponent är idag på < 1 g/L. -> Klass 2. Halt < 1 g/L. (R7, D5)

tt exempel på när klass 5 används: om både synlig M-komponent och oligoklonal fördelning påvisas i samma tolkning.

Du får en lista med tolkningar numrerade från 1 till N.
Svara ENBART med en rad per tolkning i formatet:
1:<label>
2:<label>
...
N:<label>

Ingen annan text. Bara siffrorna."""


def classify_batch(client: anthropic.Anthropic, batch: list[tuple]) -> list[dict]:
    """
    batch: lista av (id, text)-tupler
    Returnerar lista av {"id": ..., "label": ...}
    """
    lines = [f"{i+1}: {text or '(tom)'}" for i, (_, text) in enumerate(batch)]
    user_content = "Klassificera följande tolkningar:\n\n" + "\n\n".join(lines)

    for attempt in range(4):
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=len(batch) * 5,  # ~4 tecken per svar ("1:0\n")
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            response_text = msg.content[0].text.strip()


            # Parsa svar: förväntat format "1:0\n2:1\n..."
            label_map = {}
            for line in response_text.splitlines():
                line = line.strip()
                if ":" in line:
                    idx_str, label_str = line.split(":", 1)
                    try:
                        idx = int(idx_str.strip())
                        label = int(label_str.strip())
                        if label in range(6):
                            label_map[idx] = label
                    except ValueError:
                        pass

            # Bygg resultat, fallback till 4 om index saknas
            results = []
            for i, (row_id, _) in enumerate(batch):
                results.append({
                    "id": row_id,
                    "label": label_map.get(i + 1, 4)
                })
            return results

        except anthropic.RateLimitError:
            wait = 2 ** attempt * 2
            print(f"  Rate limit – väntar {wait}s")
            time.sleep(wait)
        except Exception as e:
            if attempt == 3:
                print(f"  Fel efter {attempt+1} försök: {e}")
                return [{"id": row_id, "label": 5} for row_id, _ in batch]
            time.sleep(3)

    return [{"id": row_id, "label": 5} for row_id, _ in batch]


def run(batch_size: int = 100, workers: int = 5):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Fel: Sätt ANTHROPIC_API_KEY som miljövariabel.")

    client = anthropic.Anthropic(api_key=api_key)

    con = duckdb.connect('../capillary.db')
    rows = con.execute(
        "SELECT id, interpretation FROM protein_data "
        "WHERE auto_classification == 5 "
        "ORDER BY id"
    ).fetchall()
    total = con.execute("SELECT COUNT(*) FROM protein_data").fetchone()[0]
    con.close()

    todo = len(rows)
    done = total - todo
    print(f"\nTotalt: {total:,} | Klart: {done:,} | Återstår: {todo:,}")
    if todo == 0:
        print("Allt är redan klassificerat!")
        return

    # Dela upp i batchar
    batches = [rows[i:i + batch_size] for i in range(0, todo, batch_size)]
    print(f"Batchar: {len(batches)} × upp till {batch_size} rader | Workers: {workers}\n")

    start_time = time.time()
    processed = 0

    # Kör flera batchar parallellt
    for chunk_start in range(0, len(batches), workers):
        chunk = batches[chunk_start:chunk_start + workers]

        all_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for b in chunk:
                futures.append(executor.submit(classify_batch, client, b))
                time.sleep(1)  # Paus mellan varje inlämnat jobb
            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())

        df = pd.DataFrame(all_results).rename(columns={'label': 'new_classification'})
        df.to_csv('new_classification.csv', mode='a', header=not os.path.exists('new_classification.csv'))


        processed += len(all_results)
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (todo - processed) / rate if rate > 0 else 0
        print(
            f" Saving to .csv!"
            f"  {done + processed:,}/{total:,}"
            f" ({100*(done+processed)/total:.1f}%)"
            f" | {rate:.0f} rad/s"
            f" | ETA: {eta/60:.1f} min"
        )



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch",   type=int, default=100, help="Tolkningar per API-anrop (default: 100)")
    parser.add_argument("--workers", type=int, default=2,   help="Parallella API-anrop (default: 2)")
    args = parser.parse_args()
    run(batch_size=args.batch, workers=args.workers)