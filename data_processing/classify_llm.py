"""
M-komponent klassificering med Claude API och DuckDB
=====================================================
Klassificerar ca 37 000 tolkningar enligt skala 0–4.
Skickar upp till BATCH_SIZE tolkningar per API-anrop för effektivitet.

Krav:
    pip install anthropic duckdb
    export ANTHROPIC_API_KEY="sk-ant-..."
"""

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
M-komponenter från serumkurvor. I vissa fall förekommer det bedömningen "oligoklonal fördelning", vilket också kan vara intressant att hitta.
Men det är primärt viktigast att vi detekterar ifall tolkningen diskuterar M-komponenter.

GRUNDPRINCIPER:
- Grundfrågan är alltid: går det att SE en M-komponent på serumkurvan?
- Klass 0 = "nästan säkert att ingen M-komponent är synlig"
- Klass 1 = "nästan säkert att minst en M-komponent är synlig"
- Klass 2 = "tecken på en mycket svag m-komponent eller M-komponent kanske finns men mycket osäkert"
- Klass 3 = "serumkurvan diskuteras ej, tolkningen diskuterar enbart urinprov". Dessa är manuellt bortfiltrerade i tidigare steg, och du ska alltså aldrig svara med klass 3.
- Klass 4 är sista utväg för genuint tvetydiga fall. Här får läkare klassificera manuellt.

REGLER (tillämpas i strikt prioritetsordning, stanna vid första match):

R1. Ingen M-komponent är synlig (obs gäller bara när vi vet att det inte finns någon som är synlig. Att en M-komponent inte längre är synlig hindrar inte en M-komponent av annan typ att fortfarande vara synlig) -> 0
R2. Minst en M-komponent är synlig -> 1
R3. Halt ≥ 1 g/L angiven → 1
R4. Halt < 1 g/L eller angiven i mg/L → 2
R5. Texten nämner BÅDE osynlig och synlig, ELLER osynlig men med
    halt ≥ 1 g/L → 1
R6. Lätt-kedje M-komponent (kappa/lambda utan Ig-klass) nämns
    utan synlighetskommentar och utan halt → 2
R7. Inget av ovanstående → 4

De enklare fallen hanteras automatiskt innan du ser dem. Du får de 
svårare fallen där automatiken inte räckte till. Här kommer några exempel på fall du kommer stöta på:
- Serum kan tyvärr inte analyseras eftersom serumprov saknas. Tubulärt proteinmönster i urinen. Ingen Bence-Jones proteinuri (immunfixation utförd). -> 3
- Patient med känd IgG lamda M-komponent. Inga tecken på inflammation. IgG lamda-M-komponentens halt är idag väsentligen oförändrad på 5 g/L (jmf med 2021-10-14). Bakgrundsimmunglobulin-nivån är ej sänkt.  Kvoten fria kappa/lambda-kedjor i serum är normal -> 1, 5 g/L är nästan säkert synlig
- Tecken på lätt-måttlig inflammation.  Immunglobuliner med normala halter.  Lätt oligoklonal immunglobulinfördelning. Urin: Albuminuri. Inga hållpunkter för Bence-Jones proteinuri. -> 3 (serumprov diskuteras inte)
- Påtaglig hypoalbuminemi.  Tecken på lätt inflammation.  Patientens IgM lambda M-komponent uppgår idag till 9,65 g/L. Bakgrundsimmunglobulinnivån är lätt-måttligt sänkt. -> 1
- Patient med kappa-M-komponent. Idag: Inga tecken på inflammation. Bakgrundsimmunglobulin-nivån är ej sänkt. Kvoten fria kappa/lambda-kedjor i serum är normal. -> 0 (kommentar från läkare: Det är en så kallad lätt-kedje M-komponent (kan vara kappa eller lambda) som kan synas på serumkurvan, men som i regel inte syns, när den syns så brukar vi oftast kommentera det, så default skulle vara att förvänta sig att inget syns, även om det inte är helt vattentätt.)
- Inga tecken på inflammation. Patientens IgG kappa M-komponent är ej säkert urskiljbar på dagens serumkurva (immunfixation ej utförd). Bakgrundsimmunglobulin-nivån är sänkt. Kvoten fria kappa/lambda-kedjor i serum är normal. -> 0 (går ej att se)
- Patient med känd kappa M-komonent. Inga tecken på inflammation. Bakgrundsimmunglobulin-nivån är ej sänkt. Urin: Förhöjd halt av kappakedo och kraftigt förhöjd halt av protein HC. Även förhöjda nivåer av flera andra proteiner i urinen.-> 3, möjligtvis 4. Går inte att avgöra ifall M-komponent är synlig eller inte.
- Lätt förhöjd halt av antitrypsin som enda tecken på inflammation.  Patientens IgG kappa-M-komponent är idag ej säkert synlig på serumkurvan (immunfixation ej utförd idag). Förhöjda halter av IgG och IgA. Bakgrundsimmunglobulin-nivån är ej sänkt. Kvoten fria kappa/lambda-kedjor i serum är normal. -> 2 (möjligt att vi ser en M-komponent här)
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
                        if label in range(5):
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
            wait = 2 ** attempt * 10
            print(f"  Rate limit – väntar {wait}s")
            time.sleep(wait)
        except Exception as e:
            if attempt == 3:
                print(f"  Fel efter {attempt+1} försök: {e}")
                return [{"id": row_id, "label": 4} for row_id, _ in batch]
            time.sleep(3)

    return [{"id": row_id, "label": 4} for row_id, _ in batch]


def run(batch_size: int = 100, workers: int = 5):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Fel: Sätt ANTHROPIC_API_KEY som miljövariabel.")

    client = anthropic.Anthropic(api_key=api_key)

    con = duckdb.connect('../capillary.db')
    rows = con.execute(
        "SELECT id, interpretation FROM protein_data "
        "WHERE auto_classification == 4 "
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
            futures = [executor.submit(classify_batch, client, b) for b in chunk]
            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())
        df = pd.DataFrame(all_results).rename(columns={'label': 'new_classification'})
        df.to_csv('new_classification.csv')


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
    parser.add_argument("--workers", type=int, default=2,   help="Parallella API-anrop (default: 5)")
    args = parser.parse_args()
    run(batch_size=args.batch, workers=args.workers)