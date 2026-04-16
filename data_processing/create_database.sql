-- Skapa ny tabell med konverterade arrayer
DROP TABLE IF EXISTS protein_data;
CREATE TABLE protein_data AS
SELECT
    row_number() OVER() AS id,
    PID as pid,
    TimeStamp as time_stamp,
    Interpretation as interpretation,
    Sign as sign,
    SignTime as sign_time,
    
    -- Konvertera ^-separerade strängar till INTEGER-arrayer
    list_transform(string_split(Value, '^'), lambda x: TRY_CAST(x AS INTEGER)) AS value,
    string_split(DelimitValue, '^')            AS delimit_value,
    string_split(DelimitValue2, '^')           AS delimit_value2,
    
    -- Övriga kolumner som de är
    string_split(Analysis, '^')      AS analysis,
    string_split(PValue, '^')        AS protein_value,  -- behåll som text pga <0.30 etc
    string_split(Comment, '^')       AS comment,
    string_split(Flags, '^')         AS flags,
    SampleInfo as sample_info,
    gender,
    age,
    IGComponent as m_component

FROM read_csv('proteindata.csv', delim=';', header=true, null_padding=true, parallel = false);

ALTER TABLE protein_data ADD COLUMN m_component_label INTEGER;
UPDATE protein_data
SET m_component_label = CASE
    WHEN m_component = 'NA' THEN 0
    WHEN lower(m_component) = 'se bedömning' THEN NULL
    ELSE 1
END;




ALTER TABLE protein_data ADD COLUMN auto_classification INTEGER; -- följande görs nedan

-- =============================================================================
-- MINIMAL KONSERVATIV KLASSIFICERING AV M-KOMPONENT I SERUMTOLKNING
--
-- Klasser:
--   0 = Ingen M-komponent synlig på serumprovet
--   1 = Minst en M-komponent synlig (antingen att det står, eller en halt >= 1 g/L angiven)
--   2 = Gränsfall – halt < 1 g/L eller osäker synlighet
--   3 = Tolkningen diskuterar inte serumprovet  → ej användbar för att arbeta med serumkurvor.
--   4 = Oligoklonal fördelning (görs av LLM)
--   5 = Måste tas med LLM.
--
-- STRATEGI:
--   • Klass 2 kontrolleras FÖRE klass 1 (undviker att "<1 g/L" matchas som 1)
--   • Klass 1 kräver att extraherad siffra är >= 1 g/L
--   • Klass 0 använder bara de två vanligaste stereotypa fraserna
--   • Klass 3 hanteras inte här – för många varianter, LLM klarar det bättre
--   • Klass 4&5 - svåra fall som måste kollas manuellt.
-- =============================================================================

UPDATE protein_data
SET auto_classification = 3 WHERE len(value) <= 1;
 
UPDATE protein_data
SET auto_classification = CASE
 
    -- =========================================================================
    -- KLASS 2 (före klass 1): Halt < 1 g/L eller mg/L → gränsfall
    --
    -- Täcker:
    --   "halt på <1 g/L", "halt på < 1 g/L"
    --   "halt på mindre än 1 g/L"
    --   "M-komponent <1 g/L?"
    --   Alla mg/L-angivelser (per definition < 1 g/L)
    -- =========================================================================
    WHEN lower(interpretation) LIKE '%halt på <1 g/l%'
      OR lower(interpretation) LIKE '%halt på < 1 g/l%'
      OR lower(interpretation) LIKE '%halt på  <1 g/l%'
      OR lower(interpretation) LIKE '%halt på  < 1 g/l%'
      OR lower(interpretation) LIKE '%halt på mindre än 1%g/l%'
      OR lower(interpretation) LIKE '%m-komponent <1 g/l%'
      OR lower(interpretation) LIKE '%m-komponent < 1 g/l%'
      OR lower(interpretation) LIKE '%halt%mg/l%'
    THEN 2
 
    -- =========================================================================
    -- KLASS 1: Halt STRIKT > 1 g/L → M-komponent synlig
    --
    -- Extraherar siffran efter "halt på [ca|cirka]" eller "på ca/cirka X g/L"
    -- och kräver att värdet är > 1.
    --
    -- Täcker t.ex.:
    --   "halt på ca 4 g/L"       → 4    > 1 → klass 1
    --   "halt på cirka 47 g/L"   → 47   > 1 → klass 1
    --   "på ca 60 g/L"           → 60   > 1 → klass 1
    --   "halt på 4,72 g/L"       → 4.72 > 1 → klass 1
    --   "halt på ca 1 g/L"       → 1    = 1 → 2 (gränszonen!)
    --   "halt på cirka 1 g/L"    → 1    = 1 → 2 (gränszonen!)
    -- =========================================================================
    WHEN TRY_CAST(
            replace(
                regexp_extract(
                    lower(interpretation),
                    '(?:halt på|på ca|på cirka)\s*(?:ca\s*|cirka\s*)?(\d+[,\.]\d+|\d+)\s*g/l',
                    1
                ),
                ',', '.'
            ) AS DOUBLE
         ) >= 1.0 THEN 1

    WHEN TRY_CAST(
            replace(
                regexp_extract(
                    lower(interpretation),
                    '(?:halt på|på ca|på cirka)\s*(?:ca\s*|cirka\s*)?(\d+[,\.]\d+|\d+)\s*g/l',
                    1
                ),
                ',', '.'
            ) AS DOUBLE
         ) < 1.0
    THEN 2
 
    -- =========================================================================
    -- KLASS 0: Tydligt negativ – ingen M-komponent i serum
    --
    -- Bara de två vanligaste stereotypa fraserna. Varianter som
    -- "Ingen synlig M-komponent i serum" eller "ej urskiljbar" skickas
    -- till LLM – de kan förekomma i texter som OCKSÅ nämner en positiv komponent. och 
    -- finns det en positiv komponent som är urskiljbar ska den givetvis klassas som kategori 1.
    -- =========================================================================
    WHEN lower(interpretation) LIKE '%ingen m-komponent påvisas i serum%'           THEN 0
    WHEN lower(interpretation) LIKE '%ingen påvisbar m-komponent%'                   THEN 0
    WHEN lower(interpretation) LIKE '%ingen mätbar m-komponent%'                     THEN 0
    WHEN lower(interpretation) LIKE '%ingen m-komponent kan påvisas%'                THEN 0
    WHEN lower(interpretation) LIKE '%ingen säker m-komponent påvisas%'              THEN 0
    WHEN lower(interpretation) LIKE '%ingen synlig m-komponent i serum%'             THEN 0
    WHEN lower(interpretation) LIKE '%ingen synlig m-komponent på serumkurvan%'      THEN 0
    WHEN lower(interpretation) LIKE '%ses ingen m-komponent på serumkurvan%'         THEN 0
    WHEN lower(interpretation) LIKE '%ingen m-komponent synlig i serum%'             THEN 0
    WHEN lower(interpretation) LIKE '%varken patientens%synliga på serumkurvan%'     THEN 0
    WHEN lower(interpretation) LIKE '%varken patientens%urskiljbara%'                THEN 0
    WHEN lower(interpretation) LIKE '%ingen säkert synlig m-komponent på serumkurvan%'   THEN 0
    WHEN lower(interpretation) LIKE '%ingen säkert synlig m-komponent på dagens serumkurva%' THEN 0
    WHEN lower(interpretation) LIKE '%ingen synlig m-komponent påvisas i serum%'          THEN 0
    WHEN lower(interpretation) LIKE '%m-komponenter kan ej med säkerhet urskiljas på serumkurvan%' THEN 0
    WHEN lower(interpretation) LIKE '%m-komponent är idag ej synlig på serumkurva%' THEN 0
    WHEN lower(interpretation) LIKE '%varken patientens igm kappa m-komponent eller igg kappa m-komponent är idag urskiljbar på serumkurvan%' THEN 0
    WHEN lower(interpretation) LIKE '%är idag ej säkert urskilbara på serumelektroferogrammet%' THEN 0
    WHEN lower(interpretation) LIKE '%ingen m-komponent är idag synlig på serumkurvan%' THEN 0
    WHEN lower(interpretation) LIKE '%ingen m-komponent är synlig på serumkurvan%' THEN 0
 
    -- =========================================================================
    -- ALLT ANNAT → klass 5
    -- =========================================================================
    ELSE 5
 
END
WHERE auto_classification IS NULL;


