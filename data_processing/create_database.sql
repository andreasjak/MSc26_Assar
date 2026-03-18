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

-- Skapar manuella klassificeringen

ALTER TABLE protein_data ADD COLUMN manual_classification INTEGER;

UPDATE protein_data
SET manual_classification = CASE

    -- NEGATIVA fraser FÖRST
    WHEN lower(interpretation) LIKE '%ingen m-komponent påvisas i serum%'           THEN 0
    WHEN lower(interpretation) LIKE '%ingen påvisbar m-komponent%'                   THEN 0
    WHEN lower(interpretation) LIKE '%ingen mätbar m-komponent%'                     THEN 0
    WHEN lower(interpretation) LIKE '%ingen m-komponent kan påvisas%'                THEN 0
    WHEN lower(interpretation) LIKE '%ingen säker m-komponent påvisas%'              THEN 0
    WHEN lower(interpretation) LIKE '%ingen synlig m-komponent i serum%'             THEN 0
    WHEN lower(interpretation) LIKE '%ingen synlig m-komponent på serumkurvan%'      THEN 0
    WHEN lower(interpretation) LIKE '%ses ingen m-komponent på serumkurvan%'         THEN 0
    WHEN lower(interpretation) LIKE '%ingen m-komponent synlig i serum%'             THEN 0
    WHEN lower(interpretation) LIKE '%ingen m-komponent.%'                           THEN 0
    WHEN lower(interpretation) LIKE '%ingen iga m-komponent påvisas%'                THEN 0
    WHEN lower(interpretation) LIKE '%ingen igg m-komponent påvisas%'                THEN 0
    WHEN lower(interpretation) LIKE '%ingen igm m-komponent påvisas%'                THEN 0
    WHEN lower(interpretation) LIKE '%varken patientens%synliga på serumkurvan%'     THEN 0
    WHEN lower(interpretation) LIKE '%varken patientens%urskiljbara%'                THEN 0
    WHEN lower(interpretation) LIKE '%m-komponent är idag ej säkert urskiljbar%'     THEN 0
    WHEN lower(interpretation) LIKE '%m-komponent är ej säkert urskiljbar%'          THEN 0
    WHEN lower(interpretation) LIKE '%m-komponent är idag ej säkert synlig%'         THEN 0
    WHEN lower(interpretation) LIKE '%m-komponent är idag inte säkert synlig%'       THEN 0
    WHEN lower(interpretation) LIKE '%m-komponent är idag ej urskiljbar%'            THEN 0
    WHEN lower(interpretation) LIKE '%m-komponent är ej säkert synlig%'              THEN 0
    WHEN lower(interpretation) LIKE '%m-komponent är idag ej säkert avgränsbar%'     THEN 0
    WHEN lower(interpretation) LIKE '%m-komponenter är ej säkert urskiljbara%'       THEN 0
    WHEN lower(interpretation) LIKE '%ingen säkert synlig m-komponent på serumkurvan%'   THEN 0
    WHEN lower(interpretation) LIKE '%ingen säkert synlig m-komponent på dagens serumkurva%' THEN 0
    WHEN lower(interpretation) LIKE '%ingen synlig m-komponent påvisas i serum%'          THEN 0
    WHEN lower(interpretation) LIKE '%m-komponenter kan ej med säkerhet urskiljas på serumkurvan%' THEN 0

    -- POSITIVA med konkret mätvärde (≥ 1 g/L) → synlig på kurvan
    -- Matchar "halt på X g/L" där X är ett tal >= 1
    WHEN (
        lower(interpretation) LIKE '%m-komponent%halt på%g/l%'
        AND regexp_matches(interpretation, 'halt på (\d+)[,.](\d+)?\s*g/[Ll]')
        AND TRY_CAST(
            regexp_extract(interpretation, 'halt på (\d+)', 1)
            AS INTEGER) >= 1
    ) THEN 1

    -- POSITIVA utan halt eller med tydlig formulering
    WHEN lower(interpretation) LIKE '%m-komponent har idag en halt%'
        AND lower(interpretation) NOT LIKE '%halt på < %'                            THEN 1
    WHEN lower(interpretation) LIKE '%m-komponent är idag%'                          THEN 1
    WHEN lower(interpretation) LIKE '%m-komponent uppgår%'                           THEN 1
    WHEN lower(interpretation) LIKE '%m-komponentens halt%'                          THEN 1
    WHEN lower(interpretation) LIKE '%m-komponenterna%'                              THEN 1
    WHEN lower(interpretation) LIKE '%m-komponenten till%'                           THEN 1
    WHEN lower(interpretation) LIKE '%m-komponent av typ igg%'                       THEN 1
    WHEN lower(interpretation) LIKE '%m-komponent av typ iga%'                       THEN 1
    WHEN lower(interpretation) LIKE '%m-komponent av typ igm%'                       THEN 1
    WHEN lower(interpretation) LIKE '%nyupptäckt%m-komponent%'
        AND lower(interpretation) NOT LIKE '%nyupptäckt%m-komponent%< %'             THEN 1
    WHEN lower(interpretation) LIKE '%nyupptäckt iga%'                               THEN 1
    WHEN lower(interpretation) LIKE '%nyupptäckt igg%'                               THEN 1
    WHEN lower(interpretation) LIKE '%nyupptäckt igm%'                               THEN 1

    -- OSÄKRA → NULL (< 1 g/L, svaga fynd, misstankar)
    WHEN lower(interpretation) LIKE '%halt på < %'                                   THEN NULL
    WHEN lower(interpretation) LIKE '%m-komponent < %'                               THEN NULL
    WHEN lower(interpretation) LIKE '%kan inte uteslutas%'                           THEN NULL
    WHEN lower(interpretation) LIKE '%misstanke om%m-komponent%'                     THEN NULL

    ELSE NULL
END;