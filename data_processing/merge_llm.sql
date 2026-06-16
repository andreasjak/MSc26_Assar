-- mergar med resultatet av llm-klassificeringarna
ALTER TABLE protein_data ADD COLUMN llm_label INTEGER;
DROP TABLE IF EXISTS llm_labels;
CREATE TABLE llm_labels AS

SELECT row_id, new_classification as llm_label FROM read_csv('llm_label.csv',delim=',');

UPDATE protein_data
SET llm_label = llm_labels.llm_label
FROM llm_labels
WHERE protein_data.row_id = llm_labels.row_id;

UPDATE protein_data
SET auto_classification = NULL
WHERE auto_classification = 5;


ALTER TABLE protein_data
ADD COLUMN label INTEGER;


ALTER TABLE protein_data ADD COLUMN manual_override INTEGER;

WITH vote_counts AS (
    SELECT 
        row_id,
        classification,
        COUNT(*) AS vote_count
    FROM classifications
    WHERE row_id IS NOT NULL
    GROUP BY row_id, classification
),
ranked_consensus AS (
    SELECT 
        row_id,
        classification,
        vote_count,
        ROW_NUMBER() OVER(PARTITION BY row_id ORDER BY vote_count DESC) AS vote_rank,
        COUNT(*) OVER(PARTITION BY row_id, vote_count) AS ties_count
    FROM vote_counts
)
UPDATE protein_data
SET manual_override = r.classification
FROM ranked_consensus r
WHERE protein_data.row_id = r.row_id
  AND r.vote_rank = 1
  AND r.ties_count = 1;

UPDATE protein_data SET label = COALESCE(manual_override,auto_classification,llm_label);
