-- mergar med resultatet av llm-klassificeringarna
ALTER TABLE protein_data ADD COLUMN llm_label INTEGER;
DROP TABLE IF EXISTS llm_labels;
CREATE TABLE llm_labels AS
SELECT id, new_classification as llm_label FROM read_csv('new_classification.csv',delim=',');

UPDATE protein_data
SET llm_label = llm_labels.llm_label
FROM llm_labels
WHERE protein_data.id = llm_labels.id;

UPDATE protein_data
SET auto_classification = NULL
WHERE auto_classification = 5;


ALTER TABLE protein_data
ADD COLUMN label INTEGER;

UPDATE protein_data SET label = COALESCE(auto_classification,llm_label);