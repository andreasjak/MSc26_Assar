UPDATE protein_data
SET auto_classification = NULL
WHERE auto_classification = 5;


ALTER TABLE protein_data
ADD COLUMN label INTEGER;


ALTER TABLE protein_data ADD COLUMN manual_override INTEGER;
UPDATE protein_data SET manual_override = 
    (SELECT classification FROM classifications WHERE classifications.row_id = protein_data.row_id AND username = 'Assar') 
    WHERE auto_classification IS NULL;

UPDATE protein_data SET label = COALESCE(manual_override,auto_classification);

UPDATE protein_data SET observation_nr = 1000 WHERE manual_override = 8;
