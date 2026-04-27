DROP TABLE IF EXISTS comments;
DROP TABLE IF EXISTS comment_frequencies;
DROP TABLE IF EXISTS comment_precences;

CREATE TABLE comments AS
    SELECT nr as comment_nr, kommentar as comment
FROM read_csv('kommentarer.csv', delim=',', header=true);

CREATE TABLE comment_frequencies (
    comment_nr TEXT,
    frequency INTEGER
);

CREATE TABLE comment_precences AS (
    SELECT id AS row_id, comment_nr FROM comments c
    JOIN protein_data p
        ON p.interpretation ILIKE '%' || TRIM(TRAILING '.' FROM c.comment) || '%'
    ORDER BY row_id
);

INSERT INTO comment_frequencies(comment_nr,frequency) (
    SELECT comment_nr, count(*) FROM comments c
    JOIN protein_data p
        ON p.interpretation ILIKE '%' || TRIM(TRAILING '.' FROM c.comment) || '%'
    GROUP BY c.comment_nr
    ORDER BY c.comment_nr
);

