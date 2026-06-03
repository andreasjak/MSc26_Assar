CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS classifications (
    username    TEXT,
    row_id      INTEGER,
    classification  TINYINT,
    PRIMARY KEY (username,row_id),
    FOREIGN KEY (username) REFERENCES users(username),
);

CREATE TABLE IF NOT EXISTS difficult_cases (
    row_id INTEGER PRIMARY KEY
);

INSERT OR IGNORE INTO users (username) VALUES
    ('Assar'),
    ('Oskar'),
    ('Magnus');

CREATE TABLE IF NOT EXISTS most_difficult_cases (
    row_id INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS difficult_curves AS SELECT * FROM read_parquet('difficult_cases.parquet')
