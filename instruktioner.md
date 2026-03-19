### Hur man får igång'et
1. I mappen data_processing, lägg .csv-filen med all data. Döp den till 'proteindata.csv'
2. Installera duckdb. 'brew install duckdb' i terminalen på mac/linux. Går att ladda ner från internet annars.
3. Nu måste vi få duckdb att köra filen 'create_database.sql'. På mac/linux: ./run.sh. Annars starta duckdb och kör .read create_database.sql
4. Klart! Installera alla packages och sen kör något av nätverken