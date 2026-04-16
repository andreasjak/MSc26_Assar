import re
import duckdb
import matplotlib.pyplot as plt
import pandas as pd

def parse_age(s):
    match = re.match(r"(\d+)y(\d+)m", s)
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        return years + months/12
    return None

con = duckdb.connect('capillary.db')

# hämta den första per patient!
df = con.execute(""" SELECT id,
                coalesce(auto_classification, llm_label) AS label,
                age FROM protein_data
                                """).df()
con.close()


df['age_num'] = df['age'].apply(parse_age)

plt.figure()
out, bins = pd.cut(df['age_num'], 8,retbins=True)
out.value_counts().plot(kind = 'bar')
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age distribution")
plt.show()

df = df.dropna(subset=['age_num', 'label'])
print(min(df['age_num']) )
# Binna åldern, t.ex. 10 bins
df['age_bin'] = pd.cut(df['age_num'], bins=20)

# Skapa en boolesk kolumn för M-komponent
df['has_m'] = (df['label'] == 1).astype(int)

# Grupp och beräkna andel i procent
result = df.groupby('age_bin')['has_m'].mean() * 100
print(result)

plt.figure()
result.plot(marker='o')
plt.xticks(rotation=45)
plt.ylabel("Percentage with M-component")
plt.xlabel("Age bin")
plt.title("M-component prevalence by age")
plt.tight_layout()
plt.show()