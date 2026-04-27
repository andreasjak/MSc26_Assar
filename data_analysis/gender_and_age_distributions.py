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

df = con.execute(""" SELECT id,
                label,
                age,
                gender
                FROM protein_data
                                """).df()
con.close()

df['has_m'] = (df['label'] == 1).astype(int)
## kön

men = df[df['gender'] == 'M']
women = df[df['gender'] == 'K']
men_total = len(men)
women_total = len(women)
men_m = men['has_m'].sum()
women_m = women['has_m'].sum()

men_no_m = men_total - men_m
women_no_m = women_total - women_m

fig, ax = plt.subplots(figsize=(6, 5))

ax.bar(['Men', 'Women'], [men_no_m, women_no_m], label='No M-component')
ax.bar(['Men', 'Women'], [men_m, women_m], bottom=[men_no_m, women_no_m], label='M-component')

# Procentetiketter inuti staplarna
for x, (m, total) in enumerate(zip([men_m, women_m], [men_total, women_total])):
    pct = m / total * 100
    ax.text(x, total + 15, f"{pct:.1f}%", ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel("Count")
ax.set_title("Gender distribution with M-component prevalence")
ax.legend()
plt.tight_layout()
plt.show()



## ålder
df['age_num'] = df['age'].apply(parse_age)


df = df.dropna(subset=['age_num', 'label'])

max_age = int(df['age_num'].max()) + 10
bins = range(0,max_age+10,10)
labels = [f"{i}-{i+10}" for i in range(0, max_age, 10)]

#plot 1: fördelningen av ålder i datan
df['age_bin'] = pd.cut(df['age_num'], bins=bins,labels=labels)
plt.figure(figsize=(8, 4))
df['age_bin'].value_counts().sort_index().plot(kind='bar')
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age distribution")
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.tight_layout()
plt.show()


#plot 2: andelen med M-komponent som funktion av ålder
result = df.groupby('age_bin', observed=True)['has_m'].mean() * 100

plt.figure(figsize=(8, 4))
result.plot(marker='o')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.ylabel("Percentage with M-component (%)")
plt.xlabel("Age bin")
plt.title("M-component prevalence by age")
plt.tight_layout()
plt.show()

#plot 3: fördelningen av M-komponentpatienternas ålder
m_component = df[df['has_m'] == 1]
plt.figure(figsize=(8, 4))
m_component['age_bin'].value_counts().sort_index().plot(kind='bar')
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age distribution amongst cases with M-component")
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.tight_layout()
plt.show()