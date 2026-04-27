import re
import duckdb
import matplotlib.pyplot as plt
import pandas as pd


con = duckdb.connect('capillary.db')

df = con.execute(""" SELECT comment_nr, frequency
                FROM comment_frequencies
                                """).df()
con.close()


plt.bar(df['comment_nr'], df['frequency'])
plt.title("No. of occurences per comment number")
plt.xlabel("Comment nr")
plt.ylabel("No of occurences")
plt.xticks(df['comment_nr'][::10])  # var 20e
plt.show()