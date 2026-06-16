import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from   functions.analyte import ANALYTES, get_ref
import duckdb
import sys


def choose_username():
    
    instructions = """Användning: använd knapparna 1-4 eller musen för att annotera. Man kan använda piltangenterna för att hoppa fram och tillbaka. 
    Prealbumin-regionen är inte med för att ge mer plats åt beta-gamma-regionen. Allting sparas kontinuerligt, så man kan stänga ner rutan när man känner sig klar. 
    
    Instruktioner: klassificera patientfallen enbart utifrån läkarutlåtandet.
    1. Normalt, ingen misstanke om M-kompnent.
    2. Misstänkt M-komponent, bör inmmunfixeras.
    3. Oligoklonalt mönster, ingen immunfixation.
    4. Lätta avvikande immunglobulinfördelning. Avvikande mönster utan flera tydliga band som för oligoklonalitet, men som inte motiverar immunfixation vid detta tillfälle. Svaras typiskt ut med:
    ”Lätt avvikande immunglobulinfördelning. M-komponent < 1 g/L? Specifik immunisering?
    """
    print(f"{instructions}\n\n")
    username = input("Ange användarnamn:")

    return username


def show_annotation_gui():
    con = duckdb.connect('capillary.db')
    usernames = con.execute("SELECT username FROM users").df()['username'].tolist()
    username = choose_username()
    if username is None:
        sys.exit("No username choosen!")
    if username not in usernames:
        print(f"Användarnamn: {username} finns ännu inte. Vill du spara detta som en ny användare? Svara med ja eller nej:")
        reply = input("")
        if reply == 'ja':
            print(f"Sparar en ny användare {username}")
            con.execute("INSERT INTO users VALUES (?)",[username])
            con.commit()
        else:
            sys.exit(0)
        


    print(f"Inloggad som: {username}")

    rows = con.execute("""
        WITH nbr_of_annotations(row_id,freq) AS (
            SELECT row_id,count(*) FROM classifications GROUP BY row_id    
        )       
                
        SELECT row_id,interpretation,coalesce(freq,0) AS tot_annotations FROM protein_data
            LEFT JOIN nbr_of_annotations USING (row_id)
        WHERE row_id NOT IN (SELECT row_id FROM classifications WHERE username = ?) AND auto_classification IS NULL and observation_nr = 1
        ORDER BY tot_annotations ASC;
    """, [username]).df()
    print(f"Antal fall att annotera: {len(rows)}")

    current_idx = [0]

    fig = plt.figure(figsize=(14, 8))
    ax  = fig.add_axes([0.05, 0.25, 0.65, 0.65])
    ax_status = fig.add_axes([0.05, 0.12, 0.90, 0.05])
    ax_status.axis('off')

    n = 9
    margin = 0.05
    gap = 0.005
    total_width = 1 - 2*margin
    button_width = (total_width - (n-1)*gap) / n

    x = margin
    axes_list = []
    for i in range(n):
        axes_list.append(fig.add_axes([x, 0.02, button_width, 0.07]))
        x += button_width + gap

    (ax_prev, ax_neg, ax_pos,ax_oligo,ax_dev, ax_lattkedja,  
     ax_followup,ax_ogiltigt, ax_next) = axes_list

    btn_prev     = widgets.Button(ax_prev,     '←',                       color='#e2e3e5')
    btn_neg      = widgets.Button(ax_neg,      '1: Ingen M-komp',         color='#d4edda')
    btn_pos      = widgets.Button(ax_pos,      '2: Misstänkt M-komp',     color='#f8d7da')
    btn_oligo= widgets.Button(ax_oligo,'3: Oligoklonal',     color='#cce5ff')
    btn_dev    = widgets.Button(ax_dev,    '4: Lätt avv. förd',         color='#fff3cd')
    btn_lattkedja      = widgets.Button(ax_lattkedja,      '5: lättkedje',      color='#e2e3e5')
    btn_ogiltigt = widgets.Button(ax_ogiltigt, '7: Ogiltigt prov',        color='#ff9999')
    btn_followup = widgets.Button(ax_followup, '6. Uppföljningsprov',     color="#79a3f8")
    btn_next     = widgets.Button(ax_next,     '→',                       color='#e2e3e5')

    status_text = ax_status.text(0.5, 0.5, '', ha='center', va='center',
                                  transform=ax_status.transAxes, fontsize=11)



    def draw(idx):
        ax.cla()
        ax.axis('off')
        row = rows.iloc[idx]

        ax.set_title(f"id={row['row_id']}  ({idx+1} / {len(rows)})")
        ax.text(
        0.02, 0.98, row['interpretation'],
        ha='left', va='top', wrap=True, fontsize=12,
        transform=ax.transAxes
        )


        existing = con.execute(
            "SELECT classification FROM classifications WHERE username = ? AND row_id = ?",
            [username, int(row['row_id'])]
        ).fetchone()

        label_names = {0: 'Ingen M-komponent', 1: 'Misstänkt M-komponent', 4: 'Oligoklonalt', 2: 'Lätt avvikande fördelning',6:'Lätt-kedje-M-komponent',3:'Ogiltigt prov'}
        existing_str = f"  ·  Annoterad: {label_names[existing[0]]}" if existing else "  ·  Ej annoterad"
        status_text.set_text(f'Inloggad som {username}. Fall {idx+1} av {len(rows)}{existing_str}')
        fig.canvas.draw_idle()
        

    def annotate(label: int):
        row = rows.iloc[current_idx[0]]
        label_names = {0: 'Ingen M-komponent', 1: 'Misstänkt M-komponent', 4: 'Oligoklonalt', 2: 'Lätt avvikande fördelning',6:'Lätt-kedje-M-komponent',3:'Ogiltigt prov'}

        print(f"id={row['row_id']} -> {label_names[label]}")
        con.execute(
            "INSERT OR REPLACE INTO classifications (username, row_id, classification) VALUES (?,?,?)",
            [username, int(row['row_id']), label]
        )

        if current_idx[0] < len(rows) - 1:
            current_idx[0] += 1
            draw(current_idx[0])
            

    def go_prev(e):
        if current_idx[0] > 0:
            current_idx[0] -= 1
            draw(current_idx[0])

    def go_next(e):
        if current_idx[0] < len(rows) - 1:
            current_idx[0] += 1
            draw(current_idx[0])

    def change_observation_nr(e):
        row = rows.iloc[current_idx[0]]
        con.execute("UPDATE protein_data SET observation_nr = 1000 WHERE row_id = (?)", [int(row['row_id'])] )
        print(f"id={row['row_id']} -> Ändrar observationsnr")
        if current_idx[0] < len(rows) - 1:
            current_idx[0] += 1
            draw(current_idx[0])

    def on_key(event):
        if event.key == '1': annotate(0)
        if event.key == '2': annotate(1)
        if event.key == '3': annotate(4)
        if event.key == '4': annotate(2)
        if event.key == '5': annotate(6)
        if event.key == '6': change_observation_nr(None)
        if event.key == '7': annotate(3)
        if event.key == 'left':  go_prev(None)
        if event.key == 'right': go_next(None)
        
    fig.canvas.mpl_connect('key_press_event', on_key)

    btn_neg.on_clicked(lambda e: annotate(0))
    btn_pos.on_clicked(lambda e: annotate(1))
    btn_oligo.on_clicked(lambda e: annotate(4))
    btn_dev.on_clicked(lambda e: annotate(2))
    btn_lattkedja.on_clicked(lambda e: annotate(6))
    btn_ogiltigt.on_clicked(lambda e: annotate(3))
    btn_prev.on_clicked(go_prev)
    btn_next.on_clicked(go_next)
    btn_followup.on_clicked(lambda e: change_observation_nr(e))


    draw(0)
    plt.show()
    con.close()
    


show_annotation_gui()
