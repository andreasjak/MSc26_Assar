import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from   functions.analyte import ANALYTES, get_ref
import duckdb
import sys

current_curve = [None]

y=0
x=0

def choose_username():
    
    instructions = """Användning: använd knapparna 1-4 eller musen för att annotera. Man kan använda piltangenterna för att hoppa fram och tillbaka. 
    Prealbumin-regionen är inte med för att ge mer plats åt beta-gamma-regionen. Allting sparas kontinuerligt, så man kan stänga ner rutan när man känner sig klar. 
    
    Instruktioner: klassificera patientfallen med avseende på immunglobulinmönster utifrån kurva och uppmätta proteinhalter, det finns fyra klasser att välja på. Detta är förstagångsfall utan känd M-komponent eller specifik anamnestisk information.
    Du kommer få klassificera 200 fall, varav 100 där modellen antingen visar stor osäkerhet, eller inte gjort samma klassificering som läkaren. Utöver de 100 "svåra" fallen kommer du även få klassificera 100 "normalsvåra" fall.
    Ordningen de 200 fallen presenteras i är slumpartad.
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
    con = duckdb.connect('application/application.db')
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
        SELECT * FROM difficult_cases WHERE row_id NOT IN (SELECT row_id FROM classifications WHERE username = ?)
    """,[username]).df()
    print(f"Antal fall att annotera: {len(rows)}")

    current_idx = [0]

    fig = plt.figure(figsize=(14, 8))
    ax_curve  = fig.add_axes([0.05, 0.25, 0.65, 0.65])
    ax_table  = fig.add_axes([0.72, 0.55, 0.26, 0.40])  # kortare, högre upp
    ax_gamma  = fig.add_axes([0.72, 0.25, 0.26, 0.28])  # större, precis under tabellen
    ax_status = fig.add_axes([0.05, 0.12, 0.90, 0.05])
    ax_status.axis('off')

    ax_prev    = fig.add_axes([0.05, 0.02, 0.07, 0.07])
    ax_neg     = fig.add_axes([0.14, 0.02, 0.16, 0.07])
    ax_pos     = fig.add_axes([0.32, 0.02, 0.16, 0.07])
    ax_oligo   = fig.add_axes([0.50, 0.02, 0.16, 0.07])
    ax_dev     = fig.add_axes([0.68, 0.02, 0.16, 0.07])
    ax_next    = fig.add_axes([0.86, 0.02, 0.07, 0.07])

    btn_prev   = widgets.Button(ax_prev, '←',                    color='#e2e3e5')
    btn_neg    = widgets.Button(ax_neg,  '1: Ingen M-komp',      color='#d4edda')
    btn_pos    = widgets.Button(ax_pos,  '2: Misstänkt M-komp',  color='#f8d7da')
    btn_oligo  = widgets.Button(ax_oligo,'3: Oligoklonalt',      color='#fff3cd')
    btn_dev    = widgets.Button(ax_dev,  '4: Lätt avv. förd.',   color="#9ebbf4")
    btn_next   = widgets.Button(ax_next, '→',                    color='#e2e3e5')

    status_text = ax_status.text(0.5, 0.5, '', ha='center', va='center',
                                  transform=ax_status.transAxes, fontsize=11)

    def redraw_zoom(curve):
        x_min = max(0, x - 20+50)
        x_max = min(300, x + 20+50)
        y_min = max(-1000, y-500)
        y_max = min(5000,y+500)
        ax_gamma.cla()
        ax_gamma.plot(curve[x_min:x_max], color='#ea3323', linewidth=1.5)
        ax_gamma.set_xticks([])
        ax_gamma.set_yticks([])
        ax_gamma.set_title(f'Zoom kring position {x}', fontsize=9, pad=4)
        ax_gamma.set_ylim(y_min, y_max)
        fig.canvas.draw_idle()



    def draw(idx):
        ax_curve.cla()
        ax_table.cla()
        ax_table.axis('off')
        
        row    = rows.iloc[idx]
        current_curve[0] = row['value']
        curve  = row['value']
        gender = row.get('gender', 'M')

        ax_curve.plot(curve[50:300], color='#ea3323', linewidth=1.5)
        ax_curve.set_title(f"id={row['row_id']}  ({idx+1} / {len(rows)})")

        core = ANALYTES[:8]
        table_data, row_colors = [], []
        for analyte in core:
            val       = row[analyte.col]
            low, high = get_ref(analyte, gender)
            outside   = not (low <= val <= high)
            flag      = ' *' if outside else ''
            table_data.append([f'{val:.2f}{flag}', f'{low}–{high}'])
            row_colors.append(['#ffcccc', '#ffcccc'] if outside else ['white', 'white'])

        tbl = ax_table.table(
            cellText=table_data,
            rowLabels=[a.name for a in core],
            colLabels=['Värde', 'Ref'],
            loc='center', cellLoc='center'
        )
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor('#dddddd')
            elif r > 0 and c >= 0:
                cell.set_facecolor(row_colors[r-1][c])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.2)
        tbl.auto_set_column_width([0, 1])
        ax_table.set_title('Proteiner', fontsize=10, pad=8)


        status_text.set_text(f'Inloggad som {username}. Fall {idx+1} av {len(rows)}')
        existing = con.execute(
            "SELECT classification FROM classifications WHERE username = ? AND row_id = ?",
            [username, int(row['row_id'])]
        ).fetchone()

        label_names = {0: 'Ingen M-komponent', 1: 'Misstänkt M-komponent', 4: 'Oligoklonalt', 2: 'Lätt avvikande fördelning'}
        existing_str = f"  ·  Annoterad: {label_names[existing[0]]}" if existing else "  ·  Ej annoterad"
        status_text.set_text(f'Inloggad som {username}. Fall {idx+1} av {len(rows)}{existing_str}')

        ax_gamma.cla()
        ax_gamma.plot(curve[190:285], color='steelblue', linewidth=1.5)
        ax_gamma.set_xlim(0, 95)
        ax_gamma.set_title('Inzoomat', fontsize=9, pad=4)
        ax_gamma.tick_params(labelsize=7)

        ax_curve.set_xticks([])
        ax_curve.set_yticks([])
        ax_gamma.set_xticks([])
        ax_gamma.set_yticks([])
        redraw_zoom(curve)
        

    def annotate(label: int):
        row = rows.iloc[current_idx[0]]
        label_names = {0: 'Ingen M-komponent', 1: 'Misstänkt M-komponent', 4: 'Oligoklonalt', 2: 'Lätt avvikande fördelning'}

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

   

    def on_key(event):
        if event.key == '1': annotate(0)
        if event.key == '2': annotate(1)
        if event.key == '3': annotate(4)
        if event.key == '4': annotate(2)
        if event.key == 'left':  go_prev(None)
        if event.key == 'right': go_next(None)

    def on_mouse_move(event):
        
        if event.inaxes == ax_curve and event.xdata is not None and current_curve[0] is not None:
            curve = current_curve[0]
            global x
            global y
            x = int(event.xdata)
            y = int(event.ydata)
            
            redraw_zoom(curve)

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
        
    fig.canvas.mpl_connect('key_press_event', on_key)

    btn_neg.on_clicked(lambda e: annotate(0))
    btn_pos.on_clicked(lambda e: annotate(1))
    btn_oligo.on_clicked(lambda e: annotate(4))
    btn_dev.on_clicked(lambda e: annotate(2))
    btn_prev.on_clicked(go_prev)
    btn_next.on_clicked(go_next)


    draw(0)
    plt.show()
    con.close()
    


show_annotation_gui()
