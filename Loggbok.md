## Vad som gjorts
- Funderat ut hur datan ska behandlas. Slutsats, duckDB för att spara allt i SQL-databas
- Funderat över lättaste möjliga måldata att testa med. Landade i att inledningsvis manuellt klassificera "interpretations" genom att leta efter målsträngar, typ "Inga synliga tecken på M-komponent", manuellt inspektera de som inte kunde klassificeras för att hitta fler formuleringar läkarna använder sig av. Och upprepa tills det inte kändes lönt längre. Av ca 172 000 rader kunde 157 500 klassificeras med denna metod. Vid manuell inspektion av de övriga var det antingen att man var osäker, eller vanligast, att en patient som haft M-komponent nu hade minskade tecken på den. I vetskap om att de tidigare haft en M-komponent påstår man sig därför kunna urskilja en M-komponent. 
- Med den manuellt klassificerade måldatan testade jag göra ett simpelt feed-forward-nätverk, indatan var alltså de 300 datapunkterna i "value", och måldatan var mina binära etiketter. Fick direkt 93% på validation set. Testade lite varianter, bland annat ett några convolution-layers, fler lager, mm, men lyckades inte slå 93%. Men roligt att vara igång och att faktiskt ha tränat ett nätverk!
- Nu ska jag gå igenom beskrivningen av data och fundera ut vilka saker som kan användas som indata och utdata, och göra ett lite mer genomtänkt försök.
- Jag har funderat mer på hur vi kan använda datan på olika sätt enligt ovan. Se här nedan
- Jag har även nogrannare gått igenom mina manuella klassificeringar. Först tänkte jag att ju högre andel jag klassificerade, desto bättre. Men idag gjorde jag en nogrannare analys, och jämförde mina klassificeringar med IGComponent-kolumnen (döpt till m_component i databasen). I många fall hade jag med falska positiva fall, när läkarna av olika skäl visste att det fanns en svag M-komponent, men den syntes inte på kurvan. Oftast patienter som genomgått behandling och tecknena på M-komponent börjar försvinna. Efter nogrann analys av de fall som är oklassificerade så ska antagligen alla vara det. Det vanligaste fallet är att man gjort nån slags tilläggsanalys. Oerhört nöjd att jag kan skriva SQL-queries på datan. Att köra SELECT interpretation FROM protein_data WHERE manual_classification IS NULL GROUP BY interpretation ORDER BY count(*); var mycket smidigt.
- Jag testade att nu återigen köra ett oerhört simpelt nätverk, bokstavligen copy pasteat ur pytorchs quickstart-guide. Trots att jag bara köra vanliga linjära/affina-lager eller vad det nu kallas, får jag ca 95% träffsäkerhet på validationsdatan och tydligare konvergens. Jag tror förbättringen därför beror på att jag gjorde etiketteringen av måldatan bättre. Nu är den riktigt bra. 95,5% på test-datan! Tjohej!
- Testat nu att lägga till delimit_value och delimit_value2 för att få fler features i indatan. Resultat ca 96% på validationsdata och testdata med ett simpelt MLP-nätverk. Med 1D CNN har jag lyckats nå ca 97%! Nu ska jag försöka analysera de fallen som blir fel.

## Min tolkning av datan

### Möjlig indata:
- Value (såklart)
- DelimitValue, matematiskt kan vi tolka det som "extra" features från values.
- DelimitValue2, var de olika komponenterna börjar och slutar, också "extra" features från values.
- PValue, resultatet av en analys från analysis-kolumnen. Kan rimligtvis också betraktas som features. 
- Comment, kommentarer till PValue. Oftast null. Dock i vissa fall står det "IgA kappa eller IgG kappa", vilket är en definitiv M-komponent. 
- Sampleinfo: ifall man i samband med beställningen av fler prover har en kommentar. Svårt att se hur denna ska användas inledningsvis.


### Möjlig måldata:
- Analysis, om vi grupperar efter patient-id, och väljer ut den första analysen per patient, är det spännande att se ifall man valt att beställa fler prover. Istället för att klassificera M-komponent eller inte kan vi klassificera "ska vi beställa fler prover eller inte", eftersom det kanske mer återspeglar verkliga händelseförloppet för en patient.
- IGComponent: kanske bara kolla ifall den är null eller inte?

### Övrig relevant
- Patient ID: denna är viktig, kanske spännande att bara köra på den första analysen per patient?
- sign: spännande att se ifall vissa läkare klassificerar olika andel som M-komponent osv? 
- gender: beter det sig olika för män och kvinnor? 
- age: också spännande. Kanske bara ta med det i indatan direkt?

### Data som inte ger något
- ValueTime, bara 1,2,3..300.
- Reference: referensintervall, redundanta. Allt det relevanta finns redan i Flags och Pvalue.
- Flags: ifall en analys är inom eller utanför referensområdet. Men rimligtvis bör ett nätverk "lära sig" vad som är normalt på egen hand, dvs jag tror inte flags ger så mycket mer utöver det man får från Pvalue. 


### Kommande steg
- Testa att ha "beställde man ytterligare prover?" som måldata.
- Göra ett större nätverk: ta med delimitvalues, ta med protein_value för vissa analyser. Testa mer avancerade lager: faltningar, transformers? och annat kul.