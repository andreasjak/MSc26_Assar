## Vad som gjorts
- Funderat ut hur datan ska behandlas. Slutsats, duckDB för att spara allt i SQL-databas
- Funderat över lättaste möjliga måldata att testa med. Landade i att inledningsvis manuellt klassificera "interpretations" genom att leta efter målsträngar, typ "Inga synliga tecken på M-komponent", manuellt inspektera de som inte kunde klassificeras för att hitta fler formuleringar läkarna använder sig av. Och upprepa tills det inte kändes lönt längre. Av ca 172 000 rader kunde 157 500 klassificeras med denna metod. Vid manuell inspektion av de övriga var det antingen att man var osäker, eller vanligast, att en patient som haft M-komponent nu hade minskade tecken på den. I vetskap om att de tidigare haft en M-komponent påstår man sig därför kunna urskilja en M-komponent. 
- Med den manuellt klassificerade måldatan testade jag göra ett simpelt feed-forward-nätverk, indatan var alltså de 300 datapunkterna i "value", och måldatan var mina binära etiketter. Fick direkt 93% på validation set. Testade lite varianter, bland annat ett några convolution-layers, fler lager, mm, men lyckades inte slå 93%. Men roligt att vara igång och att faktiskt ha tränat ett nätverk!
- Nu ska jag gå igenom beskrivningen av data och fundera ut vilka saker som kan användas som indata och utdata, och göra ett lite mer genomtänkt försök.

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
- Undersöka hur väl IGComponent överrensstämmer med min manuella klassificering. 