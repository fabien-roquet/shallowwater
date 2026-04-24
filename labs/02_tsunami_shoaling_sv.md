# Labb 02 - Tsunami över havet: hastighet, uppgrundning och svamplager

**Baserad på:** `notebooks/02_tsunami_shoaling.ipynb` och de tekniska skripten `scripts/03_tsunami.py` samt `scripts/03_tsunami_shoaling_sponge.py`.

## Berättelse

En tsunami kan vara låg ute på djupt hav, men ändå färdas mycket snabbt. När vågen närmar sig kusten blir vattnet grundare. Då bromsas vågen upp och vattenytan kan höjas. I modellen använder vi också ett **svamplager** vid den öppna kanten. Svamplagret dämpar vågor så att de inte studsar tillbaka på ett orealistiskt sätt.

Den här labben visar **uppgrundning**. Den är inte en riktig översvämnings- eller varningsmodell.

## Lärandemål

Efter labben ska du kunna förklara att:

- långa vågor rör sig ungefär med hastigheten `sqrt(g H)`, där `H` är vattendjupet,
- vågor går snabbare på djupt vatten än på grunt vatten,
- våghöjden kan öka när vågen kommer in över grundare vatten,
- ett svamplager kan minska konstgjorda reflektioner i en numerisk modell.

## Frågor att fundera på

- Varför är tsunamin snabbast ute på djupt vatten?
- Var blir vågen högst i experimentet?
- Vad händer om kusten görs ännu grundare?
- Vad händer om svamplagret tas bort?
- Varför visar modellen inte hur långt vattnet rinner upp på land?

## Föreslaget experiment

1. Skapa en bassäng med djupt hav till väster och en grund kust till öster.
2. Starta med en bred höjning av vattenytan ute på djupt vatten.
3. Kör modellen och följ vågen när den rör sig mot kusten.
4. Jämför vågens fart i djupt och grunt vatten.
5. Mät den största våghöjden längs x-led.
6. Kör om experimentet med ett annat kustdjup.
7. Testa till sist att stänga av svamplagret och leta efter reflektioner.

## Elevblad

Fyll i medan du arbetar:

| Fråga | Svar |
|---|---|
| Vilken parameter ändrade du? | |
| Vad trodde du skulle hända? | |
| Vad hände faktiskt? | |
| Vilken figur eller mätning visar det tydligast? | |
| Hur skulle du förklara resultatet för någon yngre? | |

## Mätning

Ett enkelt mått är den största absoluta vattenståndsändringen vid varje x-position:

```python
eta_stack = np.stack(out["eta"], axis=0)
max_abs_eta_x = np.max(np.abs(eta_stack), axis=(0, 1))
```

Rita sedan `max_abs_eta_x` tillsammans med bottendjupet. Titta efter om vågen växer där botten blir grundare.

## Utmaning

Kör tre experiment med olika kustdjup, till exempel:

```python
H_coast = 200.0
H_coast = 80.0
H_coast = 30.0
```

Vilket fall ger störst våg nära kusten? Skriv en kort förklaring med orden **djup**, **hastighet** och **uppgrundning**.

## Läraranteckningar

För yngre elever är animationen viktigast. Börja med att låta dem beskriva vad de ser: vågen är låg och snabb ute på djupt vatten, men förändras nära kusten. För äldre elever kan man koppla observationerna till formeln `c = sqrt(g H)` och diskutera varför detta fortfarande är en förenklad modell utan vågbrytning, friktion mot land eller översvämning.
