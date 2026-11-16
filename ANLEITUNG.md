# Apollon – Überblick und Anleitung

```text
                          ..  .........................................  ..
              ..........................,.,.,.,.,,.,.,.,......................... .
          ..........,,,,,,,,,,,,,,,,:,,,,,,:,:,:,,,,,,,,,,,,,,,,,,,,,,,,,,,............
        ......,..,,,::,,::,::,,::::::::;::;;;:;;;;r;:;rr;:iri;i;;;:;:::;;,,:;:,,,,,.....
      ......,,,,,,:::::;;;:;;ii:irXrrisX2Xsss222XXs22h22h22sX2552Xs223XXissirrri;::,,,.....
     .....,,:,:,;;i;;isXrsr3shrrMXh235552hXhrshX5sX525r53s3sXXX3r55s32X2shhr2XXXrXs;;::,....
    ...,,,::;i;s2X23hX53Xsh2s3h2sXXsXXsirXiiiriiii;irri;irr;;rri;;ir;;iirririiiiiii;;::,,....
   ...,::;;irssXssrXssirisii;i;rriiirii;i;;;;;:;:;:,:,:,:::::::;i;iiiiii;;;;;;;;:;;;;:::,.....
  ..,,,:;;ii;iisriirr;i;ii;iriii;;;:::;;;irrsssssXsssrssrrsssrrrii;:;:;;;;;i;iiii;;;;;::,,,...
  ...,,:,;;:i;i;i;i;ririr;i;i;iisrXss2X2X2rXs;iii;i;,;i;iir:XXsisXXisssri;ii;i;ii;;;;:::,,,....
  ...,,,,::;:i;i;iii;i;;;irssXXiXsi::,:,,::i;rr52rXXrri;i;HH32Xsi;:,:;irXX;srri;;:;;;;:::,,,...
   ...,,::::;;;;;i:i;rsrrrs;,,;r2HHHH;irsXM###h52XXX3325XriiGGGGGGHHHHM5s;,:;iirii;;:;::,,,,...
   ....,,::::;:;:iirii:,:r2HMHHGGSGSis25GH&@@9hr:,,,55hhh3rXr###SSSSGGHHHMMMM2i:,;;;::::,,.....
   ...,,,,:::,:;;i:,i2hMHHGGGSSSS###ssX33h9Gii;:,,,,,2XHHh5Xs999#9####SSGGGHHMMhh3i.:,:,,,....
   .....,,:::;:,;XMhMHHGGGSS##9##999sX23M3hh,,,,,,,,,5shHs2rs999999#9####SSGGHHMMhh3:,:,,.,...
   ...,,,,::,:2MMMHHHGGSS###999999999XX52hMG23,,,,,X5532H52X99999999#####SSSSSS;;;;::,,,,,,...
   .....,,::::MHHHGGSSSS####9999999999rX22533MMMMhhMhMM222s9999999####;i;iiiiii;i;;::::,,,,....
   ...,,,,::;:;;iiriiriiS###9###999999B9sX52hh32Xh3h33XXs9iriirrrrssrrrriri;;;;:;;::::,:,,.....
   ...,,,::::::;:;iiirissrsssssXXsXssssXssXsXssXsXssXsXsssssrsiiriiii;;i;i;i;ii:;:;::::,,,,...
  ....,,,,:::::;;i;;;ii;;iiii;iirriiirriiirrriiirrii;iiri;;ii;;;iii;;iiii;;iii;;;;;;::,:,,,....
   ...,,,,::::;;;;;;;;;;ii;;ii;;iii;;ii;;ii;;ii;;ii;;ii;;ii;;ii;;ii;;i;;ii;;ii;;;::;::,:,,,....
   .....,:::::;;;;;;:;:;:;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;:;i;i;i;;i;i;:;;;;;;:;;;;;:;:,,,,,,...
   .....,,,:,:::::;:;:;::;:;;;;;;;;:;:;::;:;:;;;;;;;;;;;;:;:;:;:;:;:;::;:;:;:;:;:;,:,:,,.,...
    ....,,,,,,,,:::;;::;;::;;::;;::;;::;;::;;::;;::;;:;;;:;;;:;;::;;::;;:::;::::,,::,,,,....
     ....,,,,,,::,:::::;;::::;:;;:;::;:;;:;::;:;:;;:;::::;:;::;:;::::;:;:;;::::::,,,,,.,.....
      ......,,,,::,,:::::;::;;::;;::;;:::::::::;::;;::::::::;;::::;;:::::::,,:::,,,,,,....
        ......,,,,,,,:,::,:,::,::::::,,::::::,,:::::::,::::,,::::,::,::,::,,,,,,,,,......
          .......,...,,,,,,,:,,,,,,,,:,,,,::,,,,::,,,,:,,,,,:,,,,,,,,,,,,,,,,...........
```

Das Auge ist das Logo von Apollon. In der Web-Oberfläche steht es als Punktbild oben in der Seitenleiste und blinzelt
dort gelegentlich; solange eine Analyse rechnet, erscheint es groß aus Zeichen auf der Bühne. Es ist nur in Weiß und
Grau auf Schwarz gezeichnet, weil Farbe in der Oberfläche den Daten vorbehalten ist. Der Code dazu liegt in
[web/src/Eye.tsx](web/src/Eye.tsx).

Dieses Dokument erklärt, was Apollon ist, wie man es startet und wie man die Oberfläche benutzt. Die englische
[README.md](README.md) beschreibt den Rechenkern im Detail, [FUNCTIONS.md](FUNCTIONS.md) die einzelnen Analysen und
[FINDINGS.md](FINDINGS.md) die bisherigen Befunde.

## Was Apollon ist

Apollon ist eine lokale Forschungssoftware. Man gibt ihr ein vorhandenes Sprachmodell und einen Prompt, wählt ein
Token der Antwort und kann dann verfolgen, wie das Modell zu genau dieser Vorhersage kommt: welche Teile dazu
beitragen, wovon sie abhängen und was sich ändert, wenn man gezielt eingreift.

Bei „Was ist 2+2?“ lässt sich leicht prüfen, ob ein Modell „4“ ausgibt. Apollon ist für die Fragen danach gedacht:

- Welche Schichten machen die „4“ wahrscheinlicher?
- Welche früheren Berechnungen sorgen dafür, dass diese Schichten richtig arbeiten?
- Was sagt das Modell, wenn man einen dieser Beiträge abschaltet?

Diese Fragen können unterschiedliche Antworten haben. Ein Teil kann kaum direkt für „4“ sprechen und trotzdem
entscheidend sein, weil spätere Schichten lesen, was er schreibt. Apollon misst deshalb für jeden Teil zwei Dinge:

| Größe | Bedeutung |
|---|---|
| Direkter Beitrag | Was der Teil für sich allein zum Ziel-Token beiträgt, ausgelesen über die letzte Norm und die Ausgabematrix. |
| Gesamteffekt | Wie sich die Vorhersage ändert, wenn der Teil entfernt und das Modell ab dieser Schicht neu gerechnet wird. |

Die Differenz ist der indirekte Effekt über spätere Schichten. Alle Werte sind Änderungen der Log-Odds des
Ziel-Tokens: mit dem Teil minus ohne ihn.

Apollon lädt Gewichte von der Festplatte (GGUF, etwa aus Ollama, oder Safetensors aus dem Hugging-Face-Cache). Der
Rust-Kern rechnet den Modelllauf selbst und kann deshalb Zwischenzustände aufzeichnen, an einer Schicht eingreifen und
von dort weiterrechnen. Es wird nichts trainiert. Jeder Lauf prüft sich selbst, zum Beispiel ob sich die Vorhersage aus
den zerlegten Beiträgen wieder zusammensetzen lässt.

## Die vier Analysen

| Analyse | Frage | Kommandozeile |
|---|---|---|
| Zerlegen | Was tragen Embedding, jede Attention und jedes MLP zur Vorhersage bei, direkt und insgesamt? | `--decompose` |
| Probe | Welche einzelnen Neuronen, Köpfe oder Dimensionen eines Blocks zählen? Sie werden einzeln und in Gruppen abgeschaltet, verstärkt oder mit Werten aus einem Kontrastprompt ersetzt. | `--probe "S12 MLP"` |
| Trace | Warum war dieses Neuron hier aktiv? Die Aktivierung wird bis auf einzelne Skalarprodukte zurückgerechnet, über frühere Neuronen und über die Positionen, die Köpfe gelesen haben. | `--trace "S12 #2328"` |
| Statisches Neuron | Was liest ein Neuron, wohin schreibt es, nur aus den Gewichten und ohne Prompt? | `--neuron "S12 #2328"` |

Ein Atlas (`--atlas`) speichert für alle Neuronen die stärksten Gewichtsverbindungen und kann von einem Trace
mitbenutzt werden.

## Installieren und starten

Voraussetzungen: Rust (stable) und Node.js.

```sh
cd web
npm ci
npm run build
cd ..
cargo run --release -- --serve
```

Danach läuft die Oberfläche unter http://127.0.0.1:8765. Unter Windows mit der MSVC-Toolchain:

```powershell
cargo +stable-x86_64-pc-windows-msvc run --release --target x86_64-pc-windows-msvc -- --serve
```

Der Server hört nur auf dem eigenen Rechner. Mit der Umgebungsvariable `APOLLON_PORT` läuft er auf einem anderen
Port. Läufe und Berichte liegen in `output/web-runs/` und werden beim Neustart wieder geladen.

Ohne Oberfläche geht alles auch direkt:

```sh
apollon -m MODELL --decompose "Was ist 2+2?"
apollon -m MODELL --probe "S12 MLP" --contrast "Was ist 2+3?" "Was ist 2+2?"
apollon -m MODELL --trace auto "Was ist 2+2?"
apollon --list
```

`apollon --help` zeigt alle Optionen. `--chat` setzt den Prompt in das Chat-Format des Modells, `--answer N` wählt das
N-te Antwort-Token, `--result-json DATEI` schreibt das Ergebnis strukturiert heraus.

## Die Oberfläche benutzen

Der Ablauf ist: Modell wählen → Prompt ausführen → Antwort-Token anklicken → Beiträge ansehen → auffälligen Teil
gezielt testen.

1. **Neue Untersuchung.** „New investigation“ öffnet das Formular: Modell, Prompt, optional das Chat-Format. Apollon
   erzeugt zuerst die Antwort.
2. **Token wählen.** Über der Bühne stehen Prompt und Antwort als Token. Ein Klick auf ein Antwort-Token zerlegt
   dessen Vorhersage. Die Position, von der aus das Modell vorhersagt, ist unterstrichen.
3. **Teil auswählen.** Ein Klick auf einen Punkt, eine Zeile oder einen Balken wählt einen Teil aus. Der Rest wird
   gedimmt, das Panel zeigt alle Werte.
4. **Eingreifen.** Im Panel starten Knöpfe den nächsten Schritt: die Einheiten einer Komponente einzeln abschalten
   (Probe), ein Neuron zurückverfolgen (Trace), seine statische Ansicht öffnen oder Werte aus einem Kontrastprompt
   einsetzen. Jeder Folgelauf hängt in der Seitenleiste unter dem Lauf, aus dem er gestartet wurde.
5. **Vorher und nachher.** Das Panel zeigt die Ausgabe des echten Laufs neben der Ausgabe ohne den gewählten Teil und
   sagt, ob die Antwort kippt.

### Ansichten

| Ansicht | Zeigt |
|---|---|
| Space | Eine drehbare 3D-Szene, je nach Analyse eine andere (siehe unten). |
| Flow | Schichten von links nach rechts, eine Zeile pro Komponententyp, Linien zum Ziel-Token. |
| Units | Eine Kachel pro Komponente, so groß wie ihr Effekt, darin ein Punkt pro Einheit. |
| Table | Jeder gemessene Teil als Zeile, nach jeder Spalte sortierbar. |

Die Space-Ansicht:

- **Zerlegung:** Schichten von links nach rechts, Token-Positionen von vorn nach hinten, die Höhe ist der Effekt auf
  das Ziel-Token. Köpfe zeigen als Bögen, was sie von welcher Position holen, und als Bodenlinien, wohin sie schauen.
- **Probe:** links alle Einheiten der Komponente; von den genauer untersuchten läuft ein Pfad durch die späteren
  Schichten und zeigt, was ihr Fehlen dort ändert.
- **Trace:** der Rückwärtsbaum vom Neuron zu früheren Neuronen, Köpfen, deren Lesepositionen und dem Input.
- **Statisches Neuron:** links, was es liest, rechts, wohin es schreibt; Höhe und Tiefe sind seine beiden
  Koordinaten p und q.

Ein Punkt steht immer für eine Einheit. Einheiten, die nicht einzeln gemessen wurden, sind klein und gedimmt. Unter
jeder Szene steht, was gezeichnet und was weggelassen wurde.

### Werkzeuge

| Werkzeug | Bedienung |
|---|---|
| Filter | Knopf „Filter“ neben der Suche: Schicht-Bereich und Mindestgröße des Effekts. Alle Ansichten folgen. |
| Markieren | In Flow ein Rechteck aufziehen, in der Tabelle Shift oder Ctrl. Markierte Teile leuchten überall. |
| Vergleich | „Compare with“ im Panel: Werte eines zweiten Laufs neben jedem Teil, als Spalte in der Tabelle und als weiße Punkte in Space. |
| Anpinnen | Knopf „Pin“ oder Taste P. Pins werden in jedem Lauf über Schicht, Typ und Einheit wiedergefunden. |
| Direct / Total | Schaltet zwischen direktem Beitrag und Gesamteffekt um. |
| Link | Die Adresse enthält Lauf, Ansicht, Metrik und Auswahl; ein Befund lässt sich so weitergeben. |

Tasten: ← → durch die Schichten, ↑ ↓ durch die Rangliste, Enter startet die Probe der gewählten Komponente, T den
Trace eines Neurons, P pinnt, Esc löscht die Auswahl, Ctrl+K springt in die Suche.

## Aufbau des Projekts

| Ort | Inhalt |
|---|---|
| `src/model.rs`, `tokenizer.rs`, `gguf.rs`, `quant.rs`, `tensor.rs`, `blas.rs` | Laden der Gewichte und der Modelllauf |
| `src/decompose.rs`, `probe.rs`, `trace.rs`, `inspect.rs`, `neuron.rs`, `atlas.rs` | Die Analysen |
| `src/export.rs` | Strukturierte Ergebnisse (JSON, Schema-Version 2) |
| `src/server.rs` | Lokaler Webserver; jede Analyse läuft als eigener, abbrechbarer Prozess |
| `web/src/api.ts`, `store.ts` | Typen, Aufrufe und Zustand der Oberfläche |
| `web/src/panel/`, `Sidebar.tsx` | Formular, Details zum gewählten Teil, Untersuchungen |
| `web/src/stage/`, `scene.ts` | Bühne: Prompt-Leiste, Flow, Units, Tabelle, Filter, 3D-Renderer |
| `web/src/space/` | Die vier 3D-Szenen |
| `web/src/noir.css`, `style.css` | Gestaltung: Schwarz, weiße Haarlinien, Farbe nur in den Daten |
| `tests/web_smoke.py` | Prüft die Web-API gegen ein echtes Modell |

Die Web-API: `GET /api/models`, `GET` und `POST /api/runs`, `GET` und `DELETE /api/runs/{id}`,
`POST /api/runs/{id}/cancel`, `GET /api/runs/{id}/artifacts` und `.../artifacts/{name}`.

## Prüfen

```sh
cargo test --release
python tests/web_smoke.py MODELLPFAD http://127.0.0.1:8765
```

Der Smoke-Test startet echte Läufe und prüft auch die Zahlen gegeneinander: Attention-Zeilen summieren zu 1, eine
Dosis von 0 entspricht dem Abschalten, die Quellgruppen eines Neurons ergeben zusammen seine Aktivierung.

## Grenzen

- Jede Aussage gilt für einen Modelllauf und einen Eingriff. Erzeugt das Abschalten eines Blocks nur Unsinn, zeigt
  das, dass er für diesen Lauf wichtig war; welche Funktion er hat, braucht gezieltere Eingriffe und weitere Prompts.
- Ob ein bei „2+2“ gefundener Mechanismus auch bei anderen Additionen greift, ist eine eigene Frage, die man mit
  Apollon untersuchen kann.
- Die Oberfläche ist bisher nur mit einem kleinen Modell (SmolLM2-135M, 30 Schichten) durchgeprüft. Bei großen
  Modellen werden die Szenen dichter und die Ergebnisse größer.
- Ergebnisse von mehreren Megabyte können für Programme abbrechen, die jede Verbindung sofort schließen
  (`Connection: close`). Der Browser ist davon nicht betroffen.
