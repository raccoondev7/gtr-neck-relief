# Guitar Neck Relief Calculator

Interactive Python app for visualising **guitar neck relief** and **string clearance** (single string, side view).

It draws:

* Neck curve (bow from **nut → 17th fret**, rigid after that)
* Frets as vertical poles (numbered)
* String as a straight equilibrium line (open or fretted)
* A vibration “worst-case” envelope based on a **modal plucked-string peak** model
  (pick point fixed at **105 mm from the bridge**)

It also renders a **clearance table** showing the **minimum clearance during pluck** at each fret. Any clearance **below 0 mm** is highlighted in light red.

---

## Features

* ✅ Single-string lateral (side) view
* ✅ Open string or fretted (select pressed fret)
* ✅ Neck bow controlled by “headstock lift” (affects nut→17th region)
* ✅ Fixed pick location: **105 mm from bridge**
* ✅ Auto-scaled vibration amplitude as speaking length shortens
* ✅ Per-fret minimum clearance table with buzz-risk highlighting
* ✅ Slider tick scales + Reset button

---

## Requirements

* Python 3.9+ recommended
* `numpy`
* `matplotlib`

Install dependencies:

```bash
pip install numpy matplotlib
```

---

## Run

```bash
python gtr-relief.py
```

---

## Controls (sliders)

* **Headstock lift (mm)**
  Controls the neck bow (only from nut → 17th fret).

* **Pressed fret (0=open)**
  `0` = open string, `1..N` = fretted note.

* **Base dev@pick (mm, open)**
  Baseline pluck displacement at the pick point for the open string.
  Auto-scales for higher frets (shorter speaking length).

* **Nut clearance (mm)**
  String height above fret tops at the nut anchor.

* **Bridge clearance (mm)**
  String height above fret tops at the bridge anchor.

* **Neck thickness / Fret height (visual)**
  Visual-only parameters for readability.

---

## Notes on the model (important)

This is a **simplified geometric + vibration envelope model**:

* Frets are assumed perfectly level.
* String equilibrium is straight segments (nut/fret/bridge).
* Neck relief is a simple curve ending at the 17th fret.
* The vibration envelope is computed as the **peak displacement over time** from a modal sum.
* “Bridge clearance” and “nut clearance” are defined **relative to fret-top height** in the model, not hardware saddle height.



---

## Screenshot



```md
<img width="3657" height="1943" alt="image" src="https://github.com/user-attachments/assets/bd74530a-eac4-4a4a-af68-244d074f1f12" />

```

---

## License
MIT
