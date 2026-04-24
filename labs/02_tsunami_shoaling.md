# Lab 02 - Tsunami Across the Ocean: Speed, Shoaling, and Sponge Layers

**Based on:** `notebooks/02_tsunami_shoaling.ipynb`, `scripts/03_tsunami.py`, and `scripts/03_tsunami_shoaling_sponge.py`

## Story

A tsunami can be low in the deep ocean but grow near the coast as the water becomes shallow. A sponge layer is used at the far boundary so outgoing waves do not bounce back unrealistically.

## Learning goals

- Tsunami speed is approximately sqrt(g H).
- Long waves slow down in shallow water.
- The wave height can grow as the wave shoals.
- Sponge layers reduce artificial reflections.

## Guiding questions

- Why is the tsunami fast in deep water?
- Where does the wave become tallest?
- What changes when the sponge layer is turned off?

## Suggested experiment

1. Create a sloping bottom that is deep offshore and shallow near the coast.
2. Start with a broad sea-surface bump in deep water.
3. Track the wave as it approaches the coast.
4. Compare runs with and without sponge damping.

## Source material

- `notebooks/02_tsunami_shoaling.ipynb`
- `scripts/03_tsunami_shoaling_sponge.py`

## Student worksheet

Fill these in while you work:

- What parameter did you change?
- What did you expect to happen?
- What actually happened?
- What simple measurement supports your conclusion?

## Challenge

Try three different coastal depths. Which one gives the largest coastal wave?

## Teacher notes

Keep the first run small enough to execute quickly. For outreach use, emphasize the animation first and the equations second. For older students, ask for one measured quantity and one comparison with theory.
