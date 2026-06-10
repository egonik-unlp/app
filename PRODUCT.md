# Product

## Register

brand

## Users
Curious listeners and the people who built the engagement model. They arrive with
two songs in mind and a question: *what does the space between them sound like?*
Context is exploratory and unhurried — one person, full screen, late evening,
poking at their own taste. Not a dashboard they live in; a place they visit.

## Product Purpose
Make an abstract ML artifact legible. The sibling project compresses every track
into a 64-dim "song-AE" latent and runs A* between two points, weighting cosine
distance, learned listening transitions, time-of-day context, and a rotation-fit
score. This app renders that journey in 3D so a non-expert can *see* the route
cross the taste map, watch it thread between genre clusters, and understand that
the path is chosen, not random. Success = a person says "oh, I get it" without
reading docs, and wants to trace another pair.

## Brand Personality
Observatory, not arcade. Wonder, precision, calm. Three words: luminous,
deliberate, deep. The interface recedes; the star map and the route carry the
moment. Copy is plain and curious, never hype.

## Anti-references
- Generic "AI dashboard": card grids, KPI tiles, gradient-on-everything.
- Spotify-clone green chrome, or neon cyberpunk arcade.
- Glassmorphic panels stacked for decoration.
- A black void with floating unexplained dots (the failure mode of 3D demos):
  every visual element must be labeled or legible in the legend.

## Design Principles
- **The map is the message.** Chrome is minimal and gets out of the way of the 3D space.
- **Nothing unexplained.** Color, motion, and links each mean one thing, stated in a legend.
- **Motion reveals structure**, not decoration: the tracer shows direction, depth fog shows distance, clusters show genre.
- **Honest about uncertainty.** When an arbitrary track is snapped to a nearby in-corpus anchor, say so plainly.

## Accessibility & Inclusion
Target WCAG AA for all 2D UI (body text ≥4.5:1 on the dark surface). Genre hues
vary in lightness as well as hue so clusters remain distinguishable for color
vision deficiency; color is never the only signal (labels + legend back it).
Full `prefers-reduced-motion` path: tracer, auto-rotate, and entrance transitions
fall back to static/instant.
