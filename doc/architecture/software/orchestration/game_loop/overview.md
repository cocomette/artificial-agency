# Game Loop Overview

The game loop is frame-unrolled. A real ARC observation bundle can produce
multiple retained frame turns. Only the final retained frame for a bundle may
submit an environment action; earlier retained frames use internal `NONE`.

The loop stores every retained turn when memory is enabled so the dashboard can
inspect animation frames, real transitions, learner traces, and replay stats.
