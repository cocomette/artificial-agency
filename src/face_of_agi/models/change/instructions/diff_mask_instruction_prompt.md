# Diff Mask Inputs

- The attached image sequence interleaves original observation frames with binary changed-pixel masks.
- For every consecutive pair of original frames, one diff-mask image is inserted
between them.
- Diff-mask images use black pixels for unchanged areas and white pixels for
areas that changed between two consecutive frames. This is deterministic, if you see white pixels, something changed for sure, you must understand what and link it to the elements you observe within the original frames.
