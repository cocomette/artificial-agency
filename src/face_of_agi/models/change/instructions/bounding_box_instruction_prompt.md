## Bounding Box Guidance
- Images have bright pink (rgb(255, 0, 255)) thick outlines around areas where pixels changed across the attached frame sequence. 
- You must focus on these outlined areas this is exactly where things have changed between 2 consecutive frames.
- You must figure out what moved or rotated or changed color or transformed in any other way. These outlined are drawn around absolute pixel changes, they may not wrap the full object.
- Never say that something did not change if you sees the pink outlines, something changed for sure and you must see the changes and include it in the `elements` mutations.
- The outlines are visual helper overlays, but you must not mention them in `elements` because they are not part of the game.
- You never mention the pink outlines in element names, descriptions, or mutations.
