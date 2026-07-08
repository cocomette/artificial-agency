## Component Guidance
- When present, the `Frame components` block is deterministic structured data generated from the same observation frames attached to this request.
- Components are 4-connected regions of the same grid symbol. Diagonal contact alone does not join components.
- Each row groups components with the same grid symbol and exact same shape. `nb` is the number of components in this group, and `box` lists their bounding boxes normalized from 0 to 1000, with x increasing right and y increasing down.
- Use components as compact evidence for element location, size, and symbol color, but the attached images remain the source of truth.
