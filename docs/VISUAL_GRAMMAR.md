# Figure visual grammar

Use this style for research figures in this project. It records the gray and green palette used for the 1M parameter paper figures. Keep the plotted measurements and axis scales faithful to the saved metrics.

## Color and hierarchy

| Role | Color |
| --- | --- |
| Canvas | white `#FFFFFF` |
| Main text and zero lines | dark slate `#233042` |
| Comparators and secondary conditions | gray `#89919e` |
| Condition emphasized by the panel | green `#087f88` |
| Grid | light gray `#e8edf1` |
| Visible axis spines | gray `#cbd4dc` |

Green marks the contrast being tested. In the reproduction curves, Ordinary NCA is green against gray Scratch and Shuffled NCA. In the confirmation figure, Designed NCA is green against gray Ordinary NCA. Label conditions directly or in a small legend so this change of emphasis is clear. Do not add a new color when a marker or line style can distinguish curves.

## Marks used in the paper

| Figure | Condition | Mark |
| --- | --- | --- |
| Reproduction curve | Scratch | solid gray line, filled triangle |
| Reproduction curve | Ordinary NCA | solid green line, filled circle |
| Reproduction curve | Shuffled NCA | dashed gray line, open square; draw last so overlap remains visible |
| Recurrence replication | high recurrence | green diamond |
| Recurrence replication | Ordinary NCA | open green circle |
| Recurrence replication | low recurrence | gray square |
| Confirmation | Ordinary NCA | gray circle |
| Confirmation | Designed NCA | green diamond |

Use thin lines: about 0.8–1.4 pt for individual curves and up to 1.9 pt for an overall mean. Keep markers about 3–5.5 pt. Show individual model runs as small, faint points (about 0.4 alpha); thin paired connectors can be fainter (about 0.28). Keep corpus means prominent. Use no broad opaque bars or decorative fills. When showing uncertainty, use a light band or narrow whiskers in the condition color.

## Layout and labeling

- Use a white background, light gridlines, dark text, and only left and bottom spines. Put the grid behind the data.
- Keep axis titles short and include units. Use `TinyStories tokens (M)` for downstream token counts and `Validation loss` for loss.
- Show the full raw learning curve as the main view. An inset may enlarge a narrow region, but it must keep the same raw values and label its x range.
- For signed gains, state the subtraction in the label or caption and draw a thin zero line. Positive gain means lower validation loss for the selected condition.
- Use line styles and markers as well as color. Keep legends compact and out of dense data regions.
- Give each panel its own short caption beneath it. Put methods and statistical detail in the manuscript text, not inside the plot.

## Statistical and export rules

Read values from saved metrics. Never hardcode an interim result into a final plot. For model-seed summaries, state whether whiskers are sample SD. For the fresh-corpus confirmation, average model seeds within each corpus replicate before showing the overall mean; seed dots are within-corpus variation, not extra independent corpora. Preserve the logged evaluation milestones and do not choose a zoom or target loss to exaggerate a difference.

Export both PDF and SVG with a tight bounding box and modest padding (`bbox_inches="tight", pad_inches=0.08`). In Matplotlib, use `pdf.fonttype = 42` and `svg.fonttype = "none"` so text remains editable. Check the result at the size used in the paper before committing it.
