# Hair_Guard_Fixer
I developed this fast object‑protection module called Hair_Guard_Fixer for depth‑estimation models. Its purpose is to correct depth‑map artifacts directly inside the model pipeline before the maps are exported. The module is primarily optimized for Depth Anything and AnyDepth SDT.

The file hair_guard_fixer.py delivers stronger results and works especially well when combined with inpaint.
The smooth version is designed for workflows that do not use inpaint, and it is also suitable for real‑time 3D conversion for gamers, thanks to its stable foreground contours and low processing overhead. In some scenes the improvement may be subtle, but it maintains consistent depth continuity.

Although the module is named Hair_Guard_Fixer—because it was originally created to fix hair‑related distortions—it has proven effective at refining the entire foreground silhouette, improving boundary accuracy across all objects, not just hair.
