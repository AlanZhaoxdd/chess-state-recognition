"""Chess game state recognition from a single image.

Shared code for the whole team. The modules are deliberately small:

* :mod:`chessrec.config`   -- paths and the constants every stage agrees on
* :mod:`chessrec.fen`      -- class names and FEN <-> 8x8 grid conversion
* :mod:`chessrec.data`     -- loading rendered boards and their labels
* :mod:`chessrec.geometry` -- perspective warp and per-square cropping
"""

__version__ = "0.1.0"
