;; O que se segue é um "manifesto" equivalente à linha de comando que você deu.
;; Você pode armazená-lo em um arquivo que você pode então passar para qualquer
;; comando 'guix' que aceite uma opção '--manifest' (ou '-m').

(specifications->manifest
  (list "python"
        "python-bokeh"
        "python-colorcet"
        "python-flask"
        "python-numpy@1"
        "python-werkzeug"
        "python-astropy"
        "python-astroquery"
        "python-photutils"
        "python-statsmodels"
        "python-pandas"
        "python-openpyxl"
        "gunicorn"
        "python-xlrd"))
