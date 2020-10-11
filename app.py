from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.events import Tap
from bokeh.models import ColumnDataSource, DataTable, TableColumn, PointDrawTool, Spinner, WheelZoomTool, RadioGroup,\
    CustomJS, Paragraph, Button, FileInput
from bokeh.layouts import column, row

import colorcet as cc

from flask import Flask, flash, render_template, request, redirect, url_for
from flask import send_from_directory, jsonify, make_response
from werkzeug.utils import secure_filename
import os

from astropy.io import fits
from astropy.visualization import HistEqStretch

import numpy as np

from pybase64 import b64decode


UPLOAD_FOLDER = './upfolder'
ALLOWED_EXTENSIONS = ['fit', 'fits','corr']


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def normal(valores):
    import numpy as np

    b = np.max(valores)
    a = np.min(valores)

    return (valores - a)/(b-a)


@app.route("/")
def interface():
    
    return render_template("interface.html")


@app.route('/corfit', methods=['POST'])
def upload_cor():

    req = request.get_json()
    print(req['url'])
    # file = request.files['file']

    # if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # return 3 # redirect(url_for('plotfits',filename=filename))

    return request.files


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')

            return redirect(request.url)

        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            return redirect(url_for('plotfits',filename=filename))

    return redirect(url_for('interface'))


@app.route('/plot/<filename>')
def plotfits(filename):
    
    with fits.open('upfolder/'+filename) as f:
        img = f[0].data
    
    stretch = HistEqStretch(img) # Histograma, melhor função para granular a imagem
    h,w = img.shape # número de linhas e colunas da matriz da imagem

    # Fonte que será usada para fazer computação
    source = ColumnDataSource(dict(x=[], y=[], tipo=[]))

    # Constrói a tabaela de dados que poderá ser usada para designar as posições do objeto, estrela e céu
    tabela = DataTable(source=source,columns=[
        TableColumn(field='x',title='x'),
        TableColumn(field='y',title='y'),
        TableColumn(field='tipo',title='tipo')
    ], editable=True)
    
    # Plota a imagem do arquivo fit
    p = figure(plot_width=w, plot_height=h,tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],\
        active_scroll='wheel_zoom')
    p.image(image=[stretch(normal(img))], x=0, y=0, dw=w, dh=h, palette=cc.CET_CBL2, level="image")
    p.x_range.range_padding = p.y_range.range_padding = 0
    p.grid.grid_line_width = 0

    # Os círculos que serão inseridos
    c = p.circle('x','y', source=source, color='red', fill_color=None, radius=8, line_width=2)
    cd = p.circle_dot('x','y', source=source, color='red', size=2)
    tool = PointDrawTool(renderers=[c,cd],empty_value='obj')
    p.add_tools(tool)
    p.toolbar.active_tap = tool

    # Muda o raio da abertura fotométrica
    spinner = Spinner(title="Raio", low=1, high=40, step=0.5, value=8, width=80)
    spinner.js_link('value', c.glyph, 'radius')

    # Selecionar o tipo de fonte luminosa: obj, src ou sky
    radio_title = Paragraph(text='Escolha o tipo:')
    LABELS = ['obj','src','sky']
    radio_group = RadioGroup(labels=LABELS, active=0)
    # radio_group.js_link('labels',tool,'empty_value',attr_selector='active')
    callback = CustomJS(args=dict(tool=tool), code='''
           tool.empty_value = cb_obj.labels[cb_obj.active]
           tool.change.emmit()
    ''')
    radio_group.js_on_change('active',callback)

    # Fazer o upload do corr.fit da imagem
    upload_cor = FileInput()
    upload_cor.js_on_change('value',CustomJS(code = '''

            var entry = {
                url: `${window.location.href}`,
                arquivo: atob(cb_obj.value),
                }
            console.log(entry)

            fetch(`${window.origin}/corfit`, {
                method: "POST",
                credentials: "include",
                body: JSON.stringify(entry),
                cache: "no-cache",
                headers: new Headers({
                    "content-type": "application/json"
                })
            })
            .then(function (response) {
                if (response.status !== 200) {
                    console.log(`Looks like there was a problem. Status code: ${response.status}`);
                    return;
                }
                response.json().then(function (data) {
                    console.log('Acabaou de chegar: ', data);
                });
            })
            .catch(function (error) {
                console.log("Fetch error: " + error);
            });
    '''))

    # o Botão de salvar irá enviar um json para o servidor que irá ler e fazer os procedimentos posteriores
    callback_botao = CustomJS(args=dict(source=source), code='''
            var dados = source.data;
            var entry = {
                tipo: dados['tipo'],
                x: dados['x'],
                y: dados['y'],
            }

            fetch(`${window.origin}/resultado`, {
                method: "POST",
                credentials: "include",
                body: JSON.stringify(entry),
                cache: "no-cache",
                headers: new Headers({
                    "content-type": "application/json"
                })
            })
            .then(function (response) {
                if (response.status !== 200) {
                    console.log(`Looks like there was a problem. Status code: ${response.status}`);
                    return;
                }
                response.json().then(function (data) {
                    console.log('Acabaou de chegar: ',data[x]);
                });
            })
            .catch(function (error) {
                console.log("Fetch error: " + error);
            });

    ''')
    salvar = Button(label='Salvar coordenadas', button_type="success")
    salvar.js_on_click(callback_botao)

    div, script = components(row(column(spinner,radio_title,radio_group,upload_cor,salvar), column(p,tabela, sizing_mode='scale_height')))
    return render_template('plot.html', the_div=div, the_script=script)


@app.route("/resultado", methods=["POST"])
def create_entry():
    '''
    Rota para receber tabela de dados a partir de um envio do navegador
    '''

    req = request.get_json()

    print(req)

    # res = make_response(jsonify({"message": "OK"}), 200)
    res = make_response(req, 200)

    return res

def click():
    print('Clicado')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == "__main__":
    app.run(debug=True)