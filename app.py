from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.events import Tap
from bokeh.models import ColumnDataSource, DataTable, TableColumn, PointDrawTool, Spinner, WheelZoomTool, RadioGroup,\
    CustomJS, Paragraph, Button, Slider
from bokeh.layouts import column, row

import colorcet as cc

from flask import Flask, flash, render_template, request, redirect, url_for
from flask import send_from_directory, jsonify, make_response 
import json
from werkzeug.utils import secure_filename
import os

from astropy.io import fits
from astropy.visualization import HistEqStretch, ContrastBiasStretch
from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_3d

import pandas as pd

import numpy as np

import base64


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
    '''
    Essa função ficou defasada quando decidi upar o corr junto com o fit
    Posteriormente farei a requisição diretamente ao astrometry.net para
    não ter mais trabalho de upar lá pra depois aqui
    '''

    req = request.get_json()
    
    # decodificando a string recebida
    decoded = base64.b64decode(req['arquivo'])

    # fazendo upload atualizando o nome do arquivo para upload
    filename = UPLOAD_FOLDER+'/'+req['url'].split('/')[-1].strip('.fit')+'.corr'
    with open(filename, 'wb') as f:
        f.write(decoded)

    # pegando a tabela com as coordenadas de estrelas selecionadas
    data = pd.DataFrame(dict(
        ra=req['ra'],
        dec=req['dec'],
        x=req['x'],
        y=req['y'],
        tipo = req['tipo']
        ))
    data = data[data['tipo']=='src']

    # tabela com os dados que acabaram de ser subidos
    cordata = Table.read(filename).to_pandas()

    # fazendo correspondência entre coordenadas
    if len(data)>0:
        m = SkyCoord([(x,y,0) for x,y in data[['x','y']]], unit='pixel', representation='cartesian')
        c = SkyCoord([(x,y,0) for x,y in cordata[['x','y']]], unit='pixel', representation='cartesian')
        idx, _, _ = match_coordinates_3d(m,c)
        cordata = cordata[idx]

        # atualisando os dados para as coordenadas do arquivo corr
        data['x'] = cordata['field_x']
        data['y'] = cordata['field_y']
        data['ra'] = cordata['field_ra']
        data['dec'] = cordata['field_dec']

        return make_response(data.to_json())

    data = cordata[['field_ra','field_dec','field_x','field_y']]
    data.columns = ['ra','dec','x','y']
    
    return make_response(data.to_json(),200)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'fits' not in request.files:
            flash('Um dos arquivos não foi postado')

            return redirect(request.url)

        ffit = request.files['fits']
        # if user does not select file, browser also
        # submit an empty part without filename
        if ffit.filename == '':
            
            flash('Ficou faltando selecionar algum arquivo')
            return redirect(request.url)

        if ffit and allowed_file(ffit.filename):
            filename = secure_filename(ffit.filename)
            ffit.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            return redirect(url_for('plotfits',filename=filename))

    return redirect(url_for('interface'))


@app.route('/plot/<filename>')
def plotfits(filename):
    
    # Dados que serão usados para fazer computação e visualizar os pontos
    source = ColumnDataSource(dict(
        fit=[], # quando salvar estado salvar tabela
        ra=[],
        dec=[],
        x=[],
        y=[],
        tipo=[], # se é obj, src ou sky
        banda=[] # o filtro da imagem
    ))

    # Abrindo imagem
    with fits.open('upfolder/'+filename) as f:
        img = f[0].data
    

    # img = np.array([[0,  1,  2,  3],
                    # [4,  5,  6,  7],
                    # [8,  9, 10, 11]])
    
    # Abrindo coordenadas se salvas
    try:
        corname = 'upfolder/'+filename.strip('[.fit|.fits]')+'.xlsx'
        cordata = pd.read_excel(corname)
        # Dados que serão usados para fazer computação e visualizar os pontos
        source = ColumnDataSource(dict(
            fit=[filename for i in range(len(cordata))], # quando salvar estado salvar tabela
            ra=['na' for i in range(len(cordata))],
            dec=['na' for i in range(len(cordata))],
            x=cordata['x'].tolist(),
            y=cordata['y'].tolist(),
            tipo=cordata['tipo'], # se é obj, src ou sky
            banda=['undef' for i in range(len(cordata))] # o filtro da imagem
        ))
        print('Coordenadas carregadas.')
    except FileNotFoundError:
        print('Não há coordenadas salvas: %s' % corname)

    stretch = HistEqStretch(img) # Histograma, melhor função para granular a imagem
    h,w = img.shape # número de linhas e colunas da matriz da imagem
    print(h,w)

    # Constrói a tabaela de dados que poderá ser usada para designar as posições do objeto, estrela e céu
    tabela = DataTable(source=source,columns=[
        TableColumn(field='x',title='x'),
        TableColumn(field='y',title='y'),
        TableColumn(field='tipo',title='tipo')
    ], editable=True)
    
    # Plota a imagem do arquivo fit
    p = figure(plot_height=h, plot_width=w, tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],\
        active_scroll='wheel_zoom')
    nimg = stretch(normal(img)).tolist()
    # print(nimg.tolist())
    # print(img.tolist())
    im = p.image(image=[nimg], x=0, y=0, dw=w, dh=h, palette='Greys256', level="image")
    # im = p.image(image=[img], x=0, y=0, dw=w, dh=h, palette=cc.CET_CBL2, level="image")
    p.x_range.range_padding = p.y_range.range_padding = 0
    p.grid.grid_line_width = 0

    # Os círculos que serão inseridos
    c = p.circle('x','y', source=source, color='red', fill_color=None, radius=8, line_width=2)
    cd = p.circle_dot('x','y', source=source, color='red', size=2)
    tool = PointDrawTool(renderers=[c,cd],empty_value='na')
    p.add_tools(tool)
    p.toolbar.active_tap = tool

    # Muda o raio da abertura fotométrica
    spinner = Spinner(title="Raio", low=1, high=40, step=0.5, value=8, width=80)
    spinner.js_link('value', c.glyph, 'radius')

    # Selecionar o tipo de fonte luminosa: obj, src ou sky
    radio_title = Paragraph(text='Escolha o tipo:')
    LABELS = ['obj','src','sky']
    radio_group = RadioGroup(labels=LABELS, active=0)

    # Evento de mudança da tabela de dados, para inserir dados padrão nas colunas inalteradas
    source.js_on_change('data', CustomJS(args=dict(radio=radio_group), code='''
            var data = cb_obj.data;

            const n = data['x'].length;
            const labels = radio.labels;
            var aux = `${window.location.pathname}`;
            aux = aux.slice(6,aux.length);
            if(data['tipo'][n-1]=='na') {
                data['fit'][n-1] = aux;
                data['tipo'][n-1] = labels[radio.active];
                data['banda'][n-1] = 'undef';
            }

            cb_obj.change.emit();
    '''))
    # Fazer o upload do corr.fit da imagem (defasado)

    contrast = Slider(start=-1, end=6, value=1, step=0.05, title="Contraste")
    contrast.js_on_change('value',CustomJS(args = dict(source=im.data_source, im=nimg), code = '''

            const x = im
            var y = source.data['image'][0];
            // console.log(y)
            const ni = y.length
            const nj = y[0].length
            const n = ni*nj
            var c = cb_obj.value
            // console.log(c)
            var max = 0 
            var min = 2**64-1

            // console.log('x: ', x) 
            // Clip values
            const clip = (y) => {
                for(let i=0;i<ni;i++) {
                    for(let j=0;j<nj;j++) {
                        if(max<y[i][j]) { max=y[i][j] }
                        if(min>y[i][j]) { min=y[i][j] }
                    }
                }
                for(let i=0;i<ni;i++) {
                    for(let j=0;j<nj;j++) {
                        y[i][j] = (y[i][j]-min)/(max-min)
                    }
                }
                return y
            }
            y = clip(y)
            for(let i=0;i<ni;i++) {
                for(let j=0;j<nj;j++) {
                    y[i][j] = Math.pow(x[i][j]+1e-8,10**c)
                }
            }
            y = clip(y)
            source.change.emit()
    '''))

    # o Botão de salvar irá enviar um json para o servidor que irá ler e fazer os procedimentos posteriores
    callback_botao = CustomJS(args=dict(source=source), code='''

            var entry = source.data;
            console.log(entry)

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
                    console.log('Acabaou de chegar: ',data['x']);
                });
            })
            .catch(function (error) {
                console.log("Fetch error: " + error);
            });

    ''')
    salvar = Button(label='Salvar tabela', button_type="success")
    salvar.js_on_click(callback_botao)

    div, script = components(row(column(spinner,contrast,radio_title,radio_group,salvar), column(p,tabela, sizing_mode='scale_height')))
    return render_template('plot.html', the_div=div, the_script=script)


@app.route("/resultado", methods=["POST"])
def create_entry():
    '''
    Rota para receber tabela de dados a partir de um envio do navegador
    '''

    req = request.get_json()

    out = pd.DataFrame(req)
    out[['x','y','tipo']].to_excel('upfolder/'+out['fit'][0].strip('[.fit|.fits]')+'.xlsx',
                                    index=False)

    print(out)

    # res = make_response(jsonify({"message": "OK"}), 200)
    res = make_response(req, 200)

    return res


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == "__main__":
    app.run(debug=True)