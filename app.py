from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.events import Tap
from bokeh.models import ColumnDataSource, DataTable, TableColumn, PointDrawTool, Spinner, WheelZoomTool, RadioGroup,\
    CustomJS, Paragraph, Button, Slider, TextInput, Toggle, Div
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
from astropy.wcs import WCS
from astroquery.astrometry_net import AstrometryNet
from astroquery.exceptions import TimeoutError

from photutils import CircularAperture, aperture_photometry

import pandas as pd

import numpy as np

import base64


UPLOAD_FOLDER = './upfolder'
ALLOWED_EXTENSIONS = ['fit', 'fits','corr']
FITs = '[.fit|.fits]'


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
    filename = UPLOAD_FOLDER+'/'+req['url'].split('/')[-1].strip(FITs)+'.corr'
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
        flux = [],
        tipo=[], # se é obj, src ou sky
        banda=[] # o filtro da imagem
    ))

    # Abrindo imagem
    with fits.open('upfolder/'+filename) as f:
        img = f[0].data
        # Caso tenha correção não ativa o botão para pedir
        celestial = WCS(f[0].header).has_celestial
    
    # Uma matriz pra fazer testes
    # img = np.array([[0,  1,  2,  3],
                    # [4,  5,  6,  7],
                    # [8,  9, 10, 11]])
    
    # Abrindo coordenadas se salvas
    try:
        corname = 'upfolder/'+filename.strip('[.fit|.fits]')+'.xlsx'
        cordata = pd.read_excel(corname)
        # Dados que serão usados para fazer computação e visualizar os pontos
        source = ColumnDataSource(cordata)

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
        TableColumn(field='ra',title='ra'),
        TableColumn(field='dec',title='dec'),
        TableColumn(field='flux',title='flux'),
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

    div_text = Div(text='Aqui vou escrever algumas instruções ou terá mais ferramentas', width=200)

    # Muda o raio da abertura fotométrica
    spinner = Spinner(title="Raio", low=1, high=40, step=0.5, value=8, width=80)
    spinner.js_link('value', c.glyph, 'radius')

    # Selecionar o tipo de fonte luminosa: obj, src ou sky
    radio_title = Paragraph(text='Escolha o tipo:')
    LABELS = ['obj','src','sky']
    radio_group = RadioGroup(labels=LABELS, active=0)

    # Evento de mudança da tabela de dados, para inserir dados padrão nas colunas inalteradas
    source.js_on_change('data', CustomJS(args=dict(radio=radio_group,r=c.glyph.radius), code='''
    source_onchange(cb_obj, radio, r);
    '''))
    # Fazer o upload do corr.fit da imagem (defasado)

    contrast = Slider(start=-1, end=6, value=1, step=0.05, title="Contraste")
    contrast.js_on_change('value',CustomJS(args = dict(source=im.data_source, im=nimg), code = '''
    contrast_onchange(cb_obj,source,im)
    '''))

    # o Botão de salvar irá enviar um json para o servidor que irá ler e fazer os procedimentos posteriores
    callback_botao = CustomJS(args=dict(source=source), code='''
    salvar_onclick(source);
    ''')
    salvar = Button(label='Salvar tabela', button_type="success")
    salvar.js_on_click(callback_botao)

    apikey_input = TextInput(title='Apikey do Astrometry.net', placeholder='digite a chave aqui')
    send_astrometry = Toggle(label='Solução de placa do astrometry.net', disabled=celestial)
    send_astrometry.js_on_click(CustomJS(args=dict(key=apikey_input, source=source), code='''
    send_astrometry(cb_obj,key,source);
    '''))

    test = Toggle(label='Teste',button_type='success')
    test.js_on_click(CustomJS(args=dict(radio=radio_group,source=source,r=c.glyph.radius), code='''
    f(cb_obj,radio,source,r);
    '''))
    print('raio: ',c.glyph.radius)
    div, script = components(row(column(spinner,contrast,radio_title,radio_group,
                                        salvar,apikey_input,send_astrometry,test),
                                 column(p,tabela, sizing_mode='scale_width'),
                                 column(div_text)))
    return render_template('plot.html', the_div=div, the_script=script)


@app.route('/add', methods=['POST'])
def add_radec():

    req = request.get_json()
    w = WCS(fits.getheader(UPLOAD_FOLDER+'/'+req['name']))

    # Pega as coordenadas celestes se houver correção
    if w.has_celestial:
        ra, dec = w.wcs_pix2world([(req['x'],req['y'])],0)[0]
        req['ra'] = ra
        req['dec'] =    dec
    print(req)
    # Faz a fotometria de abertura
    with fits.open(UPLOAD_FOLDER+'/'+req['name']) as f:
        img = f[0].data
    r = req['r']
    aperture = CircularAperture((req['x'],req['y']), r)
    fluxes = aperture_photometry(img,aperture)
    req['flux'] = fluxes['aperture_sum'][0]

    res = make_response(jsonify(req), 200)

    return res


def solveplateastrometry(key,data,force_upload=False):
    ast = AstrometryNet()
    ast.api_key = key

    try_again = True
    submission_id = None

    while try_again:
        try:
            if not submission_id:
                if isinstance(data,str):
                    wcs_header = ast.solve_from_image(data, force_image_upload=force_upload,
                                 submission_id=submission_id)
                    print('Com imagem\n',wcs_header)
                elif isinstance(data,pd.DataFrame):
                    print(data)
                    filename = data.iloc[0,2]
                    with fits.open('upfolder/'+filename) as f:
                        w, h = f[0].data.shape
                    wcs_header = ast.solve_from_source_list(data['x'], data['y'],
                                 submission_id=submission_id, image_width=w, image_height=h)
                    print('Com dados\n',wcs_header)
            else:
                wcs_header = ast.monitor_submission(submission_id,
                                 solve_timeout=120)
                print('Buscando algumas vezes\n',wcs_header)
        except TimeoutError as e:
            submission_id = e.args[1]
        else:
            # got a result, so terminate
            try_again = False

    return wcs_header


@app.route("/astrometry_net/<key>/<filename>", methods=["POST","GET"])
def astrometrysolve(key,filename):
    if request.method=='POST':

        req = request.get_json()
        data = pd.DataFrame(req)
        sdata = data[data['tipo']=='src']
        filepath = UPLOAD_FOLDER+'/'+filename

        # Primeira tentativa apenas com a lista de estrelas
        if len(data):
            print('Tentando com a lista de coordenadas')
            wcs_header = solveplateastrometry(key,sdata[['x','y','fit']])
        else:
            print('Tentando com a imagem do photutils')
            wcs_header = solveplateastrometry(key,filepath)
        print('Resultado 1\n', wcs_header)

        if not isinstance(wcs_header,fits.Header):
            print('Tentando com upload da imagem')
            wcs_header = solveplateastrometry(key,filepath,force_upload=True)
            print('Resultado 2\n', wcs_header)

        if isinstance(wcs_header,fits.Header):
            with fits.open(filepath,'update') as f:
                f[0].header = f[0].header+wcs_header
            
            return make_response({'message': 'OK'}, 200)

        return make_response(jsonify({'message': 'NO'}),200)

    # return redirect(url_for('plotfits',filename=filename))

@app.route("/resultado", methods=["POST"])
def create_entry():
    '''
    Rota para receber tabela de dados a partir de um envio do navegador
    '''

    req = request.get_json()

    out = pd.DataFrame(req)
    out.to_excel('upfolder/'+out['fit'][0].strip(FITs)+'.xlsx',
                 index=False)

    res = make_response(jsonify({"message": "Arquivo salvo"}), 200)
    # res = make_response(req, 200)

    return res


@app.route('/download/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == "__main__":
    app.run(debug=True)