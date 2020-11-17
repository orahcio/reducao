from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import ColumnDataSource, DataTable, TableColumn, PointDrawTool, Spinner, WheelZoomTool,\
    RadioGroup, CustomJS, Paragraph, Button, Slider, TextInput, Toggle, Div, Tabs, Panel, CDSView,\
    GroupFilter
from bokeh.layouts import column, row

import colorcet as cc

from flask import Flask, flash, render_template, request, redirect, url_for,\
                  send_from_directory, jsonify, make_response, session
import json
from werkzeug.utils import secure_filename
import os

from astropy.io import fits
from astropy.visualization import HistEqStretch, ContrastBiasStretch
from astropy.table import Table

from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.wcs import WCS
from astroquery.astrometry_net import AstrometryNet
from astroquery.exceptions import TimeoutError
from astroquery.irsa import Irsa

from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder
from photutils import CircularAperture, aperture_photometry

import pandas as pd

import numpy as np

import base64


UPLOAD_FOLDER = './upfolder'
ALLOWED_EXTENSIONS = ['fit', 'fits','corr']
FITs = '[.fit|.fits]'
BANDAS = ['fitsB','fitsV','fitsR']


app = Flask(__name__)
app.secret_key = '43k5jh3kUIh3h45$##ssds'
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


# Caso precise transportar uma asrquivo no json abre ele com base64.b64decode(req['arquivo'])
# salva o arquivo unsado 'wb' 


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        pathname = app.config['UPLOAD_FOLDER']+'/'+request.form.get('refname')
        if not os.path.exists(pathname):
            os.mkdir(pathname)

        # Definindo dados da análise para serem salvos junto com os arquivos
        dirdata = dict(
            name = request.form.get('refname'),
            r = 8
        )

        # Upload de arquivos e salvando dados da sessão para serem salvosjuntos
        for banda in request.files.keys():
            dirdata[banda] = []
            for ffit in request.files.getlist(banda):
                if allowed_file(ffit.filename):
                    filename = secure_filename(ffit.filename)
                    ffit.save(os.path.join(pathname, filename))
                    dirdata[banda].append(filename)

        with open(pathname+'/data.json', 'w') as f:
            json.dump(dirdata,f)

        return redirect(url_for('plotfits',dirname=dirdata['name']))

    # return redirect(url_for('interface'))


def plot():
    '''
    Constrói um plot
    '''


@app.route('/plot/<dirname>')
def plotfits(dirname):
    
    pathname = app.config['UPLOAD_FOLDER']+'/'+dirname+'/'
    with open(pathname+'data.json') as f:
        dirdata = json.load(f)

    r = dirdata['r']
    
    # Faz logo algumas estatísticas da imagem
    dirdata['stats'] = {}
    for fil in BANDAS:
        dirdata['stats'][fil] = []
        for fname in dirdata[fil]:
            img = fits.getdata(pathname+fname)
            print(fname)
            dirdata['stats'][fil].append(sigma_clipped_stats(img,sigma=3.0))

    with open(pathname+'data.json','w') as f:
        json.dump(dirdata,f)

    # Dados que serão usados para fazer computação e visualizar os pontos
    source = ColumnDataSource(dict(
        ra=[],
        dec=[],
        x=[],
        y=[],
        flux = [],
        j = [],
        k = [],
        tipo=[], # se é obj, src ou sky
        banda=[], # o filtro da imagem e arquivo
        color=[] # para colorir de acordo o tipo de objeto
    ))

    # Abrindo coordenadas se salvas
    try:
        cordata = pd.read_excel(pathname+'data.xlsx')
        # Dados que serão usados para fazer computação e visualizar os pontos
        source = ColumnDataSource(cordata)

        print('Coordenadas carregadas.')
    except FileNotFoundError:
        print('Não há coordenadas salvas em %s' % pathname)

    # Constrói a tabaela de dados que poderá ser usada para designar as posições do objeto, estrela e céu
    tabela = DataTable(source=source,columns=[
        TableColumn(field='x',title='x'),
        TableColumn(field='y',title='y'),
        TableColumn(field='ra',title='ra'),
        TableColumn(field='dec',title='dec'),
        TableColumn(field='flux',title='flux'),
        TableColumn(field='j',title='j'),
        TableColumn(field='k',title='k'),
        TableColumn(field='tipo',title='tipo')
    ], editable=True)
    
    P = [] # lista de gráficos para o plot
    Nimg = [] # lista de imagens normalizadas para o contraste
    for fil in BANDAS:
        for fname in dirdata[fil]:
            img = fits.getdata(pathname+fname)
            stretch = HistEqStretch(img) # Histograma, melhor função para granular a imagem
            h,w = img.shape # número de linhas e colunas da matriz da imagem
            nimg = stretch(normal(img)).tolist()
            p = figure(plot_width=700, active_scroll='wheel_zoom')
            p.image(image=[nimg], x=0, y=0, dw=w, dh=h, palette='Greys256', level="image")
            p.x_range.range_padding = p.y_range.range_padding = 0
            p.grid.grid_line_width = 0

            view = CDSView(source=source,filters=[GroupFilter(column_name='banda', group=fil+':'+fname)])
            c = p.circle('x','y', source=source, view=view, color='color', fill_color=None, radius=r, line_width=2)
            cd = p.circle_dot('x','y', source=source, view=view, color='color', size=2)
            tool = PointDrawTool(renderers=[c,cd],empty_value='na')
            p.add_tools(tool)
            p.toolbar.active_tap = tool
            p.toolbar.active_inspect = None

            tab = Panel(child=p, title=fil+':'+fname)

            P.append(tab)
            Nimg.append(nimg)
    
    graficos = Tabs(tabs=P)
    graficos.js_on_change('active', CustomJS(code='''
    tabs_onchange(cb_obj);
    '''))

    contrast = Slider(start=-1, end=6, value=1, step=0.05, title="Contraste")
    contrast.js_on_change('value',CustomJS(args = dict(tabs=graficos.tabs, im=Nimg), code = '''
    contrast_onchange(cb_obj,tabs,im);
    '''))

    # Selecionar o tipo de fonte luminosa: obj, src ou sky
    radio_title = Paragraph(text='Escolha o tipo:')
    LABELS = ['obj','src','sky']
    radio_group = RadioGroup(labels=LABELS, active=0)

    # Evento de mudança da tabela de dados, para inserir dados padrão nas colunas inalteradas
    source.js_on_change('data', CustomJS(args=dict(radio=radio_group, graficos=graficos), code='''
    source_onchange(cb_obj, radio, graficos);
    '''))
    
    # Muda o raio da abertura fotométrica
    spinner = Spinner(title="Raio", low=1, high=40, step=0.5, value=r, width=80)
    spinner.js_link('value', c.glyph, 'radius')
    spinner.js_on_change('value', CustomJS(args=dict(source=source), code='''
    radius_onchange(cb_obj,source);
    '''))

    # print(h,w)

    # # Plota a imagem do arquivo fit
    # p = figure(plot_height=h, plot_width=w, tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],\
    #     active_scroll='wheel_zoom')
    # nimg = stretch(normal(img)).tolist()
    # # print(nimg.tolist())
    # # print(img.tolist())
    # im = p.image(image=[nimg], x=0, y=0, dw=w, dh=h, palette='Greys256', level="image")
    # # im = p.image(image=[img], x=0, y=0, dw=w, dh=h, palette=cc.CET_CBL2, level="image")
    # p.x_range.range_padding = p.y_range.range_padding = 0
    # p.grid.grid_line_width = 0

    # # Os círculos que serão inseridos
    # c = p.circle('x','y', source=source, color='color', fill_color=None, radius=r, line_width=2)
    # cd = p.circle_dot('x','y', source=source, color='color', size=2)
    # tool = PointDrawTool(renderers=[c,cd],empty_value='na')
    # p.add_tools(tool)
    # p.toolbar.active_tap = tool
    # p.toolbar.active_inspect = None

    # # Coluna de controles de interação
    # # Muda o raio da abertura fotométrica
    # spinner = Spinner(title="Raio", low=1, high=40, step=0.5, value=r, width=80)
    # spinner.js_link('value', c.glyph, 'radius')
    # spinner.js_on_change('value', CustomJS(args=dict(source=source), code='''
    # radius_onchange(cb_obj,source);
    # '''))

    # # Selecionar o tipo de fonte luminosa: obj, src ou sky
    # radio_title = Paragraph(text='Escolha o tipo:')
    # LABELS = ['obj','src','sky']
    # radio_group = RadioGroup(labels=LABELS, active=0)

    # # Evento de mudança da tabela de dados, para inserir dados padrão nas colunas inalteradas
    # source.js_on_change('data', CustomJS(args=dict(radio=radio_group), code='''
    # source_onchange(cb_obj, radio);
    # '''))
    # # Fazer o upload do corr.fit da imagem (defasado)


    # # Coluna de requisição
    # text1 = Div(text='<b>Instruções:</b><p>1. Digite a chave do Astrometry.net')
    # apikey_input = TextInput(title='Apikey do Astrometry.net', placeholder='digite a chave aqui')

    # text2 = Div(text='2. Clique abaixo pra requisitar a correção WCS')
    # send_astrometry = Toggle(label='Solução de placa do astrometry.net', disabled=celestial)
    # send_astrometry.js_on_click(CustomJS(args=dict(key=apikey_input, source=source), code='''
    # send_astrometry(cb_obj,key,source);
    # '''))

    # text3 = Div(text='3. Após escolher as fontes no gráfico e ajustar o raio, clique abaixo pra requisitar as magnitudes j e k')
    # busca_2mass = Button(label='2MASS',button_type='success')
    # busca_2mass.js_on_click(CustomJS(args=dict(source=source), code='''
    # send_2mass(source)
    # '''))

    # text4 = Div(text='4. Salve a tabela de dados clicando abaixo para download')
    # # o Botão de salvar irá enviar um json para o servidor que irá ler e fazer os procedimentos posteriores
    # salvar = Button(label='Salvar tabela', button_type="success")
    # salvar.js_on_click(CustomJS(args=dict(source=source), code='''
    # salvar_onclick(source);
    # '''))

    # reset = Button(label='Limpar', button_type='success')
    # reset.js_on_click(CustomJS(args=dict(source=source), code='''
    # reset_onclick(source);
    # '''))

    # test = Button(label='Teste',button_type='success')
    # test.js_on_click(CustomJS(args=dict(radio=radio_group,source=source,r=c.glyph.radius), code='''
    # f(cb_obj,radio,source,r);
    # '''))
    # print('raio: ',c.glyph.radius)
    div, script = components(row(column(contrast,radio_title,radio_group),graficos))
    return render_template('plot.html', the_div=div, the_script=script,filename=dirdata['name'])


@app.route('/fluxes', methods=['POST'])
def recalc_fluxes():
    '''
    Nesta função o raio tem que vir pelo javascript pois ele é mudado num widget do bokeh
    '''

    req = request.get_json()

    data = pd.DataFrame(dict(
        x=req['x'],
        y=req['y'],
        flux=req['flux']
    ))

    with fits.open(UPLOAD_FOLDER+'/'+session['name']) as f:
        img = f[0].data
    r = req['r']
    session.modifeid = True
    session['r'] = r

    aperture = CircularAperture(data[['x','y']], r)
    fluxes = aperture_photometry(img,aperture)
    data['flux'] = fluxes['aperture_sum']

    res = make_response(data.to_json(),200)

    return res


def centralizar(img, cx, cy):
    '''
    Encontra uma fonte dentro de uma abertura com o raio definido pelo usuário
    '''

    xmax, ymax = img.shape
    dmin = np.sqrt(xmax*xmax+ymax*ymax)

    r = session['r']
    aperture = CircularAperture((cx,cy), r)
    mask = aperture.to_mask()
    print(mask)
    mean, median, std = session['stats']
    find = DAOStarFinder(fwhm=3, threshold=3*std)
    cr = find((img-median)*mask.to_image(img.shape))
    if cr:
        print(cr)
        # Pega o índice mais próxima do ponto clicado
        for i in range(len(cr)):
            x = cr[i]['xcentroid']
            y = cr[i]['ycentroid']
            d = np.sqrt((x-cx)**2+(y-cy)**2)
            if d<dmin:
                dmin = d
                idx = i
        cx = cr[idx]['xcentroid']
        cy = cr[idx]['ycentroid']

    return cx,cy


def query_2MASS(ra,dec):
    '''
    Busca no 2MASS dado um campo com o raio definido pelo usuário
    '''

    w = WCS(fits.getheader(UPLOAD_FOLDER+'/'+session['name']))
    r = session['r']
    o = SkyCoord(w.wcs_pix2world([(0,0)],1), unit='deg')
    opr = SkyCoord(w.wcs_pix2world([(r,r)],1), unit='deg')
    rw = o.separation(opr)[0]
    print('Separação',rw)

    crval = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')

    Q = Irsa.query_region(crval,catalog='fp_psc',spatial='Cone', radius=rw,\
                           selcols=['ra','dec','j_m','k_m']).to_pandas()
    print(Q)
    m = SkyCoord(ra=Q['ra'],dec=Q['dec'], unit=('deg','deg'), frame='icrs')
    idx, d2, _ = match_coordinates_sky(crval,m)

    j , k = Q.loc[idx][['j_m','k_m']]
    return j, k


@app.route('/add', methods=['POST'])
def add_radec():
    '''
    Ao adicionar uma marca no gráfico:

    1- Regula o centróide se for uma estrela ou objeto
    2- Calcula as coordenadas celestiais se houver correção astrométrica
    3- Calcula o fluxo
    '''

    req = request.get_json()

    with fits.open(UPLOAD_FOLDER+'/'+session['name']) as f:
        img = f[0].data

    # Faz a fotometria de abertura
    if not req['tipo'] == 'sky':
        req['x'], req['y'] = centralizar(img,req['x'],req['y'])
    aperture = CircularAperture((req['x'],req['y']), session['r'])
    fluxes = aperture_photometry(img,aperture)
    req['flux'] = fluxes['aperture_sum'][0]

    # Pega as coordenadas celestes se houver correção
    w = WCS(fits.getheader(UPLOAD_FOLDER+'/'+session['name']))
    if w.has_celestial:
        ra, dec = w.wcs_pix2world([(req['x'],req['y'])],1)[0]
        req['ra'] = ra
        req['dec'] = dec
        if req['tipo'] == 'src':
            req['j'], req['k'] = query_2MASS(ra,dec)
        else:
            req['j'], req['k'] = 'na','na'

    return make_response(jsonify(req), 200)


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
    '''
    Essa função faz a requisição para o astrometry.net,
    tem espaço para usar o método get, possivelmente implementarei
    para fazer forçando o upload se a solução não existir e retornar
    á página do plot de dados.
    '''
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


@app.route("/resultado", methods=["POST"])
def create_entry():
    '''
    Rota para receber tabela de dados a partir de um envio do navegador
    '''

    req = request.get_json()
    out = pd.DataFrame(req)

    if not out.empty:
        out.to_excel('upfolder/'+out['fit'][0].strip(FITs)+'.xlsx')
        res = make_response(jsonify({"message": "Arquivo salvo"}), 200)

        return res
    
    return make_response(jsonify({'message': 'Tabela vazia'}), 200)


@app.route("/busca",methods=['POST'])
def search_2MASS():
    '''
    Faz busca no catálogo 2MASS a partir das coordenadas celestes
    '''
    
    w = WCS(fits.getheader(UPLOAD_FOLDER+'/'+session['name']))
    r = session['r']
    o = SkyCoord(w.wcs_pix2world([(0,0)],1), unit='deg')
    opr = SkyCoord(w.wcs_pix2world([(r,r)],1), unit='deg')
    rw = o.separation(opr)[0]
    print('Separação',rw)

    req = request.get_json()

    data = pd.DataFrame(req)

    src = SkyCoord(ra=data['ra'], dec=data['dec'], unit='deg', frame='icrs')
    crval = SkyCoord(ra=np.mean(data['ra']), dec=np.mean(data['dec']), unit='deg', frame='icrs')
    r = 1.1*crval.separation(src).max()

    Q = Irsa.query_region(crval,catalog='fp_psc',spatial='Cone', radius=r,selcols=['ra','dec','j_m','k_m']).to_pandas()
    print(Q)
    m = SkyCoord(ra=Q['ra'],dec=Q['dec'], unit=('deg','deg'), frame='icrs')
    idx, d2, _ = match_coordinates_sky(src,m)

    Q.loc[idx[d2>=rw]] = None # retira estrela que não conseguiu chegar perto

    data[['j','k']] = Q[['j_m','k_m']].loc[idx].values
    print(data)
    res = make_response(data.to_json(), 200)

    return res


@app.route('/download/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


def main():
    # port = int(os.environ.get("PORT",5000))
    # app.run(host="0.0.0.0", port=port)
    app.run(debug=True)


if __name__ == "__main__":
   main()
