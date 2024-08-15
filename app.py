from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.events import ButtonClick
from bokeh.models import ColumnDataSource, DataTable, TableColumn, PointDrawTool, Spinner, WheelZoomTool,\
    RadioGroup, CustomJS, Paragraph, Button, Slider, TextInput, Toggle, Div, Tabs, TabPanel, CDSView,\
    GroupFilter, Select, OpenURL
from bokeh.layouts import column, row
from bokeh import __version__ as ver

import colorcet as cc

from flask import Flask, render_template, request, redirect, url_for,\
                  send_from_directory, jsonify, make_response, session, g
import json
from numpy.lib.function_base import gradient
from werkzeug.utils import secure_filename
import os

from astropy.io import fits
from astropy.visualization import HistEqStretch, ContrastBiasStretch
from astropy.table import Table
from astropy.time import Time

from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.wcs import WCS
from astroquery.astrometry_net import AstrometryNet
from astroquery.exceptions import TimeoutError
from bs4 import BeautifulSoup
from astroquery.ipac.irsa import Irsa

from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder
from photutils import CircularAperture, aperture_photometry

import statsmodels.formula.api as smf

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
app.config.update(
    # SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def normal(valores):
    import numpy as np

    n, m = valores.shape
    b = float(np.max(valores))
    a = float(np.min(valores))
    ba = b-a

    imagem = np.zeros(valores.shape)
    for i in range(n):
        for j in range(m):
            imagem[i,j] = (float(valores[i,j])-a)/ba

    return imagem


@app.route("/")
def interface():
    '''
    Essa função irá renderizar a página html que tem o formulário
    que irá pegar os arquivos
    '''
    return render_template("interface.html")


# Caso precise transportar um arquivo no json abre ele com base64.b64decode(req['arquivo'])
# salva o arquivo usando 'wb' 


@app.route('/', methods=['POST'])
def upload_file():
    '''
    Essa função irá fazer upload dos arquivos e redirecionar para o
    plot dos gráficos.
    '''

    pathname = app.config['UPLOAD_FOLDER']+'/'+request.form.get('refname')
    if not os.path.exists(pathname):
        os.mkdir(pathname)

    # Definindo table da análise para serem salvos junto com os arquivos
    dirdata = dict(
        name = request.form.get('refname'),
        r = 8
    )

    # Upload de arquivos e salvando table da sessão para serem salvosjuntos
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


@app.route('/plot/<dirname>')
def plotfits(dirname):
    
    session.modified = True
    # session.samesite = 'Lax'
    session['pathname'] = app.config['UPLOAD_FOLDER']+'/'+dirname+'/'
    session['stats'] = {}
    session['date'] = {} # pegar a data para converter em juliana e inserir nas análises

    with open(session['pathname']+'data.json') as f:
        dirdata = json.load(f)

    r = dirdata['r']
    session['r'] = r
    celestial = False
    # Faz logo algumas estatísticas da imagem
    for fil in BANDAS:
        for fname in dirdata[fil]:
            img, header = fits.getdata(session['pathname']+fname, header=True)
            session['stats'][f'{fil}:{fname}'] = sigma_clipped_stats(img,sigma=3.0)

            if not celestial:
                celestial = WCS(header).has_celestial
                session['wcs'] = session['pathname']+fname
                
            # a data de observação de cada imagem    
            session['date'][f'{fil}:{fname}'] = Time(header['DATE-OBS']).jd

    # Tabela com os dados de coordenadas
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
        sid=[], # id da estrela copiada
        colors=[], # para colorir de acordo o tipo de objeto
    ))

    # Abrindo coordenadas se salvas
    try:
        cordata = pd.read_excel(session['pathname']+'data.xlsx').to_dict('list')
        print(cordata)
        if len(cordata) > 0:
            source.stream(cordata)
            print(f'Coordenadas carregadas com {len(cordata)}')
    except FileNotFoundError:
        print('Não há coordenadas salvas em %s' % session['pathname'])

    # Constrói a tabaela de table que poderá ser usada para designar as posições do objeto, estrela e céu
    tabela = DataTable(source=source,columns=[
        TableColumn(field='x',title='x'),
        TableColumn(field='y',title='y'),
        TableColumn(field='ra',title='ra'),
        TableColumn(field='dec',title='dec'),
        TableColumn(field='j',title='j'),
        TableColumn(field='k',title='k'),
        TableColumn(field='flux',title='flux'),
        TableColumn(field='tipo',title='tipo'),
        TableColumn(field='banda',title='banda'),
        TableColumn(field='sid',title='sid')
    ], editable=False)
    

    P = [] # lista de gráficos para o plot
    Nimg = [] # lista de imagens normalizadas para o contraste
    for fil in BANDAS:
        for fname in dirdata[fil]:
            img = fits.getdata(session['pathname']+fname)
            stretch = HistEqStretch(normal(img)) # Histograma, melhor função para granular a imagem
            h,w = img.shape # número de linhas e colunas da matriz da imagem
            nimg = stretch(normal(img))
            p = figure(width=700, output_backend="webgl", active_scroll='wheel_zoom')
            p.image(image=[nimg], x=0, y=0, dw=w, dh=h, palette='Greys256', level="image")
            p.x_range.range_padding = p.y_range.range_padding = 0
            p.grid.grid_line_width = 0

            view = CDSView(filter=GroupFilter(column_name='banda', group=fil+':'+fname))
            c = p.circle('x','y', source=source, view=view, color='colors', fill_color='colors', radius=r, line_width=2, alpha=0.3, line_alpha=1)
            cd = p.scatter('x','y', source=source, marker='circle_dot', view=view, color='colors', size=2)
            tool = PointDrawTool(renderers=[c,cd],empty_value='na')
            p.add_tools(tool)
            p.toolbar.active_tap = tool
            p.toolbar.active_inspect = None

            tab = TabPanel(child=p, title=fil+':'+fname)

            P.append(tab)
            Nimg.append(nimg)
    
    graficos = Tabs(tabs=P)
    # graficos.js_on_change('active', CustomJS(code='''
    # tabs_onchange(cb_obj);
    # '''))

    contrast = Slider(start=-1, end=36, width=250, value=1, step=0.5, title="Contraste")
    contrast.js_on_change('value',CustomJS(args = dict(tabs=graficos, im=Nimg), code = '''
    contrast_onchange(cb_obj,tabs,im);
    '''))

    # Selecionar o tipo de fonte luminosa: obj, src ou sky
    radio_title = Paragraph(text='Escolha o tipo:')
    LABELS = ['obj','src','sky']
    radio_group = RadioGroup(labels=LABELS, active=0)

    # # Evento de mudança da tabela de table, para inserir table padrão nas colunas inalteradas
    source.js_on_change('change:data', CustomJS(args=dict(radio=radio_group, graficos=graficos), code='''
    source_onchange(cb_obj, radio, graficos);
    '''))
    
    # Muda o raio da abertura fotométrica
    spinner = Spinner(title="Raio", low=1, high=40, step=0.5, value=r, width=80)
    spinner.js_on_change('value', CustomJS(args=dict(source=source, tabs=graficos.tabs), code='''
    radius_onchange(cb_obj,source,tabs);
    '''))

    # Coluna de requisição
    text1 = Div(text='<b>Instruções:</b><p>1. Digite a chave do Astrometry.net')
    apikey_input = TextInput(title='Apikey do Astrometry.net', placeholder='digite a chave aqui')

    text2 = Div(text='''<p>2. Selecione qual imagem será usada como referência para o\
                astrometry.net e para o cálculo das coordenadas celestes</p>''')
    seletor = Select(title='Escolha a imagem de referência', options=[*session['stats'].keys()])

    text3 = Div(text='3. Clique abaixo pra requisitar a correção WCS')
    send_astrometry = Toggle(label='Solução de placa do astrometry.net', disabled=celestial)
    send_astrometry.js_on_click(CustomJS(args=dict(key=apikey_input, source=source, selected=seletor), code='''
    send_astrometry(cb_obj,key,source,selected);
    '''))

    # o Botão de salvar irá enviar um json para o servidor que irá ler e fazer os procedimentos posteriores
    text4 = Div(text='4. Salve a tabela de table clicando em salvar.')
    salvar = Button(label='Salvar tabela', button_type="success")
    salvar.js_on_click(CustomJS(args=dict(source=source), code='''
    salvar_onclick(source);
    '''))

    reset = Button(label='Reiniciar', button_type='success')
    reset.js_on_click(CustomJS(args=dict(source=source,tabela=tabela), code='''
    reset_onclick(source,tabela);
    '''))

    copiar = Button(label='Copiar coordenadas', button_type='success')
    copiar.js_on_click(CustomJS(args=dict(source=source, ref=seletor, radio=radio_group, active=graficos), code='''
    add_data(source,ref,radio,active);
    '''))
    
    reduzir = Button(label='Reduzir', button_type='primary')
    reduzir.js_on_click(CustomJS(args=dict(base=request.url_root, dirname=dirname), code='''
    window.open(base+"reducao/"+dirname,"_self");
    '''))
    print("URL: ", f"{request.url_root}reducao/{dirname}")
    div, script = components(row(column(contrast,spinner,radio_title,radio_group, reset,copiar,salvar, width=300),\
        column(graficos, tabela, sizing_mode='scale_width'),\
        column(text1,apikey_input,text2,seletor,text3,send_astrometry,text4, reduzir, width=200)))

    return render_template('plot.html', the_div=div, the_script=script,filename=dirdata['name'],bokeh=ver)


@app.route('/fluxes', methods=['POST'])
def recalc_fluxes():
    '''
    Nesta função o raio tem que vir pelo javascript pois ele é mudado num widget do bokeh
    '''

    req = request.get_json()

    data = pd.DataFrame(dict(
        x=req['x'],
        y=req['y'],
        flux=req['flux'],
        banda=req['banda']
    ))

    r = req['r']
    session.modifeid = True
    session['r'] = r

    # Salva o novo raio
    print(session['pathname']+'data.json')
    with open(session['pathname']+'data.json') as f:
        dirdata = json.load(f)
    dirdata['r'] = r
    with open(session['pathname']+'data.json','w') as f:
        json.dump(dirdata,f)

    for banda in data['banda']:
        fname = banda.split(':')[1]
        img = fits.getdata(session['pathname']+fname)
        aperture = CircularAperture(data[['x','y']][data['banda']==banda], r)
        fluxes = aperture_photometry(img,aperture)
        data.loc[data['banda']==banda, 'flux'] = fluxes['aperture_sum']


    res = make_response(data.to_json(),200)

    return res


def centralizar(img, banda, cx, cy):
    '''
    Encontra uma fonte dentro de uma abertura com o raio definido pelo usuário
    '''

    xmax, ymax = img.shape
    dmin = np.sqrt(xmax*xmax+ymax*ymax)

    r = session['r']
    aperture = CircularAperture((cx,cy), r)
    mask = aperture.to_mask()
    print(mask)
    _, median, std = session['stats'][banda]
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
                cx = cr[i]['xcentroid']
                cy = cr[i]['ycentroid']

    return cx,cy


def query_2MASS(ra,dec):
    '''
    Busca no 2MASS dado um campo com o raio definido pelo usuário
    '''

    w = WCS(fits.getheader(session['wcs']))
    r = session['r']
    orim = SkyCoord(w.wcs_pix2world([(0,0)],1), unit='deg')
    opr = SkyCoord(w.wcs_pix2world([(r,0)],1), unit='deg')
    rw = orim.separation(opr)[0]
    print('Separação',rw)

    crval = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')

    Q = Irsa.query_region(crval,catalog='fp_psc',spatial='Cone',\
                          radius=rw)[['ra','dec','j_m','k_m']].to_pandas()
    print(Q)
    m = SkyCoord(ra=Q['ra'],dec=Q['dec'], unit=('deg','deg'), frame='icrs')
    idx, _, _ = match_coordinates_sky(crval,m)

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
    fname = req['banda'].split(':')[1]
    img = fits.getdata(session['pathname']+fname)

    # Faz a fotometria de abertura
    if not req['tipo'] == 'sky':
        req['x'], req['y'] = centralizar(img,req['banda'],req['x'],req['y'])
    aperture = CircularAperture((req['x'],req['y']), session['r'])
    fluxes = aperture_photometry(img,aperture)
    req['flux'] = fluxes['aperture_sum'][0]

    # Pega as coordenadas celestes se houver correção
    w = WCS(fits.getheader(session['wcs']))
    if w.has_celestial:
        ra, dec = w.wcs_pix2world([(req['x'],req['y'])],1)[0]
        req['ra'] = ra
        req['dec'] = dec
        if req['tipo'] == 'src':
            req['j'], req['k'] = query_2MASS(ra,dec)
        else:
            req['j'], req['k'] = 'na','na'
    print(req)
    print(jsonify(req))
    return make_response(jsonify(req), 200)


def solveplateastrometry(key,data=None,filepath=None):
    ast = AstrometryNet()
    ast.api_key = key

    try_again = True
    submission_id = None
    wcs_header = {}

    while try_again:
        try:
            if not submission_id:
                if not data:
                    print("Requisitando com upload da imagem: ", filepath)
                    wcs_header = ast.solve_from_image(filepath, force_image_upload=True,
                                 submission_id=submission_id, solve_timeout=120)
                    print('Com imagem\n',wcs_header)
                elif isinstance(data,pd.DataFrame) and isinstance(filepath,str):
                    print(data)
                    with fits.open(filepath) as f:
                        w, h = f[0].data.shape
                    wcs_header = ast.solve_from_source_list(data['x'], data['y'],
                                 submission_id=submission_id, image_width=w, image_height=h,
                                 solve_timeout=120)
                    print('Com table\n',wcs_header)
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


@app.route("/astrometry_net/<key>/<dirname>/<filename>", methods=["POST","GET"])
def astrometrysolve(key,dirname,filename):
    '''
    Essa função faz a requisição para o astrometry.net,
    tem espaço para usar o método get, possivelmente implementarei
    para fazer forçando o upload se a solução não existir e retornar
    á página do plot de table.
    '''
    if request.method=='POST':

        req = request.get_json()
        data = pd.DataFrame(req)
        sdata = data[data['tipo']=='src']
        filepath = UPLOAD_FOLDER+'/'+dirname+'/'+filename

        # Primeira tentativa apenas com a lista de estrelas
        if len(data):
            print('Tentando com a lista de coordenadas')
            wcs_header = solveplateastrometry(key,sdata[['x','y']],filepath)
        else:
            print('Tentando com a imagem do photutils')
            wcs_header = solveplateastrometry(key,filepath=filepath)
        print('Resultado 1\n', wcs_header)

        if not isinstance(wcs_header,fits.Header):
            print('Tentando com upload da imagem')
            wcs_header = solveplateastrometry(key,filepath=filepath)
            print('Resultado 2\n', wcs_header)

        if isinstance(wcs_header,fits.Header):
            with fits.open(filepath,'update') as f:
                f[0].header = f[0].header+wcs_header
            session['wcs'] = filepath
            
            return make_response({'message': 'OK'}, 200)

        return make_response(jsonify({'message': 'NO'}),200)


@app.route("/resultado", methods=["POST"])
def create_entry():
    '''
    Rota para receber tabela de table a partir de um envio do navegador
    '''

    req = request.get_json()
    out = pd.DataFrame(req)
    print(out)
    if not out.empty:
        out.to_excel(session['pathname']+'data.xlsx',index=False)
        res = make_response(jsonify({"message": "Arquivo salvo"}), 200)

        return res
    
    return make_response(jsonify({'message': 'Tabela vazia'}), 200)


@app.route("/reiniciar", methods=["POST"])
def reset_data():
    '''
    Essa rota salva uma tabela vazia mesmo que uma já exista
    para que a análise seja reiniciada
    '''

    out = pd.DataFrame(dict(
        ra=[],
        dec=[],
        x=[],
        y=[],
        flux = [],
        j = [],
        k = [],
        tipo=[], # se é obj, src ou sky
        banda=[], # o filtro da imagem e arquivo
        sid=[], # id da estrela copiada
        colors=[], # para colorir de acordo o tipo de objeto
    ))

    out.to_excel(session['pathname']+'data.xlsx',index=False)
    
    return make_response(jsonify({"message": "Tabela reiniciada"}), 200)


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

    Q = Irsa.query_region(crval,catalog='fp_psc',spatial='Cone',\
                          radius=r)[['ra','dec','j_m','k_m']].to_pandas()
    print(Q)
    m = SkyCoord(ra=Q['ra'],dec=Q['dec'], unit=('deg','deg'), frame='icrs')
    idx, d2, _ = match_coordinates_sky(src,m)

    Q.loc[idx[d2>=rw]] = None # retira estrela que não conseguiu chegar perto

    data[['j','k']] = Q[['j_m','k_m']].loc[idx].values
    print(data)
    res = make_response(data.to_json(), 200)

    return res


@app.route('/download/<pathname>/<filename>')
def uploaded_file(pathname,filename):
    return send_from_directory(app.config['UPLOAD_FOLDER']+'/'+pathname,
                               filename)


def getcoef(data, q=0.5):
    '''
    Pega o coeficiente angular da regressão
    '''
    
    data.columns = ['income', 'depend']
    # por algum motivo a tabela não estava fazendo a regreção corretamente, estava entendendo o eixo
    # das variáveis independentes como vários eixos e não apenas um eixo
    table = {'income': data['income'].values.tolist(), 'depend': data['depend'].values.tolist()}

    mod = smf.quantreg('depend ~ income', table)
    res = mod.fit(q=q)

    return res.params['income']



@app.route('/reducao/<dirname>')
def reducao(dirname):
    '''
    Faz a redução de table caso exista a tabela. A redução é feita
    conforme a combinção BVR, VR ou BV.
    '''
    filepath = f"{app.config['UPLOAD_FOLDER']}/{dirname}/data.xlsx"
    output = f"{app.config['UPLOAD_FOLDER']}/{dirname}/result.xlsx"
    filedata = f"{app.config['UPLOAD_FOLDER']}/{dirname}/data.json"
    
    mag = lambda x: -2.5*np.log10(x)

    # try:
    with open(filedata) as f:
        datadir = json.load(f)
        hasB = len(datadir['fitsB']) > 0
        hasR = len(datadir['fitsR']) > 0
    print('B: ', hasB, '. R: ',hasR)
    if hasB:
        if not hasR:
            print('Tem B e não tem R')

    print('abriu data.json')
    #                               Adicionei essa parte para remover duplicatas que porventura surgiram na hora de clicar
    #                                                       Essas colunas garantem que só uma fonte distinta sobreviva na tabela
    try:
        table = pd.read_excel(filepath).drop_duplicates(subset=['sid','banda','j','k','flux'], keep='first')
        if len(table.index) <= 0:
            return "Salve a tabela antes para redução, ela estava sem dados"
    except FileNotFoundError:
        return "Salve a tabela antes para redução, arquivo não encontrado"
    
    table['mag'] = 0.0 # criando a coluna de magnitudes, não pode iniciar como inteiro, pois vai receber float
    

    # Selecionando estrelas dentro do intervalo
    ids = table['tipo']=='src' # índices pra selecionar estrelas
    idx = (table['tipo']!='src')|((-0.1<table['j'][ids]-table['k'][ids])&(table['j'][ids]-table['k'][ids]<1.0))
    table = table[idx]
    ids = table['tipo']=='src' # recalcula o ids
    ido = table['tipo']=='obj' # índices pra selecionar objetos
        
    # Calculando as magnitudes
    for e in set(table['banda']):
        med = np.median(table[(table['banda']==e)&(table['tipo']=='sky')]['flux'])
        idx = (table['banda']==e)&(table['tipo']!='sky')
        table.loc[idx,'mag'] = mag(table['flux'][idx]-med)


    # Calculando índices 2MASS das estrelas
    INDICES = {}
    INDICEO = {}
    # Índices de catálogo
    idref = (table['banda'] == 'fitsV:'+datadir['fitsV'][-1]) & ids
    j = table['j'][idref]
    INDICES['j'] = j.values
    k = table['k'][idref]
    INDICES['k'] = k.values
    j_k = j-k
    INDICES['j-k'] = j_k.values
    if hasR:
        V_R = 0.3458*j_k**3 - 0.5401*j_k**2 + 1.0038*j_k + 0.0451
        INDICES['V-R'] = V_R.values
    V = 1.4688*j_k**3 - 2.325*j_k**2 + 3.5143*j_k + 0.1496 + j
    INDICES['V'] = V.values
    if hasB:
        B_V = 0.2807*j_k**3 - 0.4535*j_k**2 + 1.7006*j_k + 0.0484
        INDICES['B-V'] = B_V.values
        if not hasR:
            B = B_V.values+V.values
            INDICES['B'] = B

    # Construindo tabela lado a lado

    for i in range(len(datadir['fitsV'])):
        if hasB: bstr = f"fitsB:{datadir['fitsB'][i]}"
        vstr = f"fitsV:{datadir['fitsV'][i]}"
        if hasR: rstr = f"fitsR:{datadir['fitsR'][i]}"

        # Índices instrumentais das estrelas
        if hasB: idb =  (table['banda'] == bstr) & ids
        idv = (table['banda'] == vstr) & ids
        if hasR: idr = (table['banda'] == rstr) & ids
        if hasB: b = table['mag'][idb].values
        v = table['mag'][idv].values
        if hasR:
            r = table['mag'][idr].values
            v_r = v-r 
        if hasB:
            b_v = b-v
            if not hasR:
                B_b = B-b

        # armazena b, v e r instrumentais
        if hasB: INDICES[f'b{i}'] = b
        INDICES[f'v{i}'] = v
        if hasR: INDICES[f'r{i}'] = r
        # b-v instrumental
        if hasB:
            INDICES[f'b-v_{i}'] = b_v
            if not hasR:
                INDICES[f'B-b_{i}'] = B_b
        # v-r instrumental
        if hasR: INDICES[f'v-r_{i}'] = v_r
        # V-v instrumental
        V_v = V-v
        INDICES[f'V-v_{i}'] = V_v

        # Índices instrumentais do objeto
        if hasB: idb =  ido & (table['banda'] == bstr)
        idv = ido & (table['banda'] == vstr)
        if hasR: idr = (table['banda'] == rstr) & ido
        # índice bₒ do objeto
        if hasB: bo = table['mag'][idb].values
        # índice vₒ do objeto
        vo = table['mag'][idv].values
        # índice rₒ do objeto
        if hasR: ro = table['mag'][idr].values
        # índice vₒ-rₒ do objeto
        if hasR: v_ro = vo-ro 
        # índice bₒ-vₒ do objeto
        if hasB: b_vo = bo-vo

        # armazena b, v e r instrumentais do objeto
        if hasB: INDICEO[f'b{i}'] = bo
        INDICEO[f'v{i}'] = vo
        if hasR: INDICEO[f'r{i}'] = ro
        # b-v instrumental
        if hasB: INDICEO[f'b-v_{i}'] = b_vo
        # v-r instrumental
        if hasR: INDICEO[f'v-r_{i}'] = v_ro

    S = pd.DataFrame(INDICES)
    print("S: ", S['V'])
    S.to_excel(output)
    O = pd.DataFrame(INDICEO)
    O.to_excel(f".{output.strip(r'.xlsx')}_obj.xlsx")

    # pegando coeficientes
    if hasR:
        Tvr = []
        Tv = []
    if hasB:
        Tbv = []
        if not hasR:
            Tb = []
    for i in range(len(datadir['fitsV'])):
        if hasR:
            # Tvr é o coeficiente inverso de v-r vs. V-R
            Tvr.append(1./getcoef(S[['V-R',f'v-r_{i}']]))
            # Tv é o coeficiente de V-v vs. V-R
            Tv.append(getcoef(S[['V-R',f'V-v_{i}']]))
        # Tbv é o inverso do coeficiente de b-v vs. B-V
        if hasB:
            Tbv.append(1./getcoef(S[['B-V',f'b-v_{i}']]))
            if not hasR:
                Tb.append(getcoef(S[[f'B-b_{i}','B-V']]))

    coef = pd.DataFrame()
    if hasB: coef['Tbv'] = Tbv
    if hasR:
        coef['Tv'] = Tv
        coef['Tvr'] = Tvr

    coef.to_excel('.'+output.strip(r'.xlsx')+'_coef.xlsx')

    # Calcular os índices do objeto
    OBJETO = {}
    for i in range(len(datadir['fitsV'])):
        if hasB:
            b_v = S['B-V'].values + Tbv[i]*(O[f'b-v_{i}'].values - S[f'b-v_{i}'].values)
            if not hasR:
                b = O[f'b{i}'].values + Tb[i]*(O[f'b-v_{i}'].values - S[f'b-v_{i}'].values)
                v = b-O[f'b-v_{i}'].values
        if hasR:
            v_r = S['V-R'].values + Tvr[i]*(O[f'v-r_{i}'].values - S[f'v-r_{i}'].values)
            v = O['v'+str(i)].values + S[f'V-v_{i}'] + Tv[i]*(v_r - S['V-R'].values)

        if hasB: OBJETO[f'b-v_{i}'] = b_v.tolist()
        if hasR: OBJETO[f'v-r_{i}'] = v_r.tolist()
        OBJETO[f'v_{i}'] = v.tolist()

    TEMPO = {}
    for i in session['date'].keys():
        TEMPO[i] = session['date'][i]
    print(TEMPO)

    table = pd.DataFrame(OBJETO)
    juliantable = pd.DataFrame(TEMPO, index=[0])

    with pd.ExcelWriter(f".{output.strip(r'.xlsx')}_objstar.xlsx") as writer:
        print(table)
        table.to_excel(writer)
        juliantable.to_excel(writer,startrow=len(v.tolist()))

    url1 = request.url_root+'download/'+dirname+'/result.xlsx'
    url2 = request.url_root+'download/'+dirname+'/result_obj.xlsx'
    url3 = request.url_root+'download/'+dirname+'/result_coef.xlsx'
    url4 = request.url_root+'download/'+dirname+'/result_objstar.xlsx'

    return f'''Resultados:<br>
    - <a href="{url1}">tabela de estrelas</a>;<br>
    - <a href="{url2}">tabela de índices do objeto</a>;<br>
    - <a href="{url3}">tabela de coeficientes</a>;<br>
    - <a href="{url4}">tabela de índices do objeto com as estrelas</a>.
    '''


def main():
    port = int(os.environ.get("PORT",5000))
    # app.run(host="0.0.0.0", port=port)
    app.run(debug=True)


if __name__ == "__main__":
    main()
