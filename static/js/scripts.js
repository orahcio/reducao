
function radius_onchange(cb_obj,source,tabs) {

    for(let i=0;i<tabs.length;i++) {
        tabs[i].child.renderers[1].glyph.radius = cb_obj.value
    }

    let data = source.data
    const n = data['x'].length
    let aux = `${window.location.pathname}`;
    aux = aux.slice(6,aux.length);
    
    let entry = {
        r: cb_obj.value,
        x: [].slice.call(data['x']), // converte os valores pra um array normal
        y: [].slice.call(data['y']),
        flux: [].slice.call(data['flux']),
        banda: data['banda']
    }
    console.log(entry)
    fetch(`${window.origin}/fluxes`, {
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
        response.json().then(function (table) {
            for(let i=0; i<n; i++) {
                data['flux'][i] = table['flux'][i];
            }
            source.change.emit()
            console.log('Deu certo')
        });
    })
    .catch(function (error) {
        console.log("Fetch error: " + error);
    });

}


const get_name = () => {
    let aux = `${window.location.pathname}`;
    aux = aux.slice(6,aux.length);

    return aux;
}


var activeTab = 'na'
const COLORS = ['red','yellow','blue'];
var N = 0;


function source_onchange(cb_obj, radio=null, graficos=null) {

    var data = cb_obj.data;

    const n = data['x'].length;
    if(n>N) {
        if(data['sid'][n-1]=='na') {
            const active = graficos.active
            const banda = graficos.tabs[active].title
            const labels = radio.labels;
            data['tipo'][n-1] = labels[radio.active];
            data['banda'][n-1] = banda;
            data['colors'][n-1] = COLORS[radio.active];
        }


        entry = {
            tipo: data['tipo'][n-1],
            x: data['x'][n-1],
            y: data['y'][n-1],
            ra: data['ra'][n-1],
            dec: data['dec'][n-1],
            flux: data['flux'][n-1],
            j: data['j'][n-1],
            k: data['k'][n-1],
            banda: data['banda'][n-1]
        }
        console.log(entry)
        fetch(`${window.origin}/add`, {
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
            response.json().then(function (table) {
                data['ra'][n-1] = table['ra'];
                data['dec'][n-1] = table['dec'];
                data['x'][n-1] = table['x'];
                data['y'][n-1] = table['y'];
                data['flux'][n-1] = table['flux'];
                data['j'][n-1] = table['j'];
                data['k'][n-1] = table['k'];

                cb_obj.change.emit()
                console.log('Deu certo')
            });
        })
        .catch(function (error) {
            console.log("Fetch error: " + error);
        });
    };
    // Atualiza o comprimento da tabela
    N = n;
}


const clip = (y) => {
    const ni = y.length
    const nj = y[0].length

    var max = 0 
    var min = 2**64-1

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


function contrast_onchange(cb_obj, tabs, im) {

    let a = tabs.active;
    
    const x = im[a]
    const source = tabs.tabs[a].child.renderers[0].data_source

    let y = source.data.image;
    const ni = y.length
    const nj = y[0].length
    const c = cb_obj.value

    // console.log('x: ', x) 
    for(let i=0;i<ni;i++) {
        for(let j=0;j<nj;j++) {
            y[i][j] = Math.pow(x[i*ni+j]+1e-8,2**c)
        }
    }
    // Clip values
    y = clip(y)

    source.change.emit()
}



const make_entry = (data) => {
    return {
        sid: [].slice.call(data['sid']),
        banda: data['banda'],
        tipo: data['tipo'],
        colors: data['colors'],
        x: [].slice.call(data['x']),
        y: [].slice.call(data['y']),
        ra: [].slice.call(data['ra']),
        dec: [].slice.call(data['dec']),
        flux: [].slice.call(data['flux']),
        j: [].slice.call(data['j']),
        k: [].slice.call(data['k'])
    };
}


function salvar_onclick(source) {
    var data = source.data;
    var entry = make_entry(data);
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
        response.json().then(function (res) {
            // Mudara essa resposta aqui
            console.log('Acabaou de chegar: ',res);
        });
    })
    .catch(function (error) {
        console.log("Fetch error: " + error);
    });

    // Fazer download
    url = `${window.origin}/download/${window.location.pathname.split('/')[2]}/data.xlsx`
    // fetch(url)
    console.log(url)
    var link = document.createElement("a");
    link.download = `${window.location.pathname.split('/')[2]}.xlsx`;
    link.href = url;
    link.click();

}

function send_astrometry(cb_obj, key, source, selected) {
    var entry = source.data;
    var name = `${window.location.pathname}`.split('/')[2];
    name = name + '/' + selected.value.split(':')[1];
    const url = window.origin + '/astrometry_net/'+key.value+'/'+name
    console.log(url)

    if(key.value==='') {
        cb_obj.label = 'Solução de placa do astrometry.net (sem chave)'
        cb_obj.active = false
        cb_obj.change.emit()
    }
    else {
        cb_obj.disabled = true
        cb_obj.label = 'Solução de placa do astrometry.net (aguarde...)'
        cb_obj.change.emit()
        fetch(url, {
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
                console.log(data)
                if(data['message']==='NO') {
                    cb_obj.label = 'Solução de placa do astrometry.net (erro)'
                    cb_obj.disabled = false
                    cb_obj.change.emit()
                }
                else {
                    cb_obj.label = 'Solução de placa do astrometry.net (concluído)'
                    cb_obj.disabled = true
                    cb_obj.change.emit()
                }
            });
        })
        .catch(function (error) {
            console.log("Fetch error: " + error);
        });
    }
}


function send_2mass(source) {
    var data = source.data;
    const n = data['x'].length
    entry = {
        tipo: data['tipo'],
        ra: [].slice.call(data['ra']),
        dec: [].slice.call(data['dec'])
    }

    fetch(`${window.origin}/busca`, {
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
        response.json().then(function (table) {
            // Mudara essa resposta aqui
            for(let i=0; i<n; i++) {
                if(data['tipo'][i]=='src') {
                    data['j'][i] = table['j'][i]
                    data['k'][i] = table['k'][i]
                }
            }
            source.change.emit()
            console.log('ok')
        });
    })
    .catch(function (error) {
        console.log("Fetch error: " + error);
    });

}


function reset_onclick(source,tabela) {
    // Tentei remover os dados da tabela usando javascript
    // mas isso tá muito bugado no bokeh
    fetch(`${window.origin}/reiniciar`, {
        method: "POST",
        credentials: "include",
        body: "nobody",
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
        response.json().then(function (res) {
            // Mudara essa resposta aqui
            console.log('Acabaou de chegar: ',res);
        });
    })
    .catch(function (error) {
        console.log("Fetch error: " + error);
    });

    // window.location.reload();

}


const add_data = async(source, ref, graficos) => {
    // Função que irá copiar os dados da imagem de referência

    console.log(ref.value, graficos.active);
    let data = source.data;
    let active = graficos.active;
    let refv = ref.value;

    let n = data['x'].length; // quantidades de linhas existentes na tabela

    let newdata = {
        'sid': [].slice.call(data['sid']),
        'banda': data['banda'],
        'tipo': data['tipo'],
        'colors': data['colors'],
        'x': [].slice.call(data['x']),
        'y': [].slice.call(data['y']),
        'ra': [].slice.call(data['ra']),
        'dec': [].slice.call(data['dec']),
        'flux': [].slice.call(data['flux']),
        'j': [].slice.call(data['j']),
        'k': [].slice.call(data['k']),
    }
    //let newData = make_entry(data);
    for(let i=0;i<n;i++) {
        // console.log('P: ',refv, 'T', data['banda'][i])
        if(data['banda'][i]===refv && data['tipo'][i]!=='obj') {
            // await new Promise(r => setTimeout(r,1000));
            newdata['sid'].push(i); // copia o índice do ponto original
            newdata['x'].push(data['x'][i]);
            newdata['y'].push(data['y'][i]);
            newdata['flux'].push('na');
            newdata['tipo'].push(data['tipo'][i]);
            newdata['banda'].push(graficos.tabs[active].title);
            newdata['ra'].push(data['ra'][i]);
            newdata['dec'].push(data['dec'][i]);
            newdata['j'].push(data['j'][i]);
            newdata['k'].push(data['k'][i]);
            newdata['colors'].push(data['colors'][i]);
            source.data = newdata;
            source_onchange(source);
        }
    }

    console.log('Copiado!')
}


function f(cb_obj,radio,source,r) {
    console.log('Executado');
    cb_obj.label = 'Apertado';
    // cb_obj.active = false; // Se ativer o botão executa duas vezes no onclik
    cb_obj.change.emit();

    console.log('radio', radio);
    console.log('source', source);
    console.log('circulo', r);
}
