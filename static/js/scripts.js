
function radius_onchange(cb_obj,source) {

    let data = source.data
    const n = data['x'].length
    let aux = `${window.location.pathname}`;
    aux = aux.slice(6,aux.length);
    
    let entry = {
        name: aux,
        r: cb_obj.value,
        x: data['x'],
        y: data['y'],
        flux: data['flux']
    }

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

function source_onchange(cb_obj, radio, r) {
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

    entry = {
        name: aux,
        r: r,
        x: data['x'][n-1],
        y: data['y'][n-1],
        ra: data['ra'][n-1],
        dec: data['dec'][n-1],
        flux: data['flux'][n-1]
    }
    console.log('r: ', r)
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
            data['flux'][n-1] = table['flux'];
            cb_obj.change.emit()
            console.log('Deu certo')
        });
    })
    .catch(function (error) {
        console.log("Fetch error: " + error);
    });
}

function contrast_onchange(cb_obj, source, im) {
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
}

function salvar_onclick(source) {
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
            // Mudara essa resposta aqui
            console.log('Acabaou de chegar: ',data['x']);
        });
    })
    .catch(function (error) {
        console.log("Fetch error: " + error);
    });
}

function send_astrometry(cb_obj, key, source) {
    var entry = source.data;
    var name = `${window.location.pathname}`;
    name = name.slice(6,name.length);
    const url = window.origin + '/astrometry_net/'+key.value+'/'+name

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

function f(cb_obj,radio,source,r) {
    console.log('Executado');
    cb_obj.label = 'Apertado';
    // cb_obj.active = false; // Se ativer o botão executa duas vezes no onclik
    cb_obj.change.emit();

    console.log('radio', radio);
    console.log('source', source);
    console.log('circulo', r);
}
