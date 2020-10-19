
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

function f(cb_obj) {
    cb_obj.label = 'Apertado'
    cb_obj.change.emit()
    console.log('Executado')
}
