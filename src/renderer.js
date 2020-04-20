window.$ = window.jQuery = require('jquery'); // not sure if you need this at all
var app = require('electron').remote;
var dialog = app.dialog;
var fs = require('fs');
var py = require('python-shell');
var path = require('path');


const pypath = path.resolve(__dirname, '..', 'src', 'backend', 'procesamiento.py');
const weightsPath = path.join(__dirname, '../src/weights/model-007-0.961.hdf5');
const tmpPath = path.join(__dirname, '../src/tmp');
var holder = document.getElementById('drag-file');
var file_up = document.getElementById('fileUpload');
var file = "";
const f_img_dip = tmpPath + "/deepSeg.png";
const f_img_sato = tmpPath + "/SatoSeg.png";
const f_img_line = tmpPath + "/lineas.png";

var imagen_dl = document.getElementById('imgDL');
var imagen_fran = document.getElementById('imgFran');
var imagen_linea = document.getElementById('imgLine');


const win = app.BrowserWindow.getAllWindows()[0];
const ses = win.webContents.session; 




// 1. Require the installed module
const customTitlebar = require('custom-electron-titlebar');

// 2. Create the custom titlebar with your own settings
//    To make it work, we just need to provide the backgroundColor property
//    Other properties are optional.
let MyTitleBar = new customTitlebar.Titlebar({
    backgroundColor: customTitlebar.Color.fromHex('#232323')
});

// 3. Update Titlebar text
MyTitleBar.updateTitle('Segementacion de Fondo de ojo');



file_up.onchange = (e) => {
    
    imagen_fran.src = "img/icon.png";
    imagen_dl.src = "img/ml.png";
    imagen_linea.src = "img/ml.png";
    holder.style.boxShadow = "-0px 0px 18px 12px rgba(255,255,255,0.30)";
    file = e.target.files[0]['path'];
    var reader = new FileReader(),
        image = new Image();
    reader.readAsDataURL(e.target.files[0]);
    reader.onload = function (_file) {
        document.getElementById('imgPrime').src = _file.target.result;
        // document.getElementById('imgPrime').style.display = 'inline';

    };

    var options = {
        args: [file, weightsPath, tmpPath],
    };

    py.PythonShell.run(pypath, options, function (err, results) {
        if (err) throw err;
        console.log("Listo");
        //console.log(results);

        imagen_fran.src = "../src/tmp/SatoSeg.png";
        imagen_dl.src = "../src/tmp/deepSeg.png";
        imagen_linea.src = "../src/tmp/lineas.png";

    });


};


holder.onclick = () => {
    imagen_fran.src = "../src/tmp/SatoSeg.png";
        imagen_dl.src = "../src/tmp/deepSeg.png";
        imagen_linea.src = "../src/tmp/lineas.png";
}

holder.ondragover = () => {
    imagen_fran.src = "img/icon.png";
    imagen_dl.src = "img/ml.png";
    imagen_linea.src = "img/ml.png";
    holder.style.boxShadow = "-2px 0px 39px 12px rgba(63,135,245,0.74)";

    return false;
};

holder.ondragleave = () => {
    holder.style.boxShadow = "none";
    return false;
};

holder.ondragend = () => {
    holder.style.boxShadow = "none";
    return false;
};

holder.ondrop = (e) => {
    ses.clearCache(() => {
        console.log("Cache cleared!");
      });
    e.preventDefault();

    holder.style.boxShadow = "-0px 0px 18px 12px rgba(255,255,255,0.30)";

    for (let f of e.dataTransfer.files) {
        console.log('File(s) you dragged here: ', f.path)
        file = f.path;
    }
    document.getElementById('imgPrime').src = file;

    var options = {
        args: [file, weightsPath, tmpPath],
    };

    py.PythonShell.run(pypath, options, function (err, results) {
        if (err) throw err;
        imagen_fran.src = "../src/tmp/SatoSeg.png";
        imagen_dl.src = "../src/tmp/deepSeg.png";
        imagen_linea.src = "../src/tmp/lineas.png";
        console.log("Listo");
    });


    return false;
};


