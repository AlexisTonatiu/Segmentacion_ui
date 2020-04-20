const {app, BrowserWindow} = require('electron');
const url = require('url');
const path = require('path');


let mainWindow

function createWindow () {
    mainWindow = new BrowserWindow({
      width: 1024,
      height: 600,
      resizable: true,
      frame: false,
      webPreferences: {
        preload: path.join(__dirname, 'preload.js'),
        nodeIntegration: true,
      }
      
    })
  
    mainWindow.loadFile(path.join(__dirname, '../views/index.html'))

    mainWindow.setMenuBarVisibility(false)
  
    mainWindow.on('closed', function () {
      mainWindow = null
    })
  }
  
  app.on('ready', createWindow)

  app.on('window-all-closed', function () {
  
    if (process.platform !== 'darwin') app.quit()
  })
  
  app.on('activate', function () {
    if (mainWindow === null) createWindow()
  })

  
 