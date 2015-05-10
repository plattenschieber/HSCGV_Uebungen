// get definition of our application window class called ApplicationWindow
#include "ApplicationWindow.h"
#include <QApplication>
#include "raytracer.h"

/*!
  The program entry point: the command line arguments are parsed, the scene
  description is read from the input file and the color of each pixel of
  the resulting image is computed and written to the output file.
  \param argc number of command line arguments
  \param argv array of strings of the command line arguments
 */
int
main (int argc, char *argv[])
{
    // use true color visual on SGIs
    QApplication::setColorSpec(QApplication::ManyColor);

    // create application object and pass all the arguments.
    // this will filter out all Qt-relevant commands and returns
    // a modified command line (containing your options).
    QApplication app(argc,argv);

    // create main widget of our application.
    // you have to pass the application object.
    ApplicationWindow *win = new ApplicationWindow();

    // set an initial size of the main window (not too small)
    win->resize(500,300);

    // show the window
    win->show();

    // if the last window (here the app window) gets closed by the
    // user (i.e. by clicking on the close button of the window frame) then
    // the window will emit a "lastWindowClosed" signal. We connect this
    // signal with the "quit" slot of the application object. On quit the
    // application will leave the event loop started below.
    app.connect( &app, SIGNAL(lastWindowClosed()), &app, SLOT(quit()) );

    // enter main event loop of the Qt application object.
    // this object will reveive all mesages from the GUI and pass them
    // along to our widgets (e.g. the main application window). These
    // events are called signals and they trigger the slot-methods they
    // are connected to.
    // if you trigger the "quit" slot of the application object then the
    // method call will return.
    return app.exec();
}
