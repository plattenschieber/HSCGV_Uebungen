// get definition of our application window class called ApplicationWindow
#include "ApplicationWindow.h"
#include <QApplication>
#include "raytracer.h"

//! program usage output
/*!
  Convenient definition of the usage output that should be printed, if
  there is a problem with the command line.
 */
#define USAGE \
   "%s: -f <filename> [-h]\n\
\n\
\t-f filename     input/output filename\n\
\t-a              antialiasing flag on\n\
\t-h              this help message\n"

//! how long can filenames get ?
#define MAXNAMLEN 1024

// Description:
/*!
  Check whether the filename has a suffix and 
  modify it according to action indicated by a flag.
  \param filename The filename to process
  \param suffix The suffix to process
  \param addOrRemove If set to false, remove the suffix - if set to true, add it.
 */
void
checkSuffix(char *filename, const char *suffix, bool addOrRemove)
{
   bool suffixExists = false;
   unsigned int filenameLength = strlen(filename);
   unsigned int suffixLength   = strlen(suffix);

   //! get position of suffix within filename
   int suffixIndex = filenameLength - suffixLength;
   if (suffixIndex > 0)
      suffixExists = (strcmp ((filename + suffixIndex), suffix)) ? false : true;
   else
      suffixIndex = filenameLength;

   //! add or remove suffix
   if ((addOrRemove == false) && (suffixExists == true))
      while (filename[suffixIndex] != '\0') filename[suffixIndex++] = '\0';
   else if ((addOrRemove == true) && (suffixExists == false))
      strncat (filename, suffix, suffixLength);

}

int startWindow(int argc, char *argv[])
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

// Description:
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
   int c;
   bool haveFilename = false, antialiasing = false;
   char filename[MAXNAMLEN - 5] = "";
   char outfilename[MAXNAMLEN - 5] = "";

   // parse command line
   while ((c = getopt(argc, argv, "bd:f:m:sh:a")) != EOF) {
      switch (c) {
         case 'f':
            sprintf(filename,"%s",optarg);
            haveFilename = true;
            break;
         case 'h':
            fprintf(stderr, USAGE, argv[0]);
            exit(0);
         case 'a':
            fprintf(stderr, "antialiasing is set\n");
            antialiasing = true;
            break;
         default:
            fprintf(stderr, USAGE, argv[0]);
            exit(1);
            break;
      }
   }

   // build filenames
   if (!haveFilename) {
      fprintf(stderr,"ERROR: no filename given\n");
      fprintf(stderr, USAGE,argv[0]);
      exit(1);
   }
   checkSuffix(filename, ".data", false);
   checkSuffix(filename, ".ppm", false);
   strcpy(outfilename,filename);
   checkSuffix(filename, ".data", true);
   checkSuffix(outfilename, ".ppm", true);

//    Raytracer *tracer = new Raytracer(filename, outfilename, antialiasing);
//    tracer->start();
   // start visualization
   startWindow(argc, argv);
}
