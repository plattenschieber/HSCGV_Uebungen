/* ******** Programmierpraktikum Computergrafik (CGP) **************
 * Aufgabe 1 - "Lichtblick"
 * Created by Peter Kipfer <kipfer@informatik.uni-erlangen.de>
 * Changed by Martin Aumueller <aumueller@uni-koeln.de>
 */

#include <cmath>
#include <vector>
#include <cstring>
#include <cstdlib>
#ifdef _WIN32
#include "xgetopt.h"
#else
#include <getopt.h>
#endif
#include "geoobject.h"
#include "lightobject.h"
#include "ray.h"
#include "param.h"
#include "types.h"

// get definition of our application window class called ApplicationWindow
#include "ApplicationWindow.h"
#include <QApplication>

#ifndef M_PI
#define M_PI 3.1415927
#endif


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

// Description:
// the connection to the parser
extern int openSceneFile(char *s);
extern int closeSceneFile(void);
extern int inputparse(void);
extern int cleanUp(void);

//! get scene description from the parser
/*!
  Get the scene description from the parser.
  \param inputfile Filename of the scene
 */
void
ReadScene(char *inputfile)
{
   openSceneFile(inputfile);
   inputparse();
   closeSceneFile();
}

// scene storage
//! the list of geometric objects in the scene
extern std::vector<GeoObject*>           g_objectList;
//! the list of light sources in the scene
extern std::vector<LightObject*>         g_lightList;
//! the list of surface properties of the objects
extern std::vector<GeoObjectProperties*> g_propList;

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
   startWindow(argc, argv);
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

   // parse the input file
   ReadScene(filename);

   unsigned int maxColVal   = 255;

   unsigned int cCol[3];

   // open output file
   FILE *outfile;
   if ( (outfile = fopen(outfilename,"w")) == NULL ) {
      ERR("open out file failed");
      exit(1);
   }

   // write header
   fprintf(outfile,"P3\n%d %d\n%d\n",g_scene.picture.Xresolution,g_scene.picture.Yresolution,maxColVal);

   fprintf(stderr,"%s rendering %s:\n",argv[0],outfilename);


   // TODO take view parameters from file into account and remove hardcoded values
   // setup viewport, its origin is bottom left

   // setup camera coordsys
   Vec3d eye_dir = (g_scene.view.lookat - g_scene.view.eyepoint).getNormalized();
   Vec3d eye_right = (eye_dir^g_scene.view.up).getNormalized();
   Vec3d eye_up = eye_dir^eye_right*-1;

    // calculatehe dimensions of the viewport using the scene's camera
    float height = 2 * tan(M_PI/180 * .5 * g_scene.view.fovy);
    float width = height * g_scene.view.aspect;

    // compute delta steps in each direction
    Vec3d deltaX = eye_right * (width / g_scene.picture.Xresolution);
    Vec3d deltaY = eye_up * (height / g_scene.picture.Yresolution);

    // this should be bottom left
    Vec3d bottomLeft = g_scene.view.eyepoint + eye_dir - deltaX*g_scene.picture.Xresolution/2 - deltaY*g_scene.picture.Yresolution/2;

   // normal ray tracing: the color of the center of a pixel is computed
   for (unsigned int sy=g_scene.picture.Yresolution ; sy > 0 ; --sy) {

      fprintf(stderr,"\rscanline %4d (%3d%%)",sy,(g_scene.picture.Yresolution-sy)*100/g_scene.picture.Yresolution);

      for (unsigned int sx=0 ; sx < g_scene.picture.Xresolution ; ++sx) {
         // the center of the pixel we are looking at right now
         Vec3d point = bottomLeft + deltaX*sx + deltaY*sy + deltaX/2 + deltaY/2;

         // the direction of our look
         Vec3d dir = point - g_scene.view.eyepoint;

         // create ray from view.eyepoint to view.lookat
         Ray theRay(g_scene.view.eyepoint,dir.getNormalized(),0,g_objectList,g_lightList);

         // compute the color
         Color col = theRay.shade();

         // in case we are using antialiasing, calculate the color of this pixel by averaging
         if (antialiasing) {
               // scale the midpoint color since we are going to use 5 points to average our color
               col *= 0.2;

               // besides taking shooting a ray through the midpoint 'o', we calculate
               // the pixels color by shooting 4 more rays  through the points 'x'
               // and averaging their values
               //  -------
               // | x   x |
               // |   o   |
               // | x   x |
               //  --------
               for(float dx = -1/4.; dx <= 1/4.; dx+=1/2.)
               {
                 for(float dy = -1/4.; dy <= 1/4.; dy+=1/2.)
                 {
                   Vec3d superSamplePoint = point + deltaX*dx + deltaY*dy;
                   Vec3d superSampleDir = superSamplePoint - g_scene.view.eyepoint;

                   // create ray from view.eyepoint to view.lookat
                   Ray theRay(g_scene.view.eyepoint,superSampleDir.getNormalized(),0,g_objectList,g_lightList);
                   col += theRay.shade()*0.2;//color, recursive_ray_trace(eye, ray, 0));
                 }
               }
         }

         // clamp the computed color value to 0...maxColVal
         for (unsigned int cc=0; cc<3; cc++) {
            if(col[cc] < 0.0)
               cCol[cc] = 0;
            else if(col[cc] > 1.0)
               cCol[cc] = maxColVal;
            else
               cCol[cc] = (unsigned int)(maxColVal * col[cc]);
         }

         // write the clamped color to the output file
         fprintf(outfile,"%4d %4d %4d ",cCol[0],cCol[1],cCol[2]);

      } // foreach x

      fprintf(outfile,"\n");

   } // foreach y

   fprintf(stderr,"\ndone\n");

   // clean up
   fclose(outfile);
   cleanUp();
}
