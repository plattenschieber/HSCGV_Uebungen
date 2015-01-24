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
   bool haveFilename = false;
   char filename[MAXNAMLEN - 5] = "";
   char outfilename[MAXNAMLEN - 5] = "";

   // parse command line
   while ((c = getopt(argc, argv, "bd:f:m:sh")) != EOF) {
      switch (c) {
         case 'f':
            sprintf(filename,"%s",optarg);
            haveFilename = true;
            break;
         case 'h':
            fprintf(stderr, USAGE, argv[0]);
            exit(0);
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


   // the bottom left of the virtual screen
   Vec3d bottomLeft = g_scene.view.lookat - Vec3d(150., 150., 0.);

   // normal ray tracing: the color of the center of a pixel is computed
   for (unsigned int sy=300 ; sy > 0 ; --sy) {

      fprintf(stderr,"\rscanline %4d (%3d%%)",sy,(g_scene.picture.Yresolution-sy)*100/g_scene.picture.Yresolution);

      for (unsigned int sx=0 ; sx < 300 ; ++sx) {
         // the center of the pixel we are looking at right now
         Vec3d point = bottomLeft + Vec3d(sx, sy, 0.);
         // the direction of our look
         Vec3d dir = point - g_scene.view.eyepoint;

         // create ray from view.eyepoint to view.lookat
         Ray theRay(g_scene.view.eyepoint,dir.getNormalized(),0,g_objectList,g_lightList);

         // compute the color
         Color col = theRay.shade();

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
