/* ******** Programmierpraktikum Computergrafik (CGP) **************
 * Aufgabe 3 - "Lichtblick"
 * Created by Peter Kipfer <kipfer@informatik.uni-erlangen.de>
 * Changed by Martin Aumueller <aumueller@uni-koeln.de>
 *
 * File: input.y
 *   Parse scene description
 */

%{

/* -----------------------------------------------------------------------------
 * First, we begin with setting up the language specific
 * part of the parser.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "types.h"
#include "vector.h"
#include "param.h"
#include "geoquadric.h"
#include "lightobject.h"

#include "input_yacc.h"

/* create scene storage */
std::vector<GeoObject*>           g_objectList;
std::vector<LightObject*>         g_lightList;
std::vector<GeoObjectProperties*> g_propList;
Scene                             g_scene;

/* parser state storage */
static Color currentColor = 0;
static Vec3d currentDirection = 0;
static Vec3d currentReflectance = 0;
static double currentSpecular = 0;
static double currentMirror = 0;
static std::vector<Vec3d> vertexList;

/*
 * table of the keywords that we use above
 */
struct KeywordType {
  const char *keyword;
  int  tokentype;
}
keywords[] = {
  { "resolution", RESOLUTION },
  { "eyepoint", EYEPOINT },
  { "lookat", LOOKAT  },
  { "fovy", FOVY },
  { "aspect", ASPECT  },
  { "numsurfaces", NUMSURFACES },
  { "object", OBJECT },
  { "poly", POLY  },
  { "polygon", POLY  }, /* equivalent spelling */
  { "numvertices", NUMVERTICES },
  { "vertex", VERTEX },
  { "numpolygons", NUMPOLYGONS },
  { "quadric", QUADRIC },
  { "numproperties", NUMPROPERTIES },
  { "ambient", AMBIENT },
  { "diffuse", DIFFUSE },
  { "specular", SPECULAR },
  { "mirror", MIRROR },
  { "numobjects", NUMOBJECTS },
  { "ambience", AMBIENCE },
  { "background", BACKGROUND },
  { "numlights", NUMLIGHTS },
  { "direction", DIRECTION },
  { "colour", COLOUR },
  { "color", COLOUR }, /* equivalent spelling */
  { "up", UP }, /* equivalent spelling */
  { NULL, -1 }, /* end of table */
};

/* where are we in the input */
static int linenum=0;

/* emit a warning message */
static void
yywarn(const char *s)
{
  ERR("WARNING: line " << linenum << ": " << s);
}

/* emit an error message and exit */
static void 
yyerror(const char *s)
{
  ERR("ERROR: line " << linenum << ": " << s);
  exit(1);
}

static FILE *YYIN = NULL;

/*
 * open a file for use by the parser
 */
int 
openSceneFile(char *sceneFile)
{
  /* complain if something seems wrong */
  if ( YYIN != NULL ) {
    ERR("openSceneFile : already open");
    exit(1);
  }

  if ( sceneFile == NULL || sceneFile[0] == '\0' ) {
    ERR("openSceneFile : no file name given");
    exit(1);
  }

  if ( (YYIN = fopen(sceneFile,"r")) == NULL ) {
    perror("openScene : can't open file");
    exit(1);
  }

  /* otherwise, get ready to parse */
  linenum = 1;

  return 0;
}


/*
 * close a file that we had open
 */
int 
closeSceneFile()
{
  if ( YYIN == NULL ) {
    ERR("closeSceneFile : not open");
    exit(1);
  }

  fclose(YYIN);
  YYIN = NULL;
  linenum = -1;
  return 0;
}

/*
 * free all memory allocated by the parser
 */
int 
cleanUp()
{
  /* delete allocated objects */
  for (std::vector<LightObject*>::iterator iter1 = g_lightList.begin();
       iter1 != g_lightList.end();
       iter1++)
    delete (*iter1);
  g_lightList.clear();

  for (std::vector<GeoObject*>::iterator iter2 = g_objectList.begin();
       iter2 != g_objectList.end();
       iter2++)
    delete (*iter2);
  g_objectList.clear();

  for (std::vector<GeoObjectProperties*>::iterator iter3 = g_propList.begin();
       iter3 != g_propList.end();
       iter3++)
    delete (*iter3);
  g_propList.clear();

  return 0;
}

/* what sort of tokens might we encounter */
enum { BADTOKEN, CHARTOKEN, INTTOKEN, FLOATTOKEN };

/*
 * skip upcoming white space and comments in the input
 * stream.  if after doing that we're at the end-of-file
 * mark return FALSE, otherwise, return TRUE (a potential token
 * is next in line)
 */
static int 
skipWhitespace(void)
{
  int c = getc(YYIN);
  if ( c == EOF ) return 0;

  while (1) {
    /* new line */
    if ( c == '\n' ) {
      linenum++;

      c = getc(YYIN);
      if ( c == EOF ) return 0;
      continue;
    }

    /* comment */
    if ( c == '#' ) {
      /* eat up everything until the end of the line/EOF */
      while ( c != '\n' ) {
	c = getc(YYIN);
	if ( c == EOF ) return 0;
      }

      /* try again */
      continue;
    }

    /* normal white space */
    if ( strchr("\t ",c) != NULL ) {
      while ( strchr("\t ",c) != NULL ) {
	/* get the next bit of white space. */
	c = getc(YYIN);
	if ( c == EOF ) return 0;
      }

      /* try again */
      continue;
    }

    /* if we got here, then we hit a possible token */
    if ( ungetc(c,YYIN) == EOF ) return 0;
    return 1;
  }
}


/*
 * check to see if this is really a keyword
 * if so, return it's "number", otherwise
 * return badness...
 */
static int 
checkKeyword(char *token)
{
  int i;

  /* compare the candidate against each keyword... */
  for ( i = 0 ; keywords[i].keyword != NULL ; i++ ) {
    if ( !strcmp(token,keywords[i].keyword) ) break;
  }

  /* when we get here, we've either found it or hit the
   * end of the list.  in either case, the return value
   * we want is right there... */
  return keywords[i].tokentype;
}


/*
 * collect up all characters in the token, and
 * try to figure out what it could be...
 */
static int 
collectToken(char *token)
{
  char *t = token;
  int tokentype = BADTOKEN;  /* assume the worst... */
  int c;

  /* always assume that we can get at least one character of something */
  c = getc(YYIN);
  while ( c != EOF && strchr("\t\n# ",c) == NULL ) {
    *t++ = c;
    c = getc(YYIN);
  }

  /* push that extra character we just got back into the input stream */
  if( ungetc(c,YYIN) == EOF ) return BADTOKEN;

  /* finish off the string, and start from the beginning again */
  *t = '\0';
  t = token;

  /* now, check it against our different possibilities */

  /* first, some kind of number... */
  if ( strchr("+-.",*t) != NULL || isdigit(*t) ) {
    /* assume that it's an integer until we get contrary evidence */
    tokentype = INTTOKEN;

    for ( ; *t ; t++ ) {
      if ( isdigit(*t) ) {
	continue;
      } else if ( strchr(".+-eE",*t) != NULL ) {
	/* this isn't a very complete check, but, hey it's only
	 * sample code, you should be making it better! */
	tokentype = FLOATTOKEN;
      } else {
	tokentype = BADTOKEN;
	break;
      }
    }

  } else {
    tokentype = CHARTOKEN;
    for ( ; *t ; t++ ) {
      if ( isgraph(*t) ) {
	continue;
      } else {
	tokentype = BADTOKEN;
	break;
      }
    }
  }

  /* otherwise, we go with whatever assumption we made */
  return tokentype;
}


/*
 * put together a multi-part error message...
 */
static void 
complain(const char *s, const char *t)
{
  char msg[200];

  strcpy(msg,s);
  strcat(msg,t);
  yyerror(msg);
}


/*
 * tokenizer for our grammar
 */

static int yylex(void);

/*
 * so that we can "count" through lists 
 * while we're parsing them, without having
 * to pass things up and down the parse tree
 */
static int surfaceCounter = 0;
static int propertyCounter = 0;
static int objectCounter = 0;
static int lightCounter = 0;
static int vertexCounter = 0;
static int polyCounter = 0;
static int indexCounter = 0;

/*
 * checking to make sure indices are okay
 */
static int nproperties = 0;
static int nsurfaces = 0;
static int nobjs = 0;
static int nlights = 0;
static int nvertices = 0;
static int npolys = 0;
static int nindices = 0;

/*
 * so that we can keep track if a particular parameter appeared
 * in our input or not
 */
static int resolutionSeen = 0;
static int aspectSeen = 0;
static int fovySeen = 0;
static int upSeen = 0;
static int eyepointSeen = 0;
static int lookatSeen = 0;


/* -----------------------------------------------------------------------------
 * OK - all set, now let's define our grammar
 * make your modifications here
 */

%}

/* we have to read in integers and floating point values, so we need this union */
%union
{
  int intval;
  float floatval;
}

%type <intval> index
%type <floatval> colourVal realVal angleVal zeroToOneVal

%token <intval> INTEGER
%token <floatval> FLOAT
%token RESOLUTION EYEPOINT LOOKAT UP FOVY ASPECT
%token NUMSURFACES
%token OBJECT QUADRIC POLY
%token NUMVERTICES NUMPOLYGONS VERTEX
%token NUMPROPERTIES
%token AMBIENT DIFFUSE SPECULAR MIRROR
%token NUMOBJECTS 
%token AMBIENCE BACKGROUND
%token NUMLIGHTS DIRECTION COLOUR

%%

/* a scene description consists of picture and viewing parameters,
 * the description of the ambient light, and the geometry specification
 */
scene 
  :
    {
      /* initialize scene parameters to default values */
      g_scene.ambience = 0;
      g_scene.picture.Xresolution = 300;
      g_scene.picture.Yresolution = 300;
      g_scene.picture.background = 0;
      g_scene.view.fovy = 55;
      g_scene.view.aspect = 1.0;
      g_scene.view.up = Vec3d(0.0, 1.0, 0.0);
      g_scene.view.lookat = Vec3d(-15.0, 25.0, -40.0);
      g_scene.view.eyepoint = Vec3d(1.0, 0.0, 0.0);
      g_scene.view.eyepoint = Vec3d(-100.0, 150.0, 150.0);
    }
    picture_parameters some_viewing_parameters global_lighting_parameters geometry
;

some_viewing_parameters
  :
    {
      VOUT("processing viewing parameters...");
    }
viewing_parameters
    {
      /* not all of the possible viewing parameters have to be specified,
       * but we emit a warning if some of them have been omitted
       */
      if (!lookatSeen)
	yywarn("LOOKAT missing from viewing parameters");
      if (!upSeen)
	yywarn("UP missing from viewing parameters");
      if (!aspectSeen)
	yywarn("ASPECT missing from viewing parameters");
      if (!fovySeen)
	yywarn("FOVY missing from viewing parameters");
      if (!eyepointSeen)
	yywarn("EYEPOINT missing from viewing parameters");
      if (!resolutionSeen)
	yywarn("RESOLUTION missing from viewing parameters");
    }
;

/* the following allows you to specify the viewing parameters in any order */
viewing_parameters 
  : viewing_parameters viewing_parameter
  | /* empty */
;

/* one or more picture parameters may be specified */
picture_parameters
  : picture_parameters picture_parameter
  | /* empty */
;

/* resolution and background color are the picture parameters */
picture_parameter
  : resolution
  | background
;


/* the eyepoint, lookat, the fovy, the aspect ratio and the up vector
 * make up the viewing parameters
 */
viewing_parameter
  : eyepoint
  | lookat
  | fovy
  | aspect
  | up
;

/* parse picture resolution and set flag that it was specified */
resolution
  : RESOLUTION index index
    { TOUT((stderr,"resolution %d %d\n", $2, $3));
    }
;

/* parse background color */
background
  : BACKGROUND colourVal colourVal colourVal
    { TOUT((stderr,"background %f %f %f\n", $2, $3, $4));
    }
;


/* parse the eyepoint (the observer's position) and set flag that it was specified */
eyepoint
  : EYEPOINT realVal realVal realVal
    { TOUT((stderr,"eyepoint %f %f %f\n", $2, $3, $4));
    }
;

/* parse the lookat (the point the observer is looking at) and set flag that it was
 * specified */
lookat
  : LOOKAT realVal realVal realVal
    { TOUT((stderr,"lookat %f %f %f\n", $2, $3, $4)); 
    }
;

/* parse the up vector, a vector indicating where the top of the picture is */
up
  : UP realVal realVal realVal
    { TOUT((stderr,"up %f %f %f\n", $2, $3, $4));
    }
;

/* read field of view in the y-direction, this is given in degrees */
fovy
  : FOVY angleVal
    { TOUT((stderr,"fovy %f\n", $2)); 
    }
;

/* parse the aspect ratio specification */
aspect
  : ASPECT realVal
    { TOUT((stderr,"aspect %f\n", $2));
    }
;

global_lighting_parameters
  : global_lighting_parameters global_lighting
  | /* empty */
;

/* parse ambient light specification */
global_lighting
  : AMBIENCE colourVal colourVal colourVal
    { TOUT((stderr,"ambience %f %f %f\n", $2, $3, $4));
      g_scene.ambience = Color($2, $3, $4); 
    }
;

/* process the description of the geometric objects making up
 * the scene: the surfaces are parsed first, then their different
 * properties followed by the light sources and finally these
 * properties are bound to the surfaces
 */
geometry 
  : 
    {
      VOUT("processing surfaces...");
    }
surface_section 
    {
      VOUT("processing properties...");
    }
property_section 
    {
      VOUT("processing lighting...");
    }
lighting_section
    {
      VOUT("processing objects...");
    }
object_section 
;

/* start parsing the surfaces */
surface_section
  : NUMSURFACES index
    {
      nsurfaces = $2;
      /* preallocate memory */
      g_objectList.reserve($2);
      /* we're going to be counting in another part of the parser... */
      surfaceCounter = 0;
    }
surfaces
;

/* several surfaces follow each other */
surfaces 
  : surfaces one_surface
  | one_surface
;

/* we know quadric surfaces and polygonal surfaces */
one_surface
  : quadric_surface
  | polygon_surface
;

/* a quadric surface is given by the 10 parameters for
 * its implicit representation
 */
quadric_surface
  : OBJECT QUADRIC realVal realVal realVal realVal realVal 
                   realVal realVal realVal realVal realVal
    {
      if ( surfaceCounter >= nsurfaces ) {
	yywarn("too many surfaces specified (ignoring) ");
      } else {
	g_objectList.push_back(new GeoQuadric($3, $4, $5, $6, $7,
					    $8, $9, $10, $11, $12));
	surfaceCounter++;
      }
    }
;

/* The first step for creating a polygon surface is building up
 * a list of vertices. Then a list of polygons is created via
 * indices into this vertex list. */
polygon_surface
  : OBJECT POLY 
    { 
      if ( surfaceCounter >= nsurfaces ) {
        yywarn("too many surfaces specified (ignoring)");
      } else {
        TOUT((stderr,"object poly\n")); 
      }
    }
vertex_section polygon_section
    {
      if ( surfaceCounter < nsurfaces ) {
        surfaceCounter++;
      }
    }
;

/* start a new list of vertices */
vertex_section
  : NUMVERTICES index 
    { TOUT((stderr,"numvertices %d\n", $2));
      nvertices = $2;
      /* this starts a new list of vertices: clear the old one */
      vertexList.clear();
    }
vertices
    { TOUT((stderr,"\n")); }
;

/* vertices follow each other */
vertices
  : vertices one_vertex
  | one_vertex
;

/* parse one vertex and store it into our temporary vertex list */
one_vertex
  : VERTEX realVal realVal realVal
    { TOUT((stderr,"vertex %f %f %f\n", $2, $3, $4));
      if(vertexCounter >= nvertices) {
        yywarn("too many vertices specified (ignoring)");
      } else {
        vertexList.push_back(Vec3d($2,$3,$4));
      }
    }
;

/* produce a polygon object from the vertices specified above */
polygon_section
  : NUMPOLYGONS index 
    { TOUT((stderr,"numpolygons %d\n", $2)); 
      npolys = $2;
      /* this starts a new polygon surface:
       * clear the old list of polygons */
      polyCounter = 0;
    }
polygons
    { TOUT((stderr,"\n"));
    }
;

/* a polygon object consists of several polygons */
polygons 
  : polygons one_polygon
  | one_polygon
;

/* start parsing one polygon */
one_polygon
  : POLY index 
    { TOUT((stderr,"polygon %d ", $2));
      nindices = $2;
      /* clear the old polygon as a new one is started */
      indexCounter = 0;
    }
indices
    { TOUT((stderr,"\n")); 
      /* add polygon to the list of polygons belonging to this surface */
      polyCounter++;
    }
;

/* read the indices into the vertex list that describe one polygon */
indices
  : indices index
    { TOUT((stderr," %d", $2));
      if($2<=0 || $2>nvertices) {
        yywarn("vertex index out of range (ignoring)");
      } else if(indexCounter > nindices) {
        yywarn("too many vertices specified (ignoring)");
      } else {
	/* add a vertex to a polygon */
        indexCounter++;
      }
    }
  | index
    { TOUT((stderr," %d", $1));
      if($1<=0 || $1>nvertices) {
        yywarn("vertex index out of range (ignoring)");
      } else if(indexCounter > nindices) {
        yywarn("too many vertices specified (ignoring)");
      } else {
	/* add a vertex to a polygon */
        indexCounter++;
      }
    }
;

/* parse object properties */
property_section
  : NUMPROPERTIES index
    { 
      nproperties = $2;
      /* preallocate memory */
      g_propList.reserve($2);
      propertyCounter = 0;
    }
properties
;

/* several properties can be specified */
properties
  : properties one_property
  | one_property
;

/* one property comprises several color values */
one_property
  : diffuse specular mirror
    { 
      if ( propertyCounter < nproperties ) {
	/* on to the next property */
	g_propList.push_back(new GeoObjectProperties(currentReflectance,
						   currentSpecular,
						   currentMirror));
	propertyCounter++;
      } else {
	yywarn("too many properties specified (ignorning)");
      }
    }
;

/* parse the diffuse reflection coefficients */
diffuse
  : DIFFUSE zeroToOneVal zeroToOneVal zeroToOneVal
    { 
      if ( propertyCounter < nproperties ) {
	TOUT((stderr,"diffuse %f %f %f\n", $2, $3, $4));
	currentReflectance = Vec3d($2, $3, $4);
      }
    }
;

/* parse the parameters for specular reflection */
specular
  : SPECULAR  zeroToOneVal
    {
      if ( propertyCounter < nproperties ) {
	TOUT((stderr,"specular %f\n", $2));
	currentSpecular = $2;
      }
    }
;

/* parse the mirror coefficient */
mirror
  : MIRROR zeroToOneVal
    { 
      if ( propertyCounter < nproperties ) {
	TOUT((stderr,"mirror %f\n", $2));
	currentMirror = $2;
      }
    }
;


/* start with the section describing the surface <--> property bindings */
object_section
  : NUMOBJECTS index
    { 
      nobjs = $2;
      objectCounter = 0;
/*       TOUT((stderr,"numobjects %d\n",$2)); */
      if ( $2 > nsurfaces ) {
	yyerror("more objects than surfaces ?");    
      }
    }
objects
;

/* several of these bindings can be specified */
objects 
  : objects one_object
  | one_object
;

/* parse the binding of a property to one surface */
one_object
  : OBJECT index index
    {
      if ( objectCounter >= nobjs ) {
	yywarn("too many objects specified (ignoring)");    
      } else {
	if ( $2 > nsurfaces ) {
	  yyerror("surface index out of range");    
	}
	if ( $3 > nproperties ) {
	  yyerror("property index out of range");    
	}
	(g_objectList[$2 - 1])->setProperties(g_propList[$3 - 1]);
	objectCounter++;
      }
    }
;

/* start parsing the description of the light sources */
lighting_section
  : NUMLIGHTS index 
    {
      nlights = $2;
      /* preallocate memory */
      g_lightList.reserve($2);
      lightCounter = 0;
/*       TOUT((stderr,"numlights %d\n",$2)); */
    }
lights
; 

/* several lights may be specified */
lights 
  : lights one_light
  | one_light
;

/* parse the description of one light and save it for later use */
one_light : direction colour
{   /* move on to the next one... */
  if ( lightCounter >= nlights ) {
    yywarn("too many lights specified (ignorning)");
  } else {
    g_lightList.push_back(new LightObject(currentDirection, currentColor));
    lightCounter++;
  }
}
;

/* the direction of the light */
direction : DIRECTION realVal realVal realVal
{
  if ( lightCounter < nlights ) {
    currentDirection = Vec3d($2, $3, $4);
  }
}
;

/* the color of the light */
colour : COLOUR colourVal colourVal colourVal
{
  if ( lightCounter < nlights ) {
    currentColor = Color($2, $3, $4);
  }
}
;

/*
 * some minor type conversions and range checks...
 */

colourVal : zeroToOneVal ;

zeroToOneVal : realVal
{
  if ( $1 < 0.0 || $1 > 1.0 ) {
    yyerror("value out of range (only 0 to 1 allowed)");
  }

  /* pass that value up the tree */
  $$ = $1;
}
;

angleVal : realVal
{
  if ( $1 < 0.0 || $1 > 180.0 ) {
    yyerror("value out of range (only 0 to 180 degrees allowed)");
  }
  
  /* pass that value up the tree */
  $$ = $1;
}
;

realVal 
  : FLOAT
    { $$ = $1; }
  | INTEGER
    { $$ = (float) $1; /* conversion from integers */ }
;

/* make sure each index is positive */
index : INTEGER
{
  if ( $1 < 1 ) {
    yyerror("index out of range (only 1 or more allowed)");
  }
  
  /* pass that value up the tree */
  $$ = $1;
}
;
    
%%

/* -----------------------------------------------------------------------------
 * That's all folks.
 * What's left is the implementation of the tokenizer.
 * Nothing to do here, though.
 */

static
int yylex(void)
{
  char token[100];      /* buffer storing string representing token */
  int keywordkind;      /* which keyword did we get? */

  /* first, skip any comments/whitespace that might be sitting there */
  /* if there isn't any left, then we're at the end of the file */

  while ( skipWhitespace() ) {
    /* figure out what kind of token we've just run into */
    switch ( collectToken(token) ) {
      case CHARTOKEN:
	/* okay, see if it's a keyword we know.  if not, complain */
	keywordkind = checkKeyword(token);
	if ( keywordkind == -1 ) {
	  complain("unknown keyword :",token);
	}
/* 	TOUT((stderr,"keyword %s\n",token)); */
	return keywordkind;
	break;
	    
      case INTTOKEN:
	yylval.intval = (unsigned int) atoi(token);
/* 	TOUT((stderr,"integer token=<%s>, value %d\n",token, yylval.intval)); */
	return INTEGER;
	    
      case FLOATTOKEN:
	yylval.floatval = (float) atof(token);
/* 	TOUT((stderr,"float token=<%s>, value %f\n",token, yylval.floatval)); */
	return FLOAT;
	    
      default:
	complain("unrecognized token :",token);
	break;
    }
  }
  return 0;
}

