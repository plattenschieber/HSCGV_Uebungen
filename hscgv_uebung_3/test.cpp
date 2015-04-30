#include <stdlib.h>
#include "SDL.h"

int main()
{
        SDL_Event event;
        int quit = 0;

        if( SDL_Init( SDL_INIT_VIDEO ) < 0 )
        { 
          fprintf( stderr, "Konnte SDL nicht initialisieren: %s\n", SDL_GetError() );  
          exit( -1 );  
        }
        
        if( SDL_SetVideoMode( 320, 200, 8, SDL_SWSURFACE ) < 0)
        { 
          fprintf( stderr, "Konnte video mode nicht setzen: %s\n", SDL_GetError() ); 
        }
        
        atexit(SDL_Quit);

        while( quit == 0)
        {
            while( SDL_PollEvent( &event ) )
            {
                switch( event.type )
                {
                  case SDL_KEYDOWN:
                    printf( "Press: " );
                    printf( " Name: %s\n", SDL_GetKeyName( event.key.keysym.sym ) );
                  break;
                    
                  case SDL_KEYUP:                    
                    printf( "Release: " );
                    printf( " Name: %s\n", SDL_GetKeyName( event.key.keysym.sym ) );
                  break;

                  case SDL_QUIT:  // SDL_QUIT  int ein schliessen des windows
                    quit = 1;
                  break;

                  default:
                  break;
                }
            }
        }
  exit( 0 );
}
