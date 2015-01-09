TARGET=         a2

HEADERS=        Grabber.h Gameboard.h
SOURCES=        main.cpp Grabber.cpp Gameboard.cpp

QMAKE_LFLAGS    += -F/opt/local/Library/Frameworks		
LIBS            += -framework Inventor -framework SoQt

#QMAKE_CFLAGS    *= -Wold-style-cast
#QMAKE_CXXFLAGS    *= -Wold-style-cast
#QMAKE_CXXFLAGS    *= -Weffc++  -Wno-deprecated  -Woverloaded-virtual -Wno-pmf-conversions -Wsign-promo  -Wsynth -Wno-system-headers

CONFIG          *= opengl static
