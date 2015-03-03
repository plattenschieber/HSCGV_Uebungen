/** \file
 * \brief Real-time Clock
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 4 - "LBM"
 *
 * Created by Martin Aumueller <aumueller@uni-koeln.de>
 */

#include <osg/Timer>

#include "clock.h"

double Clock::now()
{
    return osg::Timer::instance()->time_s();
}
