/** \file
 * \brief Real-time Clock
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 4 - "LBM"
 *
 * Created by Martin Aumueller <aumueller@uni-koeln.de>
 */

#ifndef CLOCK_H
#define CLOCK_H

//! Real-time clock
class Clock
{
    public:
        //! current wall-clock time
        static double now();
};

#endif
