/** \file
 * \brief main function
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 4 - "LBM"
 *
 * Created by Stefan Zellmann <zellmans@uni-koeln.de> and
 * Martin Aumueller <aumueller@uni-koeln.de>
 */

#include <QtGui/QApplication>
#include "applicationwindow.h"
#include "lbmCu.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    new ApplicationWindow;

    a.connect( &a, SIGNAL(lastWindowClosed()), &a, SLOT(quit()) );
    launchKernel();

    return a.exec();
}
